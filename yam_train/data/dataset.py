"""Dataset = tonyzhaozh/act `utils.py` (EpisodicDataset / get_norm_stats / load_data) on our converted data.

Kept from the source:
  * one item = one episode; a random start_ts is drawn on every __getitem__ (np.random.choice)
  * real-robot alignment: actions are read from max(0, start_ts - action_shift) (source: 1; our default 0)
  * padding with zeros + is_pad, mean/std normalisation, std clipped at 1e-2, torch.std (unbiased)
  * normalisation stats over all episodes of the task, 80/20 split from np.random.permutation after seed 1
Added:
  * action_mode "delta": every chunk step is relative to the follower state at start_ts
  * episodes have different lengths -> pad to chunk_size instead of episode_len (identical after the
    policy's `actions[:, :num_queries]`)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ACTION_MODES = ("abs", "delta")


class FrameStore:
    """Converted dataset (see yam_train.data.convert); JPEG files are opened lazily (DataLoader-worker safe)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        try:
            self.meta = json.loads((self.root / "meta.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read dataset meta in {self.root}: {exc}") from exc
        if self.meta.get("format") != "yam_train/jpeg-v1":
            raise RuntimeError(f"{self.root} is not a yam_train/jpeg-v1 dataset; re-run yam_train.data.convert")
        self.qpos = np.load(self.root / "qpos.npy")
        self.action = np.load(self.root / "action.npy")
        self.index = {c: np.load(self.root / f"{c}.idx.npy") for c in self.meta["cameras"]}
        self._bins: dict[str, np.memmap] = {}

    def episodes(self, task: str) -> list[dict]:
        eps = [e for e in self.meta["episodes"] if e["task"] == task]
        if not eps:
            raise ValueError(f"no episodes for task {task!r}; available: {self.meta['tasks']}")
        return sorted(eps, key=lambda e: e["name"])

    def jpeg(self, cam: str, row: int) -> np.ndarray:
        if cam not in self._bins:
            self._bins[cam] = np.memmap(self.root / f"{cam}.jpg.bin", dtype=np.uint8, mode="r")
        off, n = self.index[cam][row]
        return self._bins[cam][off:off + n]

    def image(self, cam: str, row: int) -> np.ndarray:
        """RGB uint8 (H, W, 3)."""
        import cv2

        bgr = cv2.imdecode(np.asarray(self.jpeg(cam, row)), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cannot decode {cam} frame at row {row}")
        return np.ascontiguousarray(bgr[:, :, ::-1])

    def __getstate__(self):  # do not ship open memmaps to DataLoader workers
        state = self.__dict__.copy()
        state["_bins"] = {}
        return state


def split_episodes(n: int, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """utils.py::load_data after imitate_episodes.py's set_seed(1)."""
    shuffled = np.random.RandomState(seed).permutation(n)
    k = int(train_ratio * n)
    return shuffled[:k], shuffled[k:]


def chunk_targets(qpos: np.ndarray, action: np.ndarray, start_ts: int, chunk: int, shift: int,
                  mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Raw (un-normalised) targets for one episode at start_ts -> (chunk, 14) float32, is_pad (chunk,)."""
    a0 = max(0, start_ts - shift)
    seg = action[a0:a0 + chunk]
    out = np.zeros((chunk, action.shape[1]), np.float32)
    out[:len(seg)] = seg
    if mode == "delta":
        out[:len(seg)] -= qpos[start_ts]
    is_pad = np.zeros(chunk, bool)
    is_pad[len(seg):] = True
    return out, is_pad


def norm_stats(store: FrameStore, eps: list[dict], chunk: int, shift: int, mode: str) -> dict:
    """utils.py::get_norm_stats (+ the delta variant over every (start_ts, k) the dataset can produce)."""
    if mode not in ACTION_MODES:
        raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {mode!r}")
    qs, acts, absolute = [], [], []
    for e in eps:
        s, n = e["start"], e["length"]
        q, a = store.qpos[s:s + n].astype(np.float64), store.action[s:s + n].astype(np.float64)
        qs.append(q)
        absolute.append(a)
        if mode == "abs":
            acts.append(a)
        else:
            for t in range(n):
                a0 = max(0, t - shift)
                acts.append(a[a0:a0 + chunk] - q[t])
    q, a, absolute = np.concatenate(qs), np.concatenate(acts), np.concatenate(absolute)

    def mean_std(x):
        return x.mean(0), np.clip(x.std(0, ddof=1), 1e-2, np.inf)

    a_mean, a_std = mean_std(a)
    q_mean, q_std = mean_std(q)
    return {
        "action_mean": a_mean.astype(np.float32), "action_std": a_std.astype(np.float32),
        "qpos_mean": q_mean.astype(np.float32), "qpos_std": q_std.astype(np.float32),
        # not used for training: absolute leader range, for the robot-side safety filter
        "leader_min": absolute.min(0).astype(np.float32), "leader_max": absolute.max(0).astype(np.float32),
    }


class EpisodicDataset(Dataset):
    def __init__(self, store: FrameStore, episodes: list[dict], camera_names: list[str], norm_stats: dict,
                 chunk_size: int, action_shift: int, action_mode: str, qpos_noise: float = 0.0):
        self.store = store
        self.episodes = episodes
        self.camera_names = list(camera_names)
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size
        self.action_shift = action_shift
        self.action_mode = action_mode
        # v2 option (0 = source): training-only Gaussian noise [rad] on the 12 arm joints of the qpos INPUT;
        # delta targets are taken relative to the noisy state, so the absolute target stays the human's
        # (DART-like: teaches "my state estimate is off -> still go to the right place").
        self.qpos_noise = float(qpos_noise)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index):
        ep = self.episodes[index]
        start_ts = np.random.choice(ep["length"])
        return self.sample(ep, int(start_ts), self.qpos_noise)

    def sample(self, ep: dict, start_ts: int, qpos_noise: float = 0.0):
        """(image uint8 (k, 3, H, W), qpos (14,), action (chunk, 14), is_pad (chunk,)), normalised like the source.
        Images stay uint8 here; train.py divides by 255 on the GPU (same numbers, 4x less IPC)."""
        s, n = ep["start"], ep["length"]
        q, a = self.store.qpos[s:s + n], self.store.action[s:s + n]
        action, is_pad = chunk_targets(q, a, start_ts, self.chunk_size, self.action_shift, self.action_mode)
        q_obs = q[start_ts]
        if qpos_noise > 0:
            eps = np.zeros(q_obs.shape, np.float32)
            eps[np.r_[0:6, 7:13]] = np.random.normal(0.0, qpos_noise, 12)
            q_obs = q_obs + eps
            if self.action_mode == "delta":  # target relative to the (noisy) observed state
                action[~is_pad] -= eps
        images = np.stack([self.store.image(c, s + start_ts) for c in self.camera_names])
        st = self.norm_stats
        image_data = torch.einsum("k h w c -> k c h w", torch.from_numpy(images))
        qpos_data = torch.from_numpy((q_obs - st["qpos_mean"]) / st["qpos_std"]).float()
        action_data = torch.from_numpy((action - st["action_mean"]) / st["action_std"]).float()
        return image_data, qpos_data, action_data, torch.from_numpy(is_pad)
