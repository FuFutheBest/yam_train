"""PyTorch dataset over the converted arrays (see yam_train.data.convert)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ChunkDataset(Dataset):
    """One sample = one timestep t of one episode.

    images  uint8 (n_cam, H, W, 3)       at t
    qpos    float (14,) normalised         follower state at t
    actions float (K, 14) normalised      leader targets t .. t+K-1 (padded with the last one)
    is_pad  bool (K,)                     True where the episode had already ended
    task    int
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        chunk_size: int,
        episodes: list[str] | None = None,
        action_mode: str = "abs",
        action_stats: dict | None = None,
    ):
        self.root = Path(root)
        try:
            self.meta = json.loads((self.root / "meta.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot read dataset meta in {self.root}: {exc}"
            ) from exc
        self.chunk = chunk_size
        self.images = np.load(self.root / "images.npy", mmap_mode="r")
        self.qpos = np.load(self.root / "qpos.npy")
        self.action = np.load(self.root / "action.npy")
        st = self.meta["stats"]
        self.q_mean, self.q_std = np.array(st["qpos"]["mean"], np.float32), np.array(
            st["qpos"]["std"], np.float32
        )
        self.action_mode = action_mode
        if action_mode == "delta":
            # actions are expressed relative to the current follower state: a(t+k) - qpos(t)
            st = {
                "action": action_stats
                or delta_stats(self.meta, self.qpos, self.action, chunk_size)
            }
        elif action_mode != "abs":
            raise ValueError(f"unknown action_mode {action_mode}")
        self.action_stats = st["action"]
        self.a_mean, self.a_std = np.array(st["action"]["mean"], np.float32), np.array(
            st["action"]["std"], np.float32
        )

        self.eps = [
            e
            for e in self.meta["episodes"]
            if (split == "all" or e["split"] == split)
            and (episodes is None or e["name"] in episodes)
        ]
        rows, ep_end, task = [], [], []
        for e in self.eps:
            r = np.arange(e["start"], e["start"] + e["length"])
            rows.append(r)
            ep_end.append(np.full(len(r), e["start"] + e["length"]))
            task.append(np.full(len(r), e["task_id"]))
        self.rows = np.concatenate(rows)
        self.ep_end = np.concatenate(ep_end)
        self.task = np.concatenate(task)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r, end = int(self.rows[i]), int(self.ep_end[i])
        idx = np.minimum(np.arange(r, r + self.chunk), end - 1)
        is_pad = np.arange(r, r + self.chunk) >= end
        act = (
            self.action[idx] - self.qpos[r]
            if self.action_mode == "delta"
            else self.action[idx]
        )
        return {
            "images": torch.from_numpy(np.ascontiguousarray(self.images[r])),
            "qpos": torch.from_numpy((self.qpos[r] - self.q_mean) / self.q_std),
            "actions": torch.from_numpy((act - self.a_mean) / self.a_std),
            "is_pad": torch.from_numpy(is_pad),
            "task": torch.tensor(int(self.task[i])),
        }


def delta_stats(meta: dict, qpos: np.ndarray, action: np.ndarray, chunk: int) -> dict:
    """mean/std of a(t+k) - qpos(t) over the TRAIN episodes and all k < chunk."""
    deltas = []
    for e in meta["episodes"]:
        if e["split"] != "train":
            continue
        s, n = e["start"], e["length"]
        q, a = qpos[s : s + n], action[s : s + n]
        for k in range(0, chunk, 5):
            idx = np.minimum(np.arange(n) + k, n - 1)
            deltas.append(a[idx] - q)
    d = np.concatenate(deltas)
    return {
        "mean": d.mean(0).tolist(),
        "std": np.maximum(d.std(0), 1e-2).tolist(),
        "min": d.min(0).tolist(),
        "max": d.max(0).tolist(),
    }
