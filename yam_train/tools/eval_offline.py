"""Open-loop check on recorded episodes: predicted vs human (leader) actions, for one or MANY checkpoints.

    python -m yam_train.tools.eval_offline --data DATA --out DIR --ckpt A.pt [B.pt ...] \
        [--split val|train|all] [--max-episodes N] [--k 0.01 | --no-agg] [--plots 3]

Use it to choose the checkpoint (instead of best.pt): e.g. every epoch_*.pt + last.pt of a run, or the
abs and delta runs of one task side by side.  Every frame of every selected episode is fed (z = 0, as on
the robot) and the executed action is chosen like imitate_episodes.py::eval_bc:
    temporal aggregation (default): average all chunks covering t, weights exp(-k * i), i = 0 is the OLDEST
    --no-agg                         : query every chunk_size steps and play the chunk
The target for step t is the label the model was trained on: leader(t - action_shift).
Images are decoded once per episode and shared by all checkpoints.

--split val (default) uses the validation episodes of the first checkpoint and refuses checkpoints that
were trained on any of them (e.g. a data.train_ratio=1.0 run).
Writes compare.md / compare.json (mean |error| per checkpoint and per episode) and --plots episode PNGs.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from yam_train.data.dataset import FrameStore
from yam_train.policy import ACTPolicy
from yam_train.train import CKPT_FORMAT, to_absolute

JOINT_NAMES = [f"L{i}" for i in range(6)] + ["L grip"] + [f"R{i}" for i in range(6)] + ["R grip"]


def load_policy(path: str, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != CKPT_FORMAT:
        raise SystemExit(f"{path}: not a {CKPT_FORMAT} checkpoint")
    pc = dict(ck["config"]["policy"])
    policy = ACTPolicy(pc, ck["camera_names"])
    policy.load_state_dict(ck.pop("model"))
    policy.to(device).eval()
    stats = {k: np.asarray(v, np.float32) for k, v in ck["stats"].items()}
    return policy, ck, stats


def execute(chunks: np.ndarray, agg: bool, k: float) -> np.ndarray:
    """chunks (T, K, A) predicted at every step -> (T, A) executed actions."""
    t_len, kk, _ = chunks.shape
    out = np.zeros((t_len, chunks.shape[2]))
    for t in range(t_len):
        if agg:
            starts = range(max(0, t - kk + 1), t + 1)  # oldest first, like all_time_actions[:, t]
            acts = np.stack([chunks[s, t - s] for s in starts])
            w = np.exp(-k * np.arange(len(acts)))
            out[t] = (acts * (w / w.sum())[:, None]).sum(0)
        else:
            s = t - t % kk
            out[t] = chunks[s, t - s]
    return out


def label(path: str) -> str:
    """runs/<run>/ckpt/<name>.pt -> <run>/<name>"""
    p = Path(path)
    return f"{p.parent.parent.name}/{p.stem}" if p.parent.name == "ckpt" else p.stem


def episode_images(store: FrameStore, cams: list[str], s: int, n: int, pool: ThreadPoolExecutor) -> torch.Tensor:
    """(n, n_cam, 3, H, W) uint8 for rows s..s+n-1 (decoded in parallel; cv2 releases the GIL)."""

    def one(row: int) -> np.ndarray:
        return np.stack([store.image(c, row) for c in cams])

    frames = np.stack(list(pool.map(one, range(s, s + n))))
    return torch.from_numpy(frames).permute(0, 1, 4, 2, 3).contiguous()


@torch.inference_mode()
def predict_chunks(policy, ck, stats, images: torch.Tensor, qpos: np.ndarray, device, batch: int) -> np.ndarray:
    """every frame -> absolute leader chunk (n, K, 14)"""
    qn = torch.from_numpy((qpos - stats["qpos_mean"]) / stats["qpos_std"]).float()
    out = []
    for i in range(0, len(qpos), batch):
        img = images[i : i + batch].to(device).float() / 255.0
        pred = policy.predict(qn[i : i + batch].to(device), img).float().cpu().numpy()
        out.append(to_absolute(pred, qpos[i : i + batch], stats, ck["action_mode"]))
    return np.concatenate(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, nargs="+")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--max-episodes", type=int, default=0, help="0 = all episodes of the split")
    p.add_argument("--k", type=float, default=0.01)
    p.add_argument("--no-agg", action="store_true")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--plots", type=int, default=3, help="episode PNGs to draw (0 = none)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [(label(c), *load_policy(c, device)) for c in args.ckpt]
    labels = [m[0] for m in models]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"checkpoint labels are not unique: {labels}")
    ck0 = models[0][2]
    for name, _pol, ck, _st in models[1:]:
        for key in ("task", "camera_names", "fps"):
            if ck[key] != ck0[key]:
                raise SystemExit(f"{name}: {key} {ck[key]!r} != {ck0[key]!r} of {labels[0]}")

    store = FrameStore(args.data)
    names = {"val": ck0["val_episodes"], "train": ck0["train_episodes"]}.get(args.split)
    if args.split == "val":
        if not names:
            raise SystemExit(f"{labels[0]} has no validation episodes (train_ratio 1.0?); put a split run first")
        for name, _pol, ck, _st in models:
            leaked = sorted(set(names) & set(ck["train_episodes"]))
            if leaked:
                raise SystemExit(
                    f"{name} was trained on {len(leaked)} of these validation episodes "
                    f"(e.g. {leaked[0]}); compare it on another split"
                )
    eps = [e for e in store.episodes(ck0["task"]) if names is None or e["name"] in names]
    if args.max_episodes:
        eps = eps[: args.max_episodes]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fps = ck0["fps"]
    per_ep: dict[str, dict[str, dict]] = {lb: {} for lb in labels}
    pool = ThreadPoolExecutor(16)
    for ei, e in enumerate(eps):
        s, n = e["start"], e["length"]
        qpos, leader = store.qpos[s : s + n], store.action[s : s + n]
        images = episode_images(store, ck0["camera_names"], s, n, pool)
        executed_all = {}
        for name, policy, ck, stats in models:
            chunks = predict_chunks(policy, ck, stats, images, qpos, device, args.batch)
            executed = execute(chunks, not args.no_agg, args.k)
            gt = leader[np.maximum(np.arange(n) - ck["action_shift"], 0)]
            err = np.abs(executed - gt).mean(0)
            first = np.abs(chunks[:, 0] - gt).mean(0)
            per_ep[name][e["name"]] = {
                **{nm: float(v) for nm, v in zip(JOINT_NAMES, err, strict=True)},
                "R arm": float(err[7:13].mean()),
                "L arm": float(err[0:6].mean()),
                "first-step R arm": float(first[7:13].mean()),
            }
            executed_all[name] = (executed, gt)
        print(
            f"[{ei + 1}/{len(eps)}] {e['name']}: "
            + "  ".join(f"{nm} {per_ep[nm][e['name']]['R arm']:.4f}" for nm in labels),
            flush=True,
        )
        if ei < args.plots:
            plot_episode(out, e["name"], executed_all, qpos, fps, args)
    pool.shutdown()

    keys = ["R arm", "R grip", "first-step R arm", "L arm", *JOINT_NAMES]
    mean = {nm: {k: float(np.mean([v[k] for v in per_ep[nm].values()])) for k in keys} for nm in labels}
    mode = "chunk playback" if args.no_agg else f"temporal agg k={args.k}"
    meta = {
        "split": args.split,
        "episodes": [e["name"] for e in eps],
        "execution": mode,
        "epochs": {nm: ck["epoch"] for nm, _p, ck, _s in models},
        "action_mode": {nm: ck["action_mode"] for nm, _p, ck, _s in models},
    }
    (out / "compare.json").write_text(json.dumps({**meta, "mean": mean, "per_episode": per_ep}, indent=1))
    lines = [
        (
            f"split `{args.split}` ({len(eps)} episodes), {mode}, z = 0; mean |executed - human| "
            "(rad / gripper 0..1), lower is better"
        ),
        "",
        "| checkpoint | epoch | mode | R arm | R grip | first-step R arm | L arm |",
        "|---|---|---|---|---|---|---|",
    ]
    for nm in labels:
        m = mean[nm]
        lines.append(
            f"| {nm} | {meta['epochs'][nm]} | {meta['action_mode'][nm]} | {m['R arm']:.4f} | "
            f"{m['R grip']:.4f} | {m['first-step R arm']:.4f} | {m['L arm']:.4f} |"
        )
    (out / "compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_episode(out: Path, ep_name: str, executed_all: dict, qpos: np.ndarray, fps: float, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 7, figsize=(26, 7), sharex=True)
    tt = np.arange(len(qpos)) / fps
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(executed_all)))
    gt = next(iter(executed_all.values()))[1]
    for dim, ax in enumerate(axes.flat):
        ax.plot(tt, gt[:, dim], "k-", lw=2, label="human (leader)")
        ax.plot(tt, qpos[:, dim], "k:", lw=1, label="follower state")
        for (name, (executed, _gt)), c in zip(executed_all.items(), colors, strict=True):
            ax.plot(tt, executed[:, dim], "-", color=c, lw=1, label=name)
        ax.set_title(JOINT_NAMES[dim])
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6)
    mode = "chunk playback" if args.no_agg else f"temporal agg k={args.k}"
    fig.suptitle(f"{ep_name}  open-loop, {mode}, z = 0")
    fig.tight_layout()
    fig.savefig(out / f"{ep_name.replace('/', '__')}.png", dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    main()
