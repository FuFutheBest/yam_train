"""Open-loop check on recorded episodes: predicted vs human (leader) actions.

    python -m yam_train.tools.eval_offline --ckpt runs/X/ckpt/best.pt --data DATA --out DIR \
        [--split val|train|all] [--k 0.01 | --no-agg]

At every timestep the recorded images + follower state are fed (z = 0) and the executed action is chosen
like imitate_episodes.py::eval_bc:
    temporal aggregation (default): average all chunks covering t, weights exp(-k * i), i = 0 is the OLDEST
    --no-agg                         : query every chunk_size steps and play the chunk
The target for step t is the label the model was trained on: leader(t - action_shift).
Writes one PNG per episode + summary.json (mean |error| per joint).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from yam_train.data.dataset import EpisodicDataset, FrameStore
from yam_train.policy import ACTPolicy
from yam_train.train import CKPT_FORMAT, to_absolute

JOINT_NAMES = [f"L{i}" for i in range(6)] + ["L grip"] + [f"R{i}" for i in range(6)] + ["R grip"]


def load_policy(path: str, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != CKPT_FORMAT:
        raise SystemExit(f"{path}: not a {CKPT_FORMAT} checkpoint")
    pc = dict(ck["config"]["policy"])
    policy = ACTPolicy(pc, ck["camera_names"])
    policy.load_state_dict(ck["model"])
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--max-episodes", type=int, default=20)
    p.add_argument("--k", type=float, default=0.01)
    p.add_argument("--no-agg", action="store_true")
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, ck, stats = load_policy(args.ckpt, device)
    store = FrameStore(args.data)
    names = {"val": ck["val_episodes"], "train": ck["train_episodes"]}.get(args.split)
    eps = [e for e in store.episodes(ck["task"]) if names is None or e["name"] in names]
    ds = EpisodicDataset(store, eps, ck["camera_names"], stats, ck["chunk_size"], ck["action_shift"],
                         ck["action_mode"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shift, fps = ck["action_shift"], ck["fps"]
    summary = {}
    for e in eps[: args.max_episodes]:
        s, n = e["start"], e["length"]
        qpos, leader = store.qpos[s:s + n], store.action[s:s + n]
        chunks = []
        with torch.inference_mode():
            for t0 in range(0, n, args.batch):
                items = [ds.sample(e, t) for t in range(t0, min(n, t0 + args.batch))]
                img = torch.stack([it[0] for it in items]).to(device).float() / 255.0
                qn = torch.stack([it[1] for it in items]).to(device)
                pred = policy.predict(qn, img).float().cpu().numpy()
                chunks.append(to_absolute(pred, qpos[t0:t0 + len(items)], stats, ck["action_mode"]))
        chunks = np.concatenate(chunks)
        executed = execute(chunks, not args.no_agg, args.k)
        gt = leader[np.maximum(np.arange(n) - shift, 0)]
        err = np.abs(executed - gt).mean(0)
        first = np.abs(chunks[:, 0] - gt).mean(0)
        summary[e["name"]] = {nm: float(v) for nm, v in zip(JOINT_NAMES, err, strict=True)}
        summary[e["name"]]["first_step_right_arm"] = float(first[7:13].mean())

        fig, axes = plt.subplots(2, 7, figsize=(24, 6), sharex=True)
        tt = np.arange(n) / fps
        for dim, ax in enumerate(axes.flat):
            ax.plot(tt, gt[:, dim], "k-", lw=1.5, label="human (leader)")
            ax.plot(tt, executed[:, dim], "r-", lw=1, label="model")
            ax.plot(tt, qpos[:, dim], "b:", lw=1, label="follower state")
            ax.set_title(f"{JOINT_NAMES[dim]}  err {err[dim]:.3f}")
            ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=7)
        mode = "chunk playback" if args.no_agg else f"temporal agg k={args.k}"
        fig.suptitle(f"{e['name']}  open-loop, {mode}, epoch {ck['epoch']}")
        fig.tight_layout()
        fig.savefig(out / f"{e['name'].replace('/', '__')}.png", dpi=80)
        plt.close(fig)
        print(f"{e['name']}: right arm {err[7:13].mean():.4f} rad  grip {err[13]:.4f}  "
              f"first-step right {first[7:13].mean():.4f}")
    keys = [*JOINT_NAMES, "first_step_right_arm"]
    agg = {k: float(np.mean([v[k] for v in summary.values()])) for k in keys}
    (out / "summary.json").write_text(json.dumps({"ckpt": args.ckpt, "mean": agg, "episodes": summary}, indent=1))
    print(f"MEAN right arm {np.mean([agg[f'R{i}'] for i in range(6)]):.4f} rad  grip {agg['R grip']:.4f}  "
          f"first-step right {agg['first_step_right_arm']:.4f}")


if __name__ == "__main__":
    main()
