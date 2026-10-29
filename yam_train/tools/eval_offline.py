"""Open-loop check: replay held-out demos through the model and plot predicted vs human actions.

    python -m yam_train.tools.eval_offline --ckpt CKPT --data DATA --out DIR [--split val]

For each episode: at every step feed the recorded images + follower state, take the
temporally-ensembled action (same as the robot does) and compare with the leader (human) action.
Writes one PNG per episode + summary.json (mean abs error per joint).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from yam_train.data.dataset import ChunkDataset
from yam_train.model.act import ACT, ACTConfig

JOINT_NAMES = (
    [f"L{i}" for i in range(6)] + ["L grip"] + [f"R{i}" for i in range(6)] + ["R grip"]
)


def ensembled(chunks: np.ndarray, m: float = -0.3) -> np.ndarray:
    """chunks (T,K,A) predicted at each step -> (T,A) temporally ensembled actions."""
    t_len, k, _ = chunks.shape
    out = np.zeros(chunks.shape[::2])
    for t in range(t_len):
        idx = [(s, t - s) for s in range(max(0, t - k + 1), t + 1)]
        w = np.exp(-m * np.arange(len(idx)))
        vals = np.stack([chunks[s, j] for s, j in idx])
        out[t] = (vals * w[:, None]).sum(0) / w.sum()
    return out


class Checkpoint:
    """Trained model + the normalisation stats stored in the checkpoint."""

    def __init__(self, path: str | Path):
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = ACTConfig(**{**ck["config"], "pretrained_backbone": False})
        self.model = ACT(cfg)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.chunk_size = cfg.chunk_size
        self.tasks: list[str] = ck["tasks"]
        self.fps = float(ck["fps"])
        st = ck["stats"]
        self.q_mean, self.q_std = np.asarray(st["qpos"]["mean"]), np.asarray(
            st["qpos"]["std"]
        )
        self.action_mode = ck.get("action_mode", "abs")
        norm = ck.get(
            "action_norm", st["action"]
        )  # stats of the network output (abs or delta)
        self.a_mean, self.a_std = np.asarray(norm["mean"]), np.asarray(norm["std"])

    def to_absolute(self, pred_norm: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """normalised network output (..., K, 14) -> absolute joint targets."""
        act = pred_norm * self.a_std + self.a_mean
        if self.action_mode == "delta":
            act = act + np.asarray(qpos)[..., None, :]
        return act


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-episodes", type=int, default=20)
    p.add_argument("--ensemble-m", type=float, default=-0.3)
    args = p.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy = Checkpoint(args.ckpt)
    ds = ChunkDataset(
        args.data, args.split, policy.chunk_size
    )  # abs actions = ground truth
    summary = {}
    for e in ds.eps[: args.max_episodes]:
        rows = np.arange(e["start"], e["start"] + e["length"])
        qpos = ds.qpos[rows].astype(np.float64)
        gt = ds.action[rows].astype(np.float64)
        chunks = []
        for i0 in range(0, len(rows), 64):
            r = rows[i0 : i0 + 64]
            imgs = torch.from_numpy(np.ascontiguousarray(ds.images[r])).to(
                policy.device
            )
            imgs = imgs.permute(0, 1, 4, 2, 3).float() / 255.0
            q = torch.from_numpy(
                ((qpos[i0 : i0 + 64] - policy.q_mean) / policy.q_std).astype(np.float32)
            )
            t = torch.full(
                (len(r),), policy.tasks.index(e["task"]), device=policy.device
            )
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=policy.device.type == "cuda"
            ):
                pred, _, _ = policy.model(imgs, q.to(policy.device), t)
            chunks.append(
                policy.to_absolute(pred.float().cpu().numpy(), qpos[i0 : i0 + 64])
            )
        chunks = np.concatenate(chunks)
        ens = ensembled(chunks, args.ensemble_m)
        err = np.abs(ens - gt).mean(0)
        first_err = np.abs(chunks[:, 0] - gt).mean(0)
        summary[e["name"]] = {
            n: float(v) for n, v in zip(JOINT_NAMES, err, strict=True)
        }

        fig, axes = plt.subplots(2, 7, figsize=(24, 6), sharex=True)
        tt = np.arange(len(rows)) / policy.fps
        for d, ax in enumerate(axes.flat):
            ax.plot(tt, gt[:, d], "k-", lw=1.5, label="human (leader)")
            ax.plot(tt, ens[:, d], "r-", lw=1, label="model")
            ax.plot(tt, qpos[:, d], "b:", lw=1, label="follower state")
            ax.set_title(f"{JOINT_NAMES[d]}  err {err[d]:.3f}")
            ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{e['name']}  (open-loop on recorded observations)")
        fig.tight_layout()
        fig.savefig(out / f"{e['name'].replace('/', '__')}.png", dpi=80)
        plt.close(fig)
        summary[e["name"]]["first_step_right_arm"] = float(first_err[7:13].mean())
        summary[e["name"]]["first_step_left_arm"] = float(first_err[0:6].mean())
        print(
            f"{e['name']}: ensembled right arm {err[7:13].mean():.4f} rad, grip {err[13]:.4f} | "
            f"first-step right {first_err[7:13].mean():.4f} left {first_err[0:6].mean():.4f}"
        )
    keys = [*JOINT_NAMES, "first_step_right_arm", "first_step_left_arm"]
    agg = {n: float(np.mean([s[n] for s in summary.values()])) for n in keys}
    (out / "summary.json").write_text(
        json.dumps({"mean": agg, "episodes": summary}, indent=1)
    )
    print(
        f"MEAN ensembled right arm {np.mean([agg[f'R{i}'] for i in range(6)]):.4f}  "
        f"left arm {np.mean([agg[f'L{i}'] for i in range(6)]):.4f}  right grip {agg['R grip']:.4f}  "
        f"first-step right {agg['first_step_right_arm']:.4f} left {agg['first_step_left_arm']:.4f}"
    )


if __name__ == "__main__":
    main()
