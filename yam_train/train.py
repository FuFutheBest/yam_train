"""Train ACT exactly like tonyzhaozh/act `imitate_episodes.py::main/train_bc`, configured by a YAML file.

    python -m yam_train.train --config configs/banana_abs.yaml [--set train.num_epochs=10 ...] [--out DIR]

One GPU per run (like the source); run different configs on different GPUs with CUDA_VISIBLE_DEVICES.

Per epoch (same order as the source): validation pass (posterior z, eval mode) -> best-by-val-loss
bookkeeping -> one pass over the training episodes (1 random timestep per episode).

Writes into --out (default runs/<config name>):
    config.yaml       resolved config + episode split
    metrics.jsonl     one line per epoch (train/val loss, l1, kl) + "extra" z=0 evaluations
    curves.png        refreshed at every checkpoint
    ckpt/epoch_XXXXX.pt, ckpt/last.pt, ckpt/best.pt (lowest val loss, like policy_best.ckpt)
best.pt is kept for parity with the source only: the val loss is 1 random frame per val episode
(10 samples) with the posterior z, so its minimum is mostly noise.  Choose among last.pt / epoch_*.pt
with `tools.eval_offline --ckpt A B C ...` (z = 0, temporal aggregation, every val frame).
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from yam_train.config import load_config
from yam_train.data.dataset import EpisodicDataset, FrameStore, chunk_targets, norm_stats, split_episodes
from yam_train.policy import ACTPolicy

CKPT_FORMAT = "yam_train/act_ref-v1"


def set_seed(seed: int) -> None:  # utils.py::set_seed
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_dict_mean(epoch_dicts: list[dict]) -> dict:  # utils.py::compute_dict_mean
    return {k: sum(d[k] for d in epoch_dicts) / len(epoch_dicts) for k in epoch_dicts[0]}


def augment_images(img: torch.Tensor, aug: dict) -> torch.Tensor:
    """v2 option (train.augment; null = source, no augmentation).  img (B, K, 3, H, W) in 0..1 on the GPU.
    Per image: random shift up to `shift` x size (edge-replicate pad, crop back; scale unchanged, so the
    robot's full frame stays in-distribution) + brightness/contrast/saturation jitter (x U[1-v, 1+v])."""
    b, k, c, h, w = img.shape
    x = img.reshape(b * k, c, h, w)
    n = x.shape[0]
    dev = x.device
    if aug.get("shift", 0) > 0:
        ph, pw = int(round(aug["shift"] * h)), int(round(aug["shift"] * w))
        padded = torch.nn.functional.pad(x, (pw, pw, ph, ph), mode="replicate")
        oy = torch.randint(0, 2 * ph + 1, (n,)).tolist()
        ox = torch.randint(0, 2 * pw + 1, (n,)).tolist()
        x = torch.stack([padded[i, :, oy[i]:oy[i] + h, ox[i]:ox[i] + w] for i in range(n)])

    def factor(v: float) -> torch.Tensor:
        return (1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * v) if v > 0 else torch.ones(1, device=dev)

    x = x * factor(aug.get("brightness", 0))
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * factor(aug.get("contrast", 0)) + mean
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (x - gray) * factor(aug.get("saturation", 0)) + gray
    return x.clamp(0, 1).reshape(b, k, c, h, w)


def forward_pass(data, policy, device, aug: dict | None = None):
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.to(device, non_blocking=True).float() / 255.0  # source: image_data / 255.0
    if aug:
        image_data = augment_images(image_data, aug)
    return policy(qpos_data.to(device), image_data, action_data.to(device), is_pad.to(device))


# ---------------------------------------------------------------- extra (not in the source) evaluation
def to_absolute(pred_norm: np.ndarray, qpos_raw: np.ndarray, stats: dict, mode: str) -> np.ndarray:
    """network output (..., K, 14) normalised -> absolute leader joint targets."""
    act = pred_norm * stats["action_std"] + stats["action_mean"]
    if mode == "delta":
        act = act + qpos_raw[..., None, :]
    return act


def fixed_samples(episodes: list[dict], n: int, seed: int = 12345) -> list[tuple[dict, int]]:
    """Deterministic (episode, start_ts) pairs; uses its own RNG so training randomness is untouched."""
    if not episodes:
        return []
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        ep = episodes[rng.randint(len(episodes))]
        out.append((ep, int(rng.randint(ep["length"]))))
    return out


@torch.inference_mode()
def extra_eval(policy, ds: EpisodicDataset, samples, device, batch: int) -> dict:
    """z = 0 (as on the robot) mean |error| vs the human, in rad / gripper units, over the chunk."""
    policy.eval()
    st, store = ds.norm_stats, ds.store
    err_sum = np.zeros(14)
    first_sum = np.zeros(14)
    n_valid = 0
    for i in range(0, len(samples), batch):
        part = samples[i : i + batch]
        items = [ds.sample(ep, t) for ep, t in part]
        img = torch.stack([it[0] for it in items]).to(device).float() / 255.0
        qn = torch.stack([it[1] for it in items]).to(device)
        pred = policy.predict(qn, img).float().cpu().numpy()
        for (ep, t), p in zip(part, pred, strict=True):
            s, n = ep["start"], ep["length"]
            q, a = store.qpos[s : s + n], store.action[s : s + n]
            gt, is_pad = chunk_targets(q, a, t, ds.chunk_size, ds.action_shift, "abs")
            pa = to_absolute(p, q[t], st, ds.action_mode)
            valid = ~is_pad
            err_sum += np.abs(pa - gt)[valid].sum(0)
            first_sum += np.abs(pa[0] - gt[0])
            n_valid += int(valid.sum())
    err, first = err_sum / max(n_valid, 1), first_sum / max(len(samples), 1)
    return {
        "right_arm_rad": float(err[7:13].mean()),
        "right_grip": float(err[13]),
        "left_arm_rad": float(err[0:6].mean()),
        "left_grip": float(err[6]),
        "first_step_right_arm_rad": float(first[7:13].mean()),
        "first_step_right_grip": float(first[13]),
    }


def plot_curves(out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    try:
        rows = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines() if x]
    except (OSError, json.JSONDecodeError):
        return
    ep_rows = [r for r in rows if r["kind"] == "epoch"]
    extra = [r for r in rows if r["kind"] == "extra"]
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.5))
    e = np.array([r["epoch"] for r in ep_rows])
    for i, key in enumerate(("loss", "l1", "kl")):
        ax[i].plot(e, [r["train"][key] for r in ep_rows], alpha=0.6, label="train")
        if ep_rows and ep_rows[0].get("val"):
            ax[i].plot(e, [r["val"][key] for r in ep_rows], alpha=0.6, label="validation")
        ax[i].set_yscale("log")
        ax[i].set_title(key)
    if extra:
        x = [r["epoch"] for r in extra]
        for split, style in (("val", "o-"), ("train", "s--")):
            if extra[0].get(split):
                for k in ("right_arm_rad", "first_step_right_arm_rad", "right_grip"):
                    ax[3].plot(x, [r[split][k] for r in extra], style, ms=3, label=f"{split} {k}")
        ax[3].set_title("z=0 mean |error| (rad / gripper)")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. train.seed=1")
    p.add_argument("--out", default="")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    d, pc, tc = cfg["data"], cfg["policy"], cfg["train"]
    if not d.get("task"):
        raise SystemExit("data.task is not set (one model per task)")
    out = Path(args.out or f"runs/{cfg['name']}")
    if (out / "metrics.jsonl").is_file() and (out / "metrics.jsonl").stat().st_size > 0:
        raise SystemExit(f"{out} already holds a run (metrics.jsonl); use another --out or remove it")
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    device = torch.device(tc.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    # ---- data (imitate_episodes.py::main -> set_seed(1); load_data)
    set_seed(d["split_seed"])
    store = FrameStore(d["root"])
    eps = store.episodes(d["task"])
    tr_idx, va_idx = split_episodes(len(eps), d["train_ratio"], d["split_seed"])
    train_eps, val_eps = [eps[i] for i in tr_idx], [eps[i] for i in va_idx]
    stats_eps = eps if d["norm_stats_from"] == "all" else train_eps
    stats = norm_stats(store, stats_eps, pc["chunk_size"], d["action_shift"], d["action_mode"])
    ds_kw = {
        "camera_names": d["camera_names"],
        "norm_stats": stats,
        "chunk_size": pc["chunk_size"],
        "action_shift": d["action_shift"],
        "action_mode": d["action_mode"],
    }
    train_ds = EpisodicDataset(store, train_eps, **ds_kw, qpos_noise=float(d.get("qpos_noise", 0.0)))
    val_ds = EpisodicDataset(store, val_eps, **ds_kw)
    nw = int(tc.get("num_workers", 1))
    dl_kw = {"batch_size": tc["batch_size"], "shuffle": True, "pin_memory": True, "num_workers": nw}
    if nw > 0:
        dl_kw.update(prefetch_factor=2, persistent_workers=True)
    train_loader = DataLoader(train_ds, **dl_kw)
    val_loader = DataLoader(val_ds, **dl_kw) if val_eps else None

    # ---- policy (train_bc -> set_seed(seed); make_policy; make_optimizer)
    set_seed(tc["seed"])
    policy = ACTPolicy(pc, d["camera_names"]).to(device)
    optimizer = policy.configure_optimizers()
    n_params = sum(x.numel() for x in policy.parameters() if x.requires_grad)

    split_info = {"train_episodes": [e["name"] for e in train_eps], "val_episodes": [e["name"] for e in val_eps]}
    (out / "config.yaml").write_text(
        yaml.safe_dump({**cfg, **split_info, "params_M": n_params / 1e6}, sort_keys=False, allow_unicode=True)
    )
    ckpt_common = {
        "format": CKPT_FORMAT,
        "config": cfg,
        "stats": {k: v.tolist() for k, v in stats.items()},
        "task": d["task"],
        "camera_names": list(d["camera_names"]),
        "image_size": store.meta["image_size"],
        "fps": store.meta["fps"],
        "chunk_size": pc["chunk_size"],
        "action_mode": d["action_mode"],
        "action_shift": d["action_shift"],
        "start_pose": np.mean([store.qpos[e["start"]] for e in eps], axis=0).tolist(),
        **split_info,
    }

    def save(path: Path, state_dict: dict, epoch: int) -> None:
        torch.save({**ckpt_common, "model": state_dict, "epoch": epoch}, path)

    extra_every = int(tc.get("extra_eval_every", 0))
    extra_val = fixed_samples(val_eps, int(tc.get("extra_eval_samples", 256)))
    extra_train = fixed_samples(train_eps, int(tc.get("extra_eval_samples", 256)))
    metrics = (out / "metrics.jsonl").open("a")
    print(
        f"task {d['task']}  action_mode {d['action_mode']}  params {n_params / 1e6:.2f}M  "
        f"train eps {len(train_eps)}  val eps {len(val_eps)}  device {device}",
        flush=True,
    )

    num_epochs = tc["num_epochs"]
    min_val_loss = np.inf
    best_ckpt_info = None
    t_start = time.time()
    for epoch in range(num_epochs):
        rec: dict = {"kind": "epoch", "epoch": epoch}
        # validation
        if val_loader is not None:
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = [forward_pass(data, policy, device) for data in val_loader]
                epoch_summary = compute_dict_mean(epoch_dicts)
                epoch_val_loss = epoch_summary["loss"]
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
            rec["val"] = {k: v.item() for k, v in epoch_summary.items()}

        # training
        policy.train()
        optimizer.zero_grad()
        train_dicts = []
        for data in train_loader:
            forward_dict = forward_pass(data, policy, device, tc.get("augment"))
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_dicts.append({k: v.detach() for k, v in forward_dict.items()})
        rec["train"] = {k: v.item() for k, v in compute_dict_mean(train_dicts).items()}
        rec["time"] = time.time() - t_start
        metrics.write(json.dumps(rec) + "\n")
        metrics.flush()

        last_epoch = epoch == num_epochs - 1
        if extra_every and (epoch % extra_every == 0 or last_epoch):
            ex = {
                "kind": "extra",
                "epoch": epoch,
                "val": extra_eval(policy, val_ds, extra_val, device, tc["batch_size"]) if extra_val else None,
                "train": extra_eval(policy, train_ds, extra_train, device, tc["batch_size"]),
            }
            metrics.write(json.dumps(ex) + "\n")
            metrics.flush()
            v = ex["val"] or ex["train"]
            print(
                f"  [z=0] epoch {epoch}  right arm {v['right_arm_rad']:.4f} rad  "
                f"first-step {v['first_step_right_arm_rad']:.4f}  grip {v['right_grip']:.4f}",
                flush=True,
            )
        if epoch % 10 == 0 or last_epoch:
            val_str = f"val {rec['val']['loss']:.5f}  " if "val" in rec else ""
            eta_h = rec["time"] / (epoch + 1) * (num_epochs - epoch - 1) / 3600
            print(
                f"epoch {epoch}  {val_str}train {rec['train']['loss']:.5f}  "
                f"(l1 {rec['train']['l1']:.4f} kl {rec['train']['kl']:.4f})  {rec['time'] / 60:.1f} min  "
                f"eta {eta_h:.1f} h",
                flush=True,
            )
        if epoch > 0 and epoch % tc["save_every"] == 0:
            save(out / "ckpt" / f"epoch_{epoch:05d}.pt", policy.state_dict(), epoch)
            plot_curves(out)

    save(out / "ckpt" / "last.pt", policy.state_dict(), num_epochs - 1)
    if best_ckpt_info is not None:
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        save(out / "ckpt" / "best.pt", best_state_dict, best_epoch)
        print(f"Training finished: seed {tc['seed']}, val loss {min_val_loss:.6f} at epoch {best_epoch}", flush=True)
    metrics.close()
    plot_curves(out)


if __name__ == "__main__":
    main()
