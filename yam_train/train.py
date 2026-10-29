"""Train ACT on the converted dataset.

Single GPU :  python -m yam_train.train --data DATA --out runs/NAME
4 GPUs     :  torchrun --nproc_per_node 4 -m yam_train.train --data DATA --out runs/NAME

Writes into --out:
    config.json        all hyper-parameters
    metrics.jsonl      one JSON line per log event (train every --log-every, val every --val-every)
    curves.png         loss curves (train vs val) — refreshed at each validation
    tb/                TensorBoard events (if tensorboard is installed)
    ckpt/step_XXXXX.pt , ckpt/last.pt , ckpt/best_val.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, Subset

from yam_train.data.dataset import ChunkDataset
from yam_train.model.act import ACT, ACTConfig, act_loss


# ---------------------------------------------------------------- augmentation
def augment(images: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    """images float (B, n_cam, 3, H, W) in [0,1].  Small random shift/scale/rotation +
    brightness/contrast/saturation jitter.  NO hue jitter: colour identifies the blue block/cup."""
    b, n, c, h, w = images.shape
    x = images.reshape(b * n, c, h, w)
    dev = x.device
    k = b * n
    scale = torch.empty(k, device=dev).uniform_(0.92, 1.0)
    angle = torch.empty(k, device=dev).uniform_(-3, 3) * math.pi / 180
    tx = torch.empty(k, device=dev).uniform_(-0.06, 0.06)
    ty = torch.empty(k, device=dev).uniform_(-0.06, 0.06)
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    bright = torch.empty(k, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
    contrast = torch.empty(k, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
    sat = torch.empty(k, 1, 1, 1, device=dev).uniform_(0.85, 1.15)
    x = x * bright
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * contrast + mean
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (x - gray) * sat + gray
    return x.clamp(0, 1).reshape(b, n, c, h, w)


def to_device(batch: dict, device, train: bool):
    imgs = batch["images"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3).float() / 255.0
    if train:
        imgs = augment(imgs)
    qpos = batch["qpos"].to(device, non_blocking=True)
    if train:
        qpos = qpos + 0.01 * torch.randn_like(qpos)  # tiny state noise -> robustness
    return imgs, qpos, batch["task"].to(device), batch["actions"].to(device), batch["is_pad"].to(device)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loader, device, a_std: torch.Tensor, max_batches: int) -> dict:
    """Inference-mode (z=0) L1 error.  Also reports error in real units (rad / gripper 0..1)."""
    model.eval()
    tot, n = 0.0, 0
    per_dim = torch.zeros(a_std.numel(), device=device)
    first = torch.zeros(a_std.numel(), device=device)
    n_valid = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        imgs, qpos, task, actions, is_pad = to_device(batch, device, train=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred, _, _ = model(imgs, qpos, task)
        pred = pred.float()
        err = (pred - actions).abs() * (~is_pad)[..., None]
        tot += err.sum().item() / actions.shape[-1]
        n += (~is_pad).sum().item()
        per_dim += (err * a_std).sum(dim=(0, 1))
        first += ((pred[:, 0] - actions[:, 0]).abs() * a_std).sum(0)
        n_valid += actions.shape[0]
    model.train()
    per_dim = (per_dim / max(n, 1)).cpu().numpy()
    first = (first / max(n_valid, 1)).cpu().numpy()
    return {
        "l1": tot / max(n, 1),
        "right_arm_rad": float(per_dim[7:13].mean()),
        "right_grip": float(per_dim[13]),
        "left_arm_rad": float(per_dim[0:6].mean()),
        "left_grip": float(per_dim[6]),
        "first_step_right_arm_rad": float(first[7:13].mean()),
    }


def plot_curves(out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    try:
        rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError):
        return
    tr = [r for r in rows if r["kind"] == "train"]
    va = [r for r in rows if r["kind"] == "val"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    if tr:
        s = np.array([r["step"] for r in tr])
        ax[0].plot(s, [r["l1"] for r in tr], alpha=0.4, label="train L1 (with z from CVAE)")
        ax[0].plot(s, [r["kl"] for r in tr], alpha=0.4, label="train KL")
        ax[2].plot(s, [r["lr"] for r in tr])
        ax[2].set_title("learning rate")
    if va:
        s = np.array([r["step"] for r in va])
        ax[0].plot(s, [r["val"]["l1"] for r in va], "o-", label="VAL L1 (z=0, like robot)")
        ax[0].plot(s, [r["train_eval"]["l1"] for r in va], "s-", label="TRAIN L1 (z=0)")
        ax[1].plot(s, [r["val"]["right_arm_rad"] for r in va], "o-", label="val right arm (rad)")
        ax[1].plot(s, [r["train_eval"]["right_arm_rad"] for r in va], "s--", label="train right arm (rad)")
        ax[1].plot(s, [r["val"]["right_grip"] for r in va], "o-", label="val right gripper")
        ax[1].plot(s, [r["train_eval"]["right_grip"] for r in va], "s--", label="train right gripper")
    ax[0].set_yscale("log")
    ax[0].set_title("loss (normalised units)")
    ax[1].set_title("mean |error| over the action chunk, real units")
    for a in ax:
        a.set_xlabel("step")
        a.grid(alpha=0.3)
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)


class MetricsLog:
    """Append-only JSONL writer; a no-op on non-main DDP ranks."""

    def __init__(self, path: Path | None):
        self.path = path

    def write(self, record: dict) -> None:
        if self.path is None:
            return
        try:
            with self.path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            print(f"warning: could not write metrics: {exc}", flush=True)


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=32, help="per GPU")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-backbone", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--kl-weight", type=float, default=10.0)
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument("--dec-layers", type=int, default=1)
    p.add_argument("--train-split", choices=["train", "all"], default="train",
                   help="all = also train on the validation episodes (final model; val metrics then optimistic)")
    p.add_argument("--action-mode", choices=["abs", "delta"], default="abs",
                   help="abs: absolute leader joints; delta: leader joints minus current follower state")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-batches", type=int, default=60)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default="")
    args = p.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed + rank)
    is_main = rank == 0
    out = Path(args.out)
    if is_main:
        (out / "ckpt").mkdir(parents=True, exist_ok=True)

    train_ds = ChunkDataset(args.data, args.train_split, args.chunk, action_mode=args.action_mode)
    val_ds = ChunkDataset(args.data, "val", args.chunk, action_mode=args.action_mode,
                          action_stats=train_ds.action_stats)
    meta = train_ds.meta
    cfg = ACTConfig(
        n_tasks=len(meta["tasks"]), n_cameras=len(meta["cameras"]), chunk_size=args.chunk,
        n_decoder_layers=args.dec_layers,
    )
    model = ACT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    sampler = DistributedSampler(train_ds, world, rank, shuffle=True, seed=args.seed, drop_last=True) if ddp else None
    loader = DataLoader(
        train_ds, batch_size=args.batch, sampler=sampler, shuffle=sampler is None, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4,
    )
    # fixed random subsets for comparable train-vs-val evaluation
    rng = np.random.default_rng(0)
    n_eval = args.val_batches * args.batch
    val_sub = Subset(val_ds, rng.choice(len(val_ds), min(n_eval, len(val_ds)), replace=False).tolist())
    tr_sub = Subset(train_ds, rng.choice(len(train_ds), min(n_eval, len(train_ds)), replace=False).tolist())
    val_loader = DataLoader(val_sub, batch_size=args.batch, num_workers=4)
    tr_eval_loader = DataLoader(tr_sub, batch_size=args.batch, num_workers=4)
    a_std = torch.tensor(train_ds.a_std, device=device)

    opt = torch.optim.AdamW(
        [
            {"params": model.other_parameters(), "lr": args.lr},
            {"params": list(model.backbone_parameters()), "lr": args.lr_backbone},
        ],
        weight_decay=args.weight_decay,
    )
    base_lrs = [args.lr, args.lr_backbone]

    def lr_factor(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    step = 0
    best_val = float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = ck["step"]
        best_val = ck.get("best_val", best_val)
    dmodel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local]) if ddp else model

    ckpt_common = {
        "config": cfg.to_dict(),
        "stats": meta["stats"],
        "tasks": meta["tasks"],
        "cameras": meta["cameras"],
        "image_size": meta["image_size"],
        "fps": meta["fps"],
        "start_pose": meta["start_pose"],
        "state_layout": meta["state_layout"],
        "action_mode": args.action_mode,
        "action_norm": train_ds.action_stats,  # stats of what the network outputs (abs or delta)
    }

    def save(path: Path):
        torch.save({**ckpt_common, "model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                    "best_val": best_val, "args": vars(args)}, path)

    tb = None
    if is_main:
        (out / "config.json").write_text(json.dumps({
            "args": vars(args), "model": cfg.to_dict(), "params_M": n_params / 1e6, "world_size": world,
            "train_steps_per_epoch": len(loader), "train_samples": len(train_ds), "val_samples": len(val_ds),
            "train_episodes": len(train_ds.eps), "val_episodes": [e["name"] for e in val_ds.eps],
        }, indent=1))
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb = SummaryWriter(str(out / "tb"))
        except Exception:  # noqa: BLE001 - tensorboard is optional
            tb = None
        print(f"params {n_params / 1e6:.1f}M  train {len(train_ds)}  val {len(val_ds)}  "
              f"steps/epoch {len(loader)}  world {world}", flush=True)

    metrics_f = MetricsLog(out / "metrics.jsonl" if is_main else None)
    epoch = step // max(1, len(loader))
    t_last = time.time()
    acc = {"loss": 0.0, "l1": 0.0, "kl": 0.0, "gn": 0.0, "n": 0}
    dmodel.train()
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if step >= args.steps:
                break
            f = lr_factor(step)
            for g, base in zip(opt.param_groups, base_lrs, strict=True):
                g["lr"] = base * f
            imgs, qpos, task, actions, is_pad = to_device(batch, device, train=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred, mu, logvar = dmodel(imgs, qpos, task, actions)
            loss, l1, kl = act_loss(pred.float(), actions, is_pad, mu.float(), logvar.float(), args.kl_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            step += 1
            acc["loss"] += loss.item()
            acc["l1"] += l1.item()
            acc["kl"] += kl.item()
            acc["gn"] += gn.item()
            acc["n"] += 1

            if is_main and step % args.log_every == 0:
                dt = time.time() - t_last
                t_last = time.time()
                rec = {k: acc[k] / acc["n"] for k in ("loss", "l1", "kl", "gn")}
                rec.update(kind="train", step=step, epoch=epoch, lr=opt.param_groups[0]["lr"],
                           samples_per_s=acc["n"] * args.batch * world / dt, time=time.time())
                metrics_f.write(rec)
                if tb:
                    for k in ("loss", "l1", "kl", "gn", "lr", "samples_per_s"):
                        tb.add_scalar(f"train/{k}", rec[k], step)
                acc = dict.fromkeys(acc, 0.0)
                acc["n"] = 0
                if step % (args.log_every * 10) == 0:
                    print(f"step {step} ep {epoch} loss {rec['loss']:.4f} l1 {rec['l1']:.4f} kl {rec['kl']:.3f} "
                          f"lr {rec['lr']:.2e} {rec['samples_per_s']:.0f} samp/s", flush=True)

            if is_main and (step % args.val_every == 0 or step == args.steps):
                v = evaluate(model, val_loader, device, a_std, args.val_batches)
                te = evaluate(model, tr_eval_loader, device, a_std, args.val_batches)
                metrics_f.write({"kind": "val", "step": step, "epoch": epoch, "val": v,
                                 "train_eval": te, "time": time.time()})
                if tb:
                    for k, val in v.items():
                        tb.add_scalar(f"val/{k}", val, step)
                    for k, val in te.items():
                        tb.add_scalar(f"train_eval/{k}", val, step)
                print(f"[val] step {step}  val l1 {v['l1']:.4f}  train l1 {te['l1']:.4f}  "
                      f"val right-arm {v['right_arm_rad']:.4f} rad  grip {v['right_grip']:.4f}", flush=True)
                if v["l1"] < best_val:
                    best_val = v["l1"]
                    save(out / "ckpt" / "best_val.pt")
                plot_curves(out)
            if is_main and (step % args.save_every == 0 or step == args.steps):
                save(out / "ckpt" / f"step_{step:06d}.pt")
                save(out / "ckpt" / "last.pt")
        epoch += 1

    if is_main:
        print("training finished", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
