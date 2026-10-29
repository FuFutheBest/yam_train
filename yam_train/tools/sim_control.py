"""Replay recorded episodes through the ROBOT's control stack (yam_infer) and compare with the human.

    python -m yam_train.tools.sim_control --ckpt A.pt [B.pt ...] --data DATA --out DIR \
        [--yam-infer ../yam_infer] [--split train|val|all] [--max-episodes N] \
        [--mode ensemble|chunk] [--feedback recorded|sim] [--latency-ms 30]

What is simulated (same objects / numbers as yam_infer/run.py):
    camera frame k arrives at T_k = k / fps  (recorded wrist JPEGs = what training saw)
    policy thread  : yam_infer.policy.policy.Policy (normalisation, delta -> absolute, z = 0);
                     ensemble: one prediction per frame; chunk: when scheduler.needs_new_chunk()
                     the chunk of frame k becomes available at T_k + latency, placed at t_obs = T_k
    control loop   : 100 Hz, yam_infer.control.scheduler.ActionScheduler.target_at -> SafetyFilter
                     (margin 0.15 rad, 1.5 rad/s); --freeze-left = run.yaml control.freeze_left
    qpos input     : recorded  = the recorded follower state (open loop, isolates the network + executor)
                     sim       = a simulated follower driven by OUR commands (first-order lag 0.13 s +
                                 33 ms delay + constant per-joint sag, fitted on the recordings:
                                 0.013 rad residual on the right arm), so the
                                 network sees the consequences of its own actions in the joint input.
                                 Images stay recorded -> only the proprioceptive half of the loop.
Compared with the human command leader(t - action_shift) at 100 Hz (right arm = the working arm):
    err        mean |command - human| over the episode
    lag_ms     time shift that best aligns command with human (+ = command is late)
    end_err    |final command - final human| (did it get where the human got?)
    jerk       mean |2nd difference| of the command (smoothness, rad / step^2 x 1e4)
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from yam_train.data.dataset import FrameStore

CONTROL_HZ = 100.0
RIGHT = np.r_[7:13]
LEFT = np.r_[0:7]


class SimFollower:
    """Follower joints: first-order lag tau toward the command, command delayed by `delay` s, plus the
    constant per-joint offset of the real arm (gravity sag, e.g. wrist joint3 -0.03..-0.06 rad; grippers 0).
    Without the offset a delta model would be re-anchored on a state the real arm never has and drift."""

    def __init__(self, state: np.ndarray, offset: np.ndarray, tau: float = 0.13, delay: float = 1 / 30):
        self.offset = np.asarray(offset, np.float64)
        self.x = np.asarray(state, np.float64) - self.offset  # internal = without the sag
        self.tau, self.delay = tau, delay
        self.cmds: list[tuple[float, np.ndarray]] = []

    def command(self, t: float, target: np.ndarray) -> None:
        self.cmds.append((t, target.copy()))

    def step(self, t: float, dt: float) -> np.ndarray:
        tgt = None
        for tc, c in reversed(self.cmds):
            if tc <= t - self.delay:
                tgt = c
                break
        if tgt is not None:
            self.x += (tgt - self.x) * (1 - np.exp(-dt / self.tau))
        return self.x + self.offset


def lag_filter(leader: np.ndarray, fps: float, tau: float = 0.13, delay_frames: int = 1, sub: int = 10) -> np.ndarray:
    """The SimFollower response (without offset) to a recorded leader trajectory."""
    out = np.empty_like(leader)
    x = leader[0].copy()
    lead = np.concatenate([np.repeat(leader[:1], delay_frames, 0), leader])[: len(leader)]
    a = 1 - np.exp(-1 / fps / sub / tau)
    for i in range(len(leader)):
        for _ in range(sub):
            x += (lead[i] - x) * a
        out[i] = x
    return out


def follower_offset(store: FrameStore, eps: list[dict], fps: float) -> np.ndarray:
    """mean(recorded follower - lag(leader)) per joint over the task's episodes; grippers 0."""
    diffs = []
    for e in eps:
        s, n = e["start"], e["length"]
        diffs.append(store.qpos[s:s + n] - lag_filter(store.action[s:s + n].astype(np.float64), fps))
    off = np.concatenate(diffs).mean(0)
    off[[6, 13]] = 0.0
    return off


def interp_rows(values: np.ndarray, fps: float, t: np.ndarray) -> np.ndarray:
    f = np.clip(t * fps, 0, len(values) - 1)
    lo = np.floor(f).astype(int)
    hi = np.minimum(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (f - lo)[:, None]


def batched_chunks(policy, images: np.ndarray, qpos: np.ndarray, batch: int = 32) -> np.ndarray:
    """= Policy.predict_prepared for every frame, batched (checked against it in main)."""
    out = []
    for i in range(0, len(qpos), batch):
        img = torch.from_numpy(images[i:i + batch]).to(policy.device).permute(0, 1, 4, 2, 3).float().div_(255.0)
        q = torch.from_numpy(((qpos[i:i + batch] - policy.q_mean) / policy.q_std).astype(np.float32)).to(policy.device)
        with torch.no_grad():
            raw = policy.net(q, img).float().cpu().numpy()
        out.append(policy.to_absolute(raw, qpos[i:i + batch]))
    return np.concatenate(out)


def simulate(policy, sched_cls, safety_cls, images: np.ndarray, qpos_rec: np.ndarray, fps: float, args,
             pre: np.ndarray | None, offset: np.ndarray) -> dict:
    """One episode through the control stack. `pre` = precomputed chunks (recorded-qpos mode)."""
    n = len(qpos_rec)
    sched = sched_cls(fps, policy.chunk_size, args.m, mode=args.mode, horizon=args.horizon, blend_s=args.blend_s)
    safety = safety_cls(policy.a_min, policy.a_max, 0.15, 1.5)
    hold = qpos_rec[0].copy()
    safety.reset(hold)
    follower = SimFollower(hold, offset)
    lat = args.latency_ms / 1000
    t_end = (n - 1) / fps
    taus = np.arange(0, t_end, 1 / CONTROL_HZ)
    cmds = np.empty((len(taus), 14))
    fol = np.empty((len(taus), 14))
    pending: list[tuple[float, float, np.ndarray]] = []  # (ready time, t_obs, chunk)
    fol_hist_t, fol_hist = [0.0], [hold.copy()]
    next_frame, last = 0, hold.copy()
    n_pred = 0
    for j, tau in enumerate(taus):
        # policy thread: frames that arrived by now
        while next_frame < n and next_frame / fps <= tau:
            k, t_k = next_frame, next_frame / fps
            next_frame += 1
            busy = any(r > t_k for r, _, _ in pending)
            if busy or not sched.needs_new_chunk(t_k):
                continue
            if pre is not None:
                chunk = pre[k]
            else:  # qpos = simulated follower at the image time
                q = np.array([np.interp(t_k, fol_hist_t, np.asarray(fol_hist)[:, d]) for d in range(14)])
                chunk = policy.predict_prepared(images[k], q, policy.tasks[0])
            pending.append((t_k + lat, t_k, chunk))
            n_pred += 1
        for item in [p for p in pending if p[0] <= tau]:
            sched.add(item[1], item[2])
            pending.remove(item)
        raw = sched.target_at(tau)
        target = last if raw is None else raw.copy()
        if args.freeze_left:  # yam_infer run.yaml control.freeze_left (currently false)
            target[LEFT] = hold[LEFT]
        safe = safety(target, 1 / CONTROL_HZ)
        last = safe
        cmds[j] = safe
        follower.command(tau, safe)
        fol[j] = follower.step(tau, 1 / CONTROL_HZ)
        fol_hist_t.append(tau + 1e-6)
        fol_hist.append(fol[j])
    return {"t": taus, "cmd": cmds, "follower": fol, "n_pred": n_pred}


def metrics(t: np.ndarray, cmd: np.ndarray, leader: np.ndarray, fps: float, shift: int) -> dict:
    human = interp_rows(leader, fps, t - shift / fps)
    err = np.abs(cmd - human)
    lags = np.arange(-30, 61) * 10  # ms
    lag_err = []
    for lag in lags:
        h = interp_rows(leader, fps, t - shift / fps - lag / 1000)
        lag_err.append(np.abs(cmd[:, RIGHT] - h[:, RIGHT]).mean())
    jerk = np.abs(np.diff(cmd[:, RIGHT], 2, axis=0)).mean() * 1e4
    return {
        "err": float(err[:, RIGHT].mean()), "grip_err": float(err[:, 13].mean()),
        "left_err": float(err[:, 0:6].mean()),
        "lag_ms": int(lags[int(np.argmin(lag_err))]), "err_at_best_lag": float(np.min(lag_err)),
        "end_err": float(np.abs(cmd[-1, RIGHT] - human[-1, RIGHT]).mean()),
        "jerk": float(jerk),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, nargs="+")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--yam-infer", default="../yam_infer", help="yam_infer checkout (its control code is used)")
    p.add_argument("--split", choices=["train", "val", "all"], default="train")
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--mode", choices=["ensemble", "chunk"], default="ensemble")
    p.add_argument("--m", type=float, default=0.01, help="ensemble weight exp(-m i), i = 0 oldest")
    p.add_argument("--horizon", type=int, default=30, help="chunk mode")
    p.add_argument("--blend-s", type=float, default=0.1, help="chunk mode")
    p.add_argument("--feedback", choices=["recorded", "sim"], default="recorded")
    p.add_argument("--latency-ms", type=float, default=30.0)
    p.add_argument("--freeze-left", action="store_true", help="hold the left arm (yam_infer control.freeze_left)")
    p.add_argument("--plots", type=int, default=3)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.yam_infer).resolve()))
    # sibling checkout, put on sys.path above (not a dependency of this package)
    from yam_infer.control.scheduler import ActionScheduler, SafetyFilter  # pyright: ignore[reportMissingImports]
    from yam_infer.policy.policy import Policy  # pyright: ignore[reportMissingImports]

    store = FrameStore(args.data)
    policies = {}
    for c in args.ckpt:
        pth = Path(c)
        name = f"{pth.parent.parent.name}/{pth.stem}" if pth.parent.name == "ckpt" else pth.stem
        policies[name] = (Policy(c, args.device, jpeg_roundtrip=False), torch.load(c, map_location="cpu",
                                                                                  weights_only=False, mmap=True))
    first_ck = next(iter(policies.values()))[1]
    task, cams, fps = first_ck["task"], first_ck["camera_names"], float(first_ck["fps"])
    for name, (pol, ck) in policies.items():
        if ck["task"] != task or list(ck["camera_names"]) != list(cams):
            raise SystemExit(f"{name}: different task / cameras")
    names = {"train": first_ck["train_episodes"], "val": first_ck["val_episodes"]}.get(args.split)
    eps = [e for e in store.episodes(task) if names is None or e["name"] in names]
    if args.max_episodes:
        eps = eps[: args.max_episodes]
    if not eps:
        raise SystemExit(f"no {args.split} episodes in {labels_of(policies)}")
    offset = follower_offset(store, store.episodes(task), fps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(16)
    results: dict[str, dict[str, dict]] = {nm: {} for nm in policies}
    checked = False
    for ei, e in enumerate(eps):
        s, n = e["start"], e["length"]
        qpos, leader = store.qpos[s:s + n].astype(np.float64), store.action[s:s + n].astype(np.float64)
        images = np.stack(list(pool.map(lambda r: np.stack([store.image(c, r) for c in cams]), range(s, s + n))))
        sims = {}
        for name, (pol, ck) in policies.items():
            pre = None
            if args.feedback == "recorded":
                pre = batched_chunks(pol, images, qpos)
                if not checked:  # batched path == the robot's per-frame path
                    ref = pol.predict_prepared(images[n // 2], qpos[n // 2], pol.tasks[0])
                    if np.abs(ref - pre[n // 2]).max() > 1e-3:
                        raise SystemExit("batched prediction differs from Policy.predict_prepared")
                    checked = True
            sim = simulate(pol, ActionScheduler, SafetyFilter, images, qpos, fps, args, pre, offset)
            results[name][e["name"]] = {**metrics(sim["t"], sim["cmd"], leader, fps, pol.action_shift),
                                        "n_pred": sim["n_pred"]}
            sims[name] = sim
        print(f"[{ei + 1}/{len(eps)}] {e['name']}: " + "  ".join(
            f"{nm} err {results[nm][e['name']]['err']:.4f} lag {results[nm][e['name']]['lag_ms']}ms"
            for nm in policies), flush=True)
        if ei < args.plots:
            plot(out, e["name"], sims, leader, qpos, fps, args)
    pool.shutdown()

    keys = ["err", "grip_err", "left_err", "lag_ms", "err_at_best_lag", "end_err", "jerk"]
    mean = {nm: {k: float(np.mean([v[k] for v in r.values()])) for k in keys} for nm, r in results.items()}
    setting = (f"split {args.split} ({len(eps)} eps), {args.mode}"
               + (f" m={args.m}" if args.mode == "ensemble" else f" horizon={args.horizon} blend={args.blend_s}")
               + f", qpos {args.feedback}, latency {args.latency_ms:.0f} ms"
               + (", left frozen" if args.freeze_left else ""))
    lines = [setting, "",
             "| checkpoint | epoch | mode | R arm err | R grip err | L arm err | lag ms | err @best lag | end err | jerk |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for nm, (_pol, ck) in policies.items():
        m = mean[nm]
        lines.append(f"| {nm} | {ck['epoch']} | {ck['action_mode']} | {m['err']:.4f} | {m['grip_err']:.4f} | "
                     f"{m['left_err']:.4f} | "
                     f"{m['lag_ms']:.0f} | {m['err_at_best_lag']:.4f} | {m['end_err']:.4f} | {m['jerk']:.2f} |")
    (out / "sim.md").write_text("\n".join(lines) + "\n")
    (out / "sim.json").write_text(json.dumps({"setting": setting, "mean": mean, "episodes": results}, indent=1))
    print("\n".join(lines))


def labels_of(policies: dict) -> str:
    return ", ".join(policies)


def plot(out: Path, ep: str, sims: dict, leader: np.ndarray, qpos: np.ndarray, fps: float, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = [*RIGHT.tolist(), 13]
    titles = [f"R{i}" for i in range(6)] + ["R grip"]
    fig, axes = plt.subplots(1, 7, figsize=(28, 4), sharex=True)
    tt = np.arange(len(leader)) / fps
    colors = plt.get_cmap("tab10")(np.arange(len(sims)) % 10)
    for ax, d, ti in zip(axes, dims, titles, strict=True):
        ax.plot(tt, leader[:, d], "k-", lw=2, label="human (leader)")
        ax.plot(tt, qpos[:, d], "k:", lw=1, label="recorded follower")
        for (nm, sim), c in zip(sims.items(), colors, strict=True):
            ax.plot(sim["t"], sim["cmd"][:, d], "-", color=c, lw=1, label=f"{nm} command")
            if args.feedback == "sim":
                ax.plot(sim["t"], sim["follower"][:, d], "--", color=c, lw=0.8, label=f"{nm} sim follower")
        ax.set_title(ti)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    fig.suptitle(f"{ep}  {args.mode}, qpos {args.feedback}, latency {args.latency_ms:.0f} ms")
    fig.tight_layout()
    fig.savefig(out / f"{ep.replace('/', '__')}.png", dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    main()
