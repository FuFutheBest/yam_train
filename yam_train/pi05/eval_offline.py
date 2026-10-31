"""Open-loop evaluation of an exported pi0.5 model on held-out demos (run in the openpi venv, needs a GPU).

    $PI05_PY -m yam_train.pi05.eval_offline --model models/pi05_yam_split --data data/nedf3_30hz_v3 --out runs_pi05/eval/x

Episodes: the model's excluded (held-out) demos by default (yam_model.json), or --episodes NAME ...
Two numbers per task (right arm joints 0-5, rad):

  grasp test   for the frame 0.5 / 1 / 2 s before the first gripper close of the demo, predict the chunk and
               read the pose planned for the close moment; error vs the human's pose there.  The same test
               as for act_ref (v1 delta 1 s: block 0.029, blue cup 0.028).  "aimed" = error < 0.04 rad.
               pi0.5's chunk is 50 steps (1.67 s): the 2 s case reads the chunk's last step.
  chunk error  every 15th frame: mean |prediction - leader| over the first 10 / 30 steps (0.33 / 1 s)
               and the gripper error over 30 steps.

What the robot gets is exactly this prediction (same transforms, same image preprocessing), only
the timing of the closed loop is missing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from openpi_client.image_tools import resize_with_pad

from yam_train.data.dataset import FrameStore
from yam_train.pi05.cli import load_model
from yam_train.pi05.yam_policy import CAMERAS, MODEL_IMAGE_HW, PROMPTS

LEADS_S = (0.5, 1.0, 2.0)


def observation(store: FrameStore, row: int, task: str) -> dict:
    h, w = MODEL_IMAGE_HW
    return {
        "images": {c: resize_with_pad(store.image(c, row), h, w) for c in CAMERAS},
        "state": store.qpos[row].astype(np.float32),
        "prompt": PROMPTS[task],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--episodes", nargs="*", help="default: the model's held-out demos")
    p.add_argument("--stride", type=int, default=15)
    args = p.parse_args()

    policy, meta = load_model(args.model)
    names = set(args.episodes or meta.get("excluded_episodes") or [])
    if not names:
        raise SystemExit("the model was trained on every demo; pass --episodes")
    store = FrameStore(args.data)
    fps = float(store.meta["fps"])
    eps = [e for e in store.meta["episodes"] if e["name"] in names]
    if len(eps) != len(names):
        raise SystemExit(f"episodes not in {args.data}: {sorted(names - {e['name'] for e in eps})}")

    per_task: dict[str, dict] = {}
    infer_ms = []
    t0 = time.time()
    for n, e in enumerate(sorted(eps, key=lambda e: e["name"])):
        s, L = e["start"], e["length"]
        A = store.action[s:s + L]
        right = A[:, 13].min() < A[:, 6].min()  # the arm that grasps
        g = 13 if right else 6
        arm = slice(g - 6, g)
        r = per_task.setdefault(e["task"], {"grasp": {x: [] for x in LEADS_S}, "e10": [], "e30": [], "grip30": []})
        close = int(np.argmax(A[:, g] < 0.5))
        for lead_s in LEADS_S:
            f = max(0, close - round(lead_s * fps))
            out = policy.infer(observation(store, s + f, e["task"]))
            infer_ms.append(out["policy_timing"]["infer_ms"])
            chunk = out["actions"]
            r["grasp"][lead_s].append(float(np.abs(chunk[min(close - f, len(chunk) - 1), arm] - A[close, arm]).mean()))
        for f in range(0, L - 30, args.stride):
            chunk = policy.infer(observation(store, s + f, e["task"]))["actions"]
            r["e10"].append(float(np.abs(chunk[:10, arm] - A[f:f + 10, arm]).mean()))
            r["e30"].append(float(np.abs(chunk[:30, arm] - A[f:f + 30, arm]).mean()))
            r["grip30"].append(float(np.abs(chunk[:30, g] - A[f:f + 30, g]).mean()))
        print(f"[{n + 1}/{len(eps)}] {e['name']}: grasp err 1s {r['grasp'][1.0][-1]:.3f}  ({time.time() - t0:.0f} s)",
              flush=True)

    summary = {"model": str(args.model), "step": meta.get("step"), "infer_ms_median": float(np.median(infer_ms)),
               "tasks": {}}
    lines = ["| task | demos | grasp err 0.5 s | 1 s | 2 s | aimed (<0.04) @1 s | chunk err 10 | chunk err 30 | grip err 30 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for task, r in per_task.items():
        g = {k: np.array(v) for k, v in r["grasp"].items()}
        row = {
            "demos": len(g[1.0]),
            "grasp_err": {str(k): float(v.mean()) for k, v in g.items()},
            "aimed_1s": int((g[1.0] < 0.04).sum()),
            "chunk_err_10": float(np.mean(r["e10"])),
            "chunk_err_30": float(np.mean(r["e30"])),
            "grip_err_30": float(np.mean(r["grip30"])),
        }
        summary["tasks"][task] = row
        ge = row["grasp_err"]
        lines.append(f"| {task} | {row['demos']} | {ge['0.5']:.3f} | {ge['1.0']:.3f} | {ge['2.0']:.3f} | "
                     f"{row['aimed_1s']}/{row['demos']} | {row['chunk_err_10']:.3f} | {row['chunk_err_30']:.3f} | "
                     f"{row['grip_err_30']:.3f} |")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"median inference {summary['infer_ms_median']:.0f} ms")


if __name__ == "__main__":
    main()
