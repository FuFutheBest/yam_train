"""Convert raw NEDF3 episodes into one compact training dataset.

Output directory:
    meta.json          episodes, split, fps, image size, normalisation stats, task list
    images.npy         uint8 (N, 2, H, W, 3)   cams = [right_wrist, left_wrist], RGB
    qpos.npy           float32 (N, 14)         follower state  (observation)
    action.npy         float32 (N, 14)         leader state    (action = what the human commanded)

Every episode is resampled onto a uniform `fps` grid (default 30 Hz = camera rate):
joints/grippers are linearly interpolated, each camera uses its nearest frame.

Usage:
    python -m yam_train.data.convert --raw RAW_DIR --out OUT_DIR [--workers 32]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .nedf3 import decode_jpeg_rgb, read_episode, state14

CAM_ORDER = ("right_wrist", "left_wrist")
VAL_EVERY = 10  # every 10th episode of each task goes to validation


def _nearest(ts_src: np.ndarray, ts_query: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(ts_src, ts_query)
    idx = np.clip(idx, 1, len(ts_src) - 1)
    left = ts_src[idx - 1]
    right = ts_src[idx]
    return np.where(ts_query - left <= right - ts_query, idx - 1, idx)


def convert_episode(args: tuple[str, int, int, int]) -> dict:
    import cv2

    path, fps, height, width = args
    ep = read_episode(path, load_images=True)
    starts = [ep.cameras[c][0][0] for c in CAM_ORDER]
    ends = [ep.cameras[c][0][-1] for c in CAM_ORDER]
    for role in ("leader", "follower"):
        for side in ("left", "right"):
            starts += [ep.joints[role][side][0][0], ep.grippers[role][side][0][0]]
            ends += [ep.joints[role][side][0][-1], ep.grippers[role][side][0][-1]]
    t0, t1 = max(starts), min(ends)
    step = int(round(1e9 / fps))
    grid = np.arange(t0, t1 + 1, step, dtype=np.int64)

    qpos = state14(ep, "follower", grid).astype(np.float32)
    action = state14(ep, "leader", grid).astype(np.float32)
    images = np.empty((len(grid), len(CAM_ORDER), height, width, 3), np.uint8)
    max_gap_ms = 0.0
    for ci, cam in enumerate(CAM_ORDER):
        ts, frames = ep.cameras[cam]
        nearest = _nearest(ts, grid)
        max_gap_ms = max(max_gap_ms, float(np.abs(ts[nearest] - grid).max() / 1e6))
        cache: dict[int, np.ndarray] = {}
        for i, j in enumerate(nearest):
            if j not in cache:
                rgb = decode_jpeg_rgb(frames[j])
                cache = {
                    j: cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
                }
            images[i, ci] = cache[j]
    return {
        "path": path,
        "task": ep.task,
        "task_description": ep.task_description,
        "qpos": qpos,
        "action": action,
        "images": images,
        "max_cam_gap_ms": max_gap_ms,
    }


def compute_stats(x: np.ndarray) -> dict:
    return {
        "mean": x.mean(0).tolist(),
        "std": np.maximum(x.std(0), 1e-2).tolist(),
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw", required=True, type=Path, help="dir containing <task>/episode_xxxx"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    episodes = sorted(
        p
        for p in args.raw.glob("*/episode_*")
        if p.is_dir() and not p.name.endswith("_inprogress")
    )
    tasks = sorted({p.parent.name for p in episodes})
    print(f"{len(episodes)} episodes, tasks={tasks}")
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [(str(p), args.fps, args.height, args.width) for p in episodes]
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(convert_episode, jobs):
            print(
                f"  {Path(r['path']).parent.name}/{Path(r['path']).name}: {len(r['qpos'])} steps,"
                f" max cam gap {r['max_cam_gap_ms']:.1f} ms"
            )
            results.append(r)

    total = sum(len(r["qpos"]) for r in results)
    images = np.lib.format.open_memmap(
        args.out / "images.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(total, len(CAM_ORDER), args.height, args.width, 3),
    )
    qpos = np.empty((total, 14), np.float32)
    action = np.empty((total, 14), np.float32)
    meta_eps = []
    counters = dict.fromkeys(tasks, 0)
    cursor = 0
    for r in results:
        task = Path(r["path"]).parent.name
        n = len(r["qpos"])
        images[cursor : cursor + n] = r["images"]
        qpos[cursor : cursor + n] = r["qpos"]
        action[cursor : cursor + n] = r["action"]
        split = "val" if counters[task] % VAL_EVERY == VAL_EVERY - 1 else "train"
        counters[task] += 1
        start_idx = cursor
        meta_eps.append(
            {
                "name": f"{task}/{Path(r['path']).name}",
                "task": task,
                "task_id": tasks.index(task),
                "task_description": r["task_description"],
                "start": start_idx,
                "length": n,
                "split": split,
                "start_qpos": r["qpos"][0].tolist(),
            }
        )
        cursor += n
    images.flush()
    np.save(args.out / "qpos.npy", qpos)
    np.save(args.out / "action.npy", action)

    train_rows = np.concatenate(
        [
            np.arange(e["start"], e["start"] + e["length"])
            for e in meta_eps
            if e["split"] == "train"
        ]
    )
    start_pose = {
        t: np.mean(
            [e["start_qpos"] for e in meta_eps if e["task"] == t], axis=0
        ).tolist()
        for t in tasks
    }
    meta = {
        "fps": args.fps,
        "image_size": [args.height, args.width],
        "cameras": list(CAM_ORDER),
        "tasks": tasks,
        "state_layout": "left_joint0..5,left_gripper,right_joint0..5,right_gripper",
        "total_steps": total,
        "stats": {
            "qpos": compute_stats(qpos[train_rows]),
            "action": compute_stats(action[train_rows]),
        },
        "start_pose": start_pose,
        "episodes": meta_eps,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    n_val = sum(e["split"] == "val" for e in meta_eps)
    print(f"done: {total} steps, {len(meta_eps)} episodes ({n_val} val) -> {args.out}")


if __name__ == "__main__":
    main()
