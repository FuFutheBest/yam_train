"""Convert raw NEDF3 episodes into one dataset directory (all tasks, no split — the split is per config).

Output directory:
    meta.json            fps, cameras, image size, tasks, episodes (name, task, start row, length)
    qpos.npy             float32 (N, 14)   follower state at each row (observation)
    action.npy           float32 (N, 14)   leader state at each row (what the human commanded)
    <camera>.jpg.bin     the recorded JPEG frames, byte-for-byte (480x640, no resize / re-encode)
    <camera>.idx.npy     int64 (N, 2)      (offset, length) into <camera>.jpg.bin for each row

Every episode is resampled onto a uniform `fps` grid (default 30 Hz = camera rate):
joints/grippers are linearly interpolated, each camera uses its nearest frame.

    python -m yam_train.data.convert --raw RAW_DIR --out OUT_DIR [--workers 16]
RAW_DIR contains <task>/episode_xxxx (SuperInference data/ layout).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .nedf3 import CAMERAS, decode_jpeg_rgb, read_episode, state14


def _nearest(ts_src: np.ndarray, ts_query: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(ts_src, ts_query)
    idx = np.clip(idx, 1, len(ts_src) - 1)
    left = ts_src[idx - 1]
    right = ts_src[idx]
    return np.where(ts_query - left <= right - ts_query, idx - 1, idx)


def convert_episode(args: tuple[str, int]) -> dict:
    path, fps = args
    ep = read_episode(path, load_images=True)
    missing = [c for c in CAMERAS if c not in ep.cameras]
    if missing:
        raise RuntimeError(f"{path}: no frames for cameras {missing}")
    starts = [ep.cameras[c][0][0] for c in CAMERAS]
    ends = [ep.cameras[c][0][-1] for c in CAMERAS]
    for role in ("leader", "follower"):
        for side in ("left", "right"):
            starts += [ep.joints[role][side][0][0], ep.grippers[role][side][0][0]]
            ends += [ep.joints[role][side][0][-1], ep.grippers[role][side][0][-1]]
    t0, t1 = max(starts), min(ends)
    grid = np.arange(t0, t1 + 1, round(1e9 / fps), dtype=np.int64)

    frames: dict[str, list[bytes]] = {}
    frame_of_row: dict[str, np.ndarray] = {}
    max_gap_ms = 0.0
    for cam in CAMERAS:
        ts, jpgs = ep.cameras[cam]
        nearest = _nearest(ts, grid)
        max_gap_ms = max(max_gap_ms, float(np.abs(ts[nearest] - grid).max() / 1e6))
        used, inverse = np.unique(nearest, return_inverse=True)
        frames[cam] = [jpgs[i] for i in used]
        frame_of_row[cam] = inverse
    h, w = decode_jpeg_rgb(frames[next(iter(CAMERAS))][0]).shape[:2]
    return {
        "path": path,
        "task": Path(path).parent.name,
        "task_description": ep.task_description,
        "qpos": state14(ep, "follower", grid).astype(np.float32),
        "action": state14(ep, "leader", grid).astype(np.float32),
        "frames": frames,
        "frame_of_row": frame_of_row,
        "image_size": [h, w],
        "max_cam_gap_ms": max_gap_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path, help="dir containing <task>/episode_xxxx")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    episodes = sorted(
        p for p in args.raw.glob("*/episode_*") if p.is_dir() and not p.name.endswith("_inprogress")
    )
    if not episodes:
        raise SystemExit(f"no episodes under {args.raw}/<task>/episode_*")
    tasks = sorted({p.parent.name for p in episodes})
    print(f"{len(episodes)} episodes, tasks={tasks}")
    args.out.mkdir(parents=True, exist_ok=True)

    bins = {c: open(args.out / f"{c}.jpg.bin", "wb") for c in CAMERAS}  # noqa: SIM115
    idx: dict[str, list[np.ndarray]] = {c: [] for c in CAMERAS}
    qpos, action, meta_eps = [], [], []
    image_size = None
    cursor = 0
    try:
        with ProcessPoolExecutor(args.workers) as pool:
            for r in pool.map(convert_episode, [(str(p), args.fps) for p in episodes]):
                n = len(r["qpos"])
                if image_size is None:
                    image_size = r["image_size"]
                elif image_size != r["image_size"]:
                    raise RuntimeError(f"{r['path']}: image size {r['image_size']} != {image_size}")
                for c in CAMERAS:
                    offs = np.empty((len(r["frames"][c]), 2), np.int64)
                    for i, jpg in enumerate(r["frames"][c]):
                        offs[i] = (bins[c].tell(), len(jpg))
                        bins[c].write(jpg)
                    idx[c].append(offs[r["frame_of_row"][c]])
                qpos.append(r["qpos"])
                action.append(r["action"])
                meta_eps.append({
                    "name": f"{r['task']}/{Path(r['path']).name}",
                    "task": r["task"],
                    "task_description": r["task_description"],
                    "start": cursor,
                    "length": n,
                    "max_cam_gap_ms": r["max_cam_gap_ms"],
                })
                cursor += n
                print(f"  {meta_eps[-1]['name']}: {n} steps, max cam gap {r['max_cam_gap_ms']:.1f} ms", flush=True)
    finally:
        for fh in bins.values():
            fh.close()

    for c in CAMERAS:
        np.save(args.out / f"{c}.idx.npy", np.concatenate(idx[c]))
    np.save(args.out / "qpos.npy", np.concatenate(qpos))
    np.save(args.out / "action.npy", np.concatenate(action))
    meta = {
        "format": "yam_train/jpeg-v1",
        "fps": args.fps,
        "cameras": list(CAMERAS),
        "image_size": image_size,
        "tasks": tasks,
        "state_layout": "left_joint0..5,left_gripper,right_joint0..5,right_gripper",
        "qpos": "follower", "action": "leader",
        "total_steps": cursor,
        "episodes": meta_eps,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"done: {cursor} steps, {len(meta_eps)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
