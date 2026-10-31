"""data/nedf3_30hz (yam_train.data.convert) -> LeRobot dataset for openpi training.  Run in the openpi venv.

    python -m yam_train.pi05.to_lerobot --data data/nedf3_30hz_v3 --repo-id yam/pi05_yam_all
    python -m yam_train.pi05.to_lerobot --data data/nedf3_30hz_v3 --repo-id yam/pi05_yam_split \
        --exclude configs/pi05/val_episodes.json

Per 30 Hz row of every episode:
    images.right_wrist / images.left_wrist   recorded JPEG -> RGB -> resize_with_pad 224x224 (openpi's
                                              function; yam_infer does the same at run time)
    state     follower qpos (14)
    actions   leader (14), absolute (the delta for the joints is taken in the training pipeline)
    task      the language prompt of the episode's task (yam_policy.PROMPTS)
The dataset goes to $HF_LEROBOT_HOME/<repo-id>; yam_info.json next to it lists the episodes and the absolute
leader range (for yam_infer's safety filter).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import (  # pyright: ignore[reportMissingImports]  (openpi venv only)
    HF_LEROBOT_HOME,
    LeRobotDataset,
)
from openpi_client.image_tools import resize_with_pad

from yam_train.data.dataset import FrameStore
from yam_train.pi05.yam_policy import CAMERAS, MODEL_IMAGE_HW, PROMPTS


def model_image(store: FrameStore, cam: str, row: int) -> np.ndarray:
    h, w = MODEL_IMAGE_HW
    return resize_with_pad(store.image(cam, row), h, w)


def write_info(store: FrameStore, episodes: list[dict], excluded: set[str], args, out: Path) -> None:
    """yam_info.json: what yam_infer needs besides the network (start poses, leader range for the safety filter)."""
    rows = np.concatenate([np.arange(e["start"], e["start"] + e["length"]) for e in episodes])
    act = store.action[rows]
    info = {
        "source": str(args.data.resolve()),
        "repo_id": args.repo_id,
        "episodes": [e["name"] for e in episodes],
        "excluded": sorted(excluded),
        "rows": len(rows),
        "prompts": {t: PROMPTS[t] for t in args.tasks},
        # absolute leader range of the demos (yam_infer's SafetyFilter clamps to it + a margin)
        "action_min": act.min(0).tolist(),
        "action_max": act.max(0).tolist(),
        # mean first follower state per task: yam_infer moves there before each episode
        "start_pose": {t: np.mean([store.qpos[e["start"]] for e in episodes if e["task"] == t], axis=0).tolist()
                       for t in args.tasks},
    }
    (out / "yam_info.json").write_text(json.dumps(info, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path, help="output dir of yam_train.data.convert")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--exclude", type=Path, help="json {task: [episode names]} of demos to leave out")
    p.add_argument("--tasks", nargs="+", default=list(PROMPTS))
    p.add_argument("--max-episodes", type=int, default=0, help="per task, 0 = all (smoke tests)")
    p.add_argument("--writer-processes", type=int, default=8)
    p.add_argument("--info-only", action="store_true", help="only (re)write yam_info.json of an existing dataset")
    args = p.parse_args()

    store = FrameStore(args.data)
    excluded: set[str] = set()
    if args.exclude:
        try:
            ex = json.loads(args.exclude.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read --exclude {args.exclude}: {exc}") from exc
        excluded = {n for k, v in ex.items() if not k.startswith("_") for n in v}
    episodes = []
    for task in args.tasks:
        eps = [e for e in store.episodes(task) if e["name"] not in excluded]
        episodes += eps[: args.max_episodes] if args.max_episodes else eps
    missing = excluded - {e["name"] for e in store.meta["episodes"]}
    if missing:
        raise SystemExit(f"--exclude names not in {args.data}: {sorted(missing)}")

    out = HF_LEROBOT_HOME / args.repo_id
    if args.info_only:
        write_info(store, episodes, excluded, args, out)
        return
    if out.exists():
        try:
            shutil.rmtree(out)
        except OSError as exc:
            raise SystemExit(f"cannot remove the old dataset {out}: {exc}") from exc
    h, w = MODEL_IMAGE_HW
    image_feature = {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]}
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="i2rt_yam_bimanual",
        fps=int(store.meta["fps"]),
        features={
            **{f"images.{c}": image_feature for c in CAMERAS},
            "state": {"dtype": "float32", "shape": (14,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (14,), "names": ["actions"]},
        },
        use_videos=False,
        image_writer_processes=args.writer_processes,
        image_writer_threads=4,
    )
    t0 = time.time()
    rows_total = 0
    with ThreadPoolExecutor(8) as pool:
        for i, e in enumerate(episodes):
            rows = range(e["start"], e["start"] + e["length"])
            imgs = {c: list(pool.map(lambda r, c=c: model_image(store, c, r), rows)) for c in CAMERAS}
            for k, r in enumerate(rows):
                ds.add_frame({
                    **{f"images.{c}": imgs[c][k] for c in CAMERAS},
                    "state": store.qpos[r].astype(np.float32),
                    "actions": store.action[r].astype(np.float32),
                    "task": PROMPTS[e["task"]],
                })
            ds.save_episode()
            rows_total += e["length"]
            print(f"[{i + 1}/{len(episodes)}] {e['name']}: {e['length']} rows  ({time.time() - t0:.0f} s)", flush=True)

    write_info(store, episodes, excluded, args, out)
    print(f"done: {len(episodes)} episodes, {rows_total} rows -> {out}")


if __name__ == "__main__":
    main()
