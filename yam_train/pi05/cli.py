"""pi0.5 entry points (thin wrappers around openpi's own scripts).  Run with the openpi venv:

    source scripts/pi05_env.sh
    $PI05_PY -m yam_train.pi05.cli norm-stats --config pi05_yam
    $PI05_PY -m yam_train.pi05.cli train      --config pi05_yam --exp run1 [--steps 20000 --batch 64 --fsdp 2]
    $PI05_PY -m yam_train.pi05.cli export     --ckpt $YAM_PI05_ROOT/checkpoints/pi05_yam/run1/19999 --out models/pi05_yam
    $PI05_PY -m yam_train.pi05.cli serve      --model models/pi05_yam [--port 8000]

export makes a self-contained model directory (what yam_infer's policy server loads):
    params/            network weights (EMA)
    assets/<repo_id>/  norm_stats.json (state/action normalisation used in training)
    yam_model.json     config name, training step, the server metadata (fps, cameras, prompts, action range)
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from yam_train.pi05 import config as yam_config

log = logging.getLogger("yam_train.pi05")


def _openpi_script(name: str):
    """Import openpi's scripts/<name>.py (not installed as a package; lives in $OPENPI_DIR)."""
    openpi_dir = os.environ.get("OPENPI_DIR")
    if not openpi_dir or not Path(openpi_dir, "scripts", f"{name}.py").exists():
        raise SystemExit("OPENPI_DIR is not an openpi checkout; source scripts/pi05_env.sh first")
    sys.path.insert(0, openpi_dir)
    return importlib.import_module(f"scripts.{name}")


def cmd_norm_stats(a) -> None:
    yam_config.register()
    _openpi_script("compute_norm_stats").main(a.config, max_frames=a.max_frames)


def cmd_train(a) -> None:
    overrides = {k: v for k, v in (("steps", a.steps), ("batch_size", a.batch), ("fsdp_devices", a.fsdp)) if v}
    cfg = yam_config.get(a.config, **overrides)
    cfg = dataclasses.replace(cfg, exp_name=a.exp, overwrite=a.overwrite, resume=a.resume)
    norm = Path(cfg.assets_dirs) / cfg.data.repo_id / "norm_stats.json"
    if not norm.exists():
        raise SystemExit(f"no {norm}; run: cli norm-stats --config {a.config}")
    _openpi_script("train").main(cfg)


def cmd_export(a) -> None:
    ckpt = Path(a.ckpt).resolve()
    cfg_name = ckpt.parent.parent.name  # .../checkpoints/<config>/<exp>/<step>
    cfg = yam_config.get(cfg_name)
    repo_id = cfg.data.repo_id
    lerobot_home = Path(os.environ.get("HF_LEROBOT_HOME", "lerobot"))
    try:
        yam_info = json.loads((lerobot_home / repo_id / "yam_info.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read the dataset's yam_info.json: {exc}") from exc
    for part in ("params", f"assets/{repo_id}/norm_stats.json"):
        if not (ckpt / part).exists():
            raise SystemExit(f"{ckpt} has no {part}")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first")
    out.mkdir(parents=True)
    shutil.copytree(ckpt / "params", out / "params")
    shutil.copytree(ckpt / "assets", out / "assets")
    meta = {
        **(cfg.policy_metadata or {}),
        "config": cfg_name,
        "experiment": ckpt.parent.name,
        "step": int(ckpt.name),
        "repo_id": repo_id,
        "episodes": len(yam_info["episodes"]),
        "excluded_episodes": yam_info["excluded"],
        "action_min": yam_info["action_min"],
        "action_max": yam_info["action_max"],
        "start_pose": yam_info["start_pose"],
        "openpi_commit": os.environ.get("OPENPI_COMMIT", ""),
    }
    (out / "yam_model.json").write_text(json.dumps(meta, indent=1))
    print(f"exported {ckpt} -> {out}")


def load_model(model_dir: str | Path, **sample_kwargs):
    """(openpi Policy, metadata) of an exported model directory."""
    from openpi.policies import policy_config

    model_dir = Path(model_dir).resolve()
    try:
        meta = json.loads((model_dir / "yam_model.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{model_dir} is not an exported pi05 model (yam_model.json): {exc}") from exc
    cfg = yam_config.get(meta["config"])
    policy = policy_config.create_trained_policy(cfg, model_dir, sample_kwargs=sample_kwargs or None)
    return policy, meta


def cmd_serve(a) -> None:
    from openpi.serving import websocket_policy_server

    policy, meta = load_model(a.model)
    log.info("serving %s (step %s) on port %d", a.model, meta.get("step"), a.port)
    websocket_policy_server.WebsocketPolicyServer(policy, host=a.host, port=a.port, metadata=meta).serve_forever()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("norm-stats")
    s.add_argument("--config", required=True, choices=list(yam_config.CONFIGS))
    s.add_argument("--max-frames", type=int, default=None)
    s.set_defaults(fn=cmd_norm_stats)
    s = sub.add_parser("train")
    s.add_argument("--config", required=True, choices=list(yam_config.CONFIGS))
    s.add_argument("--exp", required=True, help="experiment name (checkpoint sub-directory)")
    s.add_argument("--steps", type=int)
    s.add_argument("--batch", type=int)
    s.add_argument("--fsdp", type=int, help="shard the model over this many GPUs (default 1)")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--resume", action="store_true")
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("export")
    s.add_argument("--ckpt", required=True, help="checkpoints/<config>/<exp>/<step>")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_export)
    s = sub.add_parser("serve")
    s.add_argument("--model", required=True, help="directory written by export")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
