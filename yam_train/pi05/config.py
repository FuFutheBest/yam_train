"""openpi TrainConfigs for YAM, registered into openpi's config table (so openpi's scripts find them by name).

| config          | LeRobot dataset          | demos                                           | use                         |
|-----------------|--------------------------|-------------------------------------------------|-----------------------------|
| pi05_yam_split  | yam/pi05_yam_split       | all minus configs/pi05/val_episodes.json (2x10) | open-loop eval vs. act_ref  |
| pi05_yam        | yam/pi05_yam_all         | every demo of both tasks                        | the model to deploy         |

One model for both tasks; the task is chosen by the language prompt (yam_policy.PROMPTS).
Hyper-parameters follow openpi's fine-tuning examples (TrainConfig defaults / pi0_aloha_pen_uncap: batch 32,
AdamW, cosine schedule with 1k warmup, peak lr 2.5e-5, EMA 0.99, full fine-tuning from pi05_base), with
15k steps = 480k samples = ~8 passes over the 59k frames.  Measured on 2 H20 (FSDP over the pair): batch 64
takes 4.7 s/step, so 15k steps at batch 32 take ~10 h; checkpoints every 1k, kept every 5k (43 GB each).

Paths (all under YAM_PI05_ROOT, default ./runs_pi05):
    assets/<config>/<repo_id>/norm_stats.json     written by `cli norm-stats`
    checkpoints/<config>/<exp_name>/<step>/       written by `cli train`
The LeRobot datasets live in HF_LEROBOT_HOME (set by scripts/pi05_env.sh).
"""

from __future__ import annotations

import dataclasses
import os
import pathlib

from openpi import transforms as _transforms
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders
from typing_extensions import override

from yam_train.pi05.yam_policy import DELTA_MASK, MODEL_IMAGE_HW, PROMPTS, YamInputs, YamOutputs

ACTION_HORIZON = 50  # 50 steps at 30 Hz = 1.67 s per predicted chunk (openpi's default horizon)
BASE_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_base/params"


def root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("YAM_PI05_ROOT", "runs_pi05")).resolve()


@dataclasses.dataclass(frozen=True)
class LeRobotYamDataConfig(_config.DataConfigFactory):
    """LeRobot dataset written by yam_train.pi05.to_lerobot -> pi0.5 inputs (see yam_policy.py)."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config) -> _config.DataConfig:
        # dataset keys -> the robot-side keys yam_infer sends at inference (only applied to dataset items)
        repack = _transforms.Group(inputs=[_transforms.RepackTransform({
            "images": {"right_wrist": "images.right_wrist", "left_wrist": "images.left_wrist"},
            "state": "state",
            "actions": "actions",
            "prompt": "prompt",
        })])
        data_transforms = _transforms.Group(inputs=[YamInputs()], outputs=[YamOutputs()]).push(
            inputs=[_transforms.DeltaActions(DELTA_MASK)],
            outputs=[_transforms.AbsoluteActions(DELTA_MASK)],
        )
        model_transforms = _config.ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("actions",),
            prompt_from_task=True,
        )


def make_config(name: str, repo_id: str, *, steps: int = 15_000, batch_size: int = 32,
                fsdp_devices: int = 1) -> _config.TrainConfig:
    r = root()
    return _config.TrainConfig(
        name=name,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=ACTION_HORIZON),
        data=LeRobotYamDataConfig(repo_id=repo_id),
        weight_loader=weight_loaders.CheckpointWeightLoader(BASE_CHECKPOINT),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(1_000, steps // 10), peak_lr=2.5e-5, decay_steps=steps, decay_lr=2.5e-6
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        batch_size=batch_size,
        num_train_steps=steps,
        save_interval=1_000,
        keep_period=5_000,
        log_interval=100,
        num_workers=8,
        wandb_enabled=False,
        fsdp_devices=fsdp_devices,
        assets_base_dir=str(r / "assets"),
        checkpoint_base_dir=str(r / "checkpoints"),
        policy_metadata={
            "format": "yam_train/pi05-v1",
            "config": name,
            "fps": 30,
            "action_horizon": ACTION_HORIZON,
            "cameras": ["right_wrist", "left_wrist"],
            "image_hw": list(MODEL_IMAGE_HW),
            "prompts": PROMPTS,
            "state_layout": "left_joint0..5,left_gripper,right_joint0..5,right_gripper",
            "actions": "absolute leader targets (joints delta-decoded on the server)",
        },
    )


CONFIGS = {
    "pi05_yam_split": ("yam/pi05_yam_split",),
    "pi05_yam": ("yam/pi05_yam_all",),
}


def register(**overrides) -> None:
    """Add the YAM configs to openpi's table (idempotent).  overrides go to make_config (steps, batch_size...)."""
    for name, (repo_id,) in CONFIGS.items():
        _config._CONFIGS_DICT[name] = make_config(name, repo_id, **overrides)


def get(name: str, **overrides) -> _config.TrainConfig:
    register(**overrides)
    return _config.get_config(name)
