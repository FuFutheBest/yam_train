"""Data mapping YAM robot <-> pi0.5 model (openpi DataTransformFn), used for training and for serving.

Robot side (what the LeRobot dataset holds after the repack, and what yam_infer sends to the server):

    images/right_wrist   uint8 (224, 224, 3)  right wrist D405, resize_with_pad from 480x640
    images/left_wrist    uint8 (224, 224, 3)  left wrist D405 (the left arm hardly moves: an overview of the table)
    state                float32 (14,)        follower [left j0..5, left grip, right j0..5, right grip]
    actions              float32 (H, 14)      leader, same layout (training only)
    prompt               str

pi0.5 has three image slots: base_0_rgb (third-person view), left_wrist_0_rgb, right_wrist_0_rgb.
The left wrist camera looks at the whole table from a nearly fixed pose, so it fills the base slot;
the right wrist camera fills the right wrist slot; the left wrist slot is empty (black, mask False ->
the model ignores those tokens).  The head camera is a black placeholder in the recordings and is not used.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from openpi import transforms

# raw image size of the recordings and the model input size (openpi resizes with padding to 224x224)
RAW_IMAGE_HW = (480, 640)
MODEL_IMAGE_HW = (224, 224)
CAMERAS = ("right_wrist", "left_wrist")

# task (dataset directory name) -> language instruction given to the model
PROMPTS = {
    "put_block_in_bowl": "put the block in the bowl",
    "put_blue_block_in_blue_cup": "put the blue block in the blue cup",
    "fold_towel": "fold the towel",
    "unroll_cylinder": "hold the cylinder with both hands and pull the cap off",
}

# Which action dims are predicted relative to the current follower state (delta) - per config:
#
# DELTA_MASK (pi05_yam*, block_bowl + blue_cup: only the right arm works).
# Right arm joints are predicted relative to the current follower state (delta: openpi's convention for
# joint-space robots, and the act_ref finding that delta aims better at grasps); grippers absolute.
# The LEFT arm joints stay absolute: in these demos the left arm stands still (every grasp is done by the
# right arm), so its deltas are ~0 with a 1%-99% quantile range of only +-0.001 rad; openpi's quantile
# normalisation would blow the rare real movements up to 100-300 (training targets are meant to be ~[-1, 1]).
# Absolute left joints stay within +-1.5 after normalisation.
DELTA_MASK = transforms.make_bool_mask(-7, 6, -1)

# DELTA_MASK_BOTH (pi05_yam3*, with the bimanual fold_towel / unroll_cylinder): both arms' joints delta,
# grippers absolute (openpi's aloha convention).  The left-arm movements of the bimanual tasks give the left
# deltas a real 1%-99% range, so the normalisation problem above does not occur (checked in the norm stats).
DELTA_MASK_BOTH = transforms.make_bool_mask(6, -1, 6, -1)


def _hwc_uint8(img) -> np.ndarray:
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):  # LeRobot may hand out float CHW images
        img = (255 * img).clip(0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    return img


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        images = data["images"]
        missing = [c for c in CAMERAS if c not in images]
        if missing:
            raise ValueError(f"YAM pi0.5 needs images {CAMERAS}, missing {missing}")
        base = _hwc_uint8(images["left_wrist"])
        right = _hwc_uint8(images["right_wrist"])
        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": np.zeros_like(base),
                "right_wrist_0_rgb": right,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode() if isinstance(prompt, bytes) else prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # the model works with 32-D padded actions; the robot has 14
        return {"actions": np.asarray(data["actions"][:, :14])}


def make_example() -> dict:
    """A random robot-side observation (for smoke tests)."""
    h, w = MODEL_IMAGE_HW
    return {
        "images": {c: np.random.randint(256, size=(h, w, 3), dtype=np.uint8) for c in CAMERAS},
        "state": np.random.rand(14).astype(np.float32),
        "prompt": PROMPTS["put_block_in_bowl"],
    }
