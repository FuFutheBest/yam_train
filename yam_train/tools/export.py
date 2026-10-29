"""Copy a training checkpoint to a delivery file (drops anything robot inference does not need).

    python -m yam_train.tools.export runs/X/ckpt/best.pt models/X.pt
"""

from __future__ import annotations

import sys

import torch

DROP = ("optimizer",)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    out = {k: v for k, v in ck.items() if k not in DROP}
    torch.save(out, sys.argv[2])
    print(f"exported {out.get('task')} epoch {out.get('epoch')} ({out.get('action_mode')} actions) -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
