#!/usr/bin/env bash
# Create the training virtualenv at <repo>/.venv
#
#   bash scripts/setup_venv.sh            # training server (H20, CUDA 12.6 wheels)
#   bash scripts/setup_venv.sh cpu        # dev machine without GPU
#
# Env overrides: PYTHON (default python3), VENV (default <repo>/.venv),
#                PIP_INDEX (PyPI mirror), TORCH_INDEX (torch wheel mirror)
set -euo pipefail

VARIANT="${1:-cu126}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-python3}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
TORCH_INDEX="${TORCH_INDEX:-https://mirrors.aliyun.com/pytorch-wheels/$VARIANT/}"
TORCH_VERSION=2.7.1        # version the delivered act_v2 models were trained with
TORCHVISION_VERSION=0.22.1

if [ -L "$VENV" ]; then
  echo "error: $VENV is a symlink ($(readlink "$VENV")); remove it first: rm $VENV" >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import sys
if not (3, 10) <= sys.version_info[:2] <= (3, 13):
    raise SystemExit(f"need Python 3.10-3.13 for torch 2.7.1, got {sys.version.split()[0]}")
PY

[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
PIP=("$VENV/bin/python" -m pip install -i "$PIP_INDEX")

"${PIP[@]}" -U pip
"${PIP[@]}" -f "$TORCH_INDEX" "torch==$TORCH_VERSION+$VARIANT" "torchvision==$TORCHVISION_VERSION+$VARIANT"
"${PIP[@]}" -r "$ROOT/requirements-train.txt"

cd "$ROOT"
"$VENV/bin/python" - <<'PY'
import torch, torchvision
import yam_train.train, yam_train.data.convert, yam_train.tools.eval_offline  # noqa: F401
n = torch.cuda.device_count()
names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
print(f"torch {torch.__version__}  torchvision {torchvision.__version__}  cuda {torch.version.cuda}  "
      f"GPUs {n}{': ' + names if n else ''}")
PY
echo "venv ready: $VENV"
