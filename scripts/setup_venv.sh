#!/usr/bin/env bash
# Create the training virtualenv at <repo>/.venv
#
#   bash scripts/setup_venv.sh            # training server (H20, CUDA 12.6 wheels)
#   bash scripts/setup_venv.sh cpu        # dev machine without GPU
#
# Env overrides: PYTHON (default: first of python3.12/3.11/3.13/3.10/python3 that is 3.10-3.13), VENV (default <repo>/.venv),
#                PIP_INDEX (PyPI mirror), TORCH_INDEX (torch wheel mirror)
set -euo pipefail

VARIANT="${1:-cu126}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
TORCH_INDEX="${TORCH_INDEX:-https://mirrors.aliyun.com/pytorch-wheels/$VARIANT/}"
TORCH_VERSION=2.7.1 # version the delivered act_v2 models were trained with
TORCHVISION_VERSION=0.22.1

if [ -L "$VENV" ]; then
  echo "error: $VENV is a symlink ($(readlink "$VENV")); remove it first: rm $VENV" >&2
  exit 1
fi

# torch 2.7.1 has wheels for Python 3.10-3.13; pick the first suitable interpreter
ok_py() { "$1" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 13))' 2>/dev/null; }
if [ -z "${PYTHON:-}" ]; then
  for cand in python3.12 python3.11 python3.13 python3.10 python3; do
    if command -v "$cand" >/dev/null && ok_py "$cand"; then
      PYTHON="$cand"
      break
    fi
  done
  [ -n "${PYTHON:-}" ] || {
    echo "error: no Python 3.10-3.13 found; set PYTHON=/path/to/python" >&2
    exit 1
  }
elif ! ok_py "$PYTHON"; then
  echo "error: PYTHON=$PYTHON is not Python 3.10-3.13" >&2
  exit 1
fi
echo "using $(command -v "$PYTHON") ($("$PYTHON" --version 2>&1))"

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
