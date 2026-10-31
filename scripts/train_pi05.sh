#!/usr/bin/env bash
# pi0.5 fine-tuning on a 4-GPU server: two runs side by side, 2 GPUs each (model sharded over the pair, FSDP).
#
#   bash scripts/train_pi05.sh [EXP]          # EXP = experiment name, default run1
#
#   GPUs 0,1  pi05_yam_split  held-out demos (configs/pi05/val_episodes.json) -> eval_offline vs act_ref
#   GPUs 2,3  pi05_yam        every demo -> the model to deploy (same step)
# Prerequisites: scripts/setup_openpi.sh; LeRobot datasets (yam_train.pi05.to_lerobot); cli norm-stats.
# Logs: logs/train_<config>_<EXP>.log ("Step N: loss=...").  Stop: pkill -f 'yam_train.pi05.cli train'
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/pi05_env.sh
EXP="${1:-run1}"
mkdir -p logs
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
OPENPI_COMMIT="$(git -C "$OPENPI_DIR" rev-parse --short HEAD)"
export OPENPI_COMMIT
start() { # gpus config
  local log="logs/train_$2_$EXP.log"
  [ -e "$YAM_PI05_ROOT/checkpoints/$2/$EXP" ] && { echo "$2/$EXP exists; use another EXP" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="$1" setsid nohup "$PI05_PY" -m yam_train.pi05.cli train --config "$2" --exp "$EXP" --fsdp 2 \
    > "$log" 2>&1 < /dev/null &
  echo "started $2 on GPUs $1 -> $log"
}
start 0,1 pi05_yam_split
start 2,3 pi05_yam
