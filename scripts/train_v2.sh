#!/usr/bin/env bash
# Start configs/v2/<name>.yaml runs in the background, 2 per GPU, output runs/v2/<name>.
#
#   bash scripts/train_v2.sh NAME [NAME ...] [-- --set key=value ...]
#   e.g. bash scripts/train_v2.sh block_bowl_abs_aug blue_cup_abs_aug
# Logs: runs/v2/<name>/train.log      Stop: pkill -f '[y]am_train.train --config configs/v2'
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PY="${PY:-.venv/bin/python}"
names=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  names+=("$1")
  shift
done
[ "${1:-}" = "--" ] && shift
[ ${#names[@]} -gt 0 ] || { echo "usage: $0 NAME [NAME ...] [-- --set k=v ...]" >&2; exit 1; }
NGPU="$("$PY" -c 'import torch; print(torch.cuda.device_count())')"
[ "$NGPU" -ge 1 ] || { echo "no CUDA GPU visible" >&2; exit 1; }
for name in "${names[@]}"; do
  [ -f "configs/v2/$name.yaml" ] || { echo "no configs/v2/$name.yaml" >&2; exit 1; }
  [ -s "runs/v2/$name/metrics.jsonl" ] && { echo "runs/v2/$name already exists" >&2; exit 1; }
done
export OMP_NUM_THREADS=4
i=0
for name in "${names[@]}"; do
  gpu=$(((i / 2) % NGPU))
  mkdir -p "runs/v2/$name"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -m yam_train.train --config "configs/v2/$name.yaml" \
    --out "runs/v2/$name" "$@" >"runs/v2/$name/train.log" 2>&1 &
  echo "GPU $gpu  pid $!  runs/v2/$name"
  i=$((i + 1))
done
