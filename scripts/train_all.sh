#!/usr/bin/env bash
# Start act_ref training runs in the background on a multi-GPU server.
#
#   bash scripts/train_all.sh [--set key=value ...]                  # extra overrides go to every run
#   CONFIGS="blue_cup_abs fold_towel_delta" bash scripts/train_all.sh  # other configs (configs/<name>.yaml)
#
# Default: the 8 runs {block_bowl, blue_cup} x {abs, delta} x {split, all}.
# Each config gives 2 runs:
#   runs/<cfg>      80/20 split (source): curves + eval_offline -> choose abs/delta and the epoch
#   runs/<cfg>_all  data.train_ratio=1.0: every demo; the model to deploy (same epoch number)
# One ACT run (batch 8) uses a small part of an H20, so several runs share a GPU: runs are dealt
# round-robin over the GPUs, the (longer) _all runs first.
# Logs: runs/<name>/train.log.   Watch: tail -f runs/*/train.log | grep -E 'eta|z=0'
# Stop all: pkill -f 'yam_train.train --config'
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || { echo "no $PY; run: bash scripts/setup_venv.sh" >&2; exit 1; }
[ -f data/nedf3_30hz/meta.json ] || {
  echo "no data/nedf3_30hz; run: $PY -m yam_train.data.convert --raw ../dataset --out data/nedf3_30hz" >&2
  exit 1
}
NGPU="$("$PY" -c 'import torch; print(torch.cuda.device_count())')"
[ "$NGPU" -ge 1 ] || { echo "no CUDA GPU visible" >&2; exit 1; }

read -r -a CONFIGS <<<"${CONFIGS:-block_bowl_abs block_bowl_delta blue_cup_abs blue_cup_delta}"
for cfg in "${CONFIGS[@]}"; do
  [ -f "configs/$cfg.yaml" ] || { echo "no configs/$cfg.yaml" >&2; exit 1; }
done
for name in "${CONFIGS[@]}" "${CONFIGS[@]/%/_all}"; do # refuse to mix with an old run
  if [ -s "runs/$name/metrics.jsonl" ]; then
    echo "runs/$name already exists; move or remove it first" >&2
    exit 1
  fi
done

NRUN=$((2 * ${#CONFIGS[@]}))
OMP=$(($(nproc) / NRUN / 2)) # e.g. 64 cores, 8 runs -> 4; leave room for the data loader workers
export OMP_NUM_THREADS=$((OMP > 0 ? OMP : 1))
i=0
for variant in all split; do
  for cfg in "${CONFIGS[@]}"; do
    gpu=$((i % NGPU))
    name="$cfg"
    extra=()
    if [ "$variant" = all ]; then
      name="${cfg}_all"
      extra=(--set data.train_ratio=1.0 --set "name=$name")
    fi
    mkdir -p "runs/$name"
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -m yam_train.train --config "configs/$cfg.yaml" \
      --out "runs/$name" "${extra[@]}" "$@" >"runs/$name/train.log" 2>&1 &
    echo "GPU $gpu  pid $!  runs/$name"
    i=$((i + 1))
  done
done
echo "started; first ETA appears after epoch 10: grep eta runs/*/train.log | tail"
