#!/usr/bin/env bash
# Unattended: wait for the LeRobot datasets -> norm stats -> wait until every act_ref training of this user
# has ended -> pi0.5 training of CONFIG (split on GPUs 0,1, all on 2,3: all 4 GPUs busy).
#
#   setsid nohup bash scripts/pi05_after_act.sh [CONFIG] [EXP] > logs/pi05_after_act.log 2>&1 &
#
# CONFIG default pi05_yam3, EXP default run1.  Waits for $LEROBOT_DONE_LOG to contain ALLDONE.
# GPU memory: another user's process may sit on a GPU, so each pair gets the XLA memory fraction of
# what is free on its fuller GPU (max 0.9).  If a run dies of out-of-memory in its first 20 min, it is
# restarted on the other pair as soon as that one is free (the two runs then go one after the other).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/pi05_env.sh
CONFIG="${1:-pi05_yam3}"
EXP="${2:-run1}"
LEROBOT_DONE_LOG="${LEROBOT_DONE_LOG:-logs/to_lerobot_yam3.log}"
log() { echo "$(date '+%F %T') $*"; }

log "waiting for the LeRobot datasets ($LEROBOT_DONE_LOG)"
until grep -q ALLDONE "$LEROBOT_DONE_LOG"; do
  grep -qE "Traceback|Error" "$LEROBOT_DONE_LOG" && { log "conversion failed, see $LEROBOT_DONE_LOG"; exit 1; }
  sleep 60
done
for c in "${CONFIG}_split" "$CONFIG"; do
  if ! ls "$YAM_PI05_ROOT"/assets/"$c"/*/*/norm_stats.json > /dev/null 2>&1; then
    log "norm stats $c"
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PI05_PY" -m yam_train.pi05.cli norm-stats --config "$c" 2>&1 | tail -n 3
  fi
done
# the quantile range of every action dim must be > 0: a ~0 range blows the normalised targets up
"$PI05_PY" - "$YAM_PI05_ROOT" "$CONFIG" <<'PY'
import glob, json, sys
import numpy as np
for c in (sys.argv[2] + "_split", sys.argv[2]):
    f = glob.glob(f"{sys.argv[1]}/assets/{c}/*/*/norm_stats.json")[0]
    a = json.load(open(f))["norm_stats"]["actions"]
    w = np.array(a["q99"]) - np.array(a["q01"])
    print(c, "action q99-q01:", np.round(w[:14], 3))
    if (w[:14] < 0.005).any():
        sys.exit(f"{c}: degenerate action range (dims {np.where(w[:14] < 0.005)[0].tolist()})")
PY

log "waiting for the act_ref trainings to end"
while pgrep -u "$(id -u)" -f "yam_train.train --config" > /dev/null; do sleep 120; done
log "act_ref done; GPUs:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
sleep 30
frac() { # gpu,gpu -> memory fraction left on the fuller one (minus 2 GB), max 0.9
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$1" |
    awk -F', ' 'BEGIN{f=0.9} {x=($3-$2-2048)/$3; if (x<f) f=x} END{printf "%.2f", f}'
}
export MEM_SPLIT MEM_ALL
MEM_SPLIT="$(frac 0,1)"
MEM_ALL="$(frac 2,3)"
log "memory fraction: GPUs 0,1 $MEM_SPLIT, GPUs 2,3 $MEM_ALL"
bash scripts/train_pi05.sh "$EXP" "$CONFIG"
pids() { pgrep -u "$(id -u)" -f "cli train --config $1 --exp $EXP" || true; }
sleep 1200
for pair in "split:${CONFIG}_split:0,1:2,3" "all:${CONFIG}:2,3:0,1"; do
  IFS=: read -r which c mine other <<<"$pair"
  logf="logs/train_${c}_${EXP}.log"
  if [ -z "$(pids "$c")" ] && grep -qE "RESOURCE_EXHAUSTED|Out of memory|OOM" "$logf"; then
    log "$c ran out of memory on GPUs $mine; restarting it on GPUs $other when they are free"
    peer="$CONFIG"; [ "$which" = all ] && peer="${CONFIG}_split"
    while [ -n "$(pids "$peer")" ]; do sleep 300; done
    mv "$logf" "$logf.oom"
    rm -rf "$YAM_PI05_ROOT/checkpoints/$c/$EXP"
    if [ "$which" = split ]; then GPUS_SPLIT="$other" MEM_SPLIT=0.9 ONLY=split bash scripts/train_pi05.sh "$EXP" "$CONFIG"
    else GPUS_ALL="$other" MEM_ALL=0.9 ONLY=all bash scripts/train_pi05.sh "$EXP" "$CONFIG"; fi
  fi
done
log "pi0.5 training running; progress: grep 'Progress on' logs/train_*_${EXP}.log | tail"
