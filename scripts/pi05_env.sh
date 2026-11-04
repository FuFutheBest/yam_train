# Environment for the pi0.5 (openpi) tools:   source scripts/pi05_env.sh
# Then: "$PI05_PY" -m yam_train.pi05.cli ...   (see docs/02_repo.md)
#
# Layout (defaults; override any variable before sourcing):
#   $YAM_TRAIN_DIR            this repo (pi05 branch)
#   $OPENPI_DIR               openpi checkout + its uv venv   (scripts/setup_openpi.sh)
#   $OPENPI_DATA_HOME         openpi download cache: pi05_base checkpoint, paligemma tokenizer
#   $HF_LEROBOT_HOME          LeRobot datasets written by yam_train.pi05.to_lerobot
#   $YAM_PI05_ROOT            norm stats (assets/) and checkpoints/
YAM_TRAIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export YAM_TRAIN_DIR
export OPENPI_DIR="${OPENPI_DIR:-$YAM_TRAIN_DIR/../openpi}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$YAM_TRAIN_DIR/../.cache/openpi}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$YAM_TRAIN_DIR/lerobot}"
export YAM_PI05_ROOT="${YAM_PI05_ROOT:-$YAM_TRAIN_DIR/runs_pi05}"
export HF_HUB_OFFLINE=1          # the datasets are local; never ask the hub
export PYTHONNOUSERSITE=1        # the server's ~/.local packages must not leak in
export PYTHONPATH="$YAM_TRAIN_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PI05_PY="$OPENPI_DIR/.venv/bin/python"
[ -x "$PI05_PY" ] || echo "pi05_env: no $PI05_PY yet; run scripts/setup_openpi.sh" >&2
