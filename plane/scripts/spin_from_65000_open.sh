#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROOT="$PLANE_ROOT/logs/wheel_legged"
PYTHON_BIN="${PYTHON_BIN:-python}"

# This course deliberately follows the historical/open-source reward recipe.
# It starts from a user-selected, experimentally good model_65000.pt and does
# not use the newer world-position, wheel-mismatch, or strong spin-stability
# terms.
STAGE_YAWS=(7 10 13)
STAGE_TARGETS=(66000 67000 68000)
STAGE_RUN_NAMES=(
  spin_open_65000_longlegs_s01_yaw_7
  spin_open_65000_longlegs_s02_yaw_10
  spin_open_65000_longlegs_s03_yaw_13
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/spin_from_65000_open.sh train <path/to/model_65000.pt>
  bash scripts/spin_from_65000_open.sh train13 <path/to/model_67000.pt>
  bash scripts/spin_from_65000_open.sh play <path/to/model_66000.pt>

The training stages are:
  model_65000.pt -> yaw ±7  -> model_66000.pt
  model_66000.pt -> yaw ±10 -> model_67000.pt
  model_67000.pt -> yaw ±13 -> model_68000.pt

Only the first two stages are recommended before physical validation.
EOF
}

resolve_path() {
  local path="$1"
  [[ "$path" == /* ]] || path="$PLANE_ROOT/$path"
  [[ -f "$path" ]] || { echo "Checkpoint not found: $path" >&2; exit 1; }
  printf '%s\n' "$path"
}

checkpoint_iter() {
  local name
  name="$(basename "$1")"
  name="${name#model_}"
  printf '%s\n' "${name%.pt}"
}

apply_open_source_environment() {
  local yaw="$1"
  export WLG_HISTORICAL_DOMAIN_RAND=1
  export WLG_MESH_TYPE=plane
  export WLG_TERRAIN_PROPORTIONS=1.0,0.0,0.0,0.0,0.0,0.0
  export WLG_TERRAIN_CURRICULUM=0
  export WLG_COMMAND_CURRICULUM=0
  export WLG_COMMAND_PROFILE=independent
  export WLG_COMMAND_RESAMPLING_TIME=5.0

  export WLG_LIN_VEL_X_MIN=0.0
  export WLG_LIN_VEL_X_MAX=0.0
  export WLG_ANG_VEL_YAW_MIN="-$yaw"
  export WLG_ANG_VEL_YAW_MAX="$yaw"
  export WLG_SPIN_MOVING_YAW_MAX="$yaw"
  export WLG_HEIGHT_MIN=0.16
  export WLG_HEIGHT_MAX=0.16

  # Historical/open-source reward values from the checked-in experiment
  # snapshots (25_20_1.0_0.2_angz+ and its ±10/±13 variants).
  export WLG_TRACKING_LIN_VEL_SCALE=1.0
  export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.0
  export WLG_TRACKING_ANG_VEL_SCALE=1.0
  export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=1.0
  export WLG_TRACKING_ANG_VEL_L1_SCALE=0.0
  export WLG_BASE_HEIGHT_SCALE=1.0
  export WLG_BASE_HEIGHT_ENHANCE_SCALE=1.0
  export WLG_BASE_HEIGHT_L1_SCALE=0.0
  export WLG_ORIENTATION_SCALE=-18.0
  export WLG_ANG_VEL_XY_SCALE=-0.07
  export WLG_LIN_VEL_Z_SCALE=-1.0
  export WLG_ACTION_RATE_SCALE=-0.10
  export WLG_ACTION_SMOOTH_SCALE=-0.10
  export WLG_CLIP_SINGLE_REWARD=1.0
  export WLG_ENTROPY_COEF=0.01

  # Do not compete with yaw tracking using the newer, partially observable
  # world-position and wheel-speed terms.
  export WLG_SPIN_STATIONARY_LIN_VEL_SCALE=0.0
  export WLG_SPIN_STATIONARY_POSITION_SCALE=0.0
  export WLG_SPIN_STATIONARY_WHEEL_SPEED_MISMATCH_SCALE=0.0
  export WLG_SPIN_STATIONARY_ANG_VEL_XY_SCALE=0.0
  export WLG_SPIN_STATIONARY_ORIENTATION_SCALE=0.0
  export WLG_SPIN_STATIONARY_ACTION_RATE_SCALE=0.0
  export WLG_SPIN_STATIONARY_ACTION_SMOOTH_SCALE=0.0
  export WLG_RECOVERY_MODE=0
  export WLG_SPAWN_Z=0.12
}

latest_stage_checkpoint() {
  local run_name="$1" target="$2" path best="" best_mtime=-1 mtime dir
  while IFS= read -r -d '' path; do
    dir="$(basename "$(dirname "$path")")"
    [[ "$dir" == *_"$run_name" ]] || continue
    mtime="$(stat -c %Y "$path")"
    if (( mtime > best_mtime )); then
      best_mtime="$mtime"
      best="$path"
    fi
  done < <(find "$LOG_ROOT" -mindepth 2 -maxdepth 2 -type f -name "model_${target}.pt" -print0)
  [[ -n "$best" ]] || return 1
  printf '%s\n' "$best"
}

train_course() {
  local source_path source_iter source_run yaw target stage output_path
  source_path="$(resolve_path "$1")"
  source_iter="$(checkpoint_iter "$source_path")"
  [[ "$source_iter" == 65000 ]] || {
    echo "The starting checkpoint must be model_65000.pt, got model_${source_iter}.pt" >&2
    exit 1
  }

  cd "$PLANE_ROOT"
  for stage in 0 1 2; do
    yaw="${STAGE_YAWS[$stage]}"
    target="${STAGE_TARGETS[$stage]}"
    apply_open_source_environment "$yaw"
    source_run="$(basename "$(dirname "$source_path")")"
    echo
    echo "==============================================================="
    echo "Open-source SPIN stage $((stage + 1))/3: yaw ±${yaw}"
    echo "source=$source_path"
    echo "target=model_${target}.pt"
    echo "position penalty=${WLG_SPIN_STATIONARY_POSITION_SCALE}, sigma=0.25"
    echo "==============================================================="

    "$PYTHON_BIN" wheel_legged_gym/scripts/train.py \
      --task=wheel_legged \
      --experiment_name=wheel_legged \
      --run_name="${STAGE_RUN_NAMES[$stage]}" \
      --resume \
      --load_run="$source_run" \
      --checkpoint="$source_iter" \
      --headless \
      --max_iterations="$((target - source_iter))"

    output_path="$(latest_stage_checkpoint "${STAGE_RUN_NAMES[$stage]}" "$target" || true)"
    [[ -n "$output_path" ]] || {
      echo "Stage did not produce model_${target}.pt; stop here for validation." >&2
      exit 1
    }
    source_path="$output_path"
    source_iter="$target"

    # Stop after yaw ±10 so the user can validate before the ±13 stage.
    if (( stage == 1 )) && [[ "${WLG_SPIN_STOP_AFTER_YAW10:-1}" == "1" ]]; then
      echo "Stopped after yaw ±10. Validate this checkpoint before continuing."
      return 0
    fi
  done
}

train13() {
  local source_path source_iter source_run output_path
  source_path="$(resolve_path "$1")"
  source_iter="$(checkpoint_iter "$source_path")"
  [[ "$source_iter" == 67000 ]] || {
    echo "The yaw ±13 stage must start from model_67000.pt, got model_${source_iter}.pt" >&2
    exit 1
  }
  apply_open_source_environment 13
  source_run="$(basename "$(dirname "$source_path")")"
  cd "$PLANE_ROOT"
  echo "Open-source SPIN stage 3/3: yaw ±13"
  echo "source=$source_path"
  echo "target=model_68000.pt"
  "$PYTHON_BIN" wheel_legged_gym/scripts/train.py \
    --task=wheel_legged \
    --experiment_name=wheel_legged \
    --run_name="${STAGE_RUN_NAMES[2]}" \
    --resume \
    --load_run="$source_run" \
    --checkpoint="$source_iter" \
    --headless \
    --max_iterations=1000
  output_path="$(latest_stage_checkpoint "${STAGE_RUN_NAMES[2]}" 68000 || true)"
  [[ -n "$output_path" ]] || {
    echo "Stage did not produce model_68000.pt." >&2
    exit 1
  }
  echo "Generated $output_path"
}

play_checkpoint() {
  local path run_name iter yaw
  path="$(resolve_path "$1")"
  iter="$(checkpoint_iter "$path")"
  run_name="$(basename "$(dirname "$path")")"
  yaw="${WLG_PLAY_YAW_STEP:-10.0}"
  apply_open_source_environment "$yaw"
  export WLG_PLAY_HEIGHT="${WLG_PLAY_HEIGHT:-0.16}"
  cd "$PLANE_ROOT"
  exec "$PYTHON_BIN" wheel_legged_gym/scripts/play.py \
    --task=wheel_legged \
    --experiment_name=wheel_legged \
    --load_run="$run_name" \
    --checkpoint="$iter" \
    --num_envs="${WLG_PLAY_NUM_ENVS:-1}"
}

case "${1:-}" in
  train) train_course "${2:?Please provide the model_65000.pt path}" ;;
  train13) train13 "${2:?Please provide the model_67000.pt path}" ;;
  play) play_checkpoint "${2:?Please provide a checkpoint path}" ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
