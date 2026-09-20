#!/usr/bin/env bash
set -Eeuo pipefail

# Resume the experimentally good model_65000.pt with the exact environment
# used by the historical spin_level_62000/.../s02_yaw_10_speed run.
# This deliberately creates a new run; it never reads the degraded model_66000.pt.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${WLG_RUN_NAME:-spin_resume_65000_longlegs_s02_yaw_10_exact}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/spin_resume_65000_exact.sh train <path/to/model_65000.pt> [additional_iterations]
  bash scripts/spin_resume_65000_exact.sh train13 <path/to/model_65000.pt>
  bash scripts/spin_resume_65000_exact.sh play  <path/to/checkpoint.pt>

The default continuation is 1000 additional iterations.  Set
WLG_RESUME_ITERS or pass a third argument to change it.
EOF
}

resolve_path() {
  local path="$1"
  [[ "$path" == /* ]] || path="$PLANE_ROOT/$path"
  [[ -f "$path" ]] || { echo "Checkpoint not found: $path" >&2; exit 1; }
  printf '%s\n' "$path"
}

latest_stage_checkpoint() {
  local run_name="$1" iter="$2" path dir best="" best_mtime=-1 mtime
  while IFS= read -r -d '' path; do
    dir="$(basename "$(dirname "$path")")"
    [[ "$dir" == *_"$run_name" ]] || continue
    mtime="$(stat -c %Y "$path")"
    if (( mtime > best_mtime )); then
      best_mtime="$mtime"
      best="$path"
    fi
  done < <(find "$PLANE_ROOT/logs/wheel_legged" -mindepth 2 -maxdepth 2 \
    -type f -name "model_${iter}.pt" -print0)
  [[ -n "$best" ]] || {
    echo "Could not find model_${iter}.pt for run ${run_name}" >&2
    exit 1
  }
  printf '%s\n' "$best"
}

apply_exact_s02_environment() {
  # Common environment from commit 2b649e5.
  export WLG_HISTORICAL_DOMAIN_RAND=1
  export WLG_SPAWN_Z=0.12
  export WLG_RECOVERY_MODE=1
  export WLG_TERRAIN_CURRICULUM=0
  export WLG_SLOPE_THRESHOLD=0.45
  export WLG_COMMAND_CURRICULUM=0
  export WLG_COMMAND_PROFILE=independent
  export WLG_COMMAND_RESAMPLING_TIME=25.0
  export WLG_REVERSE_CLIMB_FIXED_HEIGHT=-1
  export WLG_STAIR_UP_FIXED_HEIGHT=-1
  export WLG_HIGHSTAND_ANCHOR_FRACTION=0.0
  export WLG_FAIL_TO_TERMINAL_TIME_S=1.0
  export WLG_TERRAIN_PITCH_TERMINATION_DEG=-1
  export WLG_TERRAIN_PITCH_EXCESS_SCALE=0.0
  export WLG_TERRAIN_PITCH_RATE_SCALE=0.0
  export WLG_MESH_TYPE=plane
  export WLG_TERRAIN_PROPORTIONS=1.0,0.0,0.0,0.0,0.0,0.0

  export WLG_LIN_VEL_X_MIN=0.0
  export WLG_LIN_VEL_X_MAX=0.0
  export WLG_ANG_VEL_YAW_MIN=-10.0
  export WLG_ANG_VEL_YAW_MAX=10.0
  export WLG_HEIGHT_MIN=0.16
  export WLG_HEIGHT_MAX=0.16

  export WLG_TRACKING_LIN_VEL_SCALE=1.5
  export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.0
  export WLG_TRACKING_ANG_VEL_SCALE=2.0
  export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=1.0
  export WLG_TRACKING_ANG_VEL_L1_SCALE=-0.10
  export WLG_CLIP_SINGLE_REWARD=5.0

  export WLG_BASE_HEIGHT_SCALE=3.0
  export WLG_BASE_HEIGHT_ENHANCE_SCALE=2.0
  export WLG_BASE_HEIGHT_L1_SCALE=-6.0
  export WLG_NOMINAL_STATE_SCALE=-1.0
  export WLG_LIN_VEL_Z_SCALE=-2.0
  export WLG_ANG_VEL_XY_SCALE=-0.25
  export WLG_ORIENTATION_SCALE=-20.0
  export WLG_TORQUES_SCALE=-0.0001
  export WLG_COLLISION_SCALE=-1.0
  export WLG_WHEEL_SUPPORT_SCALE=0.25
  export WLG_DOF_POS_LIMITS_SCALE=-1.0
  export WLG_DOF_VEL_SCALE=-0.0001
  export WLG_DOF_ACC_SCALE=-5e-7
  export WLG_ACTION_RATE_SCALE=-0.020
  export WLG_ACTION_SMOOTH_SCALE=-0.025

  export WLG_INIT_NOISE_STD=0.30
  export WLG_ENTROPY_COEF=0.0008
  export WLG_RECOVERY_POSE_SCALE=0.0

  export WLG_HIGH_STAND_LIN_VEL_SCALE=0.0
  export WLG_HIGH_STAND_ANG_VEL_XY_SCALE=0.0
  export WLG_HIGH_STAND_ORIENTATION_SCALE=0.0
  export WLG_HIGH_STAND_ACTION_RATE_SCALE=0.0
  export WLG_HIGH_STAND_ACTION_SMOOTH_SCALE=0.0

  export WLG_SPIN_LOW_HEIGHT_MIN=0.16
  export WLG_SPIN_LOW_HEIGHT_MAX=0.20
  export WLG_SPIN_HIGH_YAW_MIN=5.0
  export WLG_SPIN_FIXED_IDLE_FRACTION=0.10
  export WLG_SPIN_FIXED_NEAR_FRACTION=0.50
  export WLG_SPIN_FIXED_NEAR_MIN_RATIO=0.70
  export WLG_SPIN_HIGH_YAW_LIN_VEL_MAX=0.0
  export WLG_SPIN_MOVING_LIN_VEL_MAX=0.0
  export WLG_SPIN_MOVING_YAW_MAX=10.0
  export WLG_SPIN_MOVING_HEIGHT_MAX=0.28
  export WLG_SPIN_TERRAIN_HEIGHT_MAX=0.26
  export WLG_MIXED_TERRAIN_LIN_VEL_MAX=0.0
  export WLG_MIXED_TERRAIN_YAW_MAX=0.0

  # Keep the mild centre/attitude terms from this exact stage.  In particular,
  # the wheel-speed mismatch term stays disabled.
  export WLG_SPIN_STATIONARY_LIN_VEL_SCALE=-0.40
  export WLG_SPIN_STATIONARY_POSITION_SCALE=-0.20
  export WLG_SPIN_STATIONARY_WHEEL_SPEED_MISMATCH_SCALE=0.0
  export WLG_SPIN_STATIONARY_POSITION_DEADBAND=0.025
  export WLG_SPIN_STATIONARY_ANG_VEL_XY_SCALE=-0.25
  export WLG_SPIN_STATIONARY_ORIENTATION_SCALE=-4.0
  export WLG_SPIN_STATIONARY_ACTION_RATE_SCALE=0.0
  export WLG_SPIN_STATIONARY_ACTION_SMOOTH_SCALE=0.0
  export WLG_SPIN_MOVING_LATERAL_VEL_SCALE=0.0
  export WLG_SPIN_MOVING_WRONG_WAY_SCALE=0.0
  export WLG_SPIN_MOVING_COMMAND_THRESHOLD=0.05
}

train() {
  local source_path="$(resolve_path "$1")"
  local source_file="$(basename "$source_path")"
  local source_run="$(basename "$(dirname "$source_path")")"
  local additional="${2:-${WLG_RESUME_ITERS:-1000}}"
  [[ "$source_file" == model_65000.pt ]] || {
    echo "The source must be model_65000.pt, got: $source_file" >&2
    exit 2
  }
  [[ "$additional" =~ ^[1-9][0-9]*$ ]] || {
    echo "additional_iterations must be a positive integer" >&2
    exit 2
  }

  apply_exact_s02_environment
  cd "$PLANE_ROOT"
  echo "Resuming exact historical s02_yaw_10_speed parameters"
  echo "source=$source_path"
  echo "run_name=$RUN_NAME"
  echo "yaw=[-10,10], height=0.16, additional_iterations=$additional"

  "$PYTHON_BIN" wheel_legged_gym/scripts/train.py \
    --task=wheel_legged \
    --experiment_name=wheel_legged \
    --run_name="$RUN_NAME" \
    --resume \
    --load_run="$source_run" \
    --checkpoint=65000 \
    --headless \
    --max_iterations="$additional"
}

train13() {
  local source_path="$(resolve_path "$1")"
  local source_file="$(basename "$source_path")"
  local source_run="$(basename "$(dirname "$source_path")")"
  local source_iter=65000 target remaining yaw stage_run source_path_next
  local -a yaws=(10.5 11.0 12.0 13.0)
  local -a targets=(65500 66000 66500 67000)
  local -a runs=(
    spin_resume_65000_yaw_10p5_bridge
    spin_resume_65000_yaw_11_bridge
    spin_resume_65000_yaw_12_bridge
    spin_resume_65000_yaw_13_final
  )
  [[ "$source_file" == model_65000.pt ]] || {
    echo "The source must be model_65000.pt, got: $source_file" >&2
    exit 2
  }

  # The 65000 policy is already good.  Use small, fixed updates and reset Adam
  # at every bridge so a lucky 65000 basin is not destroyed by old momentum.
  export WLG_RESUME_LOAD_OPTIMIZER=0
  export WLG_LEARNING_RATE=1e-4
  export WLG_PPO_SCHEDULE=fixed
  export WLG_DESIRED_KL=0.002

  for stage in "${!yaws[@]}"; do
    yaw="${yaws[$stage]}"
    target="${targets[$stage]}"
    stage_run="${runs[$stage]}"
    apply_exact_s02_environment
    export WLG_ANG_VEL_YAW_MIN="-${yaw}"
    export WLG_ANG_VEL_YAW_MAX="${yaw}"
    export WLG_SPIN_MOVING_YAW_MAX="${yaw}"
    remaining="$((target - source_iter))"
    echo
    echo "[$((stage + 1))/4] source=$source_run/model_${source_iter}.pt"
    echo "yaw=[-${yaw},${yaw}], target=model_${target}.pt, lr=${WLG_LEARNING_RATE}"
    "$PYTHON_BIN" wheel_legged_gym/scripts/train.py \
      --task=wheel_legged \
      --experiment_name=wheel_legged \
      --run_name="$stage_run" \
      --resume \
      --load_run="$source_run" \
      --checkpoint="$source_iter" \
      --headless \
      --max_iterations="$remaining"
    source_path_next="$(latest_stage_checkpoint "$stage_run" "$target")"
    source_run="$(basename "$(dirname "$source_path_next")")"
    source_iter="$target"
  done
}

play() {
  local checkpoint="$(resolve_path "$1")"
  apply_exact_s02_environment
  cd "$PLANE_ROOT"
  "$PYTHON_BIN" wheel_legged_gym/scripts/play.py \
    --task=wheel_legged \
    --experiment_name=wheel_legged \
    --load_run="$(basename "$(dirname "$checkpoint")")" \
    --checkpoint="$(basename "$checkpoint" .pt | sed 's/^model_//')" \
    --num_envs="${WLG_PLAY_NUM_ENVS:-1}"
}

case "${1:-}" in
  train) train "${2:?Please provide model_65000.pt}" "${3:-}" ;;
  train13) train13 "${2:?Please provide model_65000.pt}" ;;
  play) play "${2:?Please provide a checkpoint path}" ;;
  *) usage; exit 2 ;;
esac
