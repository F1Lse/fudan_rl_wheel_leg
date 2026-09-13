#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROOT="$PLANE_ROOT/logs/wheel_legged"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLAY_NUM_ENVS="${WLG_PLAY_NUM_ENVS:-20}"

BASE_CHECKPOINT="${WLG_SPIN_BASE_CHECKPOINT:-56500}"
BASE_RUN_NAME="${WLG_SPIN_BASE_RUN_NAME:-flat_stability_longlegs_s05_final_mixed_consolidation}"

STAGE_KEYS=(
  yaw_1
  yaw_2
  yaw_3
  yaw_5
  yaw_7
  yaw_10
  yaw_13
  yaw_13_stable
)
STAGE_LABELS=(
  "低位原地旋转入门：0.16 m、yaw ±1"
  "低位原地旋转：0.16 m、yaw ±2"
  "低位原地旋转：0.16 m、yaw ±3"
  "低位原地旋转：0.16 m、yaw ±5"
  "低位高速旋转：0.16 m、yaw ±7"
  "低位高速旋转：0.16 m、yaw ±10"
  "低位高速旋转：0.16 m、yaw ±13"
  "±13 原地旋转稳定性巩固"
)
# Absolute checkpoint numbers continuing from the stable model_56500.pt.
STAGE_TARGETS=(56800 57100 57500 58000 58600 59400 60400 61400)
STAGE_RUN_NAMES=(
  spin_fixed_56500_longlegs_s01_yaw_1
  spin_fixed_56500_longlegs_s02_yaw_2
  spin_fixed_56500_longlegs_s03_yaw_3
  spin_fixed_56500_longlegs_s04_yaw_5
  spin_fixed_56500_longlegs_s05_yaw_7
  spin_fixed_56500_longlegs_s06_yaw_10
  spin_fixed_56500_longlegs_s07_yaw_13
  spin_fixed_56500_longlegs_s08_yaw_13_stable
)
STAGE_COUNT="${#STAGE_KEYS[@]}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/spin_curriculum.sh train
  bash scripts/spin_curriculum.sh status
  bash scripts/spin_curriculum.sh play
  bash scripts/spin_curriculum.sh play 58000
  bash scripts/spin_curriculum.sh play 'logs/wheel_legged/RUN/model_58000.pt'
  bash scripts/spin_curriculum.sh export

The first stage resumes from the stable flat/terrain model_56500.pt. If that
checkpoint has a different path, provide it explicitly:

  WLG_SPIN_BASE_PT=/absolute/path/model_56500.pt \
    bash scripts/spin_curriculum.sh train

Useful overrides:
  PYTHON_BIN=python
  WLG_PLAY_NUM_ENVS=20
  WLG_PLAY_HEIGHT=0.16
  WLG_PLAY_INITIAL_YAW=5.0
  WLG_PLAY_YAW_STEP=5.0
  WLG_EXPORT_OUT=export_onnx/longlegs_spin_final.onnx
EOF
}

checkpoint_iter_from_path() {
  local file_name
  file_name="$(basename "$1")"
  file_name="${file_name#model_}"
  printf '%s\n' "${file_name%.pt}"
}

stage_index_for_checkpoint() {
  local checkpoint_iter="$1" stage_index
  [[ "$checkpoint_iter" =~ ^[0-9]+$ ]] || {
    echo "Checkpoint must be an integer, got: $checkpoint_iter" >&2
    return 2
  }
  for stage_index in "${!STAGE_TARGETS[@]}"; do
    if (( checkpoint_iter <= STAGE_TARGETS[stage_index] )); then
      printf '%s\n' "$stage_index"
      return 0
    fi
  done
  printf '%s\n' "$((STAGE_COUNT - 1))"
}

resolve_checkpoint_path() {
  local path="$1"
  [[ "$path" == /* ]] || path="$PLANE_ROOT/$path"
  [[ -f "$path" ]] || {
    echo "Checkpoint not found: $path" >&2
    return 1
  }
  printf '%s\n' "$path"
}

find_base_checkpoint() {
  local explicit_path="${WLG_SPIN_BASE_PT:-}"
  local path dir_name mtime best_mtime=-1 best_path=""

  if [[ -n "$explicit_path" ]]; then
    if [[ "$explicit_path" != /* ]]; then
      explicit_path="$PLANE_ROOT/$explicit_path"
    fi
    [[ -f "$explicit_path" ]] || {
      echo "SPIN base checkpoint not found: $explicit_path" >&2
      return 1
    }
    [[ "$(checkpoint_iter_from_path "$explicit_path")" == "$BASE_CHECKPOINT" ]] || {
      echo "WLG_SPIN_BASE_PT must point to model_${BASE_CHECKPOINT}.pt" >&2
      return 1
    }
    printf '%s\n' "$explicit_path"
    return 0
  fi

  if [[ -d "$LOG_ROOT" ]]; then
    while IFS= read -r -d '' path; do
      dir_name="$(basename "$(dirname "$path")")"
      [[ "$dir_name" == *_"$BASE_RUN_NAME" ]] || continue
      mtime="$(stat -c %Y "$path")"
      if (( mtime > best_mtime )); then
        best_mtime="$mtime"
        best_path="$path"
      fi
    done < <(
      find "$LOG_ROOT" -mindepth 2 -maxdepth 2 -type f \
        -name "model_${BASE_CHECKPOINT}.pt" -print0
    )
  fi

  [[ -n "$best_path" ]] || {
    echo "Could not find the historical model_${BASE_CHECKPOINT}.pt base." >&2
    echo "Set WLG_SPIN_BASE_PT to its absolute path and rerun." >&2
    return 1
  }
  printf '%s\n' "$best_path"
}

latest_checkpoint_for_stage() {
  local stage_index="$1"
  local run_name="${STAGE_RUN_NAMES[$stage_index]}"
  local best_iter=-1 best_mtime=-1 best_path=""
  local path dir_name file_name iter mtime

  if [[ -d "$LOG_ROOT" ]]; then
    while IFS= read -r -d '' path; do
      dir_name="$(basename "$(dirname "$path")")"
      [[ "$dir_name" == *_"$run_name" ]] || continue
      file_name="$(basename "$path")"
      iter="${file_name#model_}"
      iter="${iter%.pt}"
      [[ "$iter" =~ ^[0-9]+$ ]] || continue
      mtime="$(stat -c %Y "$path")"
      if (( iter > best_iter || (iter == best_iter && mtime > best_mtime) )); then
        best_iter="$iter"
        best_mtime="$mtime"
        best_path="$path"
      fi
    done < <(find "$LOG_ROOT" -mindepth 2 -maxdepth 2 -type f -name 'model_*.pt' -print0)
  fi

  printf '%s\t%s\n' "$best_iter" "$best_path"
}

latest_checkpoint_overall() {
  local best_stage=-1 best_iter=-1 best_path=""
  local stage_index iter path
  for stage_index in "${!STAGE_KEYS[@]}"; do
    IFS=$'\t' read -r iter path < <(latest_checkpoint_for_stage "$stage_index")
    if (( iter > best_iter )); then
      best_stage="$stage_index"
      best_iter="$iter"
      best_path="$path"
    fi
  done
  printf '%s\t%s\t%s\n' "$best_stage" "$best_iter" "$best_path"
}

find_exact_checkpoint() {
  local requested_iter="$1"
  local stage_index path dir_name run_name mtime best_mtime=-1
  local best_stage=-1 best_path=""
  [[ "$requested_iter" =~ ^[0-9]+$ ]] || {
    echo "Checkpoint must be an integer, got: $requested_iter" >&2
    return 2
  }

  if [[ -d "$LOG_ROOT" ]]; then
    while IFS= read -r -d '' path; do
      dir_name="$(basename "$(dirname "$path")")"
      for stage_index in "${!STAGE_RUN_NAMES[@]}"; do
        run_name="${STAGE_RUN_NAMES[$stage_index]}"
        [[ "$dir_name" == *_"$run_name" ]] || continue
        mtime="$(stat -c %Y "$path")"
        if (( mtime > best_mtime )); then
          best_stage="$stage_index"
          best_mtime="$mtime"
          best_path="$path"
        fi
      done
    done < <(
      find "$LOG_ROOT" -mindepth 2 -maxdepth 2 -type f \
        -name "model_${requested_iter}.pt" -print0
    )
  fi
  printf '%s\t%s\t%s\n' "$best_stage" "$requested_iter" "$best_path"
}

apply_common_environment() {
  export WLG_HISTORICAL_DOMAIN_RAND=1
  export WLG_SPAWN_Z=0.12
  export WLG_RECOVERY_MODE=1
  export WLG_TERRAIN_CURRICULUM=0
  export WLG_SLOPE_THRESHOLD=0.45
  export WLG_COMMAND_CURRICULUM=0
  export WLG_COMMAND_PROFILE=spin_fixed
  # Long holds are closer to joystick operation and avoid repeated +limit to
  # -limit steps before the policy has time to settle.
  export WLG_COMMAND_RESAMPLING_TIME=8.0
  export WLG_REVERSE_CLIMB_FIXED_HEIGHT=-1
  export WLG_STAIR_UP_FIXED_HEIGHT=-1
  export WLG_HIGHSTAND_ANCHOR_FRACTION=0.0
  export WLG_FAIL_TO_TERMINAL_TIME_S=1.0
  export WLG_TERRAIN_PITCH_TERMINATION_DEG=-1
  export WLG_TERRAIN_PITCH_EXCESS_SCALE=0.0
  export WLG_TERRAIN_PITCH_RATE_SCALE=0.0

  export WLG_LIN_VEL_X_MIN=0.0
  export WLG_LIN_VEL_X_MAX=0.0
  export WLG_ANG_VEL_YAW_MIN=-1.0
  export WLG_ANG_VEL_YAW_MAX=1.0
  export WLG_HEIGHT_MIN=0.16
  export WLG_HEIGHT_MAX=0.16

  export WLG_TRACKING_LIN_VEL_SCALE=1.5
  export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.0
  export WLG_TRACKING_ANG_VEL_SCALE=2.0
  export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.5
  # The L1 companion supplies gradient when the Gaussian tracking reward is
  # effectively zero. A wider per-term clip prevents that gradient from being
  # flattened at the high-yaw stages.
  export WLG_TRACKING_ANG_VEL_L1_SCALE=-0.25
  export WLG_CLIP_SINGLE_REWARD=5.0
  export WLG_BASE_HEIGHT_SCALE=2.0
  export WLG_BASE_HEIGHT_ENHANCE_SCALE=1.5
  export WLG_NOMINAL_STATE_SCALE=-1.0
  export WLG_LIN_VEL_Z_SCALE=-0.1
  export WLG_ANG_VEL_XY_SCALE=-0.05
  export WLG_ORIENTATION_SCALE=-3.0
  export WLG_TORQUES_SCALE=-0.0001
  export WLG_COLLISION_SCALE=-1.0
  export WLG_WHEEL_SUPPORT_SCALE=0.25
  export WLG_DOF_POS_LIMITS_SCALE=-1.0
  export WLG_DOF_VEL_SCALE=-0.0001
  export WLG_DOF_ACC_SCALE=-5e-7
  export WLG_ACTION_RATE_SCALE=-0.015
  export WLG_ACTION_SMOOTH_SCALE=-0.02
  # Start with modest exploration; the resumed policy already has a stable
  # wheel-balance solution and only needs to acquire yaw authority.
  export WLG_INIT_NOISE_STD=0.30
  export WLG_ENTROPY_COEF=0.0015
  export WLG_RECOVERY_POSE_SCALE=0.0

  export WLG_HIGH_STAND_LIN_VEL_SCALE=0.0
  export WLG_HIGH_STAND_ANG_VEL_XY_SCALE=0.0
  export WLG_HIGH_STAND_ORIENTATION_SCALE=0.0
  export WLG_HIGH_STAND_ACTION_RATE_SCALE=0.0
  export WLG_HIGH_STAND_ACTION_SMOOTH_SCALE=0.0

  export WLG_SPIN_LOW_HEIGHT_MIN=0.16
  export WLG_SPIN_LOW_HEIGHT_MAX=0.20
  export WLG_SPIN_HIGH_YAW_MIN=1.0
  export WLG_SPIN_FIXED_IDLE_FRACTION=0.10
  export WLG_SPIN_FIXED_NEAR_FRACTION=0.10
  export WLG_SPIN_FIXED_NEAR_MIN_RATIO=0.90
  export WLG_SPIN_HIGH_YAW_LIN_VEL_MAX=0.0
  export WLG_SPIN_MOVING_LIN_VEL_MAX=0.0
  export WLG_SPIN_MOVING_YAW_MAX=1.0
  export WLG_SPIN_MOVING_HEIGHT_MAX=0.28
  export WLG_SPIN_TERRAIN_HEIGHT_MAX=0.26
  export WLG_MIXED_TERRAIN_LIN_VEL_MAX=0.0
  export WLG_MIXED_TERRAIN_YAW_MAX=0.0

  # Keep these deliberately mild while yaw is being acquired. They are
  # tightened stage by stage after the robot demonstrates actual rotation.
  export WLG_SPIN_STATIONARY_LIN_VEL_SCALE=-0.5
  export WLG_SPIN_STATIONARY_ANG_VEL_XY_SCALE=-0.15
  export WLG_SPIN_STATIONARY_ORIENTATION_SCALE=-1.5
  export WLG_SPIN_STATIONARY_ACTION_RATE_SCALE=-0.005
  export WLG_SPIN_STATIONARY_ACTION_SMOOTH_SCALE=-0.008
}

apply_stage_environment() {
  local stage_index="$1"
  apply_common_environment
  export WLG_MESH_TYPE=plane
  export WLG_TERRAIN_PROPORTIONS=1.0,0.0,0.0,0.0,0.0,0.0

  case "${STAGE_KEYS[$stage_index]}" in
    yaw_1)
      ;;
    yaw_2)
      export WLG_ANG_VEL_YAW_MIN=-2.0
      export WLG_ANG_VEL_YAW_MAX=2.0
      export WLG_SPIN_MOVING_YAW_MAX=2.0
      export WLG_SPIN_STATIONARY_ORIENTATION_SCALE=-2.0
      ;;
    yaw_3)
      export WLG_ANG_VEL_YAW_MIN=-3.0
      export WLG_ANG_VEL_YAW_MAX=3.0
      export WLG_SPIN_MOVING_YAW_MAX=3.0
      export WLG_SPIN_STATIONARY_ORIENTATION_SCALE=-2.5
      ;;
    yaw_5)
      export WLG_ANG_VEL_YAW_MIN=-5.0
      export WLG_ANG_VEL_YAW_MAX=5.0
      export WLG_SPIN_MOVING_YAW_MAX=5.0
      export WLG_SPIN_STATIONARY_LIN_VEL_SCALE=-0.75
      export WLG_SPIN_STATIONARY_ORIENTATION_SCALE=-3.0
      ;;
    yaw_7)
      export WLG_ANG_VEL_YAW_MIN=-7.0
      export WLG_ANG_VEL_YAW_MAX=7.0
      export WLG_SPIN_MOVING_YAW_MAX=7.0
      export WLG_SPIN_STATIONARY_LIN_VEL_SCALE=-1.0
      export WLG_SPIN_STATIONARY_ANG_VEL_XY_SCALE=-0.25
      export WLG_SPIN_STATIONARY_ORIENTATION_SCALE=-4.0
      ;;
    yaw_10)
      export WLG_ANG_VEL_YAW_MIN=-10.0
      export WLG_ANG_VEL_YAW_MAX=10.0
      export WLG_SPIN_MOVING_YAW_MAX=10.0
      export WLG_SPIN_STATIONARY_LIN_VEL_SCALE=-1.25
      export WLG_SPIN_STATIONARY_ANG_VEL_XY_SCALE=-0.30
      export WLG_SPIN_STATIONARY_ORIENTATION_SCALE=-4.5
      ;;
    yaw_13)
      export WLG_ANG_VEL_YAW_MIN=-13.0
      export WLG_ANG_VEL_YAW_MAX=13.0
      export WLG_SPIN_MOVING_YAW_MAX=13.0
      export WLG_SPIN_STATIONARY_LIN_VEL_SCALE=-1.5
      export WLG_SPIN_STATIONARY_ANG_VEL_XY_SCALE=-0.35
      export WLG_SPIN_STATIONARY_ORIENTATION_SCALE=-5.0
      export WLG_ENTROPY_COEF=0.0012
      ;;
    yaw_13_stable)
      export WLG_ANG_VEL_YAW_MIN=-13.0
      export WLG_ANG_VEL_YAW_MAX=13.0
      export WLG_SPIN_MOVING_YAW_MAX=13.0
      export WLG_TRACKING_ANG_VEL_L1_SCALE=-0.18
      export WLG_ORIENTATION_SCALE=-6.0
      export WLG_SPIN_STATIONARY_LIN_VEL_SCALE=-2.0
      export WLG_SPIN_STATIONARY_ANG_VEL_XY_SCALE=-0.50
      export WLG_SPIN_STATIONARY_ORIENTATION_SCALE=-6.0
      export WLG_SPIN_STATIONARY_ACTION_RATE_SCALE=-0.015
      export WLG_SPIN_STATIONARY_ACTION_SMOOTH_SCALE=-0.02
      export WLG_ENTROPY_COEF=0.0008
      ;;
    *)
      echo "Unknown SPIN stage: ${STAGE_KEYS[$stage_index]}" >&2
      exit 2
      ;;
  esac
}

print_stage_config() {
  local stage_index="$1"
  printf '  terrain=%s, proportions=%s, profile=%s\n' \
    "$WLG_MESH_TYPE" "$WLG_TERRAIN_PROPORTIONS" "$WLG_COMMAND_PROFILE"
  printf '  vx=[%s,%s], yaw=[%s,%s], height=[%s,%s]\n' \
    "$WLG_LIN_VEL_X_MIN" "$WLG_LIN_VEL_X_MAX" \
    "$WLG_ANG_VEL_YAW_MIN" "$WLG_ANG_VEL_YAW_MAX" \
    "$WLG_HEIGHT_MIN" "$WLG_HEIGHT_MAX"
  printf '  fixed_spin: exact=80%%, near=10%%, idle=10%%\n'
  printf '  yaw_l1=%s, reward_clip=%s, resampling=%ss\n' \
    "$WLG_TRACKING_ANG_VEL_L1_SCALE" "$WLG_CLIP_SINGLE_REWARD" \
    "$WLG_COMMAND_RESAMPLING_TIME"
  printf '  target_checkpoint=%s\n' "${STAGE_TARGETS[$stage_index]}"
}

show_status() {
  local base_path stage_index iter path target
  base_path="$(find_base_checkpoint 2>/dev/null || true)"
  echo "Long-leg SPIN curriculum status"
  echo "Base: ${base_path:-missing model_${BASE_CHECKPOINT}.pt}"
  for stage_index in "${!STAGE_KEYS[@]}"; do
    target="${STAGE_TARGETS[$stage_index]}"
    IFS=$'\t' read -r iter path < <(latest_checkpoint_for_stage "$stage_index")
    if (( iter < 0 )); then
      printf '  [%d/%d] %-44s not started (target %d)\n' \
        "$((stage_index + 1))" "$STAGE_COUNT" "${STAGE_LABELS[$stage_index]}" "$target"
    elif (( iter >= target )); then
      printf '  [%d/%d] %-44s complete: model_%d.pt\n' \
        "$((stage_index + 1))" "$STAGE_COUNT" "${STAGE_LABELS[$stage_index]}" "$iter"
    else
      printf '  [%d/%d] %-44s current: model_%d.pt -> %d\n' \
        "$((stage_index + 1))" "$STAGE_COUNT" "${STAGE_LABELS[$stage_index]}" "$iter" "$target"
    fi
  done
}

check_python_environment() {
  if ! "$PYTHON_BIN" -c 'import torch' >/dev/null 2>&1; then
    echo "Python '$PYTHON_BIN' cannot import torch." >&2
    echo "Activate the Isaac Gym environment first (for example: conda activate leg)." >&2
    exit 1
  fi
}

train_all() {
  local base_path stage_index target iter path source_iter source_path remaining source_run
  check_python_environment
  mkdir -p "$LOG_ROOT"
  cd "$PLANE_ROOT"
  base_path="$(find_base_checkpoint)"

  for stage_index in "${!STAGE_KEYS[@]}"; do
    target="${STAGE_TARGETS[$stage_index]}"
    IFS=$'\t' read -r iter path < <(latest_checkpoint_for_stage "$stage_index")
    if (( iter >= target )); then
      echo "[skip] ${STAGE_LABELS[$stage_index]} already reached model_${iter}.pt"
      continue
    fi

    apply_stage_environment "$stage_index"
    source_iter="$iter"
    source_path="$path"
    if (( source_iter < 0 )); then
      if (( stage_index == 0 )); then
        source_iter="$BASE_CHECKPOINT"
        source_path="$base_path"
      else
        IFS=$'\t' read -r source_iter source_path < <(
          latest_checkpoint_for_stage "$((stage_index - 1))"
        )
        if (( source_iter < ${STAGE_TARGETS[$((stage_index - 1))]} )); then
          echo "Previous SPIN stage is incomplete; refusing to skip it." >&2
          exit 1
        fi
      fi
    fi

    remaining="$((target - source_iter))"
    source_run="$(basename "$(dirname "$source_path")")"
    echo
    echo "================================================================"
    echo "[$((stage_index + 1))/$STAGE_COUNT] ${STAGE_LABELS[$stage_index]}"
    print_stage_config "$stage_index"
    echo "  source=$source_path"
    echo "  add ${remaining} iterations; Ctrl+C is safe"
    echo "================================================================"

    "$PYTHON_BIN" wheel_legged_gym/scripts/train.py \
      --task=wheel_legged \
      --experiment_name=wheel_legged \
      --run_name="${STAGE_RUN_NAMES[$stage_index]}" \
      --resume \
      --load_run="$source_run" \
      --checkpoint="$source_iter" \
      --headless \
      --max_iterations="$remaining"

    IFS=$'\t' read -r iter path < <(latest_checkpoint_for_stage "$stage_index")
    if (( iter < target )); then
      echo "Stage stopped at model_${iter}.pt; rerun 'train' to continue." >&2
      exit 130
    fi
  done

  IFS=$'\t' read -r stage_index iter path < <(latest_checkpoint_overall)
  echo
  echo "All SPIN stages complete. Final checkpoint: $path"
}

play_checkpoint() {
  local requested="${1:-${WLG_SPIN_PLAY_PT:-}}"
  local stage_index iter path run_dir
  if [[ "$requested" == *.pt ]]; then
    path="$(resolve_checkpoint_path "$requested")"
    iter="$(checkpoint_iter_from_path "$path")"
    stage_index="$(stage_index_for_checkpoint "$iter")"
  elif [[ -n "$requested" ]]; then
    IFS=$'\t' read -r stage_index iter path < <(find_exact_checkpoint "$requested")
  else
    IFS=$'\t' read -r stage_index iter path < <(latest_checkpoint_overall)
  fi
  if (( stage_index < 0 )); then
    echo "No matching SPIN checkpoint was found." >&2
    exit 1
  fi

  apply_stage_environment "$stage_index"
  export WLG_PLAY_HEIGHT="${WLG_PLAY_HEIGHT:-0.16}"
  run_dir="$(basename "$(dirname "$path")")"
  cd "$PLANE_ROOT"
  echo "Playing ${STAGE_LABELS[$stage_index]}: $path"
  print_stage_config "$stage_index"
  echo "Keyboard: A/D yaw, W/S translation, X/C height, E stop, Q quit"
  exec "$PYTHON_BIN" wheel_legged_gym/scripts/play.py \
    --task=wheel_legged \
    --experiment_name=wheel_legged \
    --load_run="$run_dir" \
    --checkpoint="$iter" \
    --num_envs="$PLAY_NUM_ENVS"
}

export_latest() {
  local stage_index iter path run_dir output_path
  IFS=$'\t' read -r stage_index iter path < <(latest_checkpoint_overall)
  if (( stage_index < 0 )); then
    echo "No SPIN checkpoint was found." >&2
    exit 1
  fi
  check_python_environment
  run_dir="$(basename "$(dirname "$path")")"
  output_path="${WLG_EXPORT_OUT:-$PLANE_ROOT/export_onnx/longlegs_spin_model_${iter}.onnx}"
  cd "$PLANE_ROOT"
  "$PYTHON_BIN" export_onnx/export_onnx.py \
    --load_run="$run_dir" \
    --checkpoint="$iter" \
    --out="$output_path"
  ls -lh "$output_path"
}

on_interrupt() {
  echo
  echo "Interrupted. The newest SPIN checkpoint remains available; rerun 'train' to resume."
  exit 130
}
trap on_interrupt INT TERM

case "${1:-train}" in
  train) train_all ;;
  status) show_status ;;
  play) play_checkpoint "${2:-}" ;;
  export) export_latest ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
