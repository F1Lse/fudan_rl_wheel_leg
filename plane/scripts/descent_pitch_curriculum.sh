#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROOT="$PLANE_ROOT/logs/wheel_legged"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLAY_NUM_ENVS="${WLG_PLAY_NUM_ENVS:-20}"

BASE_CHECKPOINT="${WLG_DESCENT_BASE_CHECKPOINT:-56500}"
BASE_RUN_NAME="${WLG_DESCENT_BASE_RUN_NAME:-flat_stability_longlegs_s05_final_mixed_consolidation}"

STAGE_KEYS=(pitch_adaptation controlled_speed final_fast)
STAGE_LABELS=(
  "低速适应：认识俯仰约束"
  "中速强化：压低台阶冲击俯仰"
  "快速专训：20度目标巩固"
)
STAGE_TARGETS=(58000 60000 62500)
STAGE_RUN_NAMES=(
  descent_pitch_longlegs_s01_adaptation
  descent_pitch_longlegs_s02_controlled_speed
  descent_pitch_longlegs_s03_final_fast
)
STAGE_COUNT="${#STAGE_KEYS[@]}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/descent_pitch_curriculum.sh train
  bash scripts/descent_pitch_curriculum.sh status
  bash scripts/descent_pitch_curriculum.sh play [checkpoint]
  bash scripts/descent_pitch_curriculum.sh export

The branch starts from model_56500.pt. If auto-discovery is ambiguous, provide:

  WLG_DESCENT_BASE_PT=logs/wheel_legged/RUN/model_56500.pt \
    bash scripts/descent_pitch_curriculum.sh train

Play the two focused terrains (the default is the measured double drop):

  WLG_PLAY_FOCUS_TERRAIN=custom_drop bash scripts/descent_pitch_curriculum.sh play
  WLG_PLAY_FOCUS_TERRAIN=pyramid_climb bash scripts/descent_pitch_curriculum.sh play

Optional:
  PYTHON_BIN=python
  WLG_PLAY_NUM_ENVS=20
  WLG_PLAY_HEIGHT=0.20
  WLG_PLAY_LIN_VEL_CMD=1.5
  WLG_EXPORT_OUT=export_onnx/longlegs_descent_pitch.onnx
EOF
}

checkpoint_iter_from_path() {
  local name
  name="$(basename "$1")"
  name="${name#model_}"
  printf '%s\n' "${name%.pt}"
}

find_base_checkpoint() {
  local explicit_path="${WLG_DESCENT_BASE_PT:-}"
  local path dir_name mtime best_mtime=-1 best_path=""

  if [[ -n "$explicit_path" ]]; then
    [[ "$explicit_path" == /* ]] || explicit_path="$PLANE_ROOT/$explicit_path"
    [[ -f "$explicit_path" ]] || {
      echo "Base checkpoint not found: $explicit_path" >&2
      return 1
    }
    [[ "$(checkpoint_iter_from_path "$explicit_path")" == "$BASE_CHECKPOINT" ]] || {
      echo "WLG_DESCENT_BASE_PT must point to model_${BASE_CHECKPOINT}.pt" >&2
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
    echo "Could not find model_${BASE_CHECKPOINT}.pt from $BASE_RUN_NAME." >&2
    echo "Set WLG_DESCENT_BASE_PT to its exact path." >&2
    return 1
  }
  printf '%s\n' "$best_path"
}

latest_checkpoint_for_stage() {
  local stage_index="$1" run_name="${STAGE_RUN_NAMES[$1]}"
  local best_iter=-1 best_mtime=-1 best_path=""
  local path dir_name name iter mtime
  if [[ -d "$LOG_ROOT" ]]; then
    while IFS= read -r -d '' path; do
      dir_name="$(basename "$(dirname "$path")")"
      [[ "$dir_name" == *_"$run_name" ]] || continue
      name="$(basename "$path")"
      iter="${name#model_}"
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
  local best_stage=-1 best_iter=-1 best_path="" stage_index iter path
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
  local requested_iter="$1" stage_index path dir_name run_name mtime
  local best_stage=-1 best_mtime=-1 best_path=""
  [[ "$requested_iter" =~ ^[0-9]+$ ]] || {
    echo "Checkpoint must be an integer: $requested_iter" >&2
    return 2
  }
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
  printf '%s\t%s\t%s\n' "$best_stage" "$requested_iter" "$best_path"
}

apply_common_environment() {
  export WLG_HISTORICAL_DOMAIN_RAND=1
  export WLG_SPAWN_Z=0.12
  export WLG_RECOVERY_MODE=1

  export WLG_MESH_TYPE=trimesh
  # 35% negative-height pyramid stairs (climb away from the centre),
  # 65% measured 50 mm curb + 150/200 mm double-drop course.
  export WLG_TERRAIN_PROPORTIONS=0,0,0,0.35,0,0.65
  export WLG_CUSTOM_TERRAIN_MODE=descent_focus
  export WLG_TERRAIN_CURRICULUM=1
  export WLG_TERRAIN_PROGRESS_FRACTION=0.35
  export WLG_SLOPE_THRESHOLD=0.45

  export WLG_COMMAND_CURRICULUM=0
  export WLG_COMMAND_PROFILE=mixed_final
  export WLG_COMMAND_RESAMPLING_TIME=4.0
  export WLG_LIN_VEL_X_MIN=-2.2
  export WLG_LIN_VEL_X_MAX=2.2
  export WLG_ANG_VEL_YAW_MIN=-0.5
  export WLG_ANG_VEL_YAW_MAX=0.5
  export WLG_HEIGHT_MIN=0.16
  export WLG_HEIGHT_MAX=0.24
  export WLG_MIXED_TERRAIN_YAW_MAX=0.5
  export WLG_MIXED_CUSTOM_YAW_MAX=0.35
  export WLG_REVERSE_CLIMB_FIXED_HEIGHT=-1

  export WLG_TRACKING_LIN_VEL_SCALE=2.0
  export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=2.0
  export WLG_TRACKING_ANG_VEL_SCALE=0.8
  export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.0
  export WLG_BASE_HEIGHT_SCALE=3.0
  export WLG_BASE_HEIGHT_ENHANCE_SCALE=3.0
  export WLG_BASE_HEIGHT_L1_SCALE=-1.0
  export WLG_NOMINAL_STATE_SCALE=-1.0
  export WLG_LIN_VEL_Z_SCALE=-0.5
  export WLG_ANG_VEL_XY_SCALE=-0.25
  export WLG_ORIENTATION_SCALE=-12.0
  export WLG_TORQUES_SCALE=-0.0001
  export WLG_COLLISION_SCALE=-1.0
  export WLG_WHEEL_SUPPORT_SCALE=1.0
  export WLG_DOF_POS_LIMITS_SCALE=-1.0
  export WLG_DOF_VEL_SCALE=-0.0002
  export WLG_DOF_ACC_SCALE=-1e-6
  export WLG_ACTION_RATE_SCALE=-0.03
  export WLG_ACTION_SMOOTH_SCALE=-0.05
  export WLG_INIT_NOISE_STD=0.30
  export WLG_ENTROPY_COEF=0.001

  export WLG_HIGH_STAND_LIN_VEL_SCALE=0.0
  export WLG_HIGH_STAND_ANG_VEL_XY_SCALE=0.0
  export WLG_HIGH_STAND_ORIENTATION_SCALE=0.0
  export WLG_HIGH_STAND_ACTION_RATE_SCALE=0.0
  export WLG_HIGH_STAND_ACTION_SMOOTH_SCALE=0.0
  export WLG_SPIN_STATIONARY_LIN_VEL_SCALE=0.0
  export WLG_SPIN_STATIONARY_ANG_VEL_XY_SCALE=0.0
  export WLG_SPIN_STATIONARY_ORIENTATION_SCALE=0.0
  export WLG_SPIN_STATIONARY_ACTION_RATE_SCALE=0.0
  export WLG_SPIN_STATIONARY_ACTION_SMOOTH_SCALE=0.0
}

apply_stage_environment() {
  local stage_index="$1"
  apply_common_environment
  case "${STAGE_KEYS[$stage_index]}" in
    pitch_adaptation)
      export WLG_MAX_INIT_TERRAIN_LEVEL=3
      export WLG_MIXED_TERRAIN_LIN_VEL_MAX=0.8
      export WLG_MIXED_CUSTOM_LIN_VEL_MAX=1.0
      export WLG_STAIR_UP_FIXED_HEIGHT=0.26
      export WLG_TERRAIN_PITCH_SOFT_LIMIT_DEG=15
      export WLG_TERRAIN_PITCH_TERMINATION_DEG=45
      export WLG_FAIL_TO_TERMINAL_TIME_S=0.20
      export WLG_TERRAIN_PITCH_EXCESS_SCALE=-30.0
      export WLG_TERRAIN_PITCH_RATE_SCALE=-0.25
      ;;
    controlled_speed)
      export WLG_MAX_INIT_TERRAIN_LEVEL=5
      export WLG_MIXED_TERRAIN_LIN_VEL_MAX=1.2
      export WLG_MIXED_CUSTOM_LIN_VEL_MAX=1.6
      export WLG_STAIR_UP_FIXED_HEIGHT=0.30
      export WLG_TERRAIN_PITCH_SOFT_LIMIT_DEG=12
      export WLG_TERRAIN_PITCH_TERMINATION_DEG=32
      export WLG_FAIL_TO_TERMINAL_TIME_S=0.15
      export WLG_TERRAIN_PITCH_EXCESS_SCALE=-45.0
      export WLG_TERRAIN_PITCH_RATE_SCALE=-0.40
      ;;
    final_fast)
      export WLG_MAX_INIT_TERRAIN_LEVEL=6
      export WLG_MIXED_TERRAIN_LIN_VEL_MAX=1.6
      export WLG_MIXED_CUSTOM_LIN_VEL_MAX=2.2
      export WLG_STAIR_UP_FIXED_HEIGHT=0.33
      export WLG_TERRAIN_PITCH_SOFT_LIMIT_DEG=10
      export WLG_TERRAIN_PITCH_TERMINATION_DEG=24
      export WLG_FAIL_TO_TERMINAL_TIME_S=0.10
      export WLG_TERRAIN_PITCH_EXCESS_SCALE=-65.0
      export WLG_TERRAIN_PITCH_RATE_SCALE=-0.60
      ;;
    *) echo "Unknown stage: ${STAGE_KEYS[$stage_index]}" >&2; exit 2 ;;
  esac
}

print_stage_config() {
  local stage_index="$1"
  printf '  terrain: 35%% pyramid climb + 65%% measured double drop\n'
  printf '  climb_speed=%s, descent_speed=%s, climb_height=%s\n' \
    "$WLG_MIXED_TERRAIN_LIN_VEL_MAX" "$WLG_MIXED_CUSTOM_LIN_VEL_MAX" \
    "$WLG_STAIR_UP_FIXED_HEIGHT"
  printf '  pitch soft=%s deg, terminal=%s deg after %ss\n' \
    "$WLG_TERRAIN_PITCH_SOFT_LIMIT_DEG" \
    "$WLG_TERRAIN_PITCH_TERMINATION_DEG" "$WLG_FAIL_TO_TERMINAL_TIME_S"
  printf '  target_checkpoint=%s\n' "${STAGE_TARGETS[$stage_index]}"
}

check_python_environment() {
  if ! "$PYTHON_BIN" -c 'import torch' >/dev/null 2>&1; then
    echo "Python '$PYTHON_BIN' cannot import torch; activate the Isaac Gym environment." >&2
    exit 1
  fi
}

show_status() {
  local base_path stage_index iter path target
  base_path="$(find_base_checkpoint 2>/dev/null || true)"
  echo "Focused descent/pyramid-climb curriculum"
  echo "Base: ${base_path:-missing model_${BASE_CHECKPOINT}.pt}"
  for stage_index in "${!STAGE_KEYS[@]}"; do
    target="${STAGE_TARGETS[$stage_index]}"
    IFS=$'\t' read -r iter path < <(latest_checkpoint_for_stage "$stage_index")
    if (( iter < 0 )); then
      printf '  [%d/%d] %-34s not started -> %d\n' \
        "$((stage_index + 1))" "$STAGE_COUNT" "${STAGE_LABELS[$stage_index]}" "$target"
    elif (( iter >= target )); then
      printf '  [%d/%d] %-34s complete: model_%d.pt\n' \
        "$((stage_index + 1))" "$STAGE_COUNT" "${STAGE_LABELS[$stage_index]}" "$iter"
    else
      printf '  [%d/%d] %-34s model_%d.pt -> %d\n' \
        "$((stage_index + 1))" "$STAGE_COUNT" "${STAGE_LABELS[$stage_index]}" "$iter" "$target"
    fi
  done
}

train_all() {
  local base_path stage_index target iter path source_iter source_path source_run remaining
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
          echo "Previous stage is incomplete; refusing to skip it." >&2
          exit 1
        fi
      fi
    fi

    apply_stage_environment "$stage_index"
    remaining="$((target - source_iter))"
    source_run="$(basename "$(dirname "$source_path")")"
    echo
    echo "================================================================"
    echo "[$((stage_index + 1))/$STAGE_COUNT] ${STAGE_LABELS[$stage_index]}"
    print_stage_config "$stage_index"
    echo "  source=$source_path"
    echo "  add $remaining iterations; Ctrl+C is safe and rerun resumes"
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
      echo "Stage stopped at model_${iter}.pt; rerun train to resume." >&2
      exit 130
    fi
  done
}

play_checkpoint() {
  local requested_iter="${1:-}" stage_index iter path run_dir
  if [[ -n "$requested_iter" ]]; then
    IFS=$'\t' read -r stage_index iter path < <(find_exact_checkpoint "$requested_iter")
  else
    IFS=$'\t' read -r stage_index iter path < <(latest_checkpoint_overall)
  fi
  (( stage_index >= 0 )) || { echo "No focused checkpoint found." >&2; exit 1; }
  apply_stage_environment "$stage_index"
  export WLG_PLAY_HEIGHT="${WLG_PLAY_HEIGHT:-0.20}"
  export WLG_PLAY_LIN_VEL_CMD="${WLG_PLAY_LIN_VEL_CMD:-1.5}"
  export WLG_PLAY_FOCUS_TERRAIN="${WLG_PLAY_FOCUS_TERRAIN:-custom_drop}"
  run_dir="$(basename "$(dirname "$path")")"
  cd "$PLANE_ROOT"
  echo "Playing $path (focus=$WLG_PLAY_FOCUS_TERRAIN)"
  print_stage_config "$stage_index"
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
  (( stage_index >= 0 )) || { echo "No focused checkpoint found." >&2; exit 1; }
  check_python_environment
  run_dir="$(basename "$(dirname "$path")")"
  output_path="${WLG_EXPORT_OUT:-$PLANE_ROOT/export_onnx/longlegs_descent_pitch_model_${iter}.onnx}"
  cd "$PLANE_ROOT"
  "$PYTHON_BIN" export_onnx/export_onnx.py \
    --load_run="$run_dir" \
    --checkpoint="$iter" \
    --out="$output_path"
  ls -lh "$output_path"
}

trap 'echo; echo "Interrupted; rerun train to resume from the newest saved checkpoint."; exit 130' INT TERM

case "${1:-train}" in
  train) train_all ;;
  status) show_status ;;
  play) play_checkpoint "${2:-}" ;;
  export) export_latest ;;
  help|-h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
