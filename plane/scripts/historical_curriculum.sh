#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROOT="$PLANE_ROOT/logs/wheel_legged"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLAY_NUM_ENVS="${WLG_PLAY_NUM_ENVS:-5}"

# Each target is an absolute checkpoint number. Every stage therefore adds
# exactly 2000 PPO iterations, including after an interrupted/resumed run.
STAGE_KEYS=(
  flat_base
  stairs_base
  stairs_speed
  stairs_yaw
  mixed_v1
  mixed_v2
  mixed_v3
)
STAGE_LABELS=(
  "上台阶1：平地基础"
  "上台阶2：上下楼梯"
  "上台阶3：楼梯速度强化"
  "angz+：楼梯转向强化"
  "随机地形 v1"
  "随机地形 v2"
  "随机地形 v3：速度上限 2.8"
)
STAGE_TARGETS=(2000 4000 6000 8000 10000 12000 14000)
STAGE_RUN_NAMES=(
  hist_longlegs_s01_flat_base
  hist_longlegs_s02_stairs_base
  hist_longlegs_s03_stairs_speed
  hist_longlegs_s04_stairs_yaw
  hist_longlegs_s05_mixed_v1
  hist_longlegs_s06_mixed_v2
  hist_longlegs_s07_mixed_v3
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/historical_curriculum.sh train    # start or resume, then run all remaining stages
  bash scripts/historical_curriculum.sh status   # show detected stage/checkpoints
  bash scripts/historical_curriculum.sh play     # play the newest checkpoint (default: 5 robots)
  bash scripts/historical_curriculum.sh play 2000 # play one exact checkpoint
  bash scripts/historical_curriculum.sh export   # export the newest checkpoint to ONNX

Environment overrides:
  PYTHON_BIN=python                 Python in the Isaac Gym environment
  WLG_PLAY_NUM_ENVS=5               number of robots used by play
  WLG_EXPORT_OUT=/path/policy.onnx  optional ONNX output path
EOF
}

latest_checkpoint_for_stage() {
  local stage_index="$1"
  local run_name="${STAGE_RUN_NAMES[$stage_index]}"
  local best_iter=-1
  local best_mtime=-1
  local best_path=""
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
  local best_stage=-1
  local best_iter=-1
  local best_path=""
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
  local stage_index path dir_name run_name best_mtime=-1 mtime
  local best_stage=-1
  local best_path=""

  [[ "$requested_iter" =~ ^[0-9]+$ ]] || {
    echo "Checkpoint must be a non-negative integer, got: $requested_iter" >&2
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
    done < <(find "$LOG_ROOT" -mindepth 2 -maxdepth 2 -type f \
      -name "model_${requested_iter}.pt" -print0)
  fi

  printf '%s\t%s\t%s\n' "$best_stage" "$requested_iter" "$best_path"
}

apply_common_environment() {
  # Match the historical Stable training dynamics. These are intentionally
  # stronger than the recent reduced-randomization experiments.
  export WLG_HISTORICAL_DOMAIN_RAND=1
  export WLG_TERRAIN_CURRICULUM=1
  export WLG_TERRAIN_PROGRESS_FRACTION=0.5
  export WLG_MAX_INIT_TERRAIN_LEVEL=5
  export WLG_SLOPE_THRESHOLD=0.75
  export WLG_COMMAND_PROFILE=independent
  export WLG_REVERSE_CLIMB_FIXED_HEIGHT=-1
  export WLG_TRACKING_ANG_VEL_SCALE=1.0
  export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.0
  export WLG_NOMINAL_STATE_SCALE=-1.0
  export WLG_LIN_VEL_Z_SCALE=-0.1
  export WLG_TORQUES_SCALE=-0.0001
  export WLG_COLLISION_SCALE=-1.0
  export WLG_DOF_POS_LIMITS_SCALE=-1.0
  export WLG_CUSTOM_TERRAIN_MODE=descent_discrete
}

apply_stage_environment() {
  local stage_index="$1"
  apply_common_environment

  # Defaults shared by historical stages 1-4.
  export WLG_MESH_TYPE=trimesh
  export WLG_TERRAIN_PROPORTIONS=0.0,0.0,0.0,0.5,0.5,0.0
  export WLG_COMMAND_CURRICULUM=0
  export WLG_BASIC_MAX_CURRICULUM=2.5
  export WLG_ADVANCED_MAX_CURRICULUM=1.5
  export WLG_BASIC_MAX_ANG_VEL_CURRICULUM=6.0
  export WLG_ADVANCED_MAX_ANG_VEL_CURRICULUM=6.0
  export WLG_COMMAND_RESAMPLING_TIME=5.0
  export WLG_LIN_VEL_X_MIN=-2.0
  export WLG_LIN_VEL_X_MAX=2.0
  export WLG_ANG_VEL_YAW_MIN=-2.0
  export WLG_ANG_VEL_YAW_MAX=2.0
  export WLG_HEIGHT_MIN=0.23
  export WLG_HEIGHT_MAX=0.33
  export WLG_TRACKING_LIN_VEL_SCALE=1.0
  export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.0
  export WLG_BASE_HEIGHT_SCALE=2.0
  export WLG_BASE_HEIGHT_ENHANCE_SCALE=1.5
  export WLG_ANG_VEL_XY_SCALE=-0.05
  export WLG_ORIENTATION_SCALE=-10.0
  export WLG_DOF_VEL_SCALE=-5e-5
  export WLG_DOF_ACC_SCALE=-2.5e-7
  export WLG_ACTION_RATE_SCALE=-0.05
  export WLG_ACTION_SMOOTH_SCALE=-0.05
  export WLG_OBS_LIN_VEL_SCALE=2.0

  case "${STAGE_KEYS[$stage_index]}" in
    flat_base)
      export WLG_MESH_TYPE=plane
      export WLG_TERRAIN_PROPORTIONS=0.2,0.2,0.2,0.1,0.2,0.1
      export WLG_HEIGHT_MIN=0.10
      export WLG_HEIGHT_MAX=0.33
      ;;
    stairs_base)
      ;;
    stairs_speed)
      export WLG_LIN_VEL_X_MIN=-2.3
      export WLG_LIN_VEL_X_MAX=2.3
      export WLG_TRACKING_LIN_VEL_SCALE=1.5
      export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.5
      export WLG_BASE_HEIGHT_SCALE=1.0
      export WLG_BASE_HEIGHT_ENHANCE_SCALE=1.0
      export WLG_OBS_LIN_VEL_SCALE=3.0
      ;;
    stairs_yaw)
      export WLG_LIN_VEL_X_MIN=-2.3
      export WLG_LIN_VEL_X_MAX=2.3
      export WLG_ANG_VEL_YAW_MIN=-5.0
      export WLG_ANG_VEL_YAW_MAX=5.0
      export WLG_HEIGHT_MIN=0.17
      export WLG_TRACKING_LIN_VEL_SCALE=1.5
      export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.5
      export WLG_BASE_HEIGHT_SCALE=1.0
      export WLG_BASE_HEIGHT_ENHANCE_SCALE=1.0
      export WLG_OBS_LIN_VEL_SCALE=3.0
      ;;
    mixed_v1|mixed_v2|mixed_v3)
      export WLG_TERRAIN_PROPORTIONS=0.3,0.2,0.0,0.3,0.2,0.0
      export WLG_COMMAND_CURRICULUM=1
      export WLG_BASIC_MAX_CURRICULUM=2.5
      export WLG_ADVANCED_MAX_CURRICULUM=2.5
      export WLG_LIN_VEL_X_MIN=-2.0
      export WLG_LIN_VEL_X_MAX=2.0
      export WLG_ANG_VEL_YAW_MIN=-4.0
      export WLG_ANG_VEL_YAW_MAX=4.0
      export WLG_HEIGHT_MIN=0.09
      export WLG_HEIGHT_MAX=0.33
      # Historical code multiplied both linear tracking functions by 1.3.
      export WLG_TRACKING_LIN_VEL_SCALE=1.3
      export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.3
      export WLG_BASE_HEIGHT_SCALE=1.0
      export WLG_BASE_HEIGHT_ENHANCE_SCALE=1.0
      export WLG_ANG_VEL_XY_SCALE=-0.05
      export WLG_ORIENTATION_SCALE=-20.0
      export WLG_DOF_VEL_SCALE=-2e-5
      export WLG_DOF_ACC_SCALE=-1e-7
      export WLG_ACTION_RATE_SCALE=-0.01
      export WLG_ACTION_SMOOTH_SCALE=-0.01
      export WLG_OBS_LIN_VEL_SCALE=3.0
      if [[ "${STAGE_KEYS[$stage_index]}" == mixed_v3 ]]; then
        export WLG_BASIC_MAX_CURRICULUM=2.8
        export WLG_ADVANCED_MAX_CURRICULUM=2.8
        export WLG_COMMAND_RESAMPLING_TIME=10.0
        export WLG_ANG_VEL_XY_SCALE=-0.07
        export WLG_DOF_VEL_SCALE=-5e-5
        export WLG_DOF_ACC_SCALE=-2.5e-7
        export WLG_ACTION_RATE_SCALE=-0.05
        export WLG_ACTION_SMOOTH_SCALE=-0.05
      fi
      ;;
    *)
      echo "Unknown stage index: $stage_index" >&2
      exit 2
      ;;
  esac
}

print_stage_config() {
  local stage_index="$1"
  printf '  terrain=%s, proportions=%s\n' "$WLG_MESH_TYPE" "$WLG_TERRAIN_PROPORTIONS"
  printf '  vx=[%s,%s], yaw=[%s,%s], height=[%s,%s]\n' \
    "$WLG_LIN_VEL_X_MIN" "$WLG_LIN_VEL_X_MAX" \
    "$WLG_ANG_VEL_YAW_MIN" "$WLG_ANG_VEL_YAW_MAX" \
    "$WLG_HEIGHT_MIN" "$WLG_HEIGHT_MAX"
  printf '  command_curriculum=%s, target_checkpoint=%s\n' \
    "$WLG_COMMAND_CURRICULUM" "${STAGE_TARGETS[$stage_index]}"
}

show_status() {
  local stage_index iter path target
  echo "Historical long-leg curriculum status"
  echo "Log root: $LOG_ROOT"
  for stage_index in "${!STAGE_KEYS[@]}"; do
    target="${STAGE_TARGETS[$stage_index]}"
    IFS=$'\t' read -r iter path < <(latest_checkpoint_for_stage "$stage_index")
    if (( iter < 0 )); then
      printf '  [%d/7] %-42s not started (target %d)\n' \
        "$((stage_index + 1))" "${STAGE_LABELS[$stage_index]}" "$target"
    elif (( iter >= target )); then
      printf '  [%d/7] %-42s complete: model_%d.pt\n' \
        "$((stage_index + 1))" "${STAGE_LABELS[$stage_index]}" "$iter"
    else
      printf '  [%d/7] %-42s current: model_%d.pt -> %d\n' \
        "$((stage_index + 1))" "${STAGE_LABELS[$stage_index]}" "$iter" "$target"
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
  local stage_index target iter path source_iter source_path remaining
  local source_run final_stage final_iter final_path

  check_python_environment
  mkdir -p "$LOG_ROOT"
  cd "$PLANE_ROOT"

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
    if (( source_iter < 0 && stage_index > 0 )); then
      IFS=$'\t' read -r source_iter source_path < <(
        latest_checkpoint_for_stage "$((stage_index - 1))"
      )
      if (( source_iter < ${STAGE_TARGETS[$((stage_index - 1))]} )); then
        echo "Previous stage has no completed checkpoint; refusing to skip it." >&2
        exit 1
      fi
    fi

    if (( source_iter < 0 )); then
      source_iter=0
    fi
    remaining="$((target - source_iter))"
    echo
    echo "================================================================"
    echo "[$((stage_index + 1))/7] ${STAGE_LABELS[$stage_index]}"
    print_stage_config "$stage_index"
    echo "  start=model_${source_iter}.pt, add ${remaining} iterations"
    echo "  Ctrl+C is safe; rerun this script to resume from the newest saved checkpoint."
    echo "================================================================"

    if [[ -n "$source_path" ]]; then
      source_run="$(basename "$(dirname "$source_path")")"
      "$PYTHON_BIN" wheel_legged_gym/scripts/train.py \
        --task=wheel_legged \
        --experiment_name=wheel_legged \
        --run_name="${STAGE_RUN_NAMES[$stage_index]}" \
        --resume \
        --load_run="$source_run" \
        --checkpoint="$source_iter" \
        --headless \
        --max_iterations="$remaining"
    else
      "$PYTHON_BIN" wheel_legged_gym/scripts/train.py \
        --task=wheel_legged \
        --experiment_name=wheel_legged \
        --run_name="${STAGE_RUN_NAMES[$stage_index]}" \
        --headless \
        --max_iterations="$remaining"
    fi

    IFS=$'\t' read -r iter path < <(latest_checkpoint_for_stage "$stage_index")
    if (( iter < target )); then
      echo "Stage stopped at model_${iter}.pt; run 'train' again to continue." >&2
      exit 130
    fi
  done

  echo
  echo "All seven stages are complete."
  IFS=$'\t' read -r final_stage final_iter final_path < <(latest_checkpoint_overall)
  echo "Final checkpoint: $final_path"
}

play_checkpoint() {
  local requested_iter="${1:-}"
  local stage_index iter path run_dir
  if [[ -n "$requested_iter" ]]; then
    IFS=$'\t' read -r stage_index iter path < <(find_exact_checkpoint "$requested_iter")
  else
    IFS=$'\t' read -r stage_index iter path < <(latest_checkpoint_overall)
  fi
  if (( stage_index < 0 )); then
    if [[ -n "$requested_iter" ]]; then
      echo "No model_${requested_iter}.pt created by this curriculum was found." >&2
    else
      echo "No checkpoint created by this curriculum was found." >&2
    fi
    exit 1
  fi
  apply_stage_environment "$stage_index"
  run_dir="$(basename "$(dirname "$path")")"
  cd "$PLANE_ROOT"
  echo "Playing ${STAGE_LABELS[$stage_index]}: $path"
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
  if (( stage_index < 0 )); then
    echo "No checkpoint created by this curriculum was found." >&2
    exit 1
  fi
  check_python_environment
  run_dir="$(basename "$(dirname "$path")")"
  output_path="${WLG_EXPORT_OUT:-$PLANE_ROOT/export_onnx/historical_longlegs_model_${iter}.onnx}"
  cd "$PLANE_ROOT"
  "$PYTHON_BIN" export_onnx/export_onnx.py \
    --load_run="$run_dir" \
    --checkpoint="$iter" \
    --out="$output_path"
  ls -lh "$output_path"
}

on_interrupt() {
  echo
  echo "Interrupted. The newest model_*.pt remains available; use 'status', 'play', or rerun 'train'."
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
