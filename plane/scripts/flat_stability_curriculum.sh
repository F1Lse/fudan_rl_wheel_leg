#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROOT="$PLANE_ROOT/logs/wheel_legged"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLAY_NUM_ENVS="${WLG_PLAY_NUM_ENVS:-5}"

BASE_CHECKPOINT="${WLG_FLAT_STABILITY_BASE_CHECKPOINT:-47000}"
BASE_RUN_NAME="${WLG_FLAT_STABILITY_BASE_RUN_NAME:-hist_recovery_v4_longlegs_s22_highstand_anchor_consolidation}"

STAGE_KEYS=(
  idle_all_height
  low_motion
  terrain_stability_low
  terrain_stability_mid
  final_mixed_consolidation
)
STAGE_LABELS=(
  "全高度静止稳定与高度跟踪"
  "全高度静止锚点加低速恢复"
  "稳定混合地形：低速上下楼梯与弱项复习"
  "稳定混合地形：恢复中速上下楼梯"
  "最终混合巩固：全高度静止、平地与上下楼梯"
)
STAGE_TARGETS=(48000 49000 51500 54000 56500)
STAGE_RUN_NAMES=(
  flat_stability_longlegs_s01_idle_all_height
  flat_stability_longlegs_s02_low_motion
  flat_stability_longlegs_s03_terrain_stability_low
  flat_stability_longlegs_s04_terrain_stability_mid
  flat_stability_longlegs_s05_final_mixed_consolidation
)
STAGE_COUNT="${#STAGE_KEYS[@]}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/flat_stability_curriculum.sh train
  bash scripts/flat_stability_curriculum.sh status
  bash scripts/flat_stability_curriculum.sh play
  bash scripts/flat_stability_curriculum.sh play 48000
  bash scripts/flat_stability_curriculum.sh export

The first stage resumes from model_47000.pt. Override its location when needed:

  WLG_FLAT_STABILITY_BASE_PT=logs/wheel_legged/RUN/model_47000.pt \
    bash scripts/flat_stability_curriculum.sh train

Optional:
  PYTHON_BIN=python
  WLG_PLAY_NUM_ENVS=1
  WLG_PLAY_HEIGHT=0.20
  WLG_PLAY_LIN_VEL_CMD=0.5
  WLG_PLAY_YAW_STEP=1.0
  WLG_EXPORT_OUT=export_onnx/longlegs_flat_stable.onnx
EOF
}

checkpoint_iter_from_path() {
  local name
  name="$(basename "$1")"
  name="${name#model_}"
  printf '%s\n' "${name%.pt}"
}

find_base_checkpoint() {
  local explicit_path="${WLG_FLAT_STABILITY_BASE_PT:-}"
  local path dir_name mtime best_mtime=-1 best_path=""

  if [[ -n "$explicit_path" ]]; then
    [[ "$explicit_path" == /* ]] || explicit_path="$PLANE_ROOT/$explicit_path"
    [[ -f "$explicit_path" ]] || {
      echo "Base checkpoint not found: $explicit_path" >&2
      return 1
    }
    [[ "$(checkpoint_iter_from_path "$explicit_path")" == "$BASE_CHECKPOINT" ]] || {
      echo "WLG_FLAT_STABILITY_BASE_PT must point to model_${BASE_CHECKPOINT}.pt" >&2
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
    echo "Set WLG_FLAT_STABILITY_BASE_PT to the checkpoint path." >&2
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
  export WLG_MESH_TYPE=plane
  export WLG_TERRAIN_PROPORTIONS=0.2,0.2,0.2,0.1,0.2,0.1
  export WLG_TERRAIN_CURRICULUM=0
  export WLG_TERRAIN_PROGRESS_FRACTION=0.5
  export WLG_MAX_INIT_TERRAIN_LEVEL=5
  export WLG_SLOPE_THRESHOLD=0.45
  export WLG_CUSTOM_TERRAIN_MODE=descent_discrete
  export WLG_COMMAND_CURRICULUM=0
  export WLG_COMMAND_PROFILE=mixed_highstand_anchor
  export WLG_COMMAND_RESAMPLING_TIME=4.0

  export WLG_HEIGHT_MIN=0.16
  export WLG_HEIGHT_MAX=0.33
  export WLG_MIXED_FLAT_HEIGHT_MIN=0.16
  export WLG_MIXED_FLAT_HEIGHT_MAX=0.33
  export WLG_MIXED_TERRAIN_LIN_VEL_MAX=1.2
  export WLG_MIXED_TERRAIN_YAW_MAX=1.5
  export WLG_MIXED_CUSTOM_LIN_VEL_MAX=1.2
  export WLG_MIXED_CUSTOM_YAW_MAX=1.0
  export WLG_HIGHSTAND_ANCHOR_HEIGHT_MIN=0.16
  export WLG_HIGHSTAND_ANCHOR_HEIGHT_MAX=0.33
  export WLG_REVERSE_CLIMB_FIXED_HEIGHT=-1

  export WLG_LIN_VEL_X_MIN=0.0
  export WLG_LIN_VEL_X_MAX=0.0
  export WLG_ANG_VEL_YAW_MIN=0.0
  export WLG_ANG_VEL_YAW_MAX=0.0
  export WLG_HIGHSTAND_ANCHOR_FRACTION=1.0

  export WLG_TRACKING_LIN_VEL_SCALE=1.0
  export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.0
  export WLG_TRACKING_ANG_VEL_SCALE=1.0
  export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.0
  export WLG_BASE_HEIGHT_SCALE=3.0
  export WLG_BASE_HEIGHT_ENHANCE_SCALE=3.0
  export WLG_BASE_HEIGHT_L1_SCALE=-2.0
  export WLG_NOMINAL_STATE_SCALE=-1.0
  export WLG_LIN_VEL_Z_SCALE=-0.2
  export WLG_ANG_VEL_XY_SCALE=-0.15
  export WLG_ORIENTATION_SCALE=-12.0
  export WLG_TORQUES_SCALE=-0.0001
  export WLG_COLLISION_SCALE=-1.0
  export WLG_WHEEL_SUPPORT_SCALE=1.0
  export WLG_DOF_POS_LIMITS_SCALE=-1.0
  export WLG_DOF_VEL_SCALE=-0.0002
  export WLG_DOF_ACC_SCALE=-1e-6
  export WLG_ACTION_RATE_SCALE=-0.03
  export WLG_ACTION_SMOOTH_SCALE=-0.05
  export WLG_INIT_NOISE_STD=0.5
  export WLG_ENTROPY_COEF=0.001
  export WLG_RECOVERY_POSE_SCALE=0.0

  # With the anchor threshold set to 0.16 m these existing masked terms now
  # cover exact-zero commands across the full requested height range.
  export WLG_HIGH_STAND_LIN_VEL_SCALE=-2.0
  export WLG_HIGH_STAND_ANG_VEL_XY_SCALE=-0.4
  export WLG_HIGH_STAND_ORIENTATION_SCALE=-6.0
  export WLG_HIGH_STAND_ACTION_RATE_SCALE=-0.02
  export WLG_HIGH_STAND_ACTION_SMOOTH_SCALE=-0.03

  # Keep the deferred SPIN-specific rewards completely inactive here.
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
    idle_all_height)
      # First find a quiet equilibrium without periodic 2 m/s pushes. The
      # narrower mass/COM/friction randomization remains active.
      export WLG_HISTORICAL_DOMAIN_RAND=0
      ;;
    low_motion)
      export WLG_LIN_VEL_X_MIN=-0.8
      export WLG_LIN_VEL_X_MAX=0.8
      export WLG_ANG_VEL_YAW_MIN=-1.0
      export WLG_ANG_VEL_YAW_MAX=1.0
      export WLG_HIGHSTAND_ANCHOR_FRACTION=0.70
      export WLG_TRACKING_LIN_VEL_SCALE=1.5
      export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.5
      export WLG_TRACKING_ANG_VEL_SCALE=1.5
      export WLG_BASE_HEIGHT_L1_SCALE=-1.5
      export WLG_WHEEL_SUPPORT_SCALE=0.5
      export WLG_ENTROPY_COEF=0.003
      ;;
    terrain_stability_low|terrain_stability_mid|final_mixed_consolidation)
      # Starting directly at model_49000, keep exact-zero/full-height anchors
      # on flat while rehearsing both stair directions, pyramid slopes and the
      # weak forward curb/double-drop descent in every following stage.
      export WLG_MESH_TYPE=trimesh
      export WLG_TERRAIN_PROPORTIONS=0.40,0.15,0.10,0.10,0.10,0.15
      export WLG_TERRAIN_CURRICULUM=1
      export WLG_CUSTOM_TERRAIN_MODE=descent_focus
      export WLG_COMMAND_RESAMPLING_TIME=5.0
      export WLG_LIN_VEL_X_MIN=-2.5
      export WLG_LIN_VEL_X_MAX=2.5
      export WLG_ANG_VEL_YAW_MIN=-3.0
      export WLG_ANG_VEL_YAW_MAX=3.0
      export WLG_HIGHSTAND_ANCHOR_FRACTION=0.60
      export WLG_TRACKING_LIN_VEL_SCALE=1.5
      export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.5
      export WLG_TRACKING_ANG_VEL_SCALE=1.5
      export WLG_BASE_HEIGHT_SCALE=2.5
      export WLG_BASE_HEIGHT_ENHANCE_SCALE=2.0
      export WLG_BASE_HEIGHT_L1_SCALE=-0.75
      export WLG_ANG_VEL_XY_SCALE=-0.10
      export WLG_ORIENTATION_SCALE=-12.0
      export WLG_ACTION_RATE_SCALE=-0.02
      export WLG_ACTION_SMOOTH_SCALE=-0.03
      export WLG_HIGH_STAND_LIN_VEL_SCALE=-1.5
      export WLG_HIGH_STAND_ANG_VEL_XY_SCALE=-0.3
      export WLG_HIGH_STAND_ORIENTATION_SCALE=-5.0
      export WLG_HIGH_STAND_ACTION_RATE_SCALE=-0.015
      export WLG_HIGH_STAND_ACTION_SMOOTH_SCALE=-0.02
      export WLG_WHEEL_SUPPORT_SCALE=0.25
      export WLG_MAX_INIT_TERRAIN_LEVEL=3
      export WLG_MIXED_TERRAIN_LIN_VEL_MAX=1.0
      export WLG_MIXED_TERRAIN_YAW_MAX=1.0
      export WLG_MIXED_CUSTOM_LIN_VEL_MAX=1.0
      export WLG_MIXED_CUSTOM_YAW_MAX=1.0
      export WLG_ENTROPY_COEF=0.003
      if [[ "${STAGE_KEYS[$stage_index]}" == terrain_stability_mid ]]; then
        export WLG_HIGHSTAND_ANCHOR_FRACTION=0.45
        export WLG_MAX_INIT_TERRAIN_LEVEL=4
        export WLG_MIXED_TERRAIN_LIN_VEL_MAX=1.5
        export WLG_MIXED_TERRAIN_YAW_MAX=1.5
        export WLG_MIXED_CUSTOM_LIN_VEL_MAX=1.4
        export WLG_MIXED_CUSTOM_YAW_MAX=1.0
      elif [[ "${STAGE_KEYS[$stage_index]}" == final_mixed_consolidation ]]; then
        export WLG_TERRAIN_PROPORTIONS=0.50,0.10,0.10,0.10,0.10,0.10
        export WLG_HIGHSTAND_ANCHOR_FRACTION=0.30
        export WLG_MAX_INIT_TERRAIN_LEVEL=5
        export WLG_MIXED_TERRAIN_LIN_VEL_MAX=2.0
        export WLG_MIXED_TERRAIN_YAW_MAX=2.0
        export WLG_MIXED_CUSTOM_LIN_VEL_MAX=1.8
        export WLG_MIXED_CUSTOM_YAW_MAX=1.2
        export WLG_ENTROPY_COEF=0.002
      fi
      ;;
    *)
      echo "Unknown flat-stability stage: ${STAGE_KEYS[$stage_index]}" >&2
      exit 2
      ;;
  esac
}

print_stage_config() {
  local stage_index="$1"
  printf '  terrain=%s, proportions=%s, custom=%s\n' \
    "$WLG_MESH_TYPE" "$WLG_TERRAIN_PROPORTIONS" "$WLG_CUSTOM_TERRAIN_MODE"
  printf '  vx=[%s,%s], yaw=[%s,%s], height=[%s,%s]\n' \
    "$WLG_LIN_VEL_X_MIN" "$WLG_LIN_VEL_X_MAX" \
    "$WLG_ANG_VEL_YAW_MIN" "$WLG_ANG_VEL_YAW_MAX" \
    "$WLG_HEIGHT_MIN" "$WLG_HEIGHT_MAX"
  printf '  exact-zero anchors=%s, height_l1=%s, historical_randomization=%s\n' \
    "$WLG_HIGHSTAND_ANCHOR_FRACTION" "$WLG_BASE_HEIGHT_L1_SCALE" \
    "$WLG_HISTORICAL_DOMAIN_RAND"
  printf '  target_checkpoint=%s\n' "${STAGE_TARGETS[$stage_index]}"
}

show_status() {
  local base_path stage_index iter path target
  base_path="$(find_base_checkpoint 2>/dev/null || true)"
  echo "Long-leg flat stability curriculum status"
  echo "Base: ${base_path:-missing model_${BASE_CHECKPOINT}.pt}"
  for stage_index in "${!STAGE_KEYS[@]}"; do
    target="${STAGE_TARGETS[$stage_index]}"
    IFS=$'\t' read -r iter path < <(latest_checkpoint_for_stage "$stage_index")
    if (( iter < 0 )); then
      printf '  [%d/%d] %-36s not started (target %d)\n' \
        "$((stage_index + 1))" "$STAGE_COUNT" "${STAGE_LABELS[$stage_index]}" "$target"
    elif (( iter >= target )); then
      printf '  [%d/%d] %-36s complete: model_%d.pt\n' \
        "$((stage_index + 1))" "$STAGE_COUNT" "${STAGE_LABELS[$stage_index]}" "$iter"
    else
      printf '  [%d/%d] %-36s current: model_%d.pt -> %d\n' \
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
          echo "Previous stage is incomplete; refusing to skip it." >&2
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
      echo "Stage stopped at model_${iter}.pt; rerun 'train' to resume." >&2
      exit 130
    fi
  done

  IFS=$'\t' read -r stage_index iter path < <(latest_checkpoint_overall)
  echo
  echo "All flat-stability stages complete. Final checkpoint: $path"
}

play_checkpoint() {
  local requested_iter="${1:-}" stage_index iter path run_dir
  if [[ -n "$requested_iter" ]]; then
    IFS=$'\t' read -r stage_index iter path < <(find_exact_checkpoint "$requested_iter")
  else
    IFS=$'\t' read -r stage_index iter path < <(latest_checkpoint_overall)
  fi
  if (( stage_index < 0 )); then
    echo "No matching flat-stability checkpoint was found." >&2
    exit 1
  fi
  apply_stage_environment "$stage_index"
  export WLG_PLAY_HEIGHT="${WLG_PLAY_HEIGHT:-0.20}"
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
    echo "No flat-stability checkpoint was found." >&2
    exit 1
  fi
  check_python_environment
  run_dir="$(basename "$(dirname "$path")")"
  output_path="${WLG_EXPORT_OUT:-$PLANE_ROOT/export_onnx/longlegs_flat_stable_model_${iter}.onnx}"
  cd "$PLANE_ROOT"
  "$PYTHON_BIN" export_onnx/export_onnx.py \
    --load_run="$run_dir" \
    --checkpoint="$iter" \
    --out="$output_path"
  ls -lh "$output_path"
}

on_interrupt() {
  echo
  echo "Interrupted. The newest checkpoint remains available; rerun 'train' to resume."
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
