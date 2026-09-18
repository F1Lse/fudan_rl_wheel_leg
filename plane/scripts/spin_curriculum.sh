#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROOT="$PLANE_ROOT/logs/wheel_legged"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLAY_NUM_ENVS="${WLG_PLAY_NUM_ENVS:-20}"

BASE_CHECKPOINT="${WLG_SPIN_BASE_CHECKPOINT:-59000}"
BASE_RUN_NAME="${WLG_SPIN_BASE_RUN_NAME:-spin_centerfix_58000_longlegs_s01_yaw_5}"

STAGE_KEYS=(
  yaw_7_speed
  yaw_7_stability
  yaw_10_speed
  yaw_13_speed
)
STAGE_LABELS=(
  "开源式转速获取：0.16 m、yaw ±7、只强化角速度跟踪"
  "稳态整平：0.16 m、yaw ±7、强化水平姿态与左右轮一致性"
  "开源式转速获取：0.16 m、yaw ±10、只强化角速度跟踪"
  "开源式转速获取：0.16 m、yaw ±13、只强化角速度跟踪"
)
# Starting from model_59000.pt, first consolidate the steady
# posture before extending the command range again. The stability stage is
# deliberately inserted at ±7 so it does not sacrifice the newly acquired yaw
# authority while correcting the visible roll/pitch bias.
STAGE_TARGETS=(61000 64000 66000 68000)
STAGE_RUN_NAMES=(
  spin_open_59000_longlegs_s01_yaw_7_speed
  spin_open_59000_longlegs_s02_yaw_7_stability
  spin_open_59000_longlegs_s03_yaw_10_speed
  spin_open_59000_longlegs_s04_yaw_13_speed
)
STAGE_COUNT="${#STAGE_KEYS[@]}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/spin_curriculum.sh train
  bash scripts/spin_curriculum.sh train-one
  bash scripts/spin_curriculum.sh status
  bash scripts/spin_curriculum.sh play
  bash scripts/spin_curriculum.sh play 61000
  bash scripts/spin_curriculum.sh play 'logs/wheel_legged/RUN/model_61000.pt'
  bash scripts/spin_curriculum.sh export

The first stage resumes from the real-robot-validated yaw-5 model_59000.pt. If that
checkpoint has a different path, provide it explicitly:

  WLG_SPIN_BASE_PT=/absolute/path/model_59000.pt \
    bash scripts/spin_curriculum.sh train

Useful overrides:
  PYTHON_BIN=python
  WLG_PLAY_NUM_ENVS=20
  WLG_PLAY_HEIGHT=0.16
  WLG_PLAY_INITIAL_YAW=7.0
  WLG_PLAY_YAW_STEP=7.0
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
  # Match the historical/open-source sampler during speed acquisition. A
  # continuous yaw range keeps already learned lower speeds in every new stage
  # instead of sending most environments straight to the new endpoint.
  export WLG_COMMAND_PROFILE=independent
  # One command per 20 s episode avoids destructive +limit to -limit reversals.
  export WLG_COMMAND_RESAMPLING_TIME=25.0
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
  export WLG_SPIN_HIGH_YAW_MIN=5.0
  export WLG_SPIN_FIXED_IDLE_FRACTION=0.10
  export WLG_SPIN_FIXED_NEAR_FRACTION=0.10
  export WLG_SPIN_FIXED_NEAR_MIN_RATIO=0.85
  export WLG_SPIN_HIGH_YAW_LIN_VEL_MAX=0.0
  export WLG_SPIN_MOVING_LIN_VEL_MAX=0.0
  export WLG_SPIN_MOVING_YAW_MAX=1.0
  export WLG_SPIN_MOVING_HEIGHT_MAX=0.28
  export WLG_SPIN_TERRAIN_HEIGHT_MAX=0.26
  export WLG_MIXED_TERRAIN_LIN_VEL_MAX=0.0
  export WLG_MIXED_TERRAIN_YAW_MAX=0.0

  # Speed-acquisition phase: do not compete with the yaw tracker using custom
  # world-position or wheel-mismatch penalties. Those are added later.
  # Keep a mild center/low-planar-speed signal from the first stage. The
  # previous speed-only curriculum learned to rotate by driving in a circle.
  export WLG_SPIN_STATIONARY_LIN_VEL_SCALE=-0.35
  export WLG_SPIN_STATIONARY_POSITION_SCALE=-0.20
  export WLG_SPIN_STATIONARY_WHEEL_SPEED_MISMATCH_SCALE=0.0
  export WLG_SPIN_STATIONARY_POSITION_DEADBAND=0.025
  export WLG_SPIN_STATIONARY_ANG_VEL_XY_SCALE=-0.05
  export WLG_SPIN_STATIONARY_ORIENTATION_SCALE=-0.50
  export WLG_SPIN_STATIONARY_ACTION_RATE_SCALE=0.0
  export WLG_SPIN_STATIONARY_ACTION_SMOOTH_SCALE=0.0
  # Moving rotation is intentionally deferred until in-place spin is verified.
  export WLG_SPIN_MOVING_LATERAL_VEL_SCALE=0.0
  export WLG_SPIN_MOVING_WRONG_WAY_SCALE=0.0
  export WLG_SPIN_MOVING_COMMAND_THRESHOLD=0.05
}

apply_stage_environment() {
  local stage_index="$1"
  apply_common_environment
  export WLG_MESH_TYPE=plane
  export WLG_TERRAIN_PROPORTIONS=1.0,0.0,0.0,0.0,0.0,0.0

  case "${STAGE_KEYS[$stage_index]}" in
    yaw_5_speed)
      export WLG_ANG_VEL_YAW_MIN=-5.0
      export WLG_ANG_VEL_YAW_MAX=5.0
      export WLG_SPIN_MOVING_YAW_MAX=5.0
      export WLG_TRACKING_ANG_VEL_SCALE=2.0
      export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.5
      export WLG_TRACKING_ANG_VEL_L1_SCALE=0.0
      export WLG_SPIN_FIXED_NEAR_FRACTION=0.30
      export WLG_SPIN_FIXED_NEAR_MIN_RATIO=0.70
      export WLG_ENTROPY_COEF=0.0020
      ;;
    yaw_7_speed)
      export WLG_ANG_VEL_YAW_MIN=-7.0
      export WLG_ANG_VEL_YAW_MAX=7.0
      export WLG_SPIN_MOVING_YAW_MAX=7.0
      export WLG_TRACKING_ANG_VEL_SCALE=2.0
      export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=1.0
      # Overlap the previous ±5 capability: 50% of non-idle samples are
      # drawn from roughly 4.9–7 instead of jumping almost always to ±7.
      export WLG_SPIN_FIXED_NEAR_FRACTION=0.50
      export WLG_SPIN_FIXED_NEAR_MIN_RATIO=0.70
      export WLG_TRACKING_ANG_VEL_L1_SCALE=-0.10
      export WLG_ENTROPY_COEF=0.0015
      ;;
    yaw_7_stability)
      # The positive direction is currently weak. Use a recovery sampler that
      # ramps positive yaw through the whole range before showing many +7
      # endpoints, while retaining enough negative samples to preserve -7.
      export WLG_COMMAND_PROFILE=spin_recovery
      export WLG_ANG_VEL_YAW_MIN=-7.0
      export WLG_ANG_VEL_YAW_MAX=7.0
      export WLG_SPIN_MOVING_YAW_MAX=7.0
      export WLG_TRACKING_ANG_VEL_SCALE=2.0
      export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.75
      export WLG_TRACKING_ANG_VEL_L1_SCALE=-0.10
      export WLG_SPIN_FIXED_NEAR_FRACTION=0.50
      export WLG_SPIN_FIXED_NEAR_MIN_RATIO=0.70
      # The resumed policy already turns. Spend this stage on the steady
      # attitude seen in play: stronger gravity-vector and angular-rate
      # penalties, plus a mild differential-wheel penalty to remove yaw drift.
      export WLG_ORIENTATION_SCALE=-10.0
      export WLG_ANG_VEL_XY_SCALE=-0.10
      export WLG_SPIN_STATIONARY_LIN_VEL_SCALE=-0.50
      export WLG_SPIN_STATIONARY_POSITION_SCALE=-0.30
      export WLG_SPIN_STATIONARY_WHEEL_SPEED_MISMATCH_SCALE=-0.20
      export WLG_SPIN_STATIONARY_ANG_VEL_XY_SCALE=-0.18
      export WLG_SPIN_STATIONARY_ORIENTATION_SCALE=-2.5
      export WLG_WHEEL_SUPPORT_SCALE=0.40
      export WLG_ENTROPY_COEF=0.0005
      ;;
    yaw_10_speed)
      export WLG_ANG_VEL_YAW_MIN=-10.0
      export WLG_ANG_VEL_YAW_MAX=10.0
      export WLG_SPIN_MOVING_YAW_MAX=10.0
      export WLG_TRACKING_ANG_VEL_SCALE=2.0
      export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=1.0
      # Overlap the previous ±7 capability while extending to ±10.
      export WLG_SPIN_FIXED_NEAR_FRACTION=0.50
      export WLG_SPIN_FIXED_NEAR_MIN_RATIO=0.70
      export WLG_TRACKING_ANG_VEL_L1_SCALE=-0.10
      export WLG_ORIENTATION_SCALE=-6.0
      export WLG_ANG_VEL_XY_SCALE=-0.08
      export WLG_SPIN_STATIONARY_LIN_VEL_SCALE=-0.40
      export WLG_SPIN_STATIONARY_POSITION_SCALE=-0.25
      export WLG_SPIN_STATIONARY_WHEEL_SPEED_MISMATCH_SCALE=-0.10
      export WLG_SPIN_STATIONARY_ANG_VEL_XY_SCALE=-0.10
      export WLG_SPIN_STATIONARY_ORIENTATION_SCALE=-1.0
      export WLG_ENTROPY_COEF=0.0008
      ;;
    yaw_13_speed)
      export WLG_ANG_VEL_YAW_MIN=-13.0
      export WLG_ANG_VEL_YAW_MAX=13.0
      export WLG_SPIN_MOVING_YAW_MAX=13.0
      export WLG_TRACKING_ANG_VEL_SCALE=2.0
      export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=1.0
      # Keep the lower edge near 10 rad/s so the ±13 stage remains reachable.
      export WLG_SPIN_FIXED_NEAR_FRACTION=0.50
      export WLG_SPIN_FIXED_NEAR_MIN_RATIO=0.75
      export WLG_TRACKING_ANG_VEL_L1_SCALE=-0.10
      export WLG_ORIENTATION_SCALE=-5.0
      export WLG_ANG_VEL_XY_SCALE=-0.07
      export WLG_SPIN_STATIONARY_LIN_VEL_SCALE=-0.40
      export WLG_SPIN_STATIONARY_POSITION_SCALE=-0.25
      export WLG_SPIN_STATIONARY_WHEEL_SPEED_MISMATCH_SCALE=-0.08
      export WLG_SPIN_STATIONARY_ANG_VEL_XY_SCALE=-0.08
      export WLG_SPIN_STATIONARY_ORIENTATION_SCALE=-0.80
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
  if [[ "$WLG_COMMAND_PROFILE" == spin_fixed ]]; then
    printf '  fixed_spin: idle=%s, near=%s, near_min_ratio=%s, exact=rest\n' \
      "$WLG_SPIN_FIXED_IDLE_FRACTION" \
      "$WLG_SPIN_FIXED_NEAR_FRACTION" \
      "$WLG_SPIN_FIXED_NEAR_MIN_RATIO"
  elif [[ "$WLG_COMMAND_PROFILE" == independent ]]; then
    printf '  independent_spin: yaw sampled uniformly over the full range\n'
  else
    printf '  mixed_spin: idle=10%%, in_place=45%%, moving=45%%\n'
  fi
  printf '  speed phase: XY anchor=%s, wheel mismatch=%s; moving lateral=%s, wrong-way=%s\n' \
    "$WLG_SPIN_STATIONARY_POSITION_SCALE" \
    "$WLG_SPIN_STATIONARY_WHEEL_SPEED_MISMATCH_SCALE" \
    "$WLG_SPIN_MOVING_LATERAL_VEL_SCALE" \
    "$WLG_SPIN_MOVING_WRONG_WAY_SCALE"
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
    if [[ "${WLG_SPIN_STOP_AFTER_STAGE:-0}" == "1" ]]; then
      echo "Stage complete at model_${iter}.pt; stopping for validation."
      return 0
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
  train-one) WLG_SPIN_STOP_AFTER_STAGE=1 train_all ;;
  status) show_status ;;
  play) play_checkpoint "${2:-}" ;;
  export) export_latest ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
