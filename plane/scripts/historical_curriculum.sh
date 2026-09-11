#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROOT="$PLANE_ROOT/logs/wheel_legged"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLAY_NUM_ENVS="${WLG_PLAY_NUM_ENVS:-5}"

# Each target is an absolute checkpoint number. Interrupted stages therefore
# resume from the newest checkpoint without repeating completed iterations.
STAGE_KEYS=(
  recovery_low
  recovery_raise
  height_balance
  slow_motion
  speed_mid
  speed_full
  terrain_recovery
  terrain_slow
  terrain_expand
  flat_refresh_mid
  flat_refresh_full
  terrain_height_high
  terrain_height_full
  terrain_speed_1p2
  terrain_speed_1p6
  terrain_speed_2p0
)
STAGE_LABELS=(
  "起身第1步：趴姿到低位轮式平衡"
  "起身第2步：低位平衡后抬升机身"
  "纯平地：零速度多高度平衡"
  "纯平地：多高度低速行驶与转向"
  "纯平地：速度 ±1.5、转向 ±2.0"
  "纯平地：速度 ±2.0、转向 ±3.0"
  "低等级混合地形：零速起身与多高度平衡"
  "混合地形：低速移动并保持起身能力"
  "随机地形：课程扩速并加入双向特殊地形"
  "纯平地复习：恢复中速稳定性"
  "纯平地复习：恢复 ±2.0 高速稳定性"
  "混合地形：集中学习 0.26–0.33 m 高机身"
  "随机地形：覆盖 0.16–0.33 m 全高度"
  "随机地形：显式扩速到 ±1.2"
  "随机地形：显式扩速到 ±1.6"
  "随机地形：楼梯最高 ±2.0 并保留双向特殊地形"
)
STAGE_TARGETS=(3000 6000 10000 14000 16100 18100 20100 22100 24100 25100 27100 28600 30600 32600 34600 36600)
STAGE_RUN_NAMES=(
  hist_recovery_v4_longlegs_s01_recovery_low
  hist_recovery_v4_longlegs_s02_recovery_raise
  hist_recovery_v4_longlegs_s03_height_balance
  hist_recovery_v4_longlegs_s04_slow_motion
  hist_recovery_v4_longlegs_s05_speed_mid
  hist_recovery_v4_longlegs_s06_speed_full
  hist_recovery_v4_longlegs_s07_terrain_recovery
  hist_recovery_v4_longlegs_s08_terrain_slow
  hist_recovery_v4_longlegs_s09_terrain_expand
  hist_recovery_v4_longlegs_s10_flat_refresh_mid
  hist_recovery_v4_longlegs_s11_flat_refresh_full
  hist_recovery_v4_longlegs_s12_terrain_height_high
  hist_recovery_v4_longlegs_s13_terrain_height_full
  hist_recovery_v4_longlegs_s14_terrain_speed_1p2
  hist_recovery_v4_longlegs_s15_terrain_speed_1p6
  hist_recovery_v4_longlegs_s16_terrain_speed_2p0
)
STAGE_COUNT="${#STAGE_KEYS[@]}"

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
  export WLG_SPAWN_Z=0.12
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
  export WLG_WHEEL_SUPPORT_SCALE=0.0
  export WLG_DOF_POS_LIMITS_SCALE=-1.0
  export WLG_CUSTOM_TERRAIN_MODE=descent_discrete
  export WLG_RECOVERY_MODE=0
  export WLG_INIT_NOISE_STD=0.5
  export WLG_ENTROPY_COEF=0.01
  export WLG_RECOVERY_POSE_SCALE=0.0
  export WLG_RECOVERY_JOINT_TARGET=0.2,0.4,-0.2,-0.4
  export WLG_INITIAL_ACTOR_BIAS=""
}

apply_stage_environment() {
  local stage_index="$1"
  apply_common_environment

  # All stages in this focused curriculum use flat ground.
  export WLG_MESH_TYPE=plane
  export WLG_TERRAIN_PROPORTIONS=0.2,0.2,0.2,0.1,0.2,0.1
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
    recovery_low|recovery_raise)
      export WLG_MESH_TYPE=plane
      export WLG_TERRAIN_PROPORTIONS=0.2,0.2,0.2,0.1,0.2,0.1
      export WLG_RECOVERY_MODE=1
      export WLG_LIN_VEL_X_MIN=0.0
      export WLG_LIN_VEL_X_MAX=0.0
      export WLG_ANG_VEL_YAW_MIN=0.0
      export WLG_ANG_VEL_YAW_MAX=0.0
      # Do not reward lying still merely because zero velocity is tracked.
      # Recovery is driven by the staged leg pose, upright orientation and
      # commanded base height.
      export WLG_TRACKING_LIN_VEL_SCALE=0.0
      export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=0.0
      export WLG_TRACKING_ANG_VEL_SCALE=0.0
      export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.0
      # The positive Gaussian was effectively zero far from 0.20 m and its
      # enhanced term saturated at -1. Use an unsaturated L1 error instead.
      export WLG_BASE_HEIGHT_SCALE=-2.0
      export WLG_BASE_HEIGHT_ENHANCE_SCALE=0.0
      export WLG_COLLISION_SCALE=-1.0
      export WLG_WHEEL_SUPPORT_SCALE=2.0
      export WLG_NOMINAL_STATE_SCALE=0.0
      export WLG_ORIENTATION_SCALE=-2.0
      export WLG_ANG_VEL_XY_SCALE=-0.2
      export WLG_DOF_POS_LIMITS_SCALE=-0.2
      export WLG_ACTION_RATE_SCALE=-0.001
      export WLG_ACTION_SMOOTH_SCALE=-0.001
      export WLG_INIT_NOISE_STD=0.5
      export WLG_ENTROPY_COEF=0.02
      if [[ "${STAGE_KEYS[$stage_index]}" == recovery_low ]]; then
        # User-verified low pose: the chassis is clear and only wheels carry
        # the robot. Wheel angles are intentionally not pose targets.
        export WLG_HEIGHT_MIN=0.20
        export WLG_HEIGHT_MAX=0.20
        export WLG_RECOVERY_JOINT_TARGET=0.0,0.8,0.0,-0.8
        export WLG_RECOVERY_POSE_SCALE=-0.5
        # action=(low_q-default_q)/pos_action_scale
        export WLG_INITIAL_ACTOR_BIAS=0.46,2.90,0.0,-0.46,-2.90,0.0
      else
        # Resume the learned prone-to-low transition, then learn to raise the
        # body from that balanced state to the verified 0.28 m pose.
        export WLG_HEIGHT_MIN=0.28
        export WLG_HEIGHT_MAX=0.28
        export WLG_RECOVERY_JOINT_TARGET=0.2,0.4,-0.2,-0.4
        export WLG_RECOVERY_POSE_SCALE=-0.25
      fi
      ;;
    height_balance|slow_motion|speed_mid|speed_full)
      export WLG_RECOVERY_MODE=1
      export WLG_COMMAND_CURRICULUM=0
      export WLG_HEIGHT_MIN=0.16
      export WLG_HEIGHT_MAX=0.30
      export WLG_COMMAND_RESAMPLING_TIME=3.0
      export WLG_COLLISION_SCALE=-1.0
      export WLG_WHEEL_SUPPORT_SCALE=2.0
      export WLG_ORIENTATION_SCALE=-10.0
      export WLG_ANG_VEL_XY_SCALE=-0.1
      export WLG_BASE_HEIGHT_SCALE=2.0
      export WLG_BASE_HEIGHT_ENHANCE_SCALE=1.5
      export WLG_RECOVERY_POSE_SCALE=0.0
      export WLG_ACTION_RATE_SCALE=-0.01
      export WLG_ACTION_SMOOTH_SCALE=-0.01
      if [[ "${STAGE_KEYS[$stage_index]}" == height_balance ]]; then
        export WLG_LIN_VEL_X_MIN=0.0
        export WLG_LIN_VEL_X_MAX=0.0
        export WLG_ANG_VEL_YAW_MIN=0.0
        export WLG_ANG_VEL_YAW_MAX=0.0
        export WLG_TRACKING_LIN_VEL_SCALE=0.0
        export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=0.0
        export WLG_TRACKING_ANG_VEL_SCALE=0.0
        export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.0
      elif [[ "${STAGE_KEYS[$stage_index]}" == slow_motion ]]; then
        export WLG_LIN_VEL_X_MIN=-1.0
        export WLG_LIN_VEL_X_MAX=1.0
        export WLG_ANG_VEL_YAW_MIN=-1.0
        export WLG_ANG_VEL_YAW_MAX=1.0
        export WLG_TRACKING_LIN_VEL_SCALE=1.0
        export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.0
        export WLG_TRACKING_ANG_VEL_SCALE=1.0
        export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.0
      elif [[ "${STAGE_KEYS[$stage_index]}" == speed_mid ]]; then
        export WLG_COMMAND_RESAMPLING_TIME=5.0
        export WLG_LIN_VEL_X_MIN=-1.5
        export WLG_LIN_VEL_X_MAX=1.5
        export WLG_ANG_VEL_YAW_MIN=-2.0
        export WLG_ANG_VEL_YAW_MAX=2.0
        export WLG_TRACKING_LIN_VEL_SCALE=1.25
        export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.25
        export WLG_TRACKING_ANG_VEL_SCALE=1.25
        export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.0
      else
        export WLG_COMMAND_RESAMPLING_TIME=5.0
        export WLG_LIN_VEL_X_MIN=-2.0
        export WLG_LIN_VEL_X_MAX=2.0
        export WLG_ANG_VEL_YAW_MIN=-3.0
        export WLG_ANG_VEL_YAW_MAX=3.0
        export WLG_TRACKING_LIN_VEL_SCALE=1.5
        export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.5
        export WLG_TRACKING_ANG_VEL_SCALE=1.5
        export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.0
      fi
      ;;
    flat_refresh_mid|flat_refresh_full)
      # Terrain-only fine-tuning reduced recent exposure to fast flat motion.
      # Briefly return to the original plane task before mixing terrains again.
      export WLG_MESH_TYPE=plane
      export WLG_RECOVERY_MODE=1
      export WLG_COMMAND_CURRICULUM=0
      export WLG_HEIGHT_MIN=0.16
      export WLG_HEIGHT_MAX=0.30
      export WLG_COMMAND_RESAMPLING_TIME=5.0
      export WLG_COLLISION_SCALE=-1.0
      export WLG_WHEEL_SUPPORT_SCALE=2.0
      export WLG_ORIENTATION_SCALE=-10.0
      export WLG_ANG_VEL_XY_SCALE=-0.1
      export WLG_BASE_HEIGHT_SCALE=2.0
      export WLG_BASE_HEIGHT_ENHANCE_SCALE=1.5
      export WLG_ACTION_RATE_SCALE=-0.01
      export WLG_ACTION_SMOOTH_SCALE=-0.01
      export WLG_TRACKING_LIN_VEL_SCALE=1.5
      export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.5
      export WLG_TRACKING_ANG_VEL_SCALE=1.5
      export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.0
      export WLG_ENTROPY_COEF=0.005
      if [[ "${STAGE_KEYS[$stage_index]}" == flat_refresh_mid ]]; then
        export WLG_LIN_VEL_X_MIN=-1.2
        export WLG_LIN_VEL_X_MAX=1.2
        export WLG_ANG_VEL_YAW_MIN=-1.5
        export WLG_ANG_VEL_YAW_MAX=1.5
      else
        export WLG_LIN_VEL_X_MIN=-2.0
        export WLG_LIN_VEL_X_MAX=2.0
        export WLG_ANG_VEL_YAW_MIN=-3.0
        export WLG_ANG_VEL_YAW_MAX=3.0
      fi
      ;;
    terrain_recovery|terrain_slow|terrain_expand|terrain_height_high|terrain_height_full|terrain_speed_1p2|terrain_speed_1p6|terrain_speed_2p0)
      # Follow the historical recovery chain: introduce terrain with a large
      # flat share, preserve recovery resets, and only then expand commands.
      export WLG_MESH_TYPE=trimesh
      export WLG_RECOVERY_MODE=1
      export WLG_TERRAIN_CURRICULUM=1
      export WLG_TERRAIN_PROGRESS_FRACTION=0.5
      export WLG_HEIGHT_MIN=0.16
      export WLG_HEIGHT_MAX=0.30
      export WLG_COMMAND_RESAMPLING_TIME=5.0
      export WLG_COLLISION_SCALE=-1.0
      export WLG_ORIENTATION_SCALE=-15.0
      export WLG_ANG_VEL_XY_SCALE=-0.1
      export WLG_BASE_HEIGHT_SCALE=2.0
      export WLG_BASE_HEIGHT_ENHANCE_SCALE=1.5
      export WLG_RECOVERY_POSE_SCALE=0.0
      export WLG_ACTION_RATE_SCALE=-0.01
      export WLG_ACTION_SMOOTH_SCALE=-0.01

      if [[ "${STAGE_KEYS[$stage_index]}" == terrain_recovery ]]; then
        # Start on levels 0-1 and spend half of the robots on flat ground. A
        # tiny tracking scale keeps terrain-curriculum bookkeeping available
        # without letting the zero-command reward dominate recovery.
        export WLG_TERRAIN_PROPORTIONS=0.5,0.2,0.1,0.1,0.1,0.0
        export WLG_MAX_INIT_TERRAIN_LEVEL=1
        export WLG_CUSTOM_TERRAIN_MODE=descent_discrete
        export WLG_COMMAND_CURRICULUM=0
        export WLG_LIN_VEL_X_MIN=0.0
        export WLG_LIN_VEL_X_MAX=0.0
        export WLG_ANG_VEL_YAW_MIN=0.0
        export WLG_ANG_VEL_YAW_MAX=0.0
        export WLG_TRACKING_LIN_VEL_SCALE=0.000001
        export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=0.0
        export WLG_TRACKING_ANG_VEL_SCALE=0.000001
        export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.0
        export WLG_WHEEL_SUPPORT_SCALE=1.0
      elif [[ "${STAGE_KEYS[$stage_index]}" == terrain_slow ]]; then
        # Once recovery works, add conservative motion while keeping 40% flat
        # examples so the newly learned stand-up behavior is not forgotten.
        export WLG_TERRAIN_PROPORTIONS=0.4,0.2,0.1,0.15,0.15,0.0
        export WLG_MAX_INIT_TERRAIN_LEVEL=3
        export WLG_CUSTOM_TERRAIN_MODE=descent_discrete
        export WLG_COMMAND_CURRICULUM=0
        export WLG_LIN_VEL_X_MIN=-0.8
        export WLG_LIN_VEL_X_MAX=0.8
        export WLG_ANG_VEL_YAW_MIN=-1.0
        export WLG_ANG_VEL_YAW_MAX=1.0
        export WLG_TRACKING_LIN_VEL_SCALE=1.25
        export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.25
        export WLG_TRACKING_ANG_VEL_SCALE=1.25
        export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.0
        export WLG_WHEEL_SUPPORT_SCALE=0.5
      elif [[ "${STAGE_KEYS[$stage_index]}" == terrain_expand ]]; then
        # The final stage starts from the low-speed range and lets successful
        # environments expand independently. Basic terrains may reach ±2 m/s
        # and ±3 rad/s; stairs-up/custom obstacles stay at safer limits.
        export WLG_TERRAIN_PROPORTIONS=0.3,0.2,0.1,0.15,0.15,0.1
        export WLG_MAX_INIT_TERRAIN_LEVEL=5
        export WLG_CUSTOM_TERRAIN_MODE=bidirectional
        export WLG_COMMAND_CURRICULUM=1
        export WLG_LIN_VEL_X_MIN=-0.8
        export WLG_LIN_VEL_X_MAX=0.8
        export WLG_ANG_VEL_YAW_MIN=-1.0
        export WLG_ANG_VEL_YAW_MAX=1.0
        export WLG_BASIC_MAX_CURRICULUM=2.0
        export WLG_ADVANCED_MAX_CURRICULUM=1.5
        export WLG_BASIC_MAX_ANG_VEL_CURRICULUM=3.0
        export WLG_ADVANCED_MAX_ANG_VEL_CURRICULUM=2.0
        export WLG_TRACKING_LIN_VEL_SCALE=1.5
        export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.5
        export WLG_TRACKING_ANG_VEL_SCALE=1.5
        export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.0
        export WLG_WHEEL_SUPPORT_SCALE=0.25
      elif [[ "${STAGE_KEYS[$stage_index]}" == terrain_height_high || "${STAGE_KEYS[$stage_index]}" == terrain_height_full ]]; then
        # Raise the body before increasing terrain speed. A large flat share
        # preserves the strong plane policy while the remaining environments
        # teach the same height response on slopes, stairs and custom curbs.
        export WLG_CUSTOM_TERRAIN_MODE=bidirectional
        export WLG_COMMAND_CURRICULUM=0
        export WLG_COMMAND_PROFILE=mixed_final
        export WLG_TRACKING_LIN_VEL_SCALE=1.5
        export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.5
        export WLG_TRACKING_ANG_VEL_SCALE=1.5
        export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.0
        export WLG_WHEEL_SUPPORT_SCALE=0.25
        export WLG_ENTROPY_COEF=0.005
        if [[ "${STAGE_KEYS[$stage_index]}" == terrain_height_high ]]; then
          # Current generator uses the fourth entry for stair-down. Weight it
          # above stair-up because the present policy already climbs well.
          export WLG_TERRAIN_PROPORTIONS=0.6,0.05,0.05,0.15,0.05,0.1
          export WLG_MAX_INIT_TERRAIN_LEVEL=3
          export WLG_HEIGHT_MIN=0.26
          export WLG_HEIGHT_MAX=0.33
          export WLG_LIN_VEL_X_MIN=-1.2
          export WLG_LIN_VEL_X_MAX=1.2
          export WLG_ANG_VEL_YAW_MIN=-1.5
          export WLG_ANG_VEL_YAW_MAX=1.5
          export WLG_MIXED_FLAT_HEIGHT_MIN=0.26
          export WLG_MIXED_FLAT_HEIGHT_MAX=0.33
          export WLG_MIXED_TERRAIN_LIN_VEL_MAX=0.8
          export WLG_MIXED_TERRAIN_YAW_MAX=1.0
          export WLG_MIXED_CUSTOM_LIN_VEL_MAX=0.8
          export WLG_MIXED_CUSTOM_YAW_MAX=1.0
        else
          export WLG_TERRAIN_PROPORTIONS=0.5,0.1,0.05,0.15,0.1,0.1
          export WLG_MAX_INIT_TERRAIN_LEVEL=4
          export WLG_HEIGHT_MIN=0.16
          export WLG_HEIGHT_MAX=0.33
          export WLG_LIN_VEL_X_MIN=-2.0
          export WLG_LIN_VEL_X_MAX=2.0
          export WLG_ANG_VEL_YAW_MIN=-3.0
          export WLG_ANG_VEL_YAW_MAX=3.0
          export WLG_MIXED_FLAT_HEIGHT_MIN=0.16
          export WLG_MIXED_FLAT_HEIGHT_MAX=0.33
          export WLG_MIXED_TERRAIN_LIN_VEL_MAX=1.0
          export WLG_MIXED_TERRAIN_YAW_MAX=1.2
          export WLG_MIXED_CUSTOM_LIN_VEL_MAX=1.0
          export WLG_MIXED_CUSTOM_YAW_MAX=1.0
        fi
      else
        # The terrain curriculum in stage 9 deliberately started at ±0.8 m/s,
        # but difficult-level success is too sparse to reach the configured
        # caps quickly. Continue with explicit speed bands, as in the historical
        # training chain, so stairs receive enough approach momentum.
        export WLG_TERRAIN_PROPORTIONS=0.3,0.15,0.1,0.2,0.15,0.1
        export WLG_CUSTOM_TERRAIN_MODE=bidirectional
        export WLG_COMMAND_CURRICULUM=0
        export WLG_COMMAND_PROFILE=mixed_final
        export WLG_TRACKING_LIN_VEL_SCALE=1.5
        export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.5
        export WLG_TRACKING_ANG_VEL_SCALE=1.5
        export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.0
        export WLG_WHEEL_SUPPORT_SCALE=0.25
        export WLG_ENTROPY_COEF=0.005
        export WLG_HEIGHT_MIN=0.16
        export WLG_HEIGHT_MAX=0.33
        export WLG_LIN_VEL_X_MIN=-2.0
        export WLG_LIN_VEL_X_MAX=2.0
        export WLG_ANG_VEL_YAW_MIN=-3.0
        export WLG_ANG_VEL_YAW_MAX=3.0
        export WLG_MIXED_FLAT_HEIGHT_MIN=0.16
        export WLG_MIXED_FLAT_HEIGHT_MAX=0.33

        if [[ "${STAGE_KEYS[$stage_index]}" == terrain_speed_1p2 ]]; then
          export WLG_MAX_INIT_TERRAIN_LEVEL=4
          export WLG_MIXED_TERRAIN_LIN_VEL_MAX=1.2
          export WLG_MIXED_TERRAIN_YAW_MAX=1.5
          export WLG_MIXED_CUSTOM_LIN_VEL_MAX=1.2
          export WLG_MIXED_CUSTOM_YAW_MAX=1.0
        elif [[ "${STAGE_KEYS[$stage_index]}" == terrain_speed_1p6 ]]; then
          export WLG_MAX_INIT_TERRAIN_LEVEL=5
          export WLG_MIXED_TERRAIN_LIN_VEL_MAX=1.6
          export WLG_MIXED_TERRAIN_YAW_MAX=2.0
          export WLG_MIXED_CUSTOM_LIN_VEL_MAX=1.5
          export WLG_MIXED_CUSTOM_YAW_MAX=1.5
        else
          export WLG_MAX_INIT_TERRAIN_LEVEL=5
          export WLG_MIXED_TERRAIN_LIN_VEL_MAX=2.0
          export WLG_MIXED_TERRAIN_YAW_MAX=2.0
          export WLG_MIXED_CUSTOM_LIN_VEL_MAX=1.8
          export WLG_MIXED_CUSTOM_YAW_MAX=1.5
        fi
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
      printf '  [%d/%d] %-42s not started (target %d)\n' \
        "$((stage_index + 1))" "$STAGE_COUNT" "${STAGE_LABELS[$stage_index]}" "$target"
    elif (( iter >= target )); then
      printf '  [%d/%d] %-42s complete: model_%d.pt\n' \
        "$((stage_index + 1))" "$STAGE_COUNT" "${STAGE_LABELS[$stage_index]}" "$iter"
    else
      printf '  [%d/%d] %-42s current: model_%d.pt -> %d\n' \
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
    echo "[$((stage_index + 1))/$STAGE_COUNT] ${STAGE_LABELS[$stage_index]}"
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
  echo "All $STAGE_COUNT stages are complete."
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
  output_path="${WLG_EXPORT_OUT:-$PLANE_ROOT/export_onnx/longlegs_historical_curriculum_model_${iter}.onnx}"
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
