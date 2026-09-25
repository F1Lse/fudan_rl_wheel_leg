#!/usr/bin/env bash
set -Eeuo pipefail

# Train one bounded spin stage from a verified flat/stair or previous spin PT.
# Never advances to the next yaw limit automatically: evaluate first.
# Usage: bash scripts/train_spin_from_flat.sh SOURCE_PT YAW_LIMIT ITERATIONS nominal|bridge|robust [POSITIVE_FRACTION] [PLANAR_SPEED_PENALTY] [GUARD_COEF] [GUARD_SCOPE] [BRIDGE_SCALE] [GUARD_YAW_LIMIT]

if (( $# < 4 || $# > 10 )); then
  echo "Usage: $0 SOURCE_PT YAW_LIMIT ITERATIONS nominal|bridge|robust [POSITIVE_FRACTION] [PLANAR_SPEED_PENALTY] [GUARD_COEF] [GUARD_SCOPE] [BRIDGE_SCALE] [GUARD_YAW_LIMIT]" >&2
  exit 2
fi

PLANE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_pt="$1"
[[ "$source_pt" == /* ]] || source_pt="$PLANE_ROOT/$source_pt"
source_pt="$(realpath -e "$source_pt")"
[[ -f "$source_pt" ]] || { echo "Checkpoint missing: $source_pt" >&2; exit 2; }

yaw_limit="$2"
iterations="$3"
mode="$4"
positive_fraction="${5:-0.50}"
planar_penalty="${6:-0.20}"
guard_coef="${7:-0.0}"
guard_scope="${8:-nonpositive}"
bridge_scale="${9:-0.5}"
guard_yaw_limit="${10:-5}"
restitution_override="${SPIN_RESTITUTION_RANGE:-}"
[[ "$yaw_limit" =~ ^([1-9]|1[0-3])$ ]] || {
  echo "YAW_LIMIT must be an integer from 1 to 13" >&2; exit 2;
}
[[ "$iterations" =~ ^[1-9][0-9]*$ ]] && (( iterations <= 2000 )) || {
  echo "ITERATIONS must be from 1 to 2000" >&2; exit 2;
}
[[ "$mode" == nominal || "$mode" == bridge || "$mode" == robust ]] || {
  echo "MODE must be nominal, bridge, or robust" >&2; exit 2;
}
[[ "$positive_fraction" =~ ^0\.[0-9]+$|^1(\.0+)?$ ]] || {
  echo "POSITIVE_FRACTION must be from 0.0 to 1.0" >&2; exit 2;
}
[[ "$planar_penalty" =~ ^(0|[1-4])(\.[0-9]+)?$|^5(\.0+)?$ ]] || {
  echo "PLANAR_SPEED_PENALTY must be from 0.0 to 5.0" >&2; exit 2;
}
[[ "$guard_coef" =~ ^(0|[1-9])([.][0-9]+)?$ ]] || {
  echo "GUARD_COEF must be from 0 to 9" >&2; exit 2;
}
[[ "$guard_scope" == nonpositive || "$guard_scope" == nonnegative || "$guard_scope" == all || "$guard_scope" == inside ]] || {
  echo "GUARD_SCOPE must be nonpositive, nonnegative, all, or inside" >&2; exit 2;
}
[[ "$bridge_scale" =~ ^0([.][0-9]+)?$|^1([.]0+)?$ ]] || {
  echo "BRIDGE_SCALE must be from 0 to 1" >&2; exit 2;
}
[[ "$guard_yaw_limit" =~ ^([0-9]|1[0-2])([.][0-9]+)?$|^13([.]0+)?$ ]] || {
  echo "GUARD_YAW_LIMIT must be from 0 to 13" >&2; exit 2;
}

source_file="$(basename "$source_pt")"
[[ "$source_file" =~ ^model_([0-9]+)\.pt$ ]] || {
  echo "Expected model_<iteration>.pt, got $source_file" >&2; exit 2;
}
source_iter="${BASH_REMATCH[1]}"
source_run="$(basename "$(dirname "$source_pt")")"
target_iter="$((source_iter + iterations))"
run_name="spin_targeted_from_flat_yaw${yaw_limit}_${mode}_v1"
if [[ "$positive_fraction" != 0.50 ]]; then
  run_name="${run_name}_p${positive_fraction//./}"
fi
if [[ "$planar_penalty" != 0.20 ]]; then
  run_name="${run_name}_planar${planar_penalty//./}"
fi
if [[ "$guard_coef" != 0 && "$guard_coef" != 0.0 ]]; then
  run_name="${run_name}_guard${guard_coef//./}"
fi
if [[ "$guard_scope" == all ]]; then
  run_name="${run_name}_all"
fi
if [[ "$guard_scope" == nonnegative ]]; then
  run_name="${run_name}_nonnegative"
fi
if [[ "$guard_scope" == inside ]]; then
  run_name="${run_name}_inside${guard_yaw_limit//./}"
fi
if [[ "$mode" == bridge && "$bridge_scale" != 0.5 ]]; then
  run_name="${run_name}_d${bridge_scale//./}"
fi
if [[ -n "$restitution_override" ]]; then
  [[ "$restitution_override" =~ ^[01](\.[0-9]+)?,[01](\.[0-9]+)?$ ]] || {
    echo "SPIN_RESTITUTION_RANGE must be two comma-separated values in [0, 1]" >&2; exit 2;
  }
  rest_tag="${restitution_override//./}"
  run_name="${run_name}_rest${rest_tag//,/to}"
fi

# Keep every stage reproducible even if launched from an interactive shell
# that still holds a previous experiment's WLG_* overrides.
while IFS='=' read -r variable _; do
  [[ "$variable" == WLG_* ]] && unset "$variable"
done < <(env)

export WLG_HISTORICAL_DOMAIN_RAND=0
[[ "$mode" == robust ]] && export WLG_HISTORICAL_DOMAIN_RAND=1
if [[ "$mode" == bridge ]]; then
  export WLG_DOMAIN_SCALE="$bridge_scale"
  export WLG_PUSH_ROBOTS=0
fi
export WLG_MESH_TYPE=plane
export WLG_TERRAIN_PROPORTIONS=1.0,0.0,0.0,0.0,0.0,0.0
export WLG_TERRAIN_CURRICULUM=0
export WLG_COMMAND_CURRICULUM=0
export WLG_COMMAND_PROFILE=spin_fixed
export WLG_COMMAND_RESAMPLING_TIME=5.0
export WLG_LIN_VEL_X_MIN=0.0 WLG_LIN_VEL_X_MAX=0.0
export WLG_ANG_VEL_YAW_MIN="-$yaw_limit" WLG_ANG_VEL_YAW_MAX="$yaw_limit"
export WLG_HEIGHT_MIN=0.16 WLG_HEIGHT_MAX=0.16
export WLG_HIGHSTAND_ANCHOR_HEIGHT_MIN=0.16
export WLG_SPIN_FIXED_IDLE_FRACTION=0.10
export WLG_SPIN_FIXED_NEAR_FRACTION=0.55
export WLG_SPIN_FIXED_NEAR_MIN_RATIO=0.70
export WLG_SPIN_FIXED_POSITIVE_FRACTION="$positive_fraction"
export WLG_RECOVERY_MODE=0
export WLG_SPAWN_Z=0.12
export WLG_FAIL_TO_TERMINAL_TIME_S=0.30

# Preserve the source's body balance objective while giving large yaw errors
# a non-saturating reward signal. The world-position penalty stays zero because
# the actor cannot observe a world-frame return-to-origin target.
export WLG_TRACKING_LIN_VEL_SCALE=1.0
export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.0
export WLG_TRACKING_ANG_VEL_SCALE=1.5
export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=0.5
export WLG_TRACKING_ANG_VEL_L1_SCALE=-0.10
export WLG_CLIP_SINGLE_REWARD=5.0
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
export WLG_HIGH_STAND_LIN_VEL_SCALE=-2.0
export WLG_HIGH_STAND_ANG_VEL_XY_SCALE=-0.4
export WLG_HIGH_STAND_ORIENTATION_SCALE=-6.0
export WLG_HIGH_STAND_ACTION_RATE_SCALE=-0.02
export WLG_HIGH_STAND_ACTION_SMOOTH_SCALE=-0.03
export WLG_ENTROPY_COEF=0.001
export WLG_SPIN_STATIONARY_LIN_VEL_SCALE="-$planar_penalty"
export WLG_SPIN_STATIONARY_POSITION_SCALE=0.0
export WLG_SPIN_STATIONARY_WHEEL_SPEED_MISMATCH_SCALE=0.0
export WLG_SPIN_STATIONARY_ANG_VEL_XY_SCALE=-0.05
export WLG_SPIN_STATIONARY_ORIENTATION_SCALE=-0.50
export WLG_SPIN_STATIONARY_ACTION_RATE_SCALE=0.0
export WLG_SPIN_STATIONARY_ACTION_SMOOTH_SCALE=0.0
export WLG_RESUME_LOAD_OPTIMIZER=0
export WLG_SPIN_GUARD_COEF="$guard_coef"
export WLG_SPIN_GUARD_SCOPE="$guard_scope"
export WLG_SPIN_GUARD_YAW_LIMIT="$guard_yaw_limit"
if [[ -n "$restitution_override" ]]; then
  export WLG_RESTITUTION_RANGE="$restitution_override"
fi

cd "$PLANE_ROOT"
manifest_dir="$PLANE_ROOT/evaluations/spin/training_manifests"
mkdir -p "$manifest_dir"
manifest="$manifest_dir/$(date -u +%Y%m%dT%H%M%SZ)_${run_name}.txt"
{
  printf 'source=%s\n' "$source_pt"
  printf 'source_sha256=%s\n' "$(sha256sum "$source_pt" | cut -d' ' -f1)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  for code_file in \
    scripts/train_spin_from_flat.sh \
    wheel_legged_gym/envs/base/legged_robot.py \
    wheel_legged_gym/envs/base/legged_robot_config.py \
    wheel_legged_gym/rsl_rl/modules/actor_critic_sequence.py \
    wheel_legged_gym/rsl_rl/algorithms/ppo.py \
    wheel_legged_gym/rsl_rl/runners/on_policy_runner.py; do
    printf 'code_sha256[%s]=%s\n' "$code_file" \
      "$(sha256sum "$code_file" | cut -d' ' -f1)"
  done
  printf 'target=model_%s.pt yaw=±%s mode=%s\n' "$target_iter" "$yaw_limit" "$mode"
  printf 'sampler=10%% idle, 55%% yaw magnitude 70%%-100%% of limit, 35%% exact limit, positive fraction=%s\n' "$positive_fraction"
  printf 'planar_speed_penalty=%s\n' "$planar_penalty"
  printf 'spin_guard_coef=%s; teacher=source checkpoint; scope=%s\n' "$guard_coef" "$guard_scope"
  printf 'spin_guard_yaw_limit_radps=%s\n' "$guard_yaw_limit"
  printf 'bridge_domain_scale=%s\n' "$bridge_scale"
  printf 'restitution_range_override=%s\n' "${restitution_override:-default}"
  printf 'run_name=%s seed=202 iterations=%s\n' "$run_name" "$iterations"
  env | grep '^WLG_' | sort
} | tee "$manifest"
[[ "${SPIN_DRY_RUN:-0}" == 1 ]] && exit 0
python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged --experiment_name=wheel_legged \
  --run_name="$run_name" --resume --load_run="$source_run" \
  --checkpoint="$source_iter" --max_iterations="$iterations" \
  --seed=202 --headless
