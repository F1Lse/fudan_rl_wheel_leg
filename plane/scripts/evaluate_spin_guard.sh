#!/usr/bin/env bash
set -Eeuo pipefail

# Paired retained-skill evaluation for a guarded spin run.
# Usage: bash scripts/evaluate_spin_guard.sh GUARD_RUN nominal|early|bridge_pair|bridge_early|speed_pair|speed_early|moderate_push|historical_no_push|all

if (( $# != 2 )); then
  echo "Usage: $0 GUARD_RUN nominal|early|bridge_pair|bridge_early|speed_pair|speed_early|moderate_push|historical_no_push|all" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
guard_run="$1"
selection="$2"
guard_checkpoint="${SPIN_GUARD_CHECKPOINT:-56800}"
eval_seed="${SPIN_EVAL_SEED:-102}"
eval_tag="${SPIN_EVAL_TAG:-}"
parent_run="${SPIN_PARENT_RUN:-Sep25_14-33-36_spin_targeted_from_flat_yaw5_nominal_v1_p075_planar10_guard10}"
parent_checkpoint="${SPIN_PARENT_CHECKPOINT:-56700}"
[[ "$guard_checkpoint" =~ ^[0-9]+$ && "$eval_seed" =~ ^[0-9]+$ ]] || {
  echo "SPIN_GUARD_CHECKPOINT and SPIN_EVAL_SEED must be integers" >&2
  exit 2
}
[[ -z "$eval_tag" || "$eval_tag" =~ ^[A-Za-z0-9_]+$ ]] || {
  echo "SPIN_EVAL_TAG must use letters, digits, or underscores" >&2
  exit 2
}
[[ -z "$eval_tag" ]] || eval_tag="_${eval_tag}"
base_run="Sep13_01-18-42_flat_stability_longlegs_s05_final_mixed_consolidation"
old_run="Sep25_13-49-20_spin_targeted_from_flat_yaw5_nominal_v1_p075_planar10"
test -f "logs/wheel_legged/$base_run/model_56500.pt"
test -f "logs/wheel_legged/$old_run/model_56800.pt"
test -f "logs/wheel_legged/$guard_run/model_${guard_checkpoint}.pt"
test -f eval_targets/spin_retention_v1.json

export WLG_EVAL_TARGET=eval_targets/spin_retention_v1.json
export WLG_EVAL_EPISODES_PER_YAW=20

evaluate_case() {
  local setting="$1" label="$2" run="$3" checkpoint="$4"
  local output="evaluations/spin/guard_${setting}_${label}${checkpoint}${eval_tag}_seed${eval_seed}_20"
  case "$setting" in
    nominal|speed_nominal)
      export WLG_HISTORICAL_DOMAIN_RAND=0 WLG_EVAL_RANDOMIZATION=0 WLG_EVAL_PUSHES=0
      ;;
    moderate_push)
      export WLG_HISTORICAL_DOMAIN_RAND=0 WLG_EVAL_RANDOMIZATION=1 WLG_EVAL_PUSHES=1
      ;;
    intermediate)
      export WLG_HISTORICAL_DOMAIN_RAND=0 WLG_DOMAIN_SCALE=0.25 WLG_EVAL_RANDOMIZATION=1 WLG_EVAL_PUSHES=0
      ;;
    historical_no_push)
      export WLG_HISTORICAL_DOMAIN_RAND=1 WLG_EVAL_RANDOMIZATION=1 WLG_EVAL_PUSHES=0
      ;;
    *) echo "Unknown setting: $setting" >&2; exit 2 ;;
  esac
  export WLG_EVAL_OUT="$output"
  echo "START $setting $label $(date -Is)"
  python wheel_legged_gym/scripts/evaluate_spin.py \
    --task=wheel_legged --headless --num_envs=48 --seed="$eval_seed" \
    --load_run="$run" --checkpoint="$checkpoint" \
    > "/tmp/spin_guard_eval_${setting}_${label}${checkpoint}${eval_tag}_seed${eval_seed}.out" 2>&1
  echo "DONE $setting $label $(date -Is)"
}

if [[ "$selection" == nominal || "$selection" == all ]]; then
  evaluate_case nominal base "$base_run" 56500
  evaluate_case nominal old "$old_run" 56800
  evaluate_case nominal guard "$guard_run" "$guard_checkpoint"
fi
if [[ "$selection" == early ]]; then
  evaluate_case nominal guard56600 "$guard_run" 56600
  evaluate_case nominal guard56700 "$guard_run" 56700
fi
if [[ "$selection" == bridge_pair ]]; then
  test -f "logs/wheel_legged/$parent_run/model_${parent_checkpoint}.pt"
  evaluate_case nominal parent "$parent_run" "$parent_checkpoint"
  evaluate_case nominal guard "$guard_run" "$guard_checkpoint"
  evaluate_case intermediate parent "$parent_run" "$parent_checkpoint"
  evaluate_case intermediate guard "$guard_run" "$guard_checkpoint"
fi
if [[ "$selection" == bridge_early ]]; then
  evaluate_case nominal guard "$guard_run" "$guard_checkpoint"
  evaluate_case intermediate guard "$guard_run" "$guard_checkpoint"
fi
if [[ "$selection" == speed_pair ]]; then
  test -f "logs/wheel_legged/$parent_run/model_${parent_checkpoint}.pt"
  export WLG_EVAL_YAWS=-6,-5,0,5,6
  evaluate_case speed_nominal parent "$parent_run" "$parent_checkpoint"
  evaluate_case speed_nominal guard "$guard_run" "$guard_checkpoint"
fi
if [[ "$selection" == speed_early ]]; then
  export WLG_EVAL_YAWS=-6,-5,0,5,6
  evaluate_case speed_nominal guard "$guard_run" 56800
  evaluate_case speed_nominal guard "$guard_run" 56900
fi
if [[ "$selection" == moderate_push || "$selection" == all ]]; then
  evaluate_case moderate_push base "$base_run" 56500
  evaluate_case moderate_push guard "$guard_run" "$guard_checkpoint"
fi
if [[ "$selection" == historical_no_push || "$selection" == all ]]; then
  evaluate_case historical_no_push base "$base_run" 56500
  evaluate_case historical_no_push guard "$guard_run" "$guard_checkpoint"
fi
[[ "$selection" == nominal || "$selection" == early || "$selection" == bridge_pair || "$selection" == bridge_early || "$selection" == speed_pair || "$selection" == speed_early || "$selection" == moderate_push || "$selection" == historical_no_push || "$selection" == all ]] || {
  echo "Unknown selection: $selection" >&2
  exit 2
}
