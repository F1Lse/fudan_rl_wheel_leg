#!/usr/bin/env bash
set -Eeuo pipefail

# Paired evaluation for the exploratory low-restitution bridge.
# Usage: bash scripts/evaluate_spin_low_restitution_bridge.sh PARENT_RUN CHILD_RUN
if (( $# != 2 )); then
  echo "Usage: $0 PARENT_RUN CHILD_RUN" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
parent_run="$1"
child_run="$2"
test -f "logs/wheel_legged/$parent_run/model_57000.pt"
test -f "logs/wheel_legged/$child_run/model_57100.pt"
test -f "logs/wheel_legged/$child_run/model_57150.pt"

export WLG_EVAL_TARGET=eval_targets/spin_retention_v1.json
export WLG_EVAL_YAWS=-6,0,6
export WLG_EVAL_EPISODE_S=12
export WLG_EVAL_RANDOMIZATION=1
export WLG_HISTORICAL_DOMAIN_RAND=0
export WLG_DOMAIN_SCALE=0.25
export WLG_RESTITUTION_RANGE=0.0,0.3
export WLG_EVAL_PUSHES=0
export WLG_EVAL_YAW_RAMP_S=0.5
export WLG_EVAL_ABLATE_DOMAIN=none

for candidate in parent:57000 child:57100 child:57150; do
  kind="${candidate%%:*}"
  checkpoint="${candidate#*:}"
  if [[ "$kind" == parent ]]; then
    run="$parent_run"
  else
    run="$child_run"
  fi
  export WLG_EVAL_OUT="evaluations/spin/lowrest_d025_${kind}_${checkpoint}_seed102_20"
  echo "START $kind checkpoint=$checkpoint $(date -Is)"
  python wheel_legged_gym/scripts/evaluate_spin.py \
    --task=wheel_legged --headless --num_envs=48 --seed=102 \
    --load_run="$run" --checkpoint="$checkpoint" \
    > "/tmp/lowrest_d025_${kind}_${checkpoint}.out" 2>&1
  echo "DONE $kind checkpoint=$checkpoint $(date -Is)"
done
