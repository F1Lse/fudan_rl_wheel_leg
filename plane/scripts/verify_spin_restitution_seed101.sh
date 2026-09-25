#!/usr/bin/env bash
set -Eeuo pipefail

# Verify the contact restitution trend on a second seed.
# Usage: bash scripts/verify_spin_restitution_seed101.sh RUN CHECKPOINT
if (( $# != 2 )); then
  echo "Usage: $0 RUN CHECKPOINT" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
run="$1"
checkpoint="$2"
test -f "logs/wheel_legged/$run/model_${checkpoint}.pt"

export WLG_EVAL_TARGET=eval_targets/spin_retention_v1.json
export WLG_EVAL_YAWS=-6,0,6
export WLG_EVAL_EPISODE_S=12
export WLG_EVAL_RANDOMIZATION=1
export WLG_HISTORICAL_DOMAIN_RAND=0
export WLG_EVAL_PUSHES=0
export WLG_EVAL_YAW_RAMP_S=0.5
export WLG_EVAL_ABLATE_DOMAIN=none

for value in range 0.0 0.9; do
  if [[ "$value" == range ]]; then
    unset WLG_EVAL_RESTITUTION_FIXED
    label=range
  else
    export WLG_EVAL_RESTITUTION_FIXED="$value"
    label="fixed${value//./}"
  fi
  export WLG_EVAL_OUT="evaluations/spin/restitution_${label}_${checkpoint}_seed101_20"
  echo "START restitution=$value $(date -Is)"
  python wheel_legged_gym/scripts/evaluate_spin.py \
    --task=wheel_legged --headless --num_envs=48 --seed=101 \
    --load_run="$run" --checkpoint="$checkpoint" \
    > "/tmp/restitution_${label}_${checkpoint}_seed101.out" 2>&1
  echo "DONE restitution=$value $(date -Is)"
done
