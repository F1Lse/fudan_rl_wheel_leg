#!/usr/bin/env bash
set -Eeuo pipefail

# Fixed contact restitution under otherwise moderate randomized dynamics.
# Usage: bash scripts/evaluate_spin_restitution_sweep.sh RUN CHECKPOINT
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

for restitution in 0.0 0.3 0.6 0.9; do
  export WLG_EVAL_RESTITUTION_FIXED="$restitution"
  label="${restitution//./}"
  export WLG_EVAL_OUT="evaluations/spin/restitution_fixed${label}_${checkpoint}_seed102_20"
  echo "START restitution=$restitution $(date -Is)"
  python wheel_legged_gym/scripts/evaluate_spin.py \
    --task=wheel_legged --headless --num_envs=48 --seed=102 \
    --load_run="$run" --checkpoint="$checkpoint" \
    > "/tmp/restitution_fixed${label}_${checkpoint}.out" 2>&1
  echo "DONE restitution=$restitution $(date -Is)"
done
