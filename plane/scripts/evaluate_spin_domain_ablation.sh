#!/usr/bin/env bash
set -Eeuo pipefail

# One-factor group ablations of the moderate dynamics evaluation.
# Usage: bash scripts/evaluate_spin_domain_ablation.sh RUN CHECKPOINT
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

for group in contact body actuation noise; do
  export WLG_EVAL_ABLATE_DOMAIN="$group"
  export WLG_EVAL_OUT="evaluations/spin/domain_ablate_${group}_${checkpoint}_seed102_20"
  echo "START ablation=$group $(date -Is)"
  python wheel_legged_gym/scripts/evaluate_spin.py \
    --task=wheel_legged --headless --num_envs=48 --seed=102 \
    --load_run="$run" --checkpoint="$checkpoint" \
    > "/tmp/domain_ablate_${group}_${checkpoint}.out" 2>&1
  echo "DONE ablation=$group $(date -Is)"
done
