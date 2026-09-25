#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate the negative-yaw adaptation at fixed, paired moderate dynamics.
# Usage: bash scripts/evaluate_spin_negative_bridge.sh RUN
if (( $# != 1 )); then
  echo "Usage: $0 RUN" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
run="$1"
export WLG_EVAL_TARGET=eval_targets/spin_retention_v1.json
export WLG_EVAL_YAWS=-6,0,6
export WLG_EVAL_EPISODE_S=12
export WLG_EVAL_RANDOMIZATION=1
export WLG_HISTORICAL_DOMAIN_RAND=0
export WLG_EVAL_PUSHES=0
export WLG_EVAL_YAW_RAMP_S=0.5

for checkpoint in 57100 57150; do
  test -f "logs/wheel_legged/$run/model_${checkpoint}.pt"
  export WLG_EVAL_OUT="evaluations/spin/negative_bridge_mid_${checkpoint}_seed102_20"
  echo "START checkpoint=$checkpoint $(date -Is)"
  python wheel_legged_gym/scripts/evaluate_spin.py \
    --task=wheel_legged --headless --num_envs=48 --seed=102 \
    --load_run="$run" --checkpoint="$checkpoint" \
    > "/tmp/negative_bridge_mid_${checkpoint}.out" 2>&1
  echo "DONE checkpoint=$checkpoint $(date -Is)"
done
