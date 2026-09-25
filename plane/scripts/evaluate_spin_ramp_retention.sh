#!/usr/bin/env bash
set -Eeuo pipefail

# Paired step-versus-ramp test for -6, idle, and +6, with held-out seeds.
# Usage: bash scripts/evaluate_spin_ramp_retention.sh RUN CHECKPOINT
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
export WLG_EVAL_RANDOMIZATION=0
export WLG_EVAL_PUSHES=0
export WLG_HISTORICAL_DOMAIN_RAND=0

for seed in 101 102; do
  for ramp in 0 0.5; do
    export WLG_EVAL_YAW_RAMP_S="$ramp"
    label="${ramp//./}"
    export WLG_EVAL_OUT="evaluations/spin/yaw6_retention_ramp${label}_${checkpoint}_seed${seed}_20"
    echo "START seed=$seed ramp=$ramp $(date -Is)"
    python wheel_legged_gym/scripts/evaluate_spin.py \
      --task=wheel_legged --headless --num_envs=48 --seed="$seed" \
      --load_run="$run" --checkpoint="$checkpoint" \
      > "/tmp/yaw6_retention_ramp${label}_${checkpoint}_seed${seed}.out" 2>&1
    echo "DONE seed=$seed ramp=$ramp $(date -Is)"
  done
done
