#!/usr/bin/env bash
set -Eeuo pipefail

# Check the 0.5 s yaw transition under two dynamics distributions.
# Usage: bash scripts/evaluate_spin_ramp_robust.sh RUN CHECKPOINT
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
export WLG_EVAL_RANDOMIZATION=1

for mode in moderate_push historical_no_push; do
  case "$mode" in
    moderate_push)
      export WLG_HISTORICAL_DOMAIN_RAND=0 WLG_EVAL_PUSHES=1
      ;;
    historical_no_push)
      export WLG_HISTORICAL_DOMAIN_RAND=1 WLG_EVAL_PUSHES=0
      ;;
  esac
  for ramp in 0 0.5; do
    export WLG_EVAL_YAW_RAMP_S="$ramp"
    label="${ramp//./}"
    export WLG_EVAL_OUT="evaluations/spin/yaw6_${mode}_ramp${label}_${checkpoint}_seed102_20"
    echo "START mode=$mode ramp=$ramp $(date -Is)"
    python wheel_legged_gym/scripts/evaluate_spin.py \
      --task=wheel_legged --headless --num_envs=48 --seed=102 \
      --load_run="$run" --checkpoint="$checkpoint" \
      > "/tmp/yaw6_${mode}_ramp${label}_${checkpoint}.out" 2>&1
    echo "DONE mode=$mode ramp=$ramp $(date -Is)"
  done
done
