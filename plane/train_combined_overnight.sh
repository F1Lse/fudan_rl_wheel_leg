#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "ERROR: pipeline stopped at line $LINENO" >&2' ERR

# Continue one policy through every curriculum stage, then export one ONNX.
# Usage:
#   bash train_combined_overnight.sh [source_run_name_or_path] [checkpoint]
# With no arguments, the newest *longlegs_mixed_custom_curb_drop_stage6* run
# and model_7000.pt are used.

cd "$(dirname "${BASH_SOURCE[0]}")"

LOG_ROOT="logs/wheel_legged"
START_RUN="${1:-}"
START_CHECKPOINT="${2:-7000}"

if [[ -z "$START_RUN" ]]; then
    START_RUN="$(find "$LOG_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -name '*longlegs_mixed_custom_curb_drop_stage6*' \
        -printf '%T@ %p\n' | sort -nr | sed -n '1p' | cut -d' ' -f2-)"
fi
if [[ -z "$START_RUN" ]]; then
    echo "ERROR: no stage-6 run was found under $LOG_ROOT" >&2
    exit 1
fi
if [[ "$START_RUN" != */* ]]; then
    START_RUN="$LOG_ROOT/$START_RUN"
fi
if [[ ! -f "$START_RUN/model_${START_CHECKPOINT}.pt" ]]; then
    echo "ERROR: missing $START_RUN/model_${START_CHECKPOINT}.pt" >&2
    exit 1
fi

python -c 'import torch, onnx; print("torch", torch.__version__, "onnx", onnx.__version__)'

PIPELINE_LOG="combined_overnight_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "Starting combined-policy pipeline"
echo "Source: $START_RUN/model_${START_CHECKPOINT}.pt"
echo "Pipeline log: $PIPELINE_LOG"

NEXT_RUN="$START_RUN"

run_stage() {
    local source_run="$1"
    local source_checkpoint="$2"
    local extra_iterations="$3"
    local expected_checkpoint="$4"
    local run_name="$5"
    shift 5

    echo
    echo "======================================================================"
    echo "Stage: $run_name"
    echo "Load:  $source_run/model_${source_checkpoint}.pt"
    echo "Train: $extra_iterations iterations -> model_${expected_checkpoint}.pt"
    echo "======================================================================"

    env "$@" python wheel_legged_gym/scripts/train.py \
        --task=wheel_legged \
        --experiment_name=wheel_legged \
        --resume \
        --load_run="$(basename "$source_run")" \
        --checkpoint="$source_checkpoint" \
        --max_iterations="$extra_iterations" \
        --run_name="$run_name" \
        --headless

    NEXT_RUN="$(find "$LOG_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -name "*_${run_name}" -printf '%T@ %p\n' | sort -nr | sed -n '1p' | cut -d' ' -f2-)"
    if [[ -z "$NEXT_RUN" || ! -f "$NEXT_RUN/model_${expected_checkpoint}.pt" ]]; then
        echo "ERROR: stage finished without model_${expected_checkpoint}.pt" >&2
        exit 1
    fi
    echo "Completed: $NEXT_RUN/model_${expected_checkpoint}.pt"
}

# 7000 -> 7500: first extend the height target while keeping motion moderate.
run_stage "$NEXT_RUN" "$START_CHECKPOINT" 500 7500 \
    longlegs_combined_height_0p28_stage7 \
    WLG_COMMAND_CURRICULUM=0 \
    WLG_LIN_VEL_X_MIN=-0.8 WLG_LIN_VEL_X_MAX=0.8 \
    WLG_ANG_VEL_YAW_MIN=-1.0 WLG_ANG_VEL_YAW_MAX=1.0 \
    WLG_HEIGHT_MIN=0.16 WLG_HEIGHT_MAX=0.28 \
    WLG_BASE_HEIGHT_SCALE=-10.0 WLG_BASE_HEIGHT_ENHANCE_SCALE=1.0 \
    WLG_CUSTOM_TERRAIN_MODE=descent_discrete

# 7500 -> 8000: reach the requested 0.33 m maximum height.
run_stage "$NEXT_RUN" 7500 500 8000 \
    longlegs_combined_height_0p33_stage8 \
    WLG_COMMAND_CURRICULUM=0 \
    WLG_LIN_VEL_X_MIN=-0.8 WLG_LIN_VEL_X_MAX=0.8 \
    WLG_ANG_VEL_YAW_MIN=-1.0 WLG_ANG_VEL_YAW_MAX=1.0 \
    WLG_HEIGHT_MIN=0.16 WLG_HEIGHT_MAX=0.33 \
    WLG_BASE_HEIGHT_SCALE=-10.0 WLG_BASE_HEIGHT_ENHANCE_SCALE=1.0 \
    WLG_CUSTOM_TERRAIN_MODE=descent_discrete

# 8000 -> 9000: mix the original descent course with its low-start climb.
run_stage "$NEXT_RUN" 8000 1000 9000 \
    longlegs_combined_reverse_climb_stage9 \
    WLG_COMMAND_CURRICULUM=1 \
    WLG_LIN_VEL_X_MIN=-0.8 WLG_LIN_VEL_X_MAX=0.8 \
    WLG_ANG_VEL_YAW_MIN=-1.0 WLG_ANG_VEL_YAW_MAX=1.0 \
    WLG_HEIGHT_MIN=0.16 WLG_HEIGHT_MAX=0.33 \
    WLG_BASE_HEIGHT_SCALE=1.0 WLG_BASE_HEIGHT_ENHANCE_SCALE=1.0 \
    WLG_CUSTOM_TERRAIN_MODE=bidirectional

# 9000 -> 9500: introduce faster yaw without jumping directly to the limit.
run_stage "$NEXT_RUN" 9000 500 9500 \
    longlegs_combined_yaw_2p5_stage10 \
    WLG_COMMAND_CURRICULUM=1 \
    WLG_LIN_VEL_X_MIN=-0.8 WLG_LIN_VEL_X_MAX=0.8 \
    WLG_ANG_VEL_YAW_MIN=-2.5 WLG_ANG_VEL_YAW_MAX=2.5 \
    WLG_HEIGHT_MIN=0.16 WLG_HEIGHT_MAX=0.33 \
    WLG_BASE_HEIGHT_SCALE=1.0 WLG_BASE_HEIGHT_ENHANCE_SCALE=1.0 \
    WLG_CUSTOM_TERRAIN_MODE=bidirectional

# 9500 -> 10500: final combined distribution, including yaw up to +/-4 rad/s.
run_stage "$NEXT_RUN" 9500 1000 10500 \
    longlegs_combined_yaw_4p0_final_stage11 \
    WLG_COMMAND_CURRICULUM=1 \
    WLG_LIN_VEL_X_MIN=-0.8 WLG_LIN_VEL_X_MAX=0.8 \
    WLG_ANG_VEL_YAW_MIN=-4.0 WLG_ANG_VEL_YAW_MAX=4.0 \
    WLG_HEIGHT_MIN=0.16 WLG_HEIGHT_MAX=0.33 \
    WLG_BASE_HEIGHT_SCALE=1.0 WLG_BASE_HEIGHT_ENHANCE_SCALE=1.0 \
    WLG_CUSTOM_TERRAIN_MODE=bidirectional

ONNX_OUT="export_onnx/longlegs_combined_height0p33_yaw4_reverse_10500.onnx"
python export_onnx/export_onnx.py \
    --load_run="$(basename "$NEXT_RUN")" \
    --checkpoint=10500 \
    --out="$ONNX_OUT"
python - "$ONNX_OUT" <<'PY'
import sys
import onnx

path = sys.argv[1]
model = onnx.load(path)
onnx.checker.check_model(model)
print(f"ONNX check passed: {path}")
for value in model.graph.input:
    dims = [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim]
    print("input", value.name, dims)
for value in model.graph.output:
    dims = [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim]
    print("output", value.name, dims)
PY

echo
echo "All stages completed."
echo "Final checkpoint: $NEXT_RUN/model_10500.pt"
echo "Final ONNX:      $ONNX_OUT"
