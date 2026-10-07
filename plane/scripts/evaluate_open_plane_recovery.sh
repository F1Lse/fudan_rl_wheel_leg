#!/usr/bin/env bash
set -euo pipefail
cd /root/fudan_rl_wheel_leg/plane
source /opt/conda/etc/profile.d/conda.sh
conda activate leg
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export WLG_MESH_TYPE=plane WLG_TERRAIN_CURRICULUM=0
export WLG_HISTORICAL_DOMAIN_RAND=1 WLG_DOMAIN_SCALE=1 WLG_PUSH_ROBOTS=1
export WLG_LIN_VEL_X_MIN=-2.5 WLG_LIN_VEL_X_MAX=2.5
export WLG_ANG_VEL_YAW_MIN=-5 WLG_ANG_VEL_YAW_MAX=5
export WLG_HEIGHT_MIN=0.16 WLG_HEIGHT_MAX=0.28
export WLG_COMMAND_PROFILE=independent
while kill -0 1805 2>/dev/null; do sleep 5; done
base=logs/wheel_legged/Sep13_01-18-42_flat_stability_longlegs_s05_final_mixed_consolidation/model_56500.pt
latest=logs/wheel_legged/Oct07_16-38-54_open_plane_from56500_highspeed/model_57000.pt
for randomized in 0 1; do
    for label in baseline56500 flat57000; do
        checkpoint="$base"
        if [ "$label" = flat57000 ]; then checkpoint="$latest"; fi
        test -s "$checkpoint"
        export WLG_EVAL_CHECKPOINT="$checkpoint"
        export WLG_EVAL_RANDOMIZATION="$randomized"
        export WLG_EVAL_OUT="eval_results/open_plane_recovery/${label}_random${randomized}.json"
        python wheel_legged_gym/scripts/evaluate_locomotion_hold.py --task=wheel_legged --headless
    done
done
echo EVALUATION_COMPLETE
