#!/usr/bin/env bash
set -Eeuo pipefail

# Controlled continuation from the terrain checkpoint.  This first stage uses
# the open-source command/reward recipe and the first open-source actuator
# setting.  Later PD changes are separate stages so each checkpoint records the
# dynamics it was trained with.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /opt/conda/etc/profile.d/conda.sh
conda activate leg
export PYTHONPATH="$PWD"

export WLG_MESH_TYPE=plane
export WLG_TERRAIN_CURRICULUM=1
export WLG_HISTORICAL_DOMAIN_RAND=1
export WLG_DOMAIN_SCALE=1
export WLG_SPAWN_Z=0.12
export WLG_COMMAND_PROFILE=independent
export WLG_COMMAND_CURRICULUM=1
export WLG_COMMAND_RESAMPLING_TIME=5.0
export WLG_LIN_VEL_X_MIN=-2.0
export WLG_LIN_VEL_X_MAX=2.0
export WLG_ANG_VEL_YAW_MIN=-7.0
export WLG_ANG_VEL_YAW_MAX=7.0
export WLG_HEIGHT_MIN=0.10
export WLG_HEIGHT_MAX=0.30

# First open-source actuator stage.
export WLG_LEG_KP=20.0
export WLG_LEG_KD=1.0
export WLG_WHEEL_KD=0.2

# Restore the archived open-source reward balance.  The stronger action
# smoothness is intentional: it is the part most likely to suppress real
# hardware roll chatter.
export WLG_TRACKING_LIN_VEL_SCALE=1.0
export WLG_TRACKING_LIN_VEL_ENHANCE_SCALE=1.0
export WLG_TRACKING_ANG_VEL_SCALE=1.0
export WLG_TRACKING_ANG_VEL_ENHANCE_SCALE=1.0
export WLG_BASE_HEIGHT_SCALE=1.0
export WLG_BASE_HEIGHT_ENHANCE_SCALE=0.0
export WLG_ANG_VEL_XY_SCALE=-0.07
export WLG_ORIENTATION_SCALE=-18.0
export WLG_ACTION_RATE_SCALE=-0.1
export WLG_ACTION_SMOOTH_SCALE=-0.1
export WLG_NOMINAL_STATE_SCALE=-1.0
export WLG_LIN_VEL_Z_SCALE=-1.0
export WLG_TORQUES_SCALE=-0.0001
export WLG_COLLISION_SCALE=-1.0
export WLG_WHEEL_SUPPORT_SCALE=0.0
export WLG_DOF_POS_LIMITS_SCALE=-1.0
export WLG_PUSH_ROBOTS=1

python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged \
  --headless \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --num_envs=8192 \
  --experiment_name=wheel_legged \
  --run_name=open_source_pd_from59900_s01 \
  --resume \
  --load_run=Oct07_19-39-48_terrain_rough_rand_from59400_v25 \
  --checkpoint=59900 \
  --max_iterations=500
