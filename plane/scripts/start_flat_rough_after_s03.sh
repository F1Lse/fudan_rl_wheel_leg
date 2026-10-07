#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/fudan_rl_wheel_leg/plane
current=Oct07_21-29-13_open_source_pd_from61400_s03
checkpoint=logs/wheel_legged/${current}/model_62400.pt
while pgrep -f -- "--run_name=open_source_pd_from61400_s03" >/dev/null; do sleep 30; done
while [ ! -f "$checkpoint" ]; do sleep 10; done
bash scripts/open_source_pd_flat_rough_from62400_s04.sh > /tmp/open_source_pd_flat_rough_from62400_s04.log 2>&1
