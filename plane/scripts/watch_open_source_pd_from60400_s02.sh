#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/fudan_rl_wheel_leg/plane
run=Oct07_20-50-25_open_source_pd_from60400_s02
while pgrep -f -- "--run_name=${run}" >/dev/null; do sleep 30; done
source /opt/conda/etc/profile.d/conda.sh
conda activate leg
export PYTHONPATH=/root/fudan_rl_wheel_leg/plane
latest=$(ls -1 logs/wheel_legged/${run}/model_*.pt | sort -V | tail -1)
ckpt=${latest##*model_}
ckpt=${ckpt%.pt}
python export_onnx/export_onnx.py --load_run=${run} --checkpoint=${ckpt} --out=export_onnx/open_source_pd_model_${ckpt}.onnx > /tmp/open_source_pd_from60400_s02_export.log 2>&1
echo exported=${ckpt} > /tmp/open_source_pd_from60400_s02_watch.log
