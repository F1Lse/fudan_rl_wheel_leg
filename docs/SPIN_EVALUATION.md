# Spin 第一版数据闭环

目标先限定为平地、机身高度 0.16 m、前向速度 0、正反向高速原地旋转。目标和临时评分阈值放在 `plane/eval_targets/spin_stability_v1.json`，评测脚本默认读取它。现有 `plane/SPIN_CURRICULUM.md` 的 Play 依赖人工按键和肉眼观察；新增 `evaluate_spin.py` 在固定命令下自动运行并保存每回合结果。第一轮先测基线，再决定是采样、奖励还是表征需要改。

## 在训练服务器上运行

需要项目原有的 Ubuntu + Isaac Gym Preview 4 环境。在 `plane` 目录运行，给出确切 run 目录和 checkpoint 编号：

```bash
cd ~/fudan_rl_wheel_leg/plane
conda activate leg
export PYTHONPATH="$PWD:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

WLG_EVAL_OUT="evaluations/spin/baseline_65000_seed101" \
python wheel_legged_gym/scripts/evaluate_spin.py \
  --task=wheel_legged --headless --num_envs=32 --seed=101 \
  --load_run='<model_65000.pt 所在的 run 目录名>' --checkpoint=65000
```

如需测 ±13，设置 `WLG_EVAL_YAWS='7,-7,10,-10,13,-13'`，或复制一份目标 JSON 后用 `WLG_EVAL_TARGET=<目标文件>` 指定。对比不同 checkpoint 时，保持 yaw 列表、seed、环境数、随机化开关、回合长度和阈值完全相同。默认启用现有域随机化；`WLG_EVAL_RANDOMIZATION=0` 可单独测标称动力学。也可用 `WLG_EVAL_EPISODE_S`、`WLG_EVAL_HEIGHT`、`WLG_EVAL_SPAWN_Z` 调整场景。

输出目录包含：

- `manifest.json`：checkpoint 的 SHA256、代码提交、种子、命令和评测设置；
- `episodes.csv`：每回合的完成、角速度误差、方向正确率、平面漂移、倾角、高度误差、力矩饱和时间占比和重置原因；
- `summary.md`：按正负目标角速度分别汇总；
- `failures/episode_XXXX.csv`：前若干个不达标回合结束前的短时轨迹，字段包括命令、实际转速、高度、倾角、漂移、XY 位置以及 6 个关节的动作、力矩、位置和速度。

`stable` 暂用角速度平均绝对误差 ≤1 rad/s、最大漂移 ≤0.15 m、最大倾角 ≤15° 且完整跑完回合作为**临时分析阈值**。这些数值尚未按实机安全标准确认，可通过 `WLG_EVAL_YAW_MAE_MAX`、`WLG_EVAL_DRIFT_MAX`、`WLG_EVAL_TILT_MAX_DEG` 覆盖。报告原始连续数值比单个 `stable` 更重要。

## 第一轮如何用数据决策

1. 先测当前人工认为最好的 `model_65000.pt`，再测其他候选 checkpoint。优先看 +yaw 和 -yaw 是否对称，不能用绝对值平均掩盖某个方向的失败。
2. 若误差大而姿态稳定，检查命令覆盖、轮速/力矩饱和与奖励饱和；先调采样和目标梯度。若转速达标但漂移或倾角大，检查失败轨迹中的周期性摆动、轮速不对称与动作变化。
3. 若仿真通过、MuJoCo 或实机失败，记录同一命令序列下的角速度、姿态、关节与电机数据，先定位时延、PD、轮胎摩擦及质量/惯量的失配。实机数据要保存传感器原值和时间戳。
4. 再用相同环境步数对比一个单因素改动。不要同时换奖励、编码器和阶段次数，否则无法判断改进来源。

现有 Spin 脚本用 `independent` 命令采样，yaw 在整个区间均匀分布，恰好接近 ±10/±13 的高速端点占比较低；而 `spin_curriculum.sh` 的命令重采样间隔为 25 秒，超过默认 20 秒回合。这是两个明确的训练覆盖缺口。拿到定速基线后，第一组消融建议只改变**端点采样比例**，第二组再独立测试**起转、停车、反向切换**；不要把两项和奖励改动混在一次续训中。

当前工作区没有可运行的 Isaac Gym GPU 环境或实际失败记录，因此这里提供的是评测入口和数据协议，尚无基线数值结论。
