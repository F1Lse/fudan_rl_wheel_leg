# `model_47000.pt` 平地稳定与高度跟踪特训

这条独立续训链用于修复 `model_47000.pt` 在平地静止时不同高度均有前后/roll 抖动，并恢复高度命令跟踪。它不进入楼梯、特殊坎或随机地形；完成后得到的 `model_52500.pt` 可作为下一轮正常地形训练的起点。原始 `model_47000.pt` 不会被覆盖。

## 阶段

1. `47000–48500`：全部环境为平地，`vx=0、yaw=0`，高度在 0.16–0.33 m 内每 4 秒重新采样。100% 样本是静止锚点，使用较低熵系数和较窄域随机化，先找到安静的全高度平衡点。
2. `48500–50500`：70% 继续使用全高度静止锚点；其余样本恢复 `vx ±0.8 m/s、yaw ±1.0 rad/s`，并恢复完整历史域随机化。
3. `50500–52500`：40% 保留全高度静止锚点；其余样本恢复平地 `vx ±2.5 m/s、yaw ±3.0 rad/s`，避免特训后只会站立。

高度奖励同时包含精确高斯奖励、宽高斯误差项和新的不饱和 L1 高度误差。静止锚点会额外惩罚平面漂移、roll/pitch 角速度、倾斜以及动作的一阶和二阶变化；移动样本不使用这些额外惩罚。

## 使用

```bash
conda activate leg
cd ~/fudan_rl_wheel_leg/plane

bash scripts/flat_stability_curriculum.sh status
bash scripts/flat_stability_curriculum.sh train
```

脚本默认查找：

```text
logs/wheel_legged/*_hist_recovery_v4_longlegs_s22_highstand_anchor_consolidation/model_47000.pt
```

找不到时指定实际路径：

```bash
WLG_FLAT_STABILITY_BASE_PT=logs/wheel_legged/你的目录/model_47000.pt \
  bash scripts/flat_stability_curriculum.sh train
```

可以随时按 `Ctrl+C`。重新执行 `train` 会自动查找当前阶段最新 checkpoint 并继续。

建议分别验证三个阶段：

```bash
WLG_PLAY_NUM_ENVS=1 WLG_PLAY_HEIGHT=0.16 \
  bash scripts/flat_stability_curriculum.sh play 48500

WLG_PLAY_NUM_ENVS=1 WLG_PLAY_HEIGHT=0.24 \
  bash scripts/flat_stability_curriculum.sh play 50500

WLG_PLAY_NUM_ENVS=1 WLG_PLAY_HEIGHT=0.33 \
  bash scripts/flat_stability_curriculum.sh play 52500
```

Play 中使用 `X/C` 改变高度，`W/S` 测试前后速度，`A/D` 测试转向，`E` 回到零命令。应重点观察每次停止后是否能快速收敛到静止，以及 0.16、0.20、0.24、0.28、0.30、0.33 m 的高度偏差和前后/roll 摆动。

导出最新模型：

```bash
bash scripts/flat_stability_curriculum.sh export
```

默认输出为：

```text
export_onnx/longlegs_flat_stable_model_<轮数>.onnx
```
