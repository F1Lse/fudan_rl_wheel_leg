# `model_47000.pt` 平地稳定与高度跟踪特训

这条独立续训链用于修复 `model_47000.pt` 在平地静止时不同高度均有前后/roll 抖动，并恢复高度命令跟踪。纯平地修复后会立即混入重点地形复习，避免金字塔坡面、特殊下降地形和既有上楼梯能力因连续平地训练而遗忘。原始 `model_47000.pt` 不会被覆盖。

## 阶段

1. `47000–48000`：全部环境为平地，`vx=0、yaw=0`，高度在 0.16–0.33 m 内每 4 秒重新采样。100% 样本是静止锚点，使用较低熵系数和较窄域随机化，先找到安静的全高度平衡点。
2. `48000–49000`：70% 继续使用全高度静止锚点；其余样本恢复 `vx ±0.8 m/s、yaw ±1.0 rad/s`，并恢复完整历史域随机化。
3. `49000–51500`：从已经完成的 `model_49000.pt` 直接进入稳定混合地形。40% 平地、25% 平滑/粗糙金字塔坡面、10% 下楼梯、10% 上楼梯、15% 正向特殊下降地形；60% 的平地环境仍为全高度静止锚点。普通和特殊地形先限制在约 `1.0 m/s、1.0 rad/s`。
4. `51500–54000`：保持同样的地形重点，平地静止锚点降至 45%；普通地形扩展到 `1.5 m/s、1.5 rad/s`，特殊下降地形扩展到约 `1.4 m/s`。
5. `54000–56500`：50% 平地、20% 金字塔坡面、10% 下楼梯、10% 上楼梯、10% 特殊下降地形。普通地形恢复到 `2.0 m/s、2.0 rad/s`，特殊下降恢复到约 `1.8 m/s`；30% 的平地环境继续保留全高度静止锚点。

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

建议验证以下关键检查点：

```bash
WLG_PLAY_NUM_ENVS=1 WLG_PLAY_HEIGHT=0.16 \
  bash scripts/flat_stability_curriculum.sh play 48000

WLG_PLAY_NUM_ENVS=1 WLG_PLAY_HEIGHT=0.24 \
  bash scripts/flat_stability_curriculum.sh play 49000

WLG_PLAY_NUM_ENVS=20 WLG_PLAY_HEIGHT=0.24 \
  bash scripts/flat_stability_curriculum.sh play 51500

WLG_PLAY_NUM_ENVS=20 WLG_PLAY_FOCUS_TERRAIN=custom_drop \
  bash scripts/flat_stability_curriculum.sh play 56500
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
