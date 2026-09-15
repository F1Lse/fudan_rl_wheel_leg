# 长腿车型 SPIN 综合策略训练

这是一条独立于正常行驶和下台阶专用模型的续训链。它从已经学会 yaw ±5 的 `model_58000.pt` 分叉，重新塑造真正的原地旋转：

- 0.16 m 低姿态原地旋转，yaw 依次训练 ±5、±7、±10、±13 rad/s；
- 每个转速训练 1000 轮，并从第一阶段开始限制世界坐标 XY 漂移；
- 本轮不加入移动旋转、变高度或地形，先验证真正原地旋转。

最终只导出一个 SPIN ONNX。楼梯、离散障碍和双向特殊坎不进入本课程。

## 阶段

1. `58000–59000`：yaw ±5，坐标锚定权重 `-4`。
2. `59000–60000`：yaw ±7，坐标锚定权重 `-6`。
3. `60000–61000`：yaw ±10，坐标锚定权重 `-8`。
4. `61000–62000`：yaw ±13，坐标锚定权重 `-10`。

这条修正版训练链从已经学会 `yaw ±5 rad/s` 的 `model_58000.pt` 分叉，不再从通用运动模型重新学习旋转。四个阶段都采用较低探索强度，并在第一阶段就加入原地位置锚定，重点把“会转”收敛为“围绕当前位置转”。

`spin_fixed` 的每个阶段都以高概率给出该阶段的固定转速，只保留少量近邻转速和静止样本。当前修正版暂不加入平移、变高度和地形，先把 `yaw ±5/±7/±10/±13 rad/s` 的原地旋转分别练稳；这些能力验收后，再从合适的 checkpoint 分叉训练“一边转一边走”。

当前 actor 命令仍保持部署兼容的 3 维顺序：`vx、yaw、height`，没有 `vy`。两轮底盘不能直接产生独立侧向速度；若上层使用云台坐标系速度，应先转换为底盘可执行的前向速度与 yaw/航向命令。增加 `vy` 会改变观测维度、ONNX 输入和 STM32 接口，需要另开一条从头训练的策略链。

训练输出中的倾角指标含义如下：

- `Mean mean_stationary_tilt_deg`：`vx=0` 原地旋转期间的平均总倾角，最直观地表示平时有多歪；
- `Mean rms_stationary_tilt_deg`：原地旋转期间总倾角的 RMS，对持续抖动和较大摆动更敏感；
- `Mean mean_spin_position_drift_m`：原地旋转相对起点的平均 XY 漂移；
- `Mean max_spin_position_drift_m`：每回合原地旋转的最大 XY 漂移；
- `Mean max_tilt_deg`：完整回合的 roll+pitch 合成倾角峰值；
- `Mean max_abs_pitch_deg`：完整回合的 pitch 峰值，不包含 roll。

对于云台，优先观察平均值和 RMS，再用峰值排查加减速瞬间的严重晃动。

## 使用

```bash
conda activate leg
cd ~/fudan_rl_wheel_leg/plane

bash scripts/spin_curriculum.sh status
bash scripts/spin_curriculum.sh train
```

脚本会自动寻找以下路径形式的起始模型：

```text
logs/wheel_legged/*_spin_fixed_56500_longlegs_s04_yaw_5/model_58000.pt
```

如果目录名不同，直接指定 PT：

```bash
WLG_SPIN_BASE_PT=logs/wheel_legged/你的目录/model_58000.pt \
  bash scripts/spin_curriculum.sh train
```

随时可以 `Ctrl+C`，重新执行 `train` 会从该阶段最新的 `model_*.pt` 接着训练。

验证最新模型：

```bash
WLG_PLAY_NUM_ENVS=1 \
WLG_PLAY_HEIGHT=0.16 \
WLG_PLAY_YAW_STEP=5.0 \
  bash scripts/spin_curriculum.sh play 59000
```

验证指定检查点：

```bash
WLG_PLAY_NUM_ENVS=1 WLG_PLAY_YAW_STEP=13.0 \
  bash scripts/spin_curriculum.sh play 62000
```

如果目录名包含空格，或者脚本无法按 run 名自动发现 checkpoint，可以直接传入完整 PT 路径；路径必须用引号包住：

```bash
WLG_PLAY_NUM_ENVS=1 WLG_PLAY_HEIGHT=0.16 WLG_PLAY_YAW_STEP=7.0 \
  bash scripts/spin_curriculum.sh play \
  'logs/wheel_legged/<时间>_spin_centered_58000_longlegs_s02_yaw_7/model_60000.pt'
```

Play 中按住 `A/D` 旋转，`E` 停止。当前四阶段的前后速度固定为 `0`、高度固定为 `0.16 m`，因此 `W/S` 和 `X/C` 会被训练范围裁回固定值。依次验证：`59000 → yaw 5`、`60000 → yaw 7`、`61000 → yaw 10`、`62000 → yaw 13`；不要一开始就在实车使用最大命令。

导出最新 SPIN ONNX：

```bash
bash scripts/spin_curriculum.sh export
```

默认输出：

```text
export_onnx/longlegs_spin_model_<轮数>.onnx
```

指定名称：

```bash
WLG_EXPORT_OUT=export_onnx/longlegs_spin_final.onnx \
  bash scripts/spin_curriculum.sh export
```
