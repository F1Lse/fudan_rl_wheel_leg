# 长腿车型 SPIN 综合策略训练

这是一条独立于正常行驶和下台阶专用模型的续训链。当前从实车表现可用的修正版 `model_59000.pt` 分叉，先按开源式奖励把 yaw 跟踪速度学起来：

- 0.16 m 低姿态原地旋转，按 yaw ±7、±10、±13 分三级训练，每级 2000 轮；
- 第一阶段只保留标准角速度跟踪、高度、姿态和动作平滑奖励；
- 暂时关闭坐标原点和两轮轮速差奖励，避免它们与转速获取目标竞争；
- 本轮不加入移动旋转、变高度或地形，先验证真正原地旋转。

最终只导出一个 SPIN ONNX。楼梯、离散障碍和双向特殊坎不进入本课程。

## 阶段

1. `59000–61000`：yaw ±7，2000 轮。
2. `61000–63000`：yaw ±10，2000 轮。
3. `63000–65000`：yaw ±13，2000 轮。

总计 6000 轮。每级完成后先检查实际转速和方向正确率，再决定是否进入下一级。坐标漂移暂时作为诊断指标，不作为这一阶段的训练目标。

世界坐标 XY 与机体线速度没有加入 actor 观测，因此这一阶段不使用坐标原点闭环奖励。后续如果需要原地稳定，再单独增加可部署的稳定约束；不要在转速尚未建立时同时加入多个强约束。

`spin_fixed` 的每个阶段都以高概率给出该阶段的固定转速，只保留少量近邻转速和静止样本。当前课程从 yaw ±7 逐级提高到 ±13，暂不加入平移、变高度或地形；先完成低位高速原地旋转，之后再单独训练“一边转一边走”。

当前 actor 命令仍保持部署兼容的 3 维顺序：`vx、yaw、height`，没有 `vy`。两轮底盘不能直接产生独立侧向速度；若上层使用云台坐标系速度，应先转换为底盘可执行的前向速度与 yaw/航向命令。增加 `vy` 会改变观测维度、ONNX 输入和 STM32 接口，需要另开一条从头训练的策略链。

训练输出中的倾角指标含义如下：

- `Mean mean_stationary_tilt_deg`：`vx=0` 原地旋转期间的平均总倾角，最直观地表示平时有多歪；
- `Mean rms_stationary_tilt_deg`：原地旋转期间总倾角的 RMS，对持续抖动和较大摆动更敏感；
- `Mean mean_spin_position_drift_m`：原地旋转相对起点的平均 XY 漂移；
- `Mean max_spin_position_drift_m`：每回合原地旋转的最大 XY 漂移；
- `Mean mean_spin_planar_speed_mps`：原地旋转期间实际 XY 合速度；
- `Mean mean_spin_wheel_speed_mismatch_rads`：两轮关节速度半差的绝对值，越接近 0 越接近原地旋转；
- `Mean max_tilt_deg`：完整回合的 roll+pitch 合成倾角峰值；
- `Mean max_abs_pitch_deg`：完整回合的 pitch 峰值，不包含 roll。

对于云台，优先观察平均值和 RMS，再用峰值排查加减速瞬间的严重晃动。

## 使用

```bash
conda activate leg
cd ~/fudan_rl_wheel_leg/plane

bash scripts/spin_curriculum.sh status
bash scripts/spin_curriculum.sh train-one
```

`train-one` 只完成当前第一个未完成阶段，适合每 2000 轮人工 Play 验证一次。使用 `train` 会自动跳过已完成阶段，并连续训练到 `65000`。

脚本会自动寻找以下路径形式的起始模型：

```text
logs/wheel_legged/*_spin_centerfix_58000_longlegs_s01_yaw_5/model_59000.pt
```

如果目录名不同，直接指定 PT：

```bash
WLG_SPIN_BASE_PT=logs/wheel_legged/你的目录/model_59000.pt \
  bash scripts/spin_curriculum.sh train
```

随时可以 `Ctrl+C`，重新执行 `train` 会从该阶段最新的 `model_*.pt` 接着训练。

验证 yaw ±7：

```bash
WLG_PLAY_NUM_ENVS=1 \
WLG_PLAY_HEIGHT=0.16 \
WLG_PLAY_INITIAL_YAW=7.0 \
WLG_PLAY_YAW_STEP=7.0 \
  bash scripts/spin_curriculum.sh play 61000
```

验证 yaw ±10：

```bash
WLG_PLAY_NUM_ENVS=1 WLG_PLAY_INITIAL_YAW=10.0 WLG_PLAY_YAW_STEP=10.0 \
  bash scripts/spin_curriculum.sh play 63000
```

验证最终 yaw 13：

```bash
WLG_PLAY_NUM_ENVS=1 WLG_PLAY_HEIGHT=0.16 \
WLG_PLAY_INITIAL_YAW=13.0 WLG_PLAY_YAW_STEP=13.0 \
  bash scripts/spin_curriculum.sh play 65000
```

如果目录名包含空格，或者脚本无法按 run 名自动发现 checkpoint，可以直接传入完整 PT 路径；路径必须用引号包住：

```bash
WLG_PLAY_NUM_ENVS=1 WLG_PLAY_HEIGHT=0.16 WLG_PLAY_YAW_STEP=7.0 \
  bash scripts/spin_curriculum.sh play \
  'logs/wheel_legged/<时间>_spin_open_59000_longlegs_s01_yaw_7_speed/model_61000.pt'
```

Play 中按住 `A/D` 旋转，`E` 停止。当前课程的前后速度固定为 `0`、高度固定为 `0.16 m`，因此 `W/S` 和 `X/C` 会被训练范围裁回固定值。检查点依次对应：`61000→7`、`63000→10`、`65000→13 rad/s`。实车应从较低命令逐步增加。

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
