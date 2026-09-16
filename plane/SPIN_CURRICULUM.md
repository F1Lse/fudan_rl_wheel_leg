# 长腿车型 SPIN 综合策略训练

这是一条独立于正常行驶和下台阶专用模型的续训链。当前从实车表现可用的修正版 `model_59000.pt` 分叉；过度约束后得到的旧 yaw ±7 `model_60000.pt` 实际只能达到约 4.2 rad/s，因此不再继承：

- 0.16 m 低姿态原地旋转，先用 yaw ±6 过渡 500 轮，再用 yaw ±7 巩固 1000 轮；
- 在 yaw ±7 已能达到约 6.45 rad/s 后，每次只增加 1 rad/s，以 500 轮一级逐步训练到 yaw ±13；
- 增强yaw跟踪，同时适度放松瞬时平面速度、世界坐标 XY 和两轮速度差惩罚；
- 本轮不加入移动旋转、变高度或地形，先验证真正原地旋转。

最终只导出一个 SPIN ONNX。楼梯、离散障碍和双向特殊坎不进入本课程。

## 阶段

1. `59000–59500`：yaw ±6 过渡，500 轮。
2. `59500–60500`：yaw ±7 巩固，1000 轮。
3. `60500–61000`：yaw ±8，500 轮。
4. `61000–61500`：yaw ±9，500 轮。
5. `61500–62000`：yaw ±10，500 轮。
6. `62000–62500`：yaw ±11，500 轮。
7. `62500–63000`：yaw ±12，500 轮。
8. `63000–63500`：yaw ±13，500 轮。

从 `60500` 继续到 `63500` 共 3000 轮。按当前约 `0.92 s/轮` 估算约需 46 分钟。各级只缓慢提高 yaw 跟踪奖励和原地约束，避免再次出现“约束很强但转速上不去”。

世界坐标 XY 与机体线速度没有加入 actor 观测，因此策略不能依靠绝对位置闭环返回起点。修正版保留坐标锚定作为训练约束，同时利用 actor 已有的两轮关节速度观测，惩罚左右轮数值不一致。当前 URDF 的左右腿根坐标系分别旋转 `+90°/-90°`，所以两个轮关节的物理轴方向相反；真正原地旋转时，两轮关节速度数值应当近似相同。该约束不会改变 ONNX 输入或 STM32 接口。

`spin_fixed` 的每个阶段都以高概率给出该阶段的固定转速，只保留少量近邻转速和静止样本。当前课程从 yaw ±6 逐级提高到 ±13，暂不加入平移、变高度或地形；先完成低位高速原地旋转，之后再单独训练“一边转一边走”。

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

`train-one` 只完成当前第一个未完成阶段，适合每 500 轮人工 Play 验证一次。使用 `train` 会自动跳过已完成阶段，并连续训练到 `63500`。

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

验证最新模型：

```bash
WLG_PLAY_NUM_ENVS=1 \
WLG_PLAY_HEIGHT=0.16 \
WLG_PLAY_INITIAL_YAW=6.0 \
WLG_PLAY_YAW_STEP=6.0 \
  bash scripts/spin_curriculum.sh play 59500
```

验证指定检查点（例如 yaw 7）：

```bash
WLG_PLAY_NUM_ENVS=1 WLG_PLAY_INITIAL_YAW=7.0 WLG_PLAY_YAW_STEP=7.0 \
  bash scripts/spin_curriculum.sh play 60500
```

验证最终 yaw 13：

```bash
WLG_PLAY_NUM_ENVS=1 WLG_PLAY_HEIGHT=0.16 \
WLG_PLAY_INITIAL_YAW=13.0 WLG_PLAY_YAW_STEP=13.0 \
  bash scripts/spin_curriculum.sh play 63500
```

如果目录名包含空格，或者脚本无法按 run 名自动发现 checkpoint，可以直接传入完整 PT 路径；路径必须用引号包住：

```bash
WLG_PLAY_NUM_ENVS=1 WLG_PLAY_HEIGHT=0.16 WLG_PLAY_YAW_STEP=7.0 \
  bash scripts/spin_curriculum.sh play \
  'logs/wheel_legged/<时间>_spin_yawrecover_59000_longlegs_s02_yaw_7/model_60500.pt'
```

Play 中按住 `A/D` 旋转，`E` 停止。当前课程的前后速度固定为 `0`、高度固定为 `0.16 m`，因此 `W/S` 和 `X/C` 会被训练范围裁回固定值。检查点依次对应：`59500→6`、`60500→7`、`61000→8`、`61500→9`、`62000→10`、`62500→11`、`63000→12`、`63500→13 rad/s`。实车应从较低命令逐步增加。

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
