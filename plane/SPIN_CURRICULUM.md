# 长腿车型 SPIN 综合策略训练

这是一条独立于正常行驶和下台阶专用模型的续训链。当前从实车表现可用的 yaw-5 `model_59000.pt` 分叉；先在已经能转起来的范围内修正稳态倾斜，再扩大转速：

- 0.16 m 低姿态原地旋转，按 yaw ±7、±7 稳态整平、±10、±13 逐级训练；
- 第一阶段不重新学习 yaw-5，而是从已验证的 `model_59000.pt` 继续；
- 采用历史版本的 `independent` 连续采样；例如 ±7 阶段会在 `[-7, 7]` 内均匀采样，完整保留之前的低速能力；
- 每条命令保持 25 秒，覆盖一个 20 秒回合，避免训练中反复从正向高速瞬间切换到反向高速；
- 保留较弱的平面速度和坐标中心约束，避免策略通过画大圈“伪造”旋转速度；
- 在 ±7 阶段增加一个稳定性 consolidation：对正方向采用低速到 `+7` 的渐进采样，同时保留负方向能力，再强化重力向量、横滚/俯仰角速度、坐标中心和左右轮速一致性；
- 稳定性阶段之后再恢复较宽 yaw 范围，后续阶段使用较温和的轮速差约束，避免重新出现明显画圈；
- 本轮不加入移动旋转、变高度或地形，先验证真正原地旋转。

最终只导出一个 SPIN ONNX。楼梯、离散障碍和双向特殊坎不进入本课程。

## 阶段

1. `59000–61000`：yaw ±7，2000 轮。
2. `61000–64000`：yaw ±7 稳态整平，3000 轮。
3. `64000–66000`：yaw ±10，2000 轮。
4. `66000–68000`：yaw ±13，2000 轮。

总计 9000 轮。每级完成后先检查实际转速、方向正确率、平均/RMS 倾角、平面速度和坐标漂移，再决定是否进入下一级。

世界坐标 XY 与机体线速度没有加入 actor 观测，因此这一阶段不使用坐标原点闭环奖励。后续如果需要原地稳定，再单独增加可部署的稳定约束；不要在转速尚未建立时同时加入多个强约束。

`independent` 在普通速度阶段的完整 yaw 范围内连续采样，因此扩大上限时不会丢掉上一阶段已经学会的转速。稳态整平阶段使用 `spin_recovery`：约 60% 样本为正方向渐进 yaw，约 20% 直接接近 `+7`，其余样本用于负方向和静止保持。当前课程从已验证的 yaw ±5 模型继续到 ±7，再逐级提高到 ±13，暂不加入平移、变高度或地形；先完成低位高速原地旋转，之后再单独训练“一边转一边走”。

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

`train-one` 只完成当前第一个未完成阶段，适合每个阶段人工 Play 验证一次。使用 `train` 会自动跳过已完成阶段，并连续训练到 `68000`。

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
WLG_PLAY_NUM_ENVS=1 WLG_PLAY_HEIGHT=0.16 WLG_PLAY_INITIAL_YAW=7.0 WLG_PLAY_YAW_STEP=7.0 \
  bash scripts/spin_curriculum.sh play 59000
```

验证 yaw ±7 稳态整平阶段：

```bash
WLG_PLAY_NUM_ENVS=1 WLG_PLAY_HEIGHT=0.16 WLG_PLAY_INITIAL_YAW=7.0 WLG_PLAY_YAW_STEP=7.0 \
  bash scripts/spin_curriculum.sh play 64000
```

验证 yaw ±10：

```bash
WLG_PLAY_NUM_ENVS=1 WLG_PLAY_INITIAL_YAW=10.0 WLG_PLAY_YAW_STEP=10.0 \
  bash scripts/spin_curriculum.sh play 66000
```

验证最终 yaw 13：

```bash
WLG_PLAY_NUM_ENVS=1 WLG_PLAY_HEIGHT=0.16 \
WLG_PLAY_INITIAL_YAW=13.0 WLG_PLAY_YAW_STEP=13.0 \
  bash scripts/spin_curriculum.sh play 68000
```

如果目录名包含空格，或者脚本无法按 run 名自动发现 checkpoint，可以直接传入完整 PT 路径；路径必须用引号包住：

```bash
WLG_PLAY_NUM_ENVS=1 WLG_PLAY_HEIGHT=0.16 WLG_PLAY_YAW_STEP=7.0 \
  bash scripts/spin_curriculum.sh play \
  'logs/wheel_legged/<时间>_spin_open_59000_longlegs_s01_yaw_7_speed/model_61000.pt'
```

Play 中按住 `A/D` 旋转，`E` 停止。当前课程的前后速度固定为 `0`、高度固定为 `0.16 m`，因此 `W/S` 和 `X/C` 会被训练范围裁回固定值。检查点依次对应：`61000→7`、`64000→7 稳态`、`66000→10`、`68000→13 rad/s`。实车应从较低命令逐步增加。

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
