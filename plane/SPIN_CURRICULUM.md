# 长腿车型 SPIN 综合策略训练

这是一条独立于正常行驶和下台阶专用模型的续训链。它从已经稳定的通用 `model_56500.pt` 开始，在同一个 SPIN 策略中逐步加入以下能力：

- 0.16 m 低姿态原地高速旋转，yaw 依次扩展到 ±7、±10、±13 rad/s；
- 原地旋转过程中切换 0.16–0.33 m 高度命令；
- 平地上一边平移一边旋转；
- 在平缓坡面和轻度起伏地面上一边移动一边转向；
- 原地旋转使用额外稳定奖励，移动转向允许必要的动态倾斜。

最终只导出一个 SPIN ONNX。楼梯、离散障碍和双向特殊坎不进入本课程。

## 阶段

1. `56500–59500`：平地、固定 0.16 m、`vx=0`、yaw ±7；先重点消除云台可见的 roll/pitch 晃动和漂移。
2. `59500–61500`：平地、固定 0.16 m、`vx=0`、yaw ±10。
3. `61500–63500`：平地、固定 0.16 m、`vx=0`、yaw ±13。
4. `63500–65500`：平地、0.16–0.33 m、yaw ±13；保留低位高速旋转并反复重采样高度。
5. `65500–68500`：平地综合采样原地高速旋转、旋转变高度、普通移动转向和低速高 yaw 移动。
6. `68500–71500`：55% 平地、25% 平缓坡、20% 轻度起伏；地形命令限制为 `vx ±1.0 m/s、yaw ±4 rad/s`。
7. `71500–73500`：最终巩固；平地仍保留 yaw ±13，地形扩展到 `vx ±1.2 m/s、yaw ±6 rad/s`。

由于 `model_56500.pt` 已经具备较好的静止稳定性，本课程将早期探索强度降到 `entropy_coef=0.002`，随后只在扩展高速 yaw 时小幅增加，最终再降到 `0.0015`。这样做是为了学习旋转能力，同时减少重新出现原地前后/roll 抖动的风险。

`spin_mixed` 命令采样器在平地分配：10% 静止锚点、25% 低位原地高速旋转、20% 原地旋转变高度、35% 平移旋转、10% 低速高 yaw 移动。原地样本会额外惩罚平面漂移、roll/pitch 角速度、动作高频变化，并直接约束机身坐标系重力向量接近 `[0, 0, -1]`；移动样本不会使用这些额外惩罚。

训练输出中的倾角指标含义如下：

- `Mean mean_stationary_tilt_deg`：`vx=0` 原地旋转期间的平均总倾角，最直观地表示平时有多歪；
- `Mean rms_stationary_tilt_deg`：原地旋转期间总倾角的 RMS，对持续抖动和较大摆动更敏感；
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
logs/wheel_legged/*_flat_stability_longlegs_s05_final_mixed_consolidation/model_56500.pt
```

如果目录名不同，直接指定 PT：

```bash
WLG_SPIN_BASE_PT=logs/wheel_legged/你的目录/model_56500.pt \
  bash scripts/spin_curriculum.sh train
```

随时可以 `Ctrl+C`，重新执行 `train` 会从该阶段最新的 `model_*.pt` 接着训练。

验证最新模型：

```bash
WLG_PLAY_NUM_ENVS=1 \
WLG_PLAY_HEIGHT=0.16 \
WLG_PLAY_YAW_STEP=7.0 \
  bash scripts/spin_curriculum.sh play
```

验证指定检查点：

```bash
WLG_PLAY_NUM_ENVS=1 WLG_PLAY_YAW_STEP=13.0 \
  bash scripts/spin_curriculum.sh play 63500
```

Play 中按住 `A/D` 旋转，`W/S` 平移，`X/C` 改变高度，`E` 停止。先从 yaw 3、7、10 逐步验证，再测试 13；不要一开始就在实车使用最大命令。

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
