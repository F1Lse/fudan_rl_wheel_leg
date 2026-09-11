# 长腿车型平地平衡训练流程（自动续训）

当前流程只训练平地起身、平衡、高度和运动，不加入楼梯或随机地形。六个阶段连续训练同一个策略：

1. `0–3000`：从趴姿收腿到用户验证的低位轮式平衡姿态 `q=[0.0, 0.8, 0.0, -0.8]`，固定低位高度约 0.20 m。
2. `3000–6000`：继承低位起身策略，在轮式平衡基础上抬升到 `q=[0.2, 0.4, -0.2, -0.4]` 和约 0.28 m 高度。
3. `6000–10000`：零线速度、零转向，随机高度 0.16–0.30 m，每 3 s 更换一次高度命令。
4. `10000–14000`：随机高度 0.16–0.30 m，线速度和转向均为 -1.0–1.0，训练低速移动中的高度和平衡。
5. `14100–16100`：线速度扩展到 -1.5–1.5 m/s，转向扩展到 -2.0–2.0 rad/s。
6. `16100–18100`：线速度扩展到 -2.0–2.0 m/s，转向扩展到 -3.0–3.0 rad/s。

训练保持部署使用的默认关节角、PD 和强域随机化；长腿 URDF 使用当前车型设置。出生高度采用 Isaac Gym 实测正常的 0.12 m，使机器人从贴近趴姿的位置开始，而不是先从空中落下。起身和平衡阶段还会奖励“双轮接地且底盘、腿部不接地”。

## 使用

先进入 Isaac Gym 环境和 `plane` 目录：

```bash
conda activate leg
cd ~/fudan_rl_wheel_leg/plane
```

查看脚本检测到的阶段：

```bash
bash scripts/historical_curriculum.sh status
```

开始训练。一个阶段完成后会自动进入下一阶段：

```bash
bash scripts/historical_curriculum.sh train
```

可随时按 `Ctrl+C`。模型默认每 100 轮保存一次，因此中断后最多损失不足 100 轮的进度。重新运行同一条 `train` 命令，会自动查找最新检查点并补足本阶段剩余轮数。当前 `hist_recovery_v4_longlegs_s01_recovery_low` 检查点会继续使用，因此已经完成的 200 多轮不会丢失；旧的失败 recovery 和原来的 17000 轮模型不会混入。

中断训练后，自动用最新检查点进行 Isaac Gym 验证：

```bash
bash scripts/historical_curriculum.sh play
```

也可以指定某个累计检查点，例如验证第一阶段的低位平衡模型：

```bash
bash scripts/historical_curriculum.sh play 3000
```

默认显示 5 台机器人；可以临时修改：

```bash
WLG_PLAY_NUM_ENVS=1 bash scripts/historical_curriculum.sh play
```

Play 默认保留训练时的噪声、质量、质心、摩擦、PD、动作延迟和外部推力随机化。如果需要做一次完全确定性的对照测试：

```bash
WLG_PLAY_WITH_RANDOMIZATION=0 \
  bash scripts/historical_curriculum.sh play 3000
```

验证后继续：

```bash
bash scripts/historical_curriculum.sh train
```

导出当前最新检查点：

```bash
bash scripts/historical_curriculum.sh export
```

默认输出为 `export_onnx/plane_balance_longlegs_model_<轮数>.onnx`。如需指定名称：

```bash
WLG_EXPORT_OUT=export_onnx/longlegs_plane_balance_final.onnx \
  bash scripts/historical_curriculum.sh export
```

不要在训练仍占用同一块 GPU 时同时运行 `play`。先中断训练、验证，再继续训练，显存和检查点状态最稳妥。
