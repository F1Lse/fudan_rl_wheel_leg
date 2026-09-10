# 长腿车型历史训练流程（自动续训）

这个脚本先增加两个长腿车型必需的起身阶段，再按旧模型的实际演变顺序训练同一个策略，每阶段增加 2000 轮：

1. `起身第1步`：从趴姿收腿到用户验证的低位轮式平衡姿态 `q=[0.0, 0.8, 0.0, -0.8]`，固定低位高度约 0.20 m。
2. `起身第2步`：继承低位起身策略，在轮式平衡基础上抬升到 `q=[0.2, 0.4, -0.2, -0.4]` 和约 0.28 m 高度。
3. `上台阶1`：平地，速度/转向 ±2，高度 0.10–0.33 m。
4. `上台阶2`：只训练上下楼梯，高度 0.23–0.33 m。
5. `上台阶3`：楼梯线速度扩大到 ±2.3 m/s，并加强速度跟踪。
6. `上台阶3_angz+`：转向扩大到 ±5 rad/s，高度扩大到 0.17–0.33 m。
7. `随机地形 v1`：30% 平地、20% 斜坡、30% 下楼梯、20% 上楼梯，开启速度课程。
8. `随机地形 v2`：保持相同配置继续收敛。
9. `随机地形 v3`：课程线速度上限提高到 2.8 m/s，命令保持时间改为 10 s。

历史训练的默认关节角、PD 和强域随机化会保持一致；长腿 URDF 使用当前车型设置。出生高度采用 Isaac Gym 实测正常的 0.12 m，使机器人从贴近趴姿的位置开始，而不是先从空中落下。

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

可随时按 `Ctrl+C`。模型默认每 100 轮保存一次，因此中断后最多损失不足 100 轮的进度。重新运行同一条 `train` 命令，会自动查找最新检查点并补足本阶段剩余轮数。脚本采用新的 `hist_recovery_v4_longlegs_*` 目录，不会接入此前未学会起身的 recovery 检查点、失败的平地 2000 轮或原来的 17000 轮训练。

中断训练后，自动用最新检查点进行 Isaac Gym 验证：

```bash
bash scripts/historical_curriculum.sh play
```

也可以指定某个累计检查点，例如验证第一阶段的低位起身模型：

```bash
bash scripts/historical_curriculum.sh play 2000
```

默认显示 5 台机器人；可以临时修改：

```bash
WLG_PLAY_NUM_ENVS=1 bash scripts/historical_curriculum.sh play
```

Play 默认保留训练时的噪声、质量、质心、摩擦、PD、动作延迟和外部推力随机化。如果需要做一次完全确定性的对照测试：

```bash
WLG_PLAY_WITH_RANDOMIZATION=0 \
  bash scripts/historical_curriculum.sh play 2000
```

验证后继续：

```bash
bash scripts/historical_curriculum.sh train
```

导出当前最新检查点：

```bash
bash scripts/historical_curriculum.sh export
```

默认输出为 `export_onnx/historical_longlegs_model_<轮数>.onnx`。如需指定名称：

```bash
WLG_EXPORT_OUT=export_onnx/longlegs_historical_final.onnx \
  bash scripts/historical_curriculum.sh export
```

不要在训练仍占用同一块 GPU 时同时运行 `play`。先中断训练、验证，再继续训练，显存和检查点状态最稳妥。
