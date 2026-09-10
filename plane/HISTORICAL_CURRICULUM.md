# 长腿车型历史训练流程（自动续训）

这个脚本按旧模型的实际演变顺序训练同一个策略，每阶段增加 2000 轮：

1. `上台阶1`：平地，速度/转向 ±2，高度 0.10–0.33 m。
2. `上台阶2`：只训练上下楼梯，高度 0.23–0.33 m。
3. `上台阶3`：楼梯线速度扩大到 ±2.3 m/s，并加强速度跟踪。
4. `上台阶3_angz+`：转向扩大到 ±5 rad/s，高度扩大到 0.17–0.33 m。
5. `随机地形 v1`：30% 平地、20% 斜坡、30% 下楼梯、20% 上楼梯，开启速度课程。
6. `随机地形 v2`：保持相同配置继续收敛。
7. `随机地形 v3`：课程线速度上限提高到 2.8 m/s，命令保持时间改为 10 s。

历史训练的默认关节角、PD 和强域随机化会保持一致；长腿 URDF、0.30 m 出生高度以及防止跪姿投机的接触配置仍使用当前车型设置。

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

可随时按 `Ctrl+C`。模型默认每 100 轮保存一次，因此中断后最多损失不足 100 轮的进度。重新运行同一条 `train` 命令，会自动查找最新检查点并补足本阶段剩余轮数，不会接入原来的 17000 轮训练。

中断训练后，自动用最新检查点进行 Isaac Gym 验证：

```bash
bash scripts/historical_curriculum.sh play
```

默认显示 5 台机器人；可以临时修改：

```bash
WLG_PLAY_NUM_ENVS=1 bash scripts/historical_curriculum.sh play
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
