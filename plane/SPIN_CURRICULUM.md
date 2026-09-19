# 长腿车型 SPIN 定高整平训练

这条分支从已经比较可用的 `model_62000.pt` 开始，直接继续训练 yaw ±7；不经过 ±4，也不使用刚才机身上下窜动、倾斜明显的 `model_64000.pt` 作为起点。当前目标是先把低位原地旋转的机身高度、roll/pitch 和角速度稳定下来，再逐步扩大到 ±10、±13。

## 训练阶段

1. `62000–64000`：高度固定 `0.16 m`，yaw `[-7, 7]`，2000 轮。重点加强高度绝对误差、竖直速度、机身姿态和 roll/pitch 角速度惩罚；不增加两轮轮速差惩罚。
2. `64000–66000`：保持定高和水平目标，yaw `[-10, 10]`，2000 轮。
3. `66000–68000`：保持定高和水平目标，yaw `[-13, 13]`，2000 轮。

第一阶段完成后建议先 Play 检查：`h_real` 是否接近 `0.16 m`，roll/pitch 是否接近 0，且 `+7` 和 `-7` 是否都能转起来。若第一阶段仍然抖动，就停在 `model_64000.pt`，不要自动进入更高转速阶段。

训练时的前向速度固定为 0、高度固定为 0.16 m，命令使用正负对称的连续 yaw 范围。坐标中心约束保留，用于抑制画圈；两轮轮速半差惩罚保持为 0，避免再次把角速度能力压掉。训练阶段不加入地形、变高度或移动旋转。

## 使用

```bash
conda activate leg
cd ~/fudan_rl_wheel_leg/plane

bash scripts/spin_curriculum.sh status
bash scripts/spin_curriculum.sh train-one   # 只跑 62000→64000
bash scripts/spin_curriculum.sh train       # 自动跑完 68000
```

如果脚本找不到历史 checkpoint，显式指定 `model_62000.pt`：

```bash
WLG_SPIN_BASE_PT='logs/wheel_legged/<原来的run目录>/model_62000.pt' \
  bash scripts/spin_curriculum.sh train-one
```

Play 第一阶段结果：

```bash
WLG_PLAY_NUM_ENVS=1 \
WLG_PLAY_HEIGHT=0.16 \
WLG_PLAY_INITIAL_YAW=0 \
WLG_PLAY_YAW_STEP=7.0 \
  bash scripts/spin_curriculum.sh play 64000
```

Play 会在终端周期打印 `h_cmd`、`h_real`、`roll`、`pitch` 和实际 yaw。按 `A/D` 旋转，`E` 停止，`Q` 退出。确认 ±7 稳定后，再验证：

```bash
WLG_PLAY_NUM_ENVS=1 WLG_PLAY_HEIGHT=0.16 WLG_PLAY_YAW_STEP=10.0 \
  bash scripts/spin_curriculum.sh play 66000

WLG_PLAY_NUM_ENVS=1 WLG_PLAY_HEIGHT=0.16 WLG_PLAY_YAW_STEP=13.0 \
  bash scripts/spin_curriculum.sh play 68000
```

导出当前课程最新 ONNX：

```bash
bash scripts/spin_curriculum.sh export
```

也可以指定输出文件：

```bash
WLG_EXPORT_OUT=export_onnx/longlegs_spin_level_final.onnx \
  bash scripts/spin_curriculum.sh export
```

重点观察指标：

- `mean_stationary_tilt_deg`、`rms_stationary_tilt_deg`：原地旋转时的平均倾斜和 RMS；
- `max_abs_pitch_deg`、`max_tilt_deg`：单回合峰值，排查突然磕碰或失衡；
- `mean_spin_position_drift_m`、`mean_spin_planar_speed_mps`：原地转是否画圈；
- `mean_spin_real_abs_yaw`、`mean_spin_yaw_abs_error`、`spin_direction_accuracy`：±7 的正反向转速能力；
- `rew_base_height_l1`、`rew_lin_vel_z`、`rew_orientation`、`rew_ang_vel_xy`：定高和机身水平是否改善。
