# Spin 第一版数据闭环

目标先限定为平地、机身高度 0.16 m、前向速度 0、正反向高速原地旋转。目标和临时评分阈值放在 `plane/eval_targets/spin_stability_v1.json`，评测脚本默认读取它。现有 `plane/SPIN_CURRICULUM.md` 的 Play 依赖人工按键和肉眼观察；新增 `evaluate_spin.py` 在固定命令下自动运行并保存每回合结果。第一轮先测基线，再决定是采样、奖励还是表征需要改。

## 在训练服务器上运行

需要项目原有的 Ubuntu + Isaac Gym Preview 4 环境。在 `plane` 目录运行，给出确切 run 目录和 checkpoint 编号：

```bash
cd ~/fudan_rl_wheel_leg/plane
conda activate leg
export PYTHONPATH="$PWD:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

WLG_EVAL_OUT="evaluations/spin/baseline_65000_seed101" \
python wheel_legged_gym/scripts/evaluate_spin.py \
  --task=wheel_legged --headless --num_envs=32 --seed=101 \
  --load_run='<model_65000.pt 所在的 run 目录名>' --checkpoint=65000
```

如需测 ±13，设置 `WLG_EVAL_YAWS='7,-7,10,-10,13,-13'`，或复制一份目标 JSON 后用 `WLG_EVAL_TARGET=<目标文件>` 指定。对比不同 checkpoint 时，保持 yaw 列表、seed、环境数、随机化开关、回合长度和阈值完全相同。默认启用配置中的**中等范围**域随机化；`WLG_EVAL_RANDOMIZATION=0` 可单独测标称动力学。历史宽范围需另设 `WLG_HISTORICAL_DOMAIN_RAND=1`；`WLG_EVAL_PUSHES=1` 可独立打开推搡。实际摩擦、质量、PD 和推搡参数记录在 `manifest.json` 中，不能只凭“随机化开启”判断场景相同。也可用 `WLG_EVAL_EPISODE_S`、`WLG_EVAL_HEIGHT`、`WLG_EVAL_SPAWN_Z` 调整场景。

`WLG_EVAL_YAWS='0'` 可测静止保留能力，此时 yaw MAE 是静止阶段的平均绝对实际转速，漂移是静止阶段相对起点的最大位移。静止验收应单独设置更严格的 `WLG_EVAL_YAW_MAE_MAX` 等阈值，并检查原始连续指标。

若实际 Play 是先站稳再下达旋转命令，用 `WLG_EVAL_WARMUP_S=2` 复现；评测会先保持 yaw=0 两秒，再下达固定目标。`rms_tilt_deg`、`max_tilt_deg` 和 roll/pitch 角速度只统计站稳期之后，并另存 `warmup_max_tilt_deg`，避免把起立瞬态误判为旋转时晃动。从起转 1 秒后计算世界坐标下机身竖轴的水平摆动 `steady_world_axis_sway_rms_deg`，用于衡量头部朝向摇摆。`steady_wobble_rms_deg` 是机身坐标中的 roll/pitch 波动；偏航可能使固定方向的倾斜也在该数值中变化，不能单独据此判断头部晃动。冷启动与先站稳两种协议应分别保存结果。`WLG_EVAL_PUSHES=0` 可在保留其他域随机化的同时单独关闭外力推搡，用来定位晃动来源。

用 `WLG_EVAL_YAW_RAMP_S=0.5` 可让旋转目标在站稳期后用 0.5 秒从 0 线性升到指定 yaw。默认值 0 为立即跳变；过渡时长写入 manifest，比较脚本也会核对它。`time_to_90pct_yaw_s` 始终以**最终目标**的 90% 为准；yaw MAE 则对每一时刻正在执行的命令计算，所以不同过渡时长的 MAE 不可直接视为同一个定速误差。

目前标称仿真中，`Sep25_15-17-52_spin_targeted_from_flat_yaw6_nominal_v1_p075_planar10_guard10_inside5/model_57000.pt` 配合 0.5 秒命令过渡，在 seed 101 和 102、每格 20 回合的 -6/0/+6 测试全部达到暂定稳定门槛；立即给 +6 时只有 2/20 回合通过。完整数值和限制见 `docs/SPIN_RESTART_PLAN.md`。交互 Play 可设置 `WLG_PLAY_YAW_RAMP_S=0.5`，并用 `WLG_PLAY_YAW_STEP=6` 设置按键旋转速度。该 Play 选项只平滑发送给策略的命令，不改变 PT 权重。

在服务器带图形显示的终端中复现当前标称平地候选：

```bash
cd ~/fudan_rl_wheel_leg/plane
WLG_MESH_TYPE=plane WLG_PLAY_WITH_RANDOMIZATION=0 \
WLG_PLAY_HEIGHT=0.16 WLG_PLAY_YAW_STEP=6 WLG_PLAY_YAW_RAMP_S=0.5 \
WLG_PLAY_HUD=1 \
python wheel_legged_gym/scripts/play.py --task=wheel_legged \
  --load_run=Sep25_15-17-52_spin_targeted_from_flat_yaw6_nominal_v1_p075_planar10_guard10_inside5 \
  --checkpoint=57000
```

按住 `a` 为正向 yaw，按住 `d` 为负向 yaw；松开后同样用 0.5 秒过渡回到 0。按 `h` 可切换实时指标。HUD 绘在 Isaac Gym 的 3D viewer 中，跟随聚焦的机器人，显示设定和执行中的 yaw 命令、实际 yaw、跟踪误差、倾角、roll/pitch 角速度、左右轮速度及速度上限、左右轮力矩及力矩上限。`WL/WR` 的速度单位为 rad/s，`TL/TR` 的力矩单位为 N·m；接近各自上限时从绿色变为橙色和红色。该 HUD 是 3D 线条面板，不是固定在屏幕像素上的文字层，也不改变训练或推理。图形 Play 是定性检查，量化结论仍以固定种子评测为准。

## 推搡前后诊断

在服务器上执行 `bash scripts/evaluate_spin_push_diagnostics.sh <run目录名> <checkpoint编号>`。脚本用 12 秒回合、前 2 秒站稳、0.5 秒 yaw 过渡、中等动力学随机化和 7 秒时的一次随机推搡；每个方向 20 回合。`WLG_EVAL_TRACE_ALL=1` 让评测保存完整轨迹，`push_impulse_ns` 标出实际推搡的时刻和冲量。随后 `analyze_spin_push.py` 输出逐事件 `push_diagnostics.csv` 和汇总 `push_diagnostics.md`，分别统计推搡前 1 秒、后 0–1 秒及后 3–4 秒的 yaw 误差、倾角、姿态角速度、力矩与位置移动。

推搡后的世界坐标位置偏移与后段是否持续移动是两件事。判断恢复时应先看后 3–4 秒的运动与姿态是否回到推搡前水平，再看偏移是否符合具体任务允许范围；机器人策略没有世界坐标回原点的输入。

若某个速度在随机化下退化，`WLG_EVAL_ABLATE_DOMAIN` 可在评测时逐组关闭 `contact`（摩擦、恢复系数）、`body`（质量、惯量、质心）、`actuation`（PD、力矩、关节初值、延迟）或 `noise`，也可只关闭 `friction`、`restitution`。保持其余设置一致；输出 manifest 记录实际启用的参数。`WLG_EVAL_RESTITUTION_FIXED=0.3` 则在其余随机化保留时把接触恢复系数固定为 0.3。批量脚本为 `evaluate_spin_domain_ablation.sh`、`evaluate_spin_contact_ablation.sh`、`evaluate_spin_restitution_sweep.sh` 和 `verify_spin_restitution_seed101.sh`。关闭随机化与指定数值不是同一实验条件。

## 下一批真实数据

要让仿真参数与训练范围有依据，至少记录一次真实轮地接触和双向起转过程。每个样本保留统一时间戳、目标和实际 yaw、roll/pitch 及角速度、机身高度、六个关节的位置与速度、轮电机实际力矩／电流、控制输出、接触地面材质和是否发生外部碰撞；采样频率与控制周期也要一起注明。若现有系统暂时无法导出全部字段，先导出时间戳、目标 yaw、IMU 姿态与角速度、两轮速度和电流，并保留原始文件。对于接触参数，保存同一地面上的接触／回弹视频或传感器记录以及测试条件，供后续在仿真中标定；不要直接把这次仿真扫描的数值当作实物参数。

输出目录包含：

- `manifest.json`：checkpoint 的 SHA256、代码提交、种子、命令和评测设置；
- `episodes.csv`：每回合的完成、角速度误差、首次达到目标转速 90% 所需时间、方向正确率、平面漂移、倾角、roll/pitch 摆动角速度、高度误差、力矩饱和时间占比和重置原因；
- `summary.md`：按正负目标角速度分别汇总；
- `failures/episode_XXXX.csv`：前若干个不达标回合结束前的短时轨迹，字段包括命令、实际转速、高度、roll/pitch 角度与角速度、漂移、XY 位置以及 6 个关节的动作、力矩、位置和速度。

`stable` 暂用角速度平均绝对误差 ≤1 rad/s、最大漂移 ≤0.15 m、最大倾角 ≤15°、roll/pitch 合成角速度 RMS ≤1 rad/s 且完整跑完回合作为**临时分析阈值**。这些数值尚未按实机安全标准确认，可通过 `WLG_EVAL_YAW_MAE_MAX`、`WLG_EVAL_DRIFT_MAX`、`WLG_EVAL_TILT_MAX_DEG`、`WLG_EVAL_RP_RATE_RMS_MAX` 覆盖。报告原始连续数值比单个 `stable` 更重要。

## 第一轮如何用数据决策

1. 先测当前人工认为最好的 `model_65000.pt`，再测其他候选 checkpoint。优先看 +yaw 和 -yaw 是否对称，不能用绝对值平均掩盖某个方向的失败。
2. 若误差大而姿态稳定，检查命令覆盖、轮速/力矩饱和与奖励饱和；先调采样和目标梯度。若转速达标但漂移或倾角大，检查失败轨迹中的周期性摆动、轮速不对称与动作变化。
3. 若仿真通过、MuJoCo 或实机失败，记录同一命令序列下的角速度、姿态、关节与电机数据，先定位时延、PD、轮胎摩擦及质量/惯量的失配。实机数据要保存传感器原值和时间戳。
4. 再用相同环境步数对比一个单因素改动。不要同时换奖励、编码器和阶段次数，否则无法判断改进来源。

现有 Spin 脚本用 `independent` 命令采样，yaw 在整个区间均匀分布，恰好接近 ±10/±13 的高速端点占比较低；而 `spin_curriculum.sh` 的命令重采样间隔为 25 秒，超过默认 20 秒回合。这是两个明确的训练覆盖缺口。拿到定速基线后，第一组消融建议只改变**端点采样比例**，第二组再独立测试**起转、停车、反向切换**；不要把两项和奖励改动混在一次续训中。

2026-09-25 已在训练服务器的 Isaac Gym 环境完成最新 `model_70000.pt` 评测。具体协议、数值与下一步实验见 `docs/SPIN_MODEL_70000_AUDIT.md`；原始 CSV 与 manifest 保存在服务器和本地忽略目录 `plane/evaluations/spin/`。
