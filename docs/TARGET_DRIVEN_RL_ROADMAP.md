# 轮腿机器人目标驱动训练改造方案

## 要解决的问题

现在的工作流主要是修改奖励权重、命令范围和训练阶段，然后延长 PPO 训练。它可以逐步改善已覆盖的情况，但无法回答三个关键问题：目标是否真的达到、失败集中在哪些状态、额外训练是否产生了可泛化的改进。

目标工作流应变为：**自然语言目标 → 可测的任务规格 → 固定评测 → 失败轨迹 → 有针对性的采样或模型改造 → 同一评测复验**。训练轮数是预算，不是成功指标。

## 代码现状与能力边界

- `plane/wheel_legged_gym/envs/base/legged_robot_config.py` 中 actor 每步观测为 25 维，历史长度为 5，编码潜变量为 3 维。`compute_proprioception_observations()` 不包含前方地形；测得的高度进入 privileged observation，供 critic 使用。
- `plane/wheel_legged_gym/rsl_rl/modules/actor_critic_sequence.py` 的 PPO 动作路径使用 `latent.detach()`。编码器主要通过 `ppo.py` 的辅助损失拟合 privileged observation 前 3 维线速度。它不是一个已经学会提前理解地形的表征模型。
- 地形课程在 `legged_robot.py::_update_terrain_curriculum()` 用起点到当前位置的平面位移决定升级，并用线速度跟踪奖励决定降级。它不能代表原地旋转、定点站立、跳跃落地等任务的成功；即便在穿越任务中，位移也不能替代越过障碍、跌倒率和姿态安全指标。
- 现有 TensorBoard 有奖励和部分诊断量，`play.py` 主要靠人工操作与控制台采样观察；仓库未见固定测试场景、机器可读的逐轨迹失败记录和跨 checkpoint 的统一评分。因此目前不能从仓库数据判断哪个算法改动会带来多少收益。

## 第一阶段：把目标变成实验规格

每次只定义一个主目标，同时规定不能退化的能力。建议存为版本化的 YAML/JSON，与 checkpoint、代码提交和评测结果绑定。例如：

```yaml
id: spin_stability_v1
task: 高速原地旋转时保持水平、定高且不画圈
scenario:
  terrain: [flat]
  commands: {vx_mps: [0], yaw_radps: [7, -7, 10, -10], height_m: [0.16]}
  variations: [friction, mass, motor_gain, delay, spawn_pose]
metrics:
  primary: stable_spin_rate
  guards: [yaw_mae_radps, max_drift_m, max_tilt_deg, height_error_m, torque_limit_fraction]
acceptance:
  stable_spin_rate_min: 0.90
  no_regression_on: [idle_hold, flat_drive]
evaluation:
  fixed_seeds: [101, 102, 103]
  episodes_per_cell: 20
```

上面的数值仅是**示例**，必须按机器人能力、实际场景和安全要求定；不要把它们直接作为训练参数。每个 checkpoint 用同一组保留场景和种子评测，另留未见过的场景测试泛化。报告按地形 × 命令 × 扰动分格给出样本数、成功率及失败类型，不能只看总平均奖励。

Spin 当前还有一个特别值得测的分布缺口：`plane/scripts/spin_curriculum.sh` 把命令重采样间隔设为 25 秒，而默认回合长 20 秒，所以通常一个回合内不会练到“起转、停车、反向切换”。第一版先测定速稳态；随后应增加这些过渡测试，再决定是否把过渡命令纳入训练。

## 第二阶段：把“更好的数据”送进诊断回路

逐时刻记录最少字段：时间戳、环境/场景/随机种子、地形参数、目标命令、机身位置与姿态、线/角速度、高度、关节角与角速度、动作、力矩、接触状态、重置原因。实机数据再记录实际可得的 IMU/编码器、电机电流或估计力矩、电池电压、控制周期和通信延迟。标明每个字段来自仿真真值、估计器还是实机传感器。

每次失败保存失败前后的一段轨迹和可选视频，而非只保存平均曲线。每个实验包至少包含：`target.yaml`、`manifest.json`（提交号、checkpoint 哈希、URDF、PD 参数、仿真器版本、配置、种子）、`episodes.csv/parquet`（一行一回合）、`trajectories/`（失败及成功对照）、`summary.md`。优先给出“命令、实际响应、姿态、接触、动作/力矩”同步时间序列；单独一段视频通常不足以诊断因果。

分析时先把失败分为：感知不到、估计错误、命令分布不足、奖励漏洞、动力学/时延失配、动作或力矩饱和、策略切换问题。只有确定主要失败类型后才改训练。

## 第三阶段：按失败类型选学习方法

| 证据 | 优先改造 | 适用边界 |
| --- | --- | --- |
| 特定地形或命令格反复失败，但训练中很少出现 | 按失败率和学习进展重采样场景；保留基础场景防遗忘。参考 Prioritized Level Replay 的思想，不直接照搬其评分公式 | 先有可复现的场景 ID、逐回合结果和固定评测 |
| 摩擦、载荷、执行器偏差变化导致实机或 MuJoCo 失效 | 先用实测轨迹标定仿真参数与时延，再试 privileged teacher + 历史适应模块（RMA 类） | 部署时的 actor 只能读取真实可得的传感器；教师真值不能泄漏进部署输入 |
| 过坎或落差需要在接触前准备，而 proprioception 直到碰撞才提供线索 | 增加真实可部署的前视深度/高度/接触预估输入，再做感知教师与学生蒸馏 | 若没有前视传感器，只能优化被动反射与速度上限，无法要求提前预判不可观测障碍 |
| 训练奖励上升但目标指标不升，或出现投机动作 | 重写成功条件与奖励，加入防退化约束；固定预算做消融 | 不先换更大网络或盲目加轮次 |
| 平地、台阶、旋转、跳跃互相干扰 | 先比较单策略条件化与多策略加显式切换；测切换瞬态 | 以评测结果决定是否合并策略 |

## 推荐实现顺序和验收门槛

1. **评测与数据协议**：为一个具体目标建立无键盘、固定种子批量评测，输出逐回合指标与失败片段。先跑当前 checkpoint 得到基线。验收：重复运行评分一致，能解释最差的场景格。
2. **目标驱动采样**：根据基线失败分布调节地形和命令采样；固定一部分普通场景，避免专项能力提升时丢失已有能力。验收：相同环境步数下，保留测试集主指标提升，防退化指标达标。
3. **表征/算法消融**：先测试当前速度估计编码器、去除 `detach` 并联合优化、以及 privileged teacher → student/RMA 类方案。每次只变一个主要因素，保持动作接口和评测集固定。验收：部署可用输入上的收益稳定，MuJoCo 仍通过。
4. **平台迁移**：把 Isaac Gym Preview 4 → Isaac Lab 作为单独工程，先复刻现有动力学、观测、奖励和基线，再比较新方法。迁移本身不会修复目标定义或数据缺口。

## 第一轮实验的决策规则

- 每个方案与基线使用相同训练环境步数和相同评测集；额外报告真实耗时。
- 选择 checkpoint 依据保留测试集主指标及防退化指标，不依据最后一轮或训练 reward 最高点。
- 若目标指标没有改善，先查看各场景格的失败轨迹和观测可辨识性，再决定加数据、改奖励或换方法。
- 有实机数据时先做仿真对齐与误差定位。PPO 的 on-policy 更新不能直接把离线实机轨迹当作普通 rollout 混入；这些轨迹可用于系统辨识、场景复现、监督估计器或行为克隆实验。

## 相关原始资料

- Rudin 等，2022，[Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://proceedings.mlr.press/v164/rudin22a.html)：本仓库继承的并行 PPO 与地形课程基础。
- Jiang 等，2021，[Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html)：按场景学习潜力重采样的参考；在本机器人上需重新定义场景和评分。
- Kumar 等，2021，[RMA: Rapid Motor Adaptation for Legged Robots](https://www.roboticsproceedings.org/rss17/p011.pdf)：利用历史观测适应环境与动力学变化的两阶段方法。
- Miki 等，2022，[Learning robust perceptive locomotion for quadrupedal robots in the wild](https://arxiv.org/abs/2201.08117)：融合本体与外部感知，适合分析前视地形信息的价值。
- [Isaac Lab 官方说明](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/reinforcement_learning.html)与[legged_gym 上游说明](https://github.com/leggedrobotics/legged_gym)：长期平台迁移参考。
