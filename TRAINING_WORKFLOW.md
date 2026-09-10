# 轮腿机器人训练与 ONNX 导出流程

本文记录本仓库约定的训练路线。目标是为长腿版机器人分别生成三个策略：

1. 正常行驶策略：平地、坡面、上下楼梯和普通转向。
2. Spin 策略：高速原地旋转。
3. Jump 策略：起跳、腾空和落地。

三个策略分别导出为：

```text
longlegs_locomotion.onnx
longlegs_spin.onnx
longlegs_jump.onnx
```

## 1. 固定的机器人参数

长腿版机器人使用：

```text
plane/resources/robots/infantry_V4/urdf/infantry_V4_long_legs_0p21_0p25.urdf
```

`plane/wheel_legged_gym/envs/wheel_legged/wheel_legged_config.py` 中必须保留：

```python
file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/infantry_V4/urdf/infantry_V4_long_legs_0p21_0p25.urdf"
l1 = 0.21
l2 = 0.25
```

当前配置使用的控制参数为：

```python
stiffness = {"f0": 20.0, "f1": 20.0, "wheel": 0.0}
damping = {"f0": 1.0, "f1": 1.0, "wheel": 0.2}
```

除非经过专门验证，不要在切换训练阶段时把 URDF、`l1/l2`、PD 参数和默认关节姿态覆盖成历史旧车配置。

## 2. 地形配置的含义

`mesh_type` 只决定仿真地形，不决定机器人行为：

```python
mesh_type = "plane"       # 平地
mesh_type = "heightfield" # 高度场
mesh_type = "trimesh"     # 三角网格复杂地形
mesh_type = None           # 不创建地面
```

不要写成字符串 `"none"`。

只有 `heightfield` 和 `trimesh` 会启用复杂地形及地形课程。楼梯训练使用 `trimesh`。

地形比例顺序为：

```text
[平地, 光滑坡, 粗糙坡, 上楼梯, 下楼梯, 离散障碍]
```

## 3. 公共环境准备

训练环境要求：Ubuntu、Python 3.8、Isaac Gym Preview 4、CUDA 兼容的 PyTorch 和 NVIDIA GPU。

每次进入对应工程后设置：

```bash
conda activate leg
export PYTHONPATH=$PWD:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

`plane` 和 `jump` 都提供名为 `wheel_legged_gym` 的 Python 包。切换目录后应在对应目录重新执行 `pip install -e .`，避免误用另一个工程的包。

## 4. 正常行驶策略

正常行驶和 Spin 使用 `plane` 工程：

```bash
cd ~/fudan_rl_wheel_leg/plane
pip install -e .
```

### 重要：默认训练不等于最终混合地形策略

直接使用开源默认配置训练时：

```python
mesh_type = "plane"
```

因此一次默认训练只会得到平地正常行驶策略。此时即使配置文件中存在 `terrain_proportions`，它也不会生效，机器人不会在楼梯上训练。

最终正常行驶策略必须连续完成以下三个阶段：

```text
阶段一：默认平地从头训练
        ↓ 加载阶段一 checkpoint
阶段二：上下楼梯各 50% 续训
        ↓ 加载阶段二 checkpoint
阶段三：平地、坡面和上下楼梯混合续训
        ↓
导出 longlegs_locomotion.onnx
```

阶段一和阶段二的模型主要作为下一阶段的起点。最终应从阶段三中选择综合表现最好的 checkpoint，导出为 `longlegs_locomotion.onnx`。它与另外单独训练的 `longlegs_spin.onnx`、`longlegs_jump.onnx` 共同构成最终三个策略。

### 4.1 阶段一：开源默认平地训练

开源上游当前默认配置为：

```python
mesh_type = "plane"
```

从头训练，不使用 `--resume`：

```bash
python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged \
  --headless \
  --run_name=longlegs_flat_stage1 \
  --max_iterations=50000
```

可以先运行较少迭代并用 `play.py` 检查。默认每 100 轮保存一次 checkpoint。

### 4.2 阶段二：上下楼梯专项续训

历史“上台阶2/3”采用上、下楼梯各一半：

```python
mesh_type = "trimesh"
curriculum = True
max_init_terrain_level = 5
terrain_proportions = [0.0, 0.0, 0.0, 0.5, 0.5, 0.0]
```

历史阶段使用的命令范围为：

```python
commands.curriculum = False
lin_vel_x = [-2.0, 2.0]
ang_vel_yaw = [-2.0, 2.0]
height = [0.23, 0.33]
```

从阶段一的 checkpoint 继续训练：

```bash
python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged \
  --headless \
  --resume \
  --experiment_name=wheel_legged \
  --load_run=<阶段一目录名> \
  --checkpoint=<阶段一checkpoint> \
  --run_name=longlegs_stairs_stage2 \
  --max_iterations=<本阶段迭代数>
```

### 4.3 阶段三：混合地形泛化

历史后期使用过以下混合比例：

```python
mesh_type = "trimesh"
terrain_proportions = [0.3, 0.2, 0.0, 0.3, 0.2, 0.0]
commands.curriculum = True
```

它包含 30% 平地、20% 光滑坡、30% 上楼梯和 20% 下楼梯。基于阶段二的最佳 checkpoint 继续训练，使最终正常行驶策略同时适应平地和上下楼梯。

### 4.4 验证正常行驶策略

```bash
python wheel_legged_gym/scripts/play.py \
  --task=wheel_legged \
  --experiment_name=wheel_legged \
  --load_run=<训练目录名> \
  --checkpoint=<checkpoint编号>
```

不要只根据最后一轮选择策略；应比较多个 checkpoint 在平地、转向和楼梯上的实际表现。

### 4.5 导出正常行驶 ONNX

```bash
python export_onnx/export_onnx.py \
  --load_run=<训练目录名> \
  --checkpoint=<checkpoint编号> \
  --out=export_onnx/longlegs_locomotion.onnx
```

## 5. Spin 策略

Spin 没有独立的源码工程，仍使用 `plane`。它通常从正常行驶策略继续训练，通过逐级扩大 `ang_vel_yaw` 范围获得。

仓库历史实验使用过：

```python
ang_vel_yaw = [-7, 7]
ang_vel_yaw = [-10, 10]
ang_vel_yaw = [-13, 13]
```

推荐沿历史方式逐级续训，而不是从普通范围直接跳到 `[-13, 13]`。每一级都加载前一级表现最好的 checkpoint。

训练命令形式：

```bash
python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged \
  --headless \
  --resume \
  --experiment_name=wheel_legged \
  --load_run=<上一级目录名> \
  --checkpoint=<上一级checkpoint> \
  --run_name=longlegs_spin_stageN \
  --max_iterations=<本阶段迭代数>
```

导出：

```bash
python export_onnx/export_onnx.py \
  --load_run=<Spin训练目录名> \
  --checkpoint=<checkpoint编号> \
  --out=export_onnx/longlegs_spin.onnx
```

## 6. Jump 策略

Jump 使用独立的 `jump` 工程，因为它包含专门的跳跃检测和奖励：

```text
check_jump
flight
encourage_jump
base_height_flight
leg_tuck
takeoff_extend
line_z
```

进入 Jump 工程并重新安装对应包：

```bash
cd ~/fudan_rl_wheel_leg/jump
pip install -e .
export PYTHONPATH=$PWD:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

确认 `jump/wheel_legged_gym/envs/wheel_legged/wheel_legged_config.py` 使用长腿 URDF，并保持 `l1 = 0.21`、`l2 = 0.25`。

从头训练 Jump：

```bash
python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged \
  --headless \
  --run_name=longlegs_jump \
  --max_iterations=50000
```

验证后导出：

```bash
python export_onnx/export_onnx.py \
  --load_run=<Jump训练目录名> \
  --checkpoint=<checkpoint编号> \
  --out=export_onnx/longlegs_jump.onnx
```

## 7. ONNX 接口

当前序列策略导出的典型接口为：

```text
obs:         [batch, 25]
obs_history: [batch, 125]
actions:     [batch, 6]
```

部署端必须与训练端保持一致：

- 关节名称与顺序。
- 默认关节角。
- 观测顺序及缩放。
- 历史观测长度。
- 动作缩放。
- PD 参数。
- 控制周期。
- URDF/MuJoCo 模型中的腿长、质量、惯量和力矩限制。

## 8. 最终验收

最终分别检查：

- `longlegs_locomotion.onnx`：平地前后行驶、普通转向、上下楼梯。
- `longlegs_spin.onnx`：左右高速旋转，并能稳定退出旋转状态。
- `longlegs_jump.onnx`：可靠起跳、腾空收腿、稳定落地。

MuJoCo 运行端可以按行为切换策略。当前 `mujoco/python_tools/onnx_mj_chuanlian.py` 已实现正常行驶与 Jump 的临时切换；独立 Spin 策略需要在部署时加入相应的策略选择入口。
