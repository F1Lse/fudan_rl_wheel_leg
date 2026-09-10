# 远程训练服务器可视化与模型 Play 指南

本文记录在 Windows 上通过 MobaXterm 连接远程训练服务器，并显示强化学习 `play` 仿真窗口的方法。

## 1. 连接服务器并启用 X11

在 MobaXterm 中确认 X Server 已启动，然后从 MobaXterm 的本地终端连接：

```bash
ssh -Y -p 31317 root@183.147.142.40
```

参数说明：

- `-Y`：启用可信 X11 图形转发。
- `-p 31317`：指定 SSH 端口。
- `root@183.147.142.40`：远程用户名和服务器地址。

不要在服务器终端内再次执行这条 SSH 命令，应当从 MobaXterm 的本地终端连接。

## 2. 修正远端 DISPLAY

MobaXterm 本地执行：

```bash
echo $DISPLAY
```

本次环境的正常结果为：

```text
127.0.0.1:0.0
```

服务器登录后，环境变量曾被平台覆盖为失效的：

```text
:20
```

诊断结果表明 SSH 的 X11 转发实际使用显示编号 `10`：

- `xauth` 中存在 `gpufree-container/unix:10`。
- SSH 服务监听 TCP 端口 `6010`。
- X11 端口和显示编号的对应关系为 `6000 + 10 = 6010`。

因此，在本次 SSH 会话中修正为：

```bash
export DISPLAY=localhost:10.0
```

检查：

```bash
echo $DISPLAY
```

预期输出：

```text
localhost:10.0
```

> 不建议将 `localhost:10.0` 永久写入 `.bashrc`。同时建立多个 SSH 会话时，编号可能变成 `11`、`12` 等。

## 3. 测试图形转发

服务器具备 Tkinter 时，可以运行：

```bash
python -c "import tkinter as tk; w=tk.Tk(); w.title('X11 Test'); w.geometry('300x150'); w.mainloop()"
```

Windows 桌面弹出测试窗口，说明 X11 转发正常。关闭窗口可以点击窗口关闭按钮，或在终端按 `Ctrl+C`。

如果出现：

```text
_tkinter.TclError: couldn't connect to display ":20"
```

说明仍在使用平台遗留的无效 `DISPLAY=:20`，需要重新执行：

```bash
export DISPLAY=localhost:10.0
```

## 4. 播放 Plane 的 PT 模型

本次模型文件：

```text
/root/fudan_rl_wheel_leg/plane/logs/wheel_legged/Sep08_22-49-25_longlegs_flat_stage1/model_2800.pt
```

先激活训练时使用的 Conda 环境。不要默认使用缺少依赖的 `(base)` 环境：

```bash
conda env list
conda activate <训练环境名>
```

验证基本依赖：

```bash
python -c "import numpy, torch; print('numpy:', numpy.__version__); print('torch:', torch.__version__)"
```

进入 Plane 工程并运行：

```bash
export DISPLAY=localhost:10.0
cd /root/fudan_rl_wheel_leg/plane

python wheel_legged_gym/scripts/play.py \
  --task=wheel_legged \
  --load_run=Sep08_22-49-25_longlegs_flat_stage1 \
  --checkpoint=2800
```

参数说明：

- `--task=wheel_legged`：选择轮腿机器人任务。
- `--load_run=Sep08_22-49-25_longlegs_flat_stage1`：指定训练批次目录。
- `--checkpoint=2800`：加载 `model_2800.pt`。
- 不要添加 `--headless`，否则不会显示仿真窗口。

## 5. 测试导出的 ONNX

本次 ONNX 文件：

```text
/root/fudan_rl_wheel_leg/plane/export_onnx/longlegs_flat_2800.onnx
```

`wheel_legged_gym/scripts/play.py` 主要加载训练 checkpoint。直接测试 ONNX 时，应使用 MuJoCo 推理脚本。

并联腿版本：

```bash
export DISPLAY=localhost:10.0
cd /root/fudan_rl_wheel_leg

python mujoco/python_tools/onnx_mj_binglian.py \
  --onnx plane/export_onnx/longlegs_flat_2800.onnx
```

串联虚拟腿版本：

```bash
python mujoco/python_tools/onnx_mj_chuanlian.py \
  --onnx plane/export_onnx/longlegs_flat_2800.onnx
```

## 6. 常见问题

### `ModuleNotFoundError: No module named 'numpy'`

当前 Python 环境不是训练环境。先检查并激活正确的 Conda 环境：

```bash
conda env list
grep -E "conda activate|train.py" ~/.bash_history | tail -40
```

不要只在错误的 `(base)` 环境中逐个安装缺失包，因为后续通常还需要匹配版本的 PyTorch、Isaac Gym、rsl_rl 等依赖。

### `can't open file './scripts/play.py'`

脚本不在工程根目录的 `scripts` 下。Plane 的正确路径是：

```text
/root/fudan_rl_wheel_leg/plane/wheel_legged_gym/scripts/play.py
```

### `cannot open display` 或 GLFW/GLX/OpenGL 报错

依次检查：

```bash
echo $DISPLAY
xauth list | awk '{print $1}'
ss -ltnp | grep -E ':60[0-9][0-9]'
```

如果 SSH X11 已建立但变量仍为 `:20`，根据监听端口恢复正确编号。比如监听 `6010`，使用：

```bash
export DISPLAY=localhost:10.0
```

如果服务器明确提示 `X11 forwarding request failed`，或者 Isaac/OpenGL 渲染不支持 X11，应改用平台提供的 VNC、NoMachine 或其他远程桌面。

## 7. 每次运行的简化流程

```bash
# MobaXterm 本地终端
ssh -Y -p 31317 root@183.147.142.40

# 服务器终端
conda activate <训练环境名>
export DISPLAY=localhost:10.0
cd /root/fudan_rl_wheel_leg/plane
python wheel_legged_gym/scripts/play.py --task=wheel_legged \
  --load_run=Sep08_22-49-25_longlegs_flat_stage1 --checkpoint=2800
```

