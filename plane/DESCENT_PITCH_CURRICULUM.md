# 下台阶低俯仰专用训练

这是一条从通用 `model_56500.pt` 分叉的专用策略，不覆盖通用模型。训练地形只包含：

- 65% 实测特殊下降：50 mm 坎顶、150 mm 落差、0.5 m 平台、再下降 200 mm；
- 35% 标准金字塔台阶爬升。

训练会记录 `Mean max_abs_pitch_deg`。它表示每个已结束回合的最大绝对 pitch 的平均值，是本次最重要的判断指标；最终希望 play 时绝大多数越障过程小于 `20°`。不要只看 `Mean rew_orientation`。

## 分阶段目标

1. `56500–58000`：低速适应，pitch 超出 `15°` 开始额外惩罚；特殊下降最高 `1.0 m/s`。
2. `58000–60000`：中速强化，软目标收紧到 `12°`；特殊下降最高 `1.6 m/s`。
3. `60000–62500`：快速巩固，软目标 `10°`；特殊下降最高 `2.2 m/s`，持续超过 `24°` 约 `0.10 s` 会重置。

终止阈值不是允许长期保持的目标角度。它保留跨越台阶瞬间的学习空间；软惩罚会从更小角度开始持续推动策略减小 pitch。

## 云端运行

```bash
cd ~/fudan_rl_wheel_leg/plane
conda activate leg

WLG_DESCENT_BASE_PT=logs/wheel_legged/你的56500目录/model_56500.pt \
  bash scripts/descent_pitch_curriculum.sh train
```

脚本会自动完成三段训练。中途 `Ctrl+C` 后重复同一条命令，会从该阶段最新保存的 `model_*.pt` 继续。

查看进度：

```bash
bash scripts/descent_pitch_curriculum.sh status
```

## Play 验证

验证特殊下降：

```bash
WLG_PLAY_FOCUS_TERRAIN=custom_drop \
WLG_PLAY_LIN_VEL_CMD=1.5 \
WLG_PLAY_HEIGHT=0.20 \
  bash scripts/descent_pitch_curriculum.sh play 62500
```

验证金字塔爬升：

```bash
WLG_PLAY_FOCUS_TERRAIN=pyramid_climb \
WLG_PLAY_LIN_VEL_CMD=1.2 \
  bash scripts/descent_pitch_curriculum.sh play 62500
```

先按 `W` 做正向，再按 `S` 做反向。重点看峰值 pitch、落地后二次摆动、轮子是否持续接地；不要只以“没有摔倒”作为通过标准。

## 导出 ONNX

```bash
bash scripts/descent_pitch_curriculum.sh export
```

默认输出为 `export_onnx/longlegs_descent_pitch_model_62500.onnx`。可通过 `WLG_EXPORT_OUT` 修改文件名。
