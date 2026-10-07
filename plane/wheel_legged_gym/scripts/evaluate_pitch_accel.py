"""Compare deterministic flat-ground hold and locomotion checkpoints.

Set WLG_EVAL_CHECKPOINT and WLG_EVAL_OUT; optionally WLG_EVAL_RANDOMIZATION=1.
No actor observations, reward definitions, or controller gains are changed.
"""
import json
import os
from pathlib import Path

import isaacgym  # noqa: F401
import torch

from wheel_legged_gym.envs import *  # noqa: F401,F403
from wheel_legged_gym.utils import get_args, task_registry


def main():
    args = get_args()
    pairs = [(0, 0), (-1, 0), (1, 0), (-2.5, 0), (2.5, 0),
             (0, -5), (0, 5), (2.5, -5), (-2.5, 5)]
    cases = [(vx, yaw, h) for h in (0.16, 0.22, 0.28) for vx, yaw in pairs]
    args.num_envs = len(cases) * 4
    args.headless = True
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    cfg.terrain.mesh_type = "plane"
    cfg.terrain.curriculum = False
    cfg.commands.curriculum = False
    cfg.env.episode_length_s = 20
    randomized = os.getenv("WLG_EVAL_RANDOMIZATION", "0") == "1"
    if not randomized:
        cfg.noise.add_noise = False
        for name in vars(cfg.domain_rand):
            if name.startswith("randomize_") or name == "push_robots":
                setattr(cfg.domain_rand, name, False)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    commands = torch.tensor(cases * 4, device=env.device, dtype=torch.float)
    sums = torch.zeros(env.num_envs, 8, device=env.device)
    counts = torch.zeros(env.num_envs, device=env.device)
    failures = torch.zeros(env.num_envs, device=env.device)
    pitch_sum = torch.zeros(env.num_envs, device=env.device)
    pitch_sq_sum = torch.zeros(env.num_envs, device=env.device)
    enabled = False
    ids = torch.arange(env.num_envs, device=env.device)

    def fixed_commands(selected):
        age = env.episode_length_buf[selected].float() * env.dt
        ramp = torch.clamp((age - 3.0) / 0.5, 0, 1)
        env.commands[selected, :2] = commands[selected, :2] * ramp[:, None]
        env.commands[selected, 2] = commands[selected, 2]

    callback = env._post_physics_step_callback
    original_reset = env.reset_idx

    def record_callback():
        callback()
        fixed_commands(ids)
        if not enabled:
            return
        valid = (env.episode_length_buf.float() * env.dt >= 4.0).float()
        tilt = torch.acos(torch.clamp(-env.projected_gravity[:, 2], -1, 1))
        pitch = torch.atan2(-env.projected_gravity[:, 0], torch.sqrt(torch.clamp(env.projected_gravity[:, 1].square() + env.projected_gravity[:, 2].square(), min=1e-9)))
        leg_speed = env.dof_vel[:, [0, 1, 3, 4]].square().mean(dim=1)
        leg_asym = ((env.dof_pos[:, :2] + env.dof_pos[:, 3:5])
                    .square().mean(dim=1))
        values = torch.stack([
            (env.base_lin_vel[:, 0] - commands[:, 0]).abs(),
            (env.base_ang_vel[:, 2] - commands[:, 1]).abs(),
            env.base_lin_vel[:, :2].norm(dim=1),
            tilt.square(), leg_speed, leg_asym,
            env._reward_action_rate(), env._reward_action_smooth(),
        ], dim=1)
        sums.add_(values * valid[:, None])
        pitch_sum.add_(pitch * valid)
        pitch_sq_sum.add_(pitch.square() * valid)
        counts.add_(valid)

    def record_reset(selected):
        if enabled and len(selected):
            failures[selected] += env.fail_buf[selected].float()
        original_reset(selected)
        fixed_commands(selected)

    env._resample_commands = fixed_commands
    env._post_physics_step_callback = record_callback
    env.reset_idx = record_reset
    train_cfg.runner.resume = False
    args.resume = False
    runner, _ = task_registry.make_alg_runner(
        env, args=args, train_cfg=train_cfg, log_root=None)
    checkpoint = Path(os.environ["WLG_EVAL_CHECKPOINT"])
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    env.reset_idx(ids)
    env.compute_observations()
    obs, history = env.get_observations()
    enabled = True
    with torch.inference_mode():
        for _ in range(round(12.0 / env.dt)):
            actions, _ = policy(obs, history)
            obs, _, _, _, _, history = env.step(actions)
    rows = []
    for index, (vx, yaw, h) in enumerate(cases):
        group = torch.arange(index, env.num_envs, len(cases), device=env.device)
        samples = counts[group].sum()
        mean = (sums[group].sum(dim=0) / samples.clamp(min=1)).cpu().tolist()
        rows.append(dict(vx=vx, yaw=yaw, height=h,
                         failures=int(failures[group].sum().item()),
                         samples=int(samples.item()), vx_mae=mean[0],
                         yaw_mae=mean[1], planar_speed=mean[2],
                         tilt_rms_deg=mean[3] ** 0.5 * 180 / 3.141592653589793,
                         leg_speed_rms=mean[4] ** 0.5,
                         leg_asym_rms=mean[5] ** 0.5,
                         action_delta_sq=mean[6], action_second_delta_sq=mean[7],
                         pitch_mean_deg=(pitch_sum[group].sum() / samples.clamp(min=1)).item() * 180 / 3.141592653589793,
                         pitch_rms_deg=(pitch_sq_sum[group].sum() / samples.clamp(min=1)).sqrt().item() * 180 / 3.141592653589793))
    result = dict(checkpoint=str(checkpoint), randomized=randomized,
                  seconds=12, warmup_seconds=4, cases=rows)
    out = Path(os.environ["WLG_EVAL_OUT"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("EVAL_OUT", out)
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()


