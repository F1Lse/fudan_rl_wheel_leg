# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from wheel_legged_gym.rsl_rl.algorithms import PPO
from wheel_legged_gym.rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    ActorCriticSequence,
)
from wheel_legged_gym.rsl_rl.env import VecEnv


# Console-only presentation. TensorBoard scalar names remain unchanged so old
# dashboards and post-processing scripts continue to work.
_EPISODE_METRIC_GROUPS = (
    (
        "速度跟踪",
        (
            "rew_tracking_lin_vel",
            "rew_tracking_lin_vel_enhance",
            "rew_tracking_lin_vel_l1",
        ),
    ),
    (
        "角速度跟踪",
        (
            "rew_tracking_ang_vel",
            "rew_tracking_ang_vel_enhance",
            "rew_tracking_ang_vel_l1",
        ),
    ),
    (
        "小陀螺实测",
        (
            "mean_spin_cmd_abs_yaw",
            "mean_spin_real_abs_yaw",
            "mean_spin_yaw_abs_error",
            "spin_direction_accuracy",
        ),
    ),
    (
        "高度与姿态",
        (
            "rew_base_height",
            "rew_base_height_enhance",
            "rew_base_height_l1",
            "rew_orientation",
            "rew_ang_vel_xy",
            "rew_lin_vel_z",
            "max_abs_pitch_deg",
            "max_tilt_deg",
        ),
    ),
    (
        "碰坎收腿与地形",
        (
            "rew_terrain_impact_tuck",
            "rew_terrain_impact_tuck_velocity",
            "terrain_tuck_active_fraction",
            "mean_terrain_tuck_error",
            "mean_terrain_tuck_triggers",
            "rew_terrain_pitch_excess",
            "rew_terrain_pitch_rate",
            "rew_wheel_support",
            "rew_collision",
            "terrain_level",
        ),
    ),
    (
        "原地旋转稳定性",
        (
            "rew_spin_stationary_lin_vel",
            "rew_spin_stationary_ang_vel_xy",
            "rew_spin_stationary_orientation",
            "rew_spin_stationary_action_rate",
            "rew_spin_stationary_action_smooth",
            "mean_stationary_tilt_deg",
            "rms_stationary_tilt_deg",
        ),
    ),
    (
        "动作平滑与能耗",
        (
            "rew_action_rate",
            "rew_action_smooth",
            "rew_dof_acc",
            "rew_dof_vel",
            "rew_dof_pos_limits",
            "rew_torques",
            "rew_nominal_state",
            "rew_recovery_pose",
        ),
    ),
)

_EPISODE_METRIC_LABELS = {
    "rew_tracking_lin_vel": "线速度跟踪奖励",
    "rew_tracking_lin_vel_enhance": "线速度增强项",
    "rew_tracking_lin_vel_l1": "线速度绝对误差惩罚",
    "rew_tracking_ang_vel": "旋转速度跟踪奖励",
    "rew_tracking_ang_vel_enhance": "旋转速度增强项",
    "rew_tracking_ang_vel_l1": "旋转速度绝对误差惩罚",
    "mean_spin_cmd_abs_yaw": "平均目标旋转速度",
    "mean_spin_real_abs_yaw": "平均实际旋转速度",
    "mean_spin_yaw_abs_error": "平均旋转速度误差",
    "spin_direction_accuracy": "旋转方向正确率",
    "rew_base_height": "机身高度奖励",
    "rew_base_height_enhance": "机身高度增强项",
    "rew_base_height_l1": "机身高度绝对误差惩罚",
    "rew_orientation": "机身姿态惩罚",
    "rew_ang_vel_xy": "横滚/俯仰角速度惩罚",
    "rew_lin_vel_z": "竖直速度惩罚",
    "max_abs_pitch_deg": "最大俯仰角（度）",
    "max_tilt_deg": "最大倾斜角（度）",
    "mean_stationary_tilt_deg": "静止平均倾斜角（度）",
    "rms_stationary_tilt_deg": "静止RMS倾斜角（度）",
    "rew_terrain_impact_tuck": "碰坎收腿奖励",
    "rew_terrain_impact_tuck_velocity": "收腿方向速度奖励",
    "terrain_tuck_active_fraction": "收腿触发时间占比",
    "mean_terrain_tuck_error": "触发期间收腿姿态误差",
    "mean_terrain_tuck_triggers": "每回合平均收腿触发次数",
    "rew_terrain_pitch_excess": "地形俯仰超限惩罚",
    "rew_terrain_pitch_rate": "地形俯仰速度惩罚",
    "rew_wheel_support": "双轮支撑奖励",
    "rew_collision": "非期望碰撞惩罚",
    "terrain_level": "平均地形等级",
    "rew_spin_stationary_lin_vel": "旋转时平移漂移惩罚",
    "rew_spin_stationary_ang_vel_xy": "旋转时横滚/俯仰速度惩罚",
    "rew_spin_stationary_orientation": "旋转时姿态惩罚",
    "rew_spin_stationary_action_rate": "旋转时动作变化惩罚",
    "rew_spin_stationary_action_smooth": "旋转时动作平滑惩罚",
    "rew_action_rate": "动作变化惩罚",
    "rew_action_smooth": "动作平滑惩罚",
    "rew_dof_acc": "关节加速度惩罚",
    "rew_dof_vel": "关节速度惩罚",
    "rew_dof_pos_limits": "关节限位惩罚",
    "rew_torques": "力矩惩罚",
    "rew_nominal_state": "偏离标称姿态惩罚",
    "rew_recovery_pose": "起身姿态引导",
}


def _format_grouped_episode_metrics(values):
    """Render episode metrics in compact Chinese groups for the terminal."""
    lines = []
    used = set()
    for group_name, group_keys in _EPISODE_METRIC_GROUPS:
        present = [key for key in group_keys if key in values]
        if not present:
            continue
        # Do not print irrelevant groups such as SPIN metrics during descent.
        if all(abs(values[key]) < 1.0e-12 for key in present):
            used.update(present)
            continue
        lines.append(f"\n[{group_name}]")
        for key in present:
            label = _EPISODE_METRIC_LABELS.get(key, key)
            lines.append(f"  {label} | {key} = {values[key]:.4f}")
            used.add(key)

    remaining = [key for key in values if key not in used]
    if remaining:
        lines.append("\n[其他指标]")
        for key in remaining:
            label = _EPISODE_METRIC_LABELS.get(key, key)
            lines.append(f"  {label} | {key} = {values[key]:.4f}")
    return "\n".join(lines) + ("\n" if lines else "")


class OnPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        if self.cfg["policy_class_name"] == "ActorCriticSequence":
            num_critic_obs += self.policy_cfg["latent_dim"]
        actor_critic: ActorCritic = actor_critic_class(
            self.env.num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [num_critic_obs],
            [self.env.obs_history_length * self.env.num_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, obs_history = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, obs_history, critic_obs = (
            obs.to(self.device),
            obs_history.to(self.device),
            critic_obs.to(self.device),
        )
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):    ###############这里
                    actions = self.alg.act(obs, obs_history, critic_obs)
                    obs, privileged_obs, rewards, dones, infos, obs_history = (
                        self.env.step(actions)
                    )
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, obs_history, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        obs_history.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos, obs)

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                if self.cfg["policy_class_name"] == "ActorCriticSequence":
                    critic_obs__ = torch.cat(
                        (critic_obs, self.alg.actor_critic.encode(obs_history)), dim=-1
                    )
                else:
                    critic_obs__ = critic_obs
                self.alg.compute_returns(critic_obs__)

            mean_value_loss, mean_surrogate_loss, mean_kl, mean_extra_loss = (
                self.alg.update()
            )
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            self.current_learning_iteration = it + 1
            if self.current_learning_iteration % self.save_interval == 0:
                self.save(
                    os.path.join(
                        self.log_dir,
                        "model_{}.pt".format(self.current_learning_iteration),
                    )
                )
            ep_infos.clear()
        self.current_learning_iteration = tot_iter
        self.save(os.path.join(self.log_dir, "model_{}.pt".format(tot_iter)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        episode_values = {}
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                episode_values[key] = value.item()
        if os.getenv("WLG_GROUPED_CONSOLE_LOG", "1").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        ):
            ep_string = "".join(
                f"{f'Mean {key}:':>{pad}} {value:.4f}\n"
                for key, value in episode_values.items()
            )
        else:
            ep_string = _format_grouped_episode_metrics(episode_values)
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/encoder", locs["mean_extra_loss"], locs["it"])
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Policy/mean_kl", locs["mean_kl"], locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )

        str = f" \033[1m 训练轮次 {locs['it'] + 1}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'计算速度:':>{pad}} {fps:.0f} steps/s (采样: {locs[
                            'collection_time']:.3f}s, 学习: {locs['learn_time']:.3f}s)\n"""
                f"""{'价值函数损失:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'策略代理损失:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'平均动作噪声:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'平均总奖励:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'平均回合长度:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'计算速度:':>{pad}} {fps:.0f} steps/s (采样: {locs[
                            'collection_time']:.3f}s, 学习: {locs['learn_time']:.3f}s)\n"""
                f"""{'价值函数损失:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'策略代理损失:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'平均动作噪声:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'累计仿真步数:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'本轮耗时:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'累计耗时:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'预计剩余:':>{pad}} {self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['tot_iter'] - locs['it'] - 1):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
