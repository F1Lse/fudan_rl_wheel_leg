"""Headless, repeatable spin evaluation with per-episode and failure data.

Run from the plane directory. Evaluation settings use WLG_EVAL_* environment
variables so the existing Isaac Gym argument parser remains unchanged.
"""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from datetime import datetime, timezone

import isaacgym  # noqa: F401 - Isaac Gym must load before torch
import torch

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.envs import *  # noqa: F401,F403 - registers wheel_legged
from wheel_legged_gym.utils import get_args, task_registry


TRACE_COLUMNS = (
    "time_s", "cmd_yaw_radps", "actual_yaw_radps", "height_m",
    "roll_deg", "pitch_deg", "tilt_deg", "roll_rate_radps",
    "pitch_rate_radps", "body_axis_world_x", "body_axis_world_y",
    "drift_m", "torque_limit_fraction", "x_m", "y_m",
    "push_impulse_ns",
) + tuple(f"action_{i}" for i in range(6)) + tuple(
    f"torque_{i}_nm" for i in range(6)
) + tuple(f"joint_pos_{i}_rad" for i in range(6)) + tuple(
    f"joint_vel_{i}_radps" for i in range(6)
)

DOMAIN_RAND_FIELDS = (
    "randomize_friction", "friction_range", "randomize_restitution",
    "restitution_range", "randomize_base_mass", "added_mass_range",
    "randomize_inertia", "randomize_inertia_range", "randomize_base_com",
    "rand_com_vec", "push_robots", "push_interval_s", "max_push_vel_xy",
    "randomize_Kp", "randomize_Kp_range", "randomize_Kd",
    "randomize_Kd_range", "randomize_motor_torque",
    "randomize_motor_torque_range", "randomize_default_dof_pos",
    "randomize_default_dof_pos_range", "randomize_action_delay",
    "delay_ms_range",
)

CODE_FILES = (
    "wheel_legged_gym/scripts/evaluate_spin.py",
    "wheel_legged_gym/envs/base/legged_robot.py",
    "wheel_legged_gym/envs/base/legged_robot_config.py",
    "wheel_legged_gym/rsl_rl/modules/actor_critic_sequence.py",
    "wheel_legged_gym/rsl_rl/algorithms/ppo.py",
)


def env_float(name, default):
    value = float(os.getenv(name, default))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def env_int(name, default):
    value = int(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def env_bool(name, default):
    value = os.getenv(name, "1" if default else "0").strip().lower()
    if value not in ("0", "1", "false", "true", "no", "yes", "off", "on"):
        raise ValueError(f"{name} must be a boolean")
    return value in ("1", "true", "yes", "on")


def parse_yaws(default):
    values = [
        float(x.strip())
        for x in os.getenv("WLG_EVAL_YAWS", ",".join(map(str, default))).split(",")
    ]
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("WLG_EVAL_YAWS must contain finite yaw rates")
    if len(set(values)) != len(values):
        raise ValueError("WLG_EVAL_YAWS must not contain duplicates")
    return values


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_target():
    raw_path = Path(os.getenv("WLG_EVAL_TARGET", "eval_targets/spin_stability_v1.json"))
    path = raw_path if raw_path.is_absolute() else Path(WHEEL_LEGGED_GYM_ROOT_DIR) / raw_path
    with path.open(encoding="utf-8") as source:
        target = json.load(source)
    if not isinstance(target, dict) or target.get("task") != "spin_stability":
        raise ValueError(f"Not a spin_stability target: {path}")
    return path.resolve(), target


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=WHEEL_LEGGED_GYM_ROOT_DIR,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty():
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=WHEEL_LEGGED_GYM_ROOT_DIR,
            text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        return None


class SpinRecorder:
    """Capture pre-reset episode stats; env.step resets finished robots internally."""

    def __init__(
        self, env, yaws, height, trace_seconds, max_traces, thresholds,
        warmup_seconds=0.0, yaw_ramp_seconds=0.0, trace_all=False,
    ):
        if env.num_actions != 6 or env.dof_pos.shape[1] != 6:
            raise ValueError("Spin trace schema expects the six-action wheel-legged robot")
        self.env = env
        self.yaws = yaws
        self.height = height
        self.max_traces = max_traces
        self.thresholds = thresholds
        self.warmup_steps = math.ceil(warmup_seconds / env.dt)
        self.yaw_ramp_seconds = yaw_ramp_seconds
        self.trace_all = trace_all
        self.enabled = False
        self.rows = []
        self.traces = []
        self.yaw_for_env = torch.tensor(
            [yaws[i % len(yaws)] for i in range(env.num_envs)],
            device=env.device, dtype=torch.float,
        )
        self.height_error_sum = torch.zeros(env.num_envs, device=env.device)
        self.torque_fraction_max = torch.zeros(env.num_envs, device=env.device)
        self.torque_saturation_steps = torch.zeros(env.num_envs, device=env.device)
        self.roll_pitch_rate_sq_sum = torch.zeros(env.num_envs, device=env.device)
        self.roll_pitch_rate_max = torch.zeros(env.num_envs, device=env.device)
        self.active_tilt_sq_sum = torch.zeros(env.num_envs, device=env.device)
        self.active_tilt_max = torch.zeros(env.num_envs, device=env.device)
        self.warmup_tilt_max = torch.zeros(env.num_envs, device=env.device)
        self.steady_steps = torch.zeros(env.num_envs, device=env.device)
        self.steady_roll_sum = torch.zeros(env.num_envs, device=env.device)
        self.steady_pitch_sum = torch.zeros(env.num_envs, device=env.device)
        self.steady_roll_sq_sum = torch.zeros(env.num_envs, device=env.device)
        self.steady_pitch_sq_sum = torch.zeros(env.num_envs, device=env.device)
        self.steady_axis_x_sum = torch.zeros(env.num_envs, device=env.device)
        self.steady_axis_y_sum = torch.zeros(env.num_envs, device=env.device)
        self.steady_axis_x_sq_sum = torch.zeros(env.num_envs, device=env.device)
        self.steady_axis_y_sq_sum = torch.zeros(env.num_envs, device=env.device)
        self.steady_tilt_sum = torch.zeros(env.num_envs, device=env.device)
        self.steady_tilt_sq_sum = torch.zeros(env.num_envs, device=env.device)
        self.active_steps = torch.zeros(env.num_envs, device=env.device)
        self.idle_yaw_error_sum = torch.zeros(env.num_envs, device=env.device)
        self.idle_drift_max = torch.zeros(env.num_envs, device=env.device)
        self.spinup_step = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
        self.spin_started = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.trace_length = max(1, round(trace_seconds / env.dt))
        self.trace = torch.zeros(
            env.num_envs, self.trace_length, len(TRACE_COLUMNS), device=env.device
        )
        self.trace_pos = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.trace_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.push_impulse = torch.zeros(env.num_envs, device=env.device)
        self.all_ids = torch.arange(env.num_envs, device=env.device)

    def fixed_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self.env.commands[env_ids, 0] = 0.0
        self.env.commands[env_ids, 2] = self.height
        active = self.env.episode_length_buf[env_ids] >= self.warmup_steps
        if self.yaw_ramp_seconds > 0:
            seconds_after_warmup = (
                (self.env.episode_length_buf[env_ids] - self.warmup_steps)
                .clamp(min=0).float() * self.env.dt
            )
            ramp_fraction = torch.clamp(
                seconds_after_warmup / self.yaw_ramp_seconds, 0.0, 1.0
            )
            yaw_command = self.yaw_for_env[env_ids] * ramp_fraction
        else:
            yaw_command = self.yaw_for_env[env_ids]
        self.env.commands[env_ids, 1] = torch.where(active, yaw_command, 0.0)
        starting = env_ids[active & ~self.spin_started[env_ids]]
        if len(starting) and self.warmup_steps > 0:
            self.env.spin_position_anchor[starting] = self.env.base_position[starting, :2]
        self.spin_started[starting] = True

    def on_step(self):
        env = self.env
        active = (env.episode_length_buf >= self.warmup_steps).float()
        self.active_steps += active
        self.idle_yaw_error_sum += torch.abs(env.base_ang_vel[:, 2]) * active
        spinning = active.bool() & (torch.abs(env.commands[:, 1]) > 0.25)
        reached = torch.abs(env.base_ang_vel[:, 2] - self.yaw_for_env) <= (
            0.10 * torch.abs(self.yaw_for_env)
        )
        first_reach = spinning & reached & (self.spinup_step < 0)
        self.spinup_step[first_reach] = (
            env.episode_length_buf[first_reach] - self.warmup_steps
        )
        self.height_error_sum += torch.abs(env.base_height - self.height) * active
        torque_limit = torch.clamp(env.torque_limits, min=1e-6)
        torque_fraction = torch.max(torch.abs(env.torques) / torque_limit, dim=1).values
        self.torque_fraction_max = torch.maximum(self.torque_fraction_max, torque_fraction)
        self.torque_saturation_steps += (torque_fraction >= 0.95).float() * active
        gravity = env.projected_gravity
        roll = torch.atan2(gravity[:, 1], -gravity[:, 2]) * (180.0 / math.pi)
        pitch = torch.atan2(
            -gravity[:, 0], torch.hypot(gravity[:, 1], gravity[:, 2])
        ) * (180.0 / math.pi)
        tilt = torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0)) * (180.0 / math.pi)
        roll_pitch_rate = torch.norm(env.base_ang_vel[:, :2], dim=1)
        self.active_tilt_sq_sum += tilt.square() * active
        self.active_tilt_max = torch.maximum(self.active_tilt_max, tilt * active)
        self.warmup_tilt_max = torch.maximum(
            self.warmup_tilt_max, tilt * (1.0 - active)
        )
        # Skip the first second after spin starts. Body-frame roll/pitch varies
        # with yaw even for a world-fixed lean, so also track the robot's body
        # z-axis in the world frame as a heading-independent sway measure.
        steady = (
            env.episode_length_buf >= self.warmup_steps + math.ceil(1.0 / env.dt)
        ).float()
        self.steady_steps += steady
        self.steady_roll_sum += roll * steady
        self.steady_pitch_sum += pitch * steady
        self.steady_roll_sq_sum += roll.square() * steady
        self.steady_pitch_sq_sum += pitch.square() * steady
        quat = env.base_quat
        axis_x = 2.0 * (quat[:, 0] * quat[:, 2] + quat[:, 3] * quat[:, 1])
        axis_y = 2.0 * (quat[:, 1] * quat[:, 2] - quat[:, 3] * quat[:, 0])
        self.steady_axis_x_sum += axis_x * steady
        self.steady_axis_y_sum += axis_y * steady
        self.steady_axis_x_sq_sum += axis_x.square() * steady
        self.steady_axis_y_sq_sum += axis_y.square() * steady
        self.steady_tilt_sum += tilt * steady
        self.steady_tilt_sq_sum += tilt.square() * steady
        self.roll_pitch_rate_sq_sum += roll_pitch_rate.square() * active
        self.roll_pitch_rate_max = torch.maximum(
            self.roll_pitch_rate_max, roll_pitch_rate * active
        )
        drift = torch.norm(env.base_position[:, :2] - env.spin_position_anchor, dim=1)
        self.idle_drift_max = torch.maximum(self.idle_drift_max, drift * active)
        values = torch.stack(
            (
                env.episode_length_buf.float() * env.dt,
                env.commands[:, 1], env.base_ang_vel[:, 2], env.base_height,
                roll, pitch, tilt, env.base_ang_vel[:, 0], env.base_ang_vel[:, 1],
                axis_x, axis_y,
                drift, torque_fraction,
                env.base_position[:, 0], env.base_position[:, 1],
                self.push_impulse,
            ), dim=1,
        )
        values = torch.cat((values, env.actions, env.torques, env.dof_pos, env.dof_vel), dim=1)
        self.trace[self.all_ids, self.trace_pos] = values
        self.trace_pos = (self.trace_pos + 1) % self.trace_length
        self.trace_count = torch.clamp(self.trace_count + 1, max=self.trace_length)
        self.push_impulse.zero_()

    def clear(self, env_ids):
        self.height_error_sum[env_ids] = 0.0
        self.torque_fraction_max[env_ids] = 0.0
        self.torque_saturation_steps[env_ids] = 0.0
        self.roll_pitch_rate_sq_sum[env_ids] = 0.0
        self.roll_pitch_rate_max[env_ids] = 0.0
        self.active_tilt_sq_sum[env_ids] = 0.0
        self.active_tilt_max[env_ids] = 0.0
        self.warmup_tilt_max[env_ids] = 0.0
        self.steady_steps[env_ids] = 0.0
        self.steady_roll_sum[env_ids] = 0.0
        self.steady_pitch_sum[env_ids] = 0.0
        self.steady_roll_sq_sum[env_ids] = 0.0
        self.steady_pitch_sq_sum[env_ids] = 0.0
        self.steady_axis_x_sum[env_ids] = 0.0
        self.steady_axis_y_sum[env_ids] = 0.0
        self.steady_axis_x_sq_sum[env_ids] = 0.0
        self.steady_axis_y_sq_sum[env_ids] = 0.0
        self.steady_tilt_sum[env_ids] = 0.0
        self.steady_tilt_sq_sum[env_ids] = 0.0
        self.active_steps[env_ids] = 0.0
        self.idle_yaw_error_sum[env_ids] = 0.0
        self.idle_drift_max[env_ids] = 0.0
        self.spinup_step[env_ids] = -1
        self.spin_started[env_ids] = False
        self.trace_pos[env_ids] = 0
        self.trace_count[env_ids] = 0
        self.push_impulse[env_ids] = 0.0

    def on_reset(self, env_ids):
        if not self.enabled or len(env_ids) == 0:
            return
        env = self.env
        for env_id in env_ids.tolist():
            steps = int(env.episode_length_buf[env_id].item())
            if steps < 2:
                continue
            spin_steps = max(float(env.episode_spin_steps[env_id].item()), 1.0)
            active_steps = max(float(self.active_steps[env_id].item()), 1.0)
            steady_steps = max(float(self.steady_steps[env_id].item()), 1.0)
            mean_roll_deg = float(self.steady_roll_sum[env_id].item()) / steady_steps
            mean_pitch_deg = float(self.steady_pitch_sum[env_id].item()) / steady_steps
            wobble_variance = max(
                0.0,
                float(self.steady_roll_sq_sum[env_id].item()) / steady_steps
                - mean_roll_deg * mean_roll_deg,
            ) + max(
                0.0,
                float(self.steady_pitch_sq_sum[env_id].item()) / steady_steps
                - mean_pitch_deg * mean_pitch_deg,
            )
            mean_axis_x = float(self.steady_axis_x_sum[env_id].item()) / steady_steps
            mean_axis_y = float(self.steady_axis_y_sum[env_id].item()) / steady_steps
            axis_sway_variance = max(
                0.0,
                float(self.steady_axis_x_sq_sum[env_id].item()) / steady_steps
                - mean_axis_x * mean_axis_x,
            ) + max(
                0.0,
                float(self.steady_axis_y_sq_sum[env_id].item()) / steady_steps
                - mean_axis_y * mean_axis_y,
            )
            mean_tilt = float(self.steady_tilt_sum[env_id].item()) / steady_steps
            tilt_variance = max(
                0.0,
                float(self.steady_tilt_sq_sum[env_id].item()) / steady_steps
                - mean_tilt * mean_tilt,
            )
            timed_out = bool(env.time_out_buf[env_id].item())
            is_idle = self.yaws[env_id % len(self.yaws)] == 0.0
            row = {
                "episode_id": len(self.rows) + 1,
                "env_id": env_id,
                "yaw_cmd_radps": self.yaws[env_id % len(self.yaws)],
                "height_cmd_m": self.height,
                "duration_s": steps * env.dt,
                "timed_out": timed_out,
                "reset_reason": (
                    "timeout" if timed_out else
                    "edge" if bool(env.edge_reset_buf[env_id].item()) else
                    "fall" if int(env.fail_buf[env_id].item()) > 0 else "other"
                ),
                "yaw_mae_radps": (
                    float(self.idle_yaw_error_sum[env_id].item()) / active_steps
                    if is_idle else
                    float(env.episode_spin_yaw_abs_error_sum[env_id].item()) / spin_steps
                ),
                "real_abs_yaw_radps": (
                    float(self.idle_yaw_error_sum[env_id].item()) / active_steps
                    if is_idle else
                    float(env.episode_spin_real_abs_yaw_sum[env_id].item()) / spin_steps
                ),
                "direction_accuracy": (
                    None if is_idle else
                    float(env.episode_spin_direction_ok_sum[env_id].item()) / spin_steps
                ),
                "max_drift_m": (
                    float(self.idle_drift_max[env_id].item())
                    if is_idle else
                    float(env.episode_spin_position_drift_max[env_id].item())
                ),
                "time_to_90pct_yaw_s": (
                    float(self.spinup_step[env_id].item()) * env.dt
                    if self.spinup_step[env_id].item() >= 0 else None
                ),
                "rms_tilt_deg": math.sqrt(float(self.active_tilt_sq_sum[env_id].item()) / active_steps),
                "max_tilt_deg": float(self.active_tilt_max[env_id].item()),
                "warmup_max_tilt_deg": float(self.warmup_tilt_max[env_id].item()),
                "steady_mean_roll_deg": mean_roll_deg,
                "steady_mean_pitch_deg": mean_pitch_deg,
                "steady_wobble_rms_deg": math.sqrt(wobble_variance),
                "steady_world_axis_sway_rms_deg": math.degrees(math.sqrt(axis_sway_variance)),
                "steady_tilt_std_deg": math.sqrt(tilt_variance),
                "rms_roll_pitch_rate_radps": math.sqrt(float(self.roll_pitch_rate_sq_sum[env_id].item()) / active_steps),
                "max_roll_pitch_rate_radps": float(self.roll_pitch_rate_max[env_id].item()),
                "mean_height_error_m": float(self.height_error_sum[env_id].item()) / active_steps,
                "max_torque_limit_fraction": float(self.torque_fraction_max[env_id].item()),
                "torque_saturation_fraction": float(self.torque_saturation_steps[env_id].item()) / active_steps,
            }
            row["stable"] = (
                timed_out
                and row["yaw_mae_radps"] <= self.thresholds["yaw_mae_radps"]
                and row["max_drift_m"] <= self.thresholds["max_drift_m"]
                and row["max_tilt_deg"] <= self.thresholds["max_tilt_deg"]
                and row["rms_roll_pitch_rate_radps"] <= self.thresholds["rms_roll_pitch_rate_radps"]
            )
            tags = []
            if not timed_out:
                tags.append(row["reset_reason"])
            for name in (
                "yaw_mae_radps", "max_drift_m", "max_tilt_deg",
                "rms_roll_pitch_rate_radps",
            ):
                if row[name] > self.thresholds[name]:
                    tags.append(name)
            row["failure_tags"] = "|".join(tags)
            self.rows.append(row)
            if (self.trace_all or not row["stable"]) and len(self.traces) < self.max_traces:
                count = int(self.trace_count[env_id].item())
                end = int(self.trace_pos[env_id].item())
                start = (end - count) % self.trace_length
                order = [(start + j) % self.trace_length for j in range(count)]
                self.traces.append((len(self.rows), self.trace[env_id, order].cpu().tolist()))


def configure_randomization(cfg, enabled):
    if enabled:
        return
    cfg.noise.add_noise = False
    for name in (
        "randomize_friction", "randomize_restitution", "randomize_base_mass",
        "randomize_inertia", "randomize_base_com", "randomize_Kp", "randomize_Kd",
        "randomize_motor_torque", "randomize_default_dof_pos",
        "randomize_action_delay", "push_robots",
    ):
        setattr(cfg.domain_rand, name, False)


def ablate_domain_group(cfg, group):
    groups = {
        "none": (),
        "contact": ("randomize_friction", "randomize_restitution"),
        "friction": ("randomize_friction",),
        "restitution": ("randomize_restitution",),
        "body": ("randomize_base_mass", "randomize_inertia", "randomize_base_com"),
        "actuation": (
            "randomize_Kp", "randomize_Kd", "randomize_motor_torque",
            "randomize_default_dof_pos", "randomize_action_delay",
        ),
        "noise": (),
    }
    if group not in groups:
        raise ValueError(f"WLG_EVAL_ABLATE_DOMAIN must be one of {sorted(groups)}")
    if group == "noise":
        cfg.noise.add_noise = False
    for name in groups[group]:
        setattr(cfg.domain_rand, name, False)


def write_outputs(out_dir, rows, traces, manifest):
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    with (out_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failure_dir = out_dir / ("traces" if manifest["trace_all"] else "failures")
    failure_dir.mkdir(exist_ok=True)
    for episode_id, trace in traces:
        with (failure_dir / f"episode_{episode_id:04d}.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(TRACE_COLUMNS)
            writer.writerows(trace)

    lines = ["# Spin evaluation", "", f"Checkpoint: `{manifest['checkpoint']}`", ""]
    lines.append("| yaw command | episodes | complete | reached 90% yaw | stable* | yaw MAE | p90 peak drift | p90 peak tilt | p90 world-axis sway | p90 body-frame roll/pitch variation | p90 RMS roll/pitch rate |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for yaw in manifest["yaw_commands_radps"]:
        group = [row for row in rows if row["yaw_cmd_radps"] == yaw]
        mean_mae = statistics.mean(row["yaw_mae_radps"] for row in group)
        drift = sorted(row["max_drift_m"] for row in group)
        tilt = sorted(row["max_tilt_deg"] for row in group)
        wobble = sorted(row["steady_wobble_rms_deg"] for row in group)
        axis_sway = sorted(row["steady_world_axis_sway_rms_deg"] for row in group)
        rp_rate = sorted(row["rms_roll_pitch_rate_radps"] for row in group)
        index = math.ceil(0.9 * len(group)) - 1
        p90_drift = drift[index]
        p90_tilt = tilt[index]
        p90_wobble = wobble[index]
        p90_axis_sway = axis_sway[index]
        p90_rp_rate = rp_rate[index]
        complete = sum(row["timed_out"] for row in group) / len(group)
        stable = sum(row["stable"] for row in group) / len(group)
        reached = sum(row["time_to_90pct_yaw_s"] is not None for row in group) / len(group)
        reached_text = "n/a" if yaw == 0 else f"{reached:.1%}"
        lines.append(
            f"| {yaw:+g} rad/s | {len(group)} | {complete:.1%} | {reached_text} | {stable:.1%} | "
            f"{mean_mae:.2f} rad/s | {p90_drift:.3f} m | {p90_tilt:.1f}° | "
            f"{p90_axis_sway:.2f}° | {p90_wobble:.2f}° | "
            f"{p90_rp_rate:.2f} rad/s |"
        )
    lines.extend((
        "", "*Stable uses the provisional thresholds in `manifest.json`; set them for the real robot before using this as an acceptance gate.",
        "", f"See `episodes.csv` and `{failure_dir.name}/` for trajectories.", "",
    ))
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    target_path, target = read_target()
    yaws = parse_yaws(target["yaw_commands_radps"])
    episodes_per_yaw = env_int("WLG_EVAL_EPISODES_PER_YAW", target["episodes_per_yaw"])
    episode_seconds = env_float("WLG_EVAL_EPISODE_S", target["episode_seconds"])
    warmup_seconds = env_float("WLG_EVAL_WARMUP_S", target.get("warmup_seconds", 0.0))
    yaw_ramp_seconds = env_float("WLG_EVAL_YAW_RAMP_S", 0.0)
    height = env_float("WLG_EVAL_HEIGHT", target["height_m"])
    trace_seconds = env_float("WLG_EVAL_TRACE_S", target["trace_seconds"])
    if episode_seconds <= 0 or trace_seconds <= 0 or height <= 0:
        raise ValueError("Episode duration, trace duration and height must be positive")
    if warmup_seconds < 0 or warmup_seconds >= episode_seconds:
        raise ValueError("Warmup duration must be nonnegative and shorter than the episode")
    if yaw_ramp_seconds < 0 or yaw_ramp_seconds >= episode_seconds - warmup_seconds:
        raise ValueError("Yaw ramp must fit inside the active episode")
    max_traces = env_int("WLG_EVAL_MAX_TRACES", target["max_traces"])
    trace_all = env_bool("WLG_EVAL_TRACE_ALL", False)
    thresholds = {
        "yaw_mae_radps": env_float("WLG_EVAL_YAW_MAE_MAX", target["stable_thresholds"]["yaw_mae_radps"]),
        "max_drift_m": env_float("WLG_EVAL_DRIFT_MAX", target["stable_thresholds"]["max_drift_m"]),
        "max_tilt_deg": env_float("WLG_EVAL_TILT_MAX_DEG", target["stable_thresholds"]["max_tilt_deg"]),
        "rms_roll_pitch_rate_radps": env_float(
            "WLG_EVAL_RP_RATE_RMS_MAX", target["stable_thresholds"]["rms_roll_pitch_rate_radps"]
        ),
    }
    if any(value < 0 for value in thresholds.values()):
        raise ValueError("Stability thresholds must be nonnegative")
    randomization = env_bool("WLG_EVAL_RANDOMIZATION", True)
    domain_ablation = os.getenv("WLG_EVAL_ABLATE_DOMAIN", "none").strip().lower()
    fixed_restitution_raw = os.getenv("WLG_EVAL_RESTITUTION_FIXED")
    fixed_restitution = (
        env_float("WLG_EVAL_RESTITUTION_FIXED", 0.0)
        if fixed_restitution_raw is not None else None
    )
    if fixed_restitution is not None and not 0.0 <= fixed_restitution <= 1.0:
        raise ValueError("WLG_EVAL_RESTITUTION_FIXED must be in [0, 1]")
    args = get_args()
    if args.load_run is None or args.checkpoint is None:
        raise ValueError("Specify --load_run and --checkpoint explicitly")
    args.resume = False
    if args.num_envs is None:
        args.num_envs = len(yaws) * 4
    if args.num_envs < len(yaws):
        raise ValueError("--num_envs must be at least the number of yaw commands")
    out_raw = os.getenv("WLG_EVAL_OUT")
    out_dir = Path(out_raw) if out_raw else (
        Path(WHEEL_LEGGED_GYM_ROOT_DIR) / "evaluations" / "spin"
        / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_model_{args.checkpoint}"
    )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {out_dir}")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.episode_length_s = episode_seconds
    env_cfg.env.fail_to_terminal_time_s = 0.30
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.commands.curriculum = False
    env_cfg.init_state.pos[2] = env_float("WLG_EVAL_SPAWN_Z", 0.12)
    configure_randomization(env_cfg, randomization)
    ablate_domain_group(env_cfg, domain_ablation)
    if fixed_restitution is not None:
        if not randomization or domain_ablation in ("contact", "restitution"):
            raise ValueError("Fixed restitution requires contact randomization to be enabled")
        env_cfg.domain_rand.randomize_restitution = True
        env_cfg.domain_rand.restitution_range = [fixed_restitution] * 2
    env_cfg.domain_rand.push_robots = env_bool(
        "WLG_EVAL_PUSHES", env_cfg.domain_rand.push_robots
    )
    train_cfg.runner.resume = False
    if args.experiment_name:
        train_cfg.runner.experiment_name = args.experiment_name
    checkpoint = (
        Path(WHEEL_LEGGED_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name
        / args.load_run / f"model_{args.checkpoint}.pt"
    ).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    recorder = SpinRecorder(
        env, yaws, height, trace_seconds, max_traces, thresholds,
        warmup_seconds=warmup_seconds, yaw_ramp_seconds=yaw_ramp_seconds,
        trace_all=trace_all,
    )
    original_callback = env._post_physics_step_callback
    original_reset = env.reset_idx
    original_push = env._push_robots

    def record_push():
        original_push()
        interval_steps = int(env.cfg.domain_rand.push_interval_s / env.sim_params.dt)
        pushed = (env.envs_steps_buf % interval_steps == 0).nonzero(as_tuple=False).flatten()
        if recorder.enabled and len(pushed):
            force = env.rigid_body_external_forces[pushed, 0, :3]
            recorder.push_impulse[pushed] = torch.norm(force, dim=1) * env.sim_params.dt

    def fixed_resample(env_ids):
        recorder.fixed_commands(env_ids)

    def record_callback():
        original_callback()
        recorder.fixed_commands(recorder.all_ids)
        recorder.on_step()

    def record_reset(env_ids):
        recorder.on_reset(env_ids)
        original_reset(env_ids)
        recorder.clear(env_ids)
        # reset_idx resamples commands before it zeroes episode_length_buf.
        # Apply the warmup command again for the newly reset episode.
        recorder.fixed_commands(env_ids)

    env._resample_commands = fixed_resample
    env._post_physics_step_callback = record_callback
    env._push_robots = record_push
    env.reset_idx = record_reset
    runner, _ = task_registry.make_alg_runner(
        env=env, args=args, train_cfg=train_cfg, log_root=None
    )
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    obs, obs_history = env.get_observations()
    recorder.clear(recorder.all_ids)
    recorder.enabled = True
    max_steps = max(1000, 2 * episodes_per_yaw * math.ceil(episode_seconds / env.dt))
    complete = False
    with torch.inference_mode():
        for _ in range(max_steps):
            actions, _ = policy(obs, obs_history)
            obs, _, _, _, _, obs_history = env.step(actions)
            counts = {yaw: 0 for yaw in yaws}
            for row in recorder.rows:
                counts[row["yaw_cmd_radps"]] += 1
            if all(counts[yaw] >= episodes_per_yaw for yaw in yaws):
                complete = True
                break
    if not complete:
        raise RuntimeError("Evaluation ended before enough episodes were collected")

    # Keep the first N completed episodes for each case, in collection order.
    selected = [
        row for yaw in yaws
        for row in [r for r in recorder.rows if r["yaw_cmd_radps"] == yaw][:episodes_per_yaw]
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "code_sha256": {
            relative: sha256_file(Path(WHEEL_LEGGED_GYM_ROOT_DIR) / relative)
            for relative in CODE_FILES
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "target_id": target["id"],
        "target_path": str(target_path),
        "target_sha256": sha256_file(target_path),
        "seed": args.seed if args.seed is not None else train_cfg.seed,
        "num_envs": env.num_envs,
        "yaw_commands_radps": yaws,
        "height_m": height,
        "episode_seconds": episode_seconds,
        "warmup_seconds": warmup_seconds,
        "yaw_ramp_seconds": yaw_ramp_seconds,
        "tilt_metric_window": "post_warmup",
        "wobble_metric_window": "one_second_after_warmup",
        "axis_sway_frame": "world_body_z_horizontal_small_angle_deg",
        "episodes_per_yaw": episodes_per_yaw,
        "randomization": randomization,
        "domain_ablation": domain_ablation,
        "fixed_restitution": fixed_restitution,
        "observation_noise": env_cfg.noise.add_noise,
        "push_robots": env_cfg.domain_rand.push_robots,
        "domain_randomization": {
            name: getattr(env_cfg.domain_rand, name)
            for name in DOMAIN_RAND_FIELDS
        },
        "stable_thresholds": thresholds,
        "trace_seconds": trace_seconds,
        "trace_all": trace_all,
        "wlg_environment": {
            name: value for name, value in sorted(os.environ.items())
            if name.startswith("WLG_")
        },
        "robot_asset": env_cfg.asset.file,
        "robot_asset_sha256": sha256_file(
            env_cfg.asset.file.format(
                WHEEL_LEGGED_GYM_ROOT_DIR=WHEEL_LEGGED_GYM_ROOT_DIR
            )
        ),
        "control_stiffness": env_cfg.control.stiffness,
        "control_damping": env_cfg.control.damping,
    }
    selected_ids = {row["episode_id"] for row in selected}
    traces = [item for item in recorder.traces if item[0] in selected_ids]
    write_outputs(out_dir, selected, traces, manifest)
    print(f"Spin evaluation written to {out_dir}")


if __name__ == "__main__":
    main()
