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
    "tilt_deg", "drift_m", "torque_limit_fraction", "x_m", "y_m",
) + tuple(f"action_{i}" for i in range(6)) + tuple(
    f"torque_{i}_nm" for i in range(6)
) + tuple(f"joint_pos_{i}_rad" for i in range(6)) + tuple(
    f"joint_vel_{i}_radps" for i in range(6)
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
    if not values or any(not math.isfinite(x) or x == 0 for x in values):
        raise ValueError("WLG_EVAL_YAWS must contain finite, nonzero yaw rates")
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

    def __init__(self, env, yaws, height, trace_seconds, max_traces, thresholds):
        if env.num_actions != 6 or env.dof_pos.shape[1] != 6:
            raise ValueError("Spin trace schema expects the six-action wheel-legged robot")
        self.env = env
        self.yaws = yaws
        self.height = height
        self.max_traces = max_traces
        self.thresholds = thresholds
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
        self.trace_length = max(1, round(trace_seconds / env.dt))
        self.trace = torch.zeros(
            env.num_envs, self.trace_length, len(TRACE_COLUMNS), device=env.device
        )
        self.trace_pos = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.trace_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.all_ids = torch.arange(env.num_envs, device=env.device)

    def fixed_commands(self, env_ids):
        self.env.commands[env_ids, 0] = 0.0
        self.env.commands[env_ids, 1] = self.yaw_for_env[env_ids]
        self.env.commands[env_ids, 2] = self.height

    def on_step(self):
        env = self.env
        self.height_error_sum += torch.abs(env.base_height - self.height)
        torque_limit = torch.clamp(env.torque_limits, min=1e-6)
        torque_fraction = torch.max(torch.abs(env.torques) / torque_limit, dim=1).values
        self.torque_fraction_max = torch.maximum(self.torque_fraction_max, torque_fraction)
        self.torque_saturation_steps += (torque_fraction >= 0.95).float()
        gravity = env.projected_gravity
        tilt = torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0)) * (180.0 / math.pi)
        drift = torch.norm(env.base_position[:, :2] - env.spin_position_anchor, dim=1)
        values = torch.stack(
            (
                env.episode_length_buf.float() * env.dt,
                env.commands[:, 1], env.base_ang_vel[:, 2], env.base_height,
                tilt, drift, torque_fraction,
                env.base_position[:, 0], env.base_position[:, 1],
            ), dim=1,
        )
        values = torch.cat((values, env.actions, env.torques, env.dof_pos, env.dof_vel), dim=1)
        self.trace[self.all_ids, self.trace_pos] = values
        self.trace_pos = (self.trace_pos + 1) % self.trace_length
        self.trace_count = torch.clamp(self.trace_count + 1, max=self.trace_length)

    def clear(self, env_ids):
        self.height_error_sum[env_ids] = 0.0
        self.torque_fraction_max[env_ids] = 0.0
        self.torque_saturation_steps[env_ids] = 0.0
        self.trace_pos[env_ids] = 0
        self.trace_count[env_ids] = 0

    def on_reset(self, env_ids):
        if not self.enabled or len(env_ids) == 0:
            return
        env = self.env
        for env_id in env_ids.tolist():
            steps = int(env.episode_length_buf[env_id].item())
            if steps < 2:
                continue
            spin_steps = max(float(env.episode_spin_steps[env_id].item()), 1.0)
            stationary_steps = max(float(env.episode_stationary_tilt_steps[env_id].item()), 1.0)
            timed_out = bool(env.time_out_buf[env_id].item())
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
                "yaw_mae_radps": float(env.episode_spin_yaw_abs_error_sum[env_id].item()) / spin_steps,
                "real_abs_yaw_radps": float(env.episode_spin_real_abs_yaw_sum[env_id].item()) / spin_steps,
                "direction_accuracy": float(env.episode_spin_direction_ok_sum[env_id].item()) / spin_steps,
                "max_drift_m": float(env.episode_spin_position_drift_max[env_id].item()),
                "rms_tilt_deg": math.sqrt(float(env.episode_stationary_tilt_sq_sum[env_id].item()) / stationary_steps) * 180.0 / math.pi,
                "max_tilt_deg": float(env.episode_max_tilt[env_id].item()) * 180.0 / math.pi,
                "mean_height_error_m": float(self.height_error_sum[env_id].item()) / steps,
                "max_torque_limit_fraction": float(self.torque_fraction_max[env_id].item()),
                "torque_saturation_fraction": float(self.torque_saturation_steps[env_id].item()) / steps,
            }
            row["stable"] = (
                timed_out
                and row["yaw_mae_radps"] <= self.thresholds["yaw_mae_radps"]
                and row["max_drift_m"] <= self.thresholds["max_drift_m"]
                and row["max_tilt_deg"] <= self.thresholds["max_tilt_deg"]
            )
            tags = []
            if not timed_out:
                tags.append(row["reset_reason"])
            for name in ("yaw_mae_radps", "max_drift_m", "max_tilt_deg"):
                if row[name] > self.thresholds[name]:
                    tags.append(name)
            row["failure_tags"] = "|".join(tags)
            self.rows.append(row)
            if not row["stable"] and len(self.traces) < self.max_traces:
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
    failure_dir = out_dir / "failures"
    failure_dir.mkdir(exist_ok=True)
    for episode_id, trace in traces:
        with (failure_dir / f"episode_{episode_id:04d}.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(TRACE_COLUMNS)
            writer.writerows(trace)

    lines = ["# Spin evaluation", "", f"Checkpoint: `{manifest['checkpoint']}`", ""]
    lines.append("| yaw command | episodes | complete | stable* | yaw MAE | p90 peak drift | p90 peak tilt |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for yaw in manifest["yaw_commands_radps"]:
        group = [row for row in rows if row["yaw_cmd_radps"] == yaw]
        mean_mae = statistics.mean(row["yaw_mae_radps"] for row in group)
        drift = sorted(row["max_drift_m"] for row in group)
        tilt = sorted(row["max_tilt_deg"] for row in group)
        index = math.ceil(0.9 * len(group)) - 1
        p90_drift = drift[index]
        p90_tilt = tilt[index]
        complete = sum(row["timed_out"] for row in group) / len(group)
        stable = sum(row["stable"] for row in group) / len(group)
        lines.append(
            f"| {yaw:+g} rad/s | {len(group)} | {complete:.1%} | {stable:.1%} | "
            f"{mean_mae:.2f} rad/s | {p90_drift:.3f} m | {p90_tilt:.1f}° |"
        )
    lines.extend((
        "", "*Stable uses the provisional thresholds in `manifest.json`; set them for the real robot before using this as an acceptance gate.",
        "", "See `episodes.csv` for per-episode values and `failures/` for short traces.", "",
    ))
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    target_path, target = read_target()
    yaws = parse_yaws(target["yaw_commands_radps"])
    episodes_per_yaw = env_int("WLG_EVAL_EPISODES_PER_YAW", target["episodes_per_yaw"])
    episode_seconds = env_float("WLG_EVAL_EPISODE_S", target["episode_seconds"])
    height = env_float("WLG_EVAL_HEIGHT", target["height_m"])
    trace_seconds = env_float("WLG_EVAL_TRACE_S", target["trace_seconds"])
    if episode_seconds <= 0 or trace_seconds <= 0 or height <= 0:
        raise ValueError("Episode duration, trace duration and height must be positive")
    max_traces = env_int("WLG_EVAL_MAX_TRACES", target["max_traces"])
    thresholds = {
        "yaw_mae_radps": env_float("WLG_EVAL_YAW_MAE_MAX", target["stable_thresholds"]["yaw_mae_radps"]),
        "max_drift_m": env_float("WLG_EVAL_DRIFT_MAX", target["stable_thresholds"]["max_drift_m"]),
        "max_tilt_deg": env_float("WLG_EVAL_TILT_MAX_DEG", target["stable_thresholds"]["max_tilt_deg"]),
    }
    if any(value < 0 for value in thresholds.values()):
        raise ValueError("Stability thresholds must be nonnegative")
    randomization = env_bool("WLG_EVAL_RANDOMIZATION", True)
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
    recorder = SpinRecorder(env, yaws, height, trace_seconds, max_traces, thresholds)
    original_callback = env._post_physics_step_callback
    original_reset = env.reset_idx

    def fixed_resample(env_ids):
        recorder.fixed_commands(env_ids)

    def record_callback():
        original_callback()
        recorder.on_step()

    def record_reset(env_ids):
        recorder.on_reset(env_ids)
        original_reset(env_ids)
        recorder.clear(env_ids)

    env._resample_commands = fixed_resample
    env._post_physics_step_callback = record_callback
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
        "episodes_per_yaw": episodes_per_yaw,
        "randomization": randomization,
        "stable_thresholds": thresholds,
        "trace_seconds": trace_seconds,
        "wlg_environment": {
            name: value for name, value in sorted(os.environ.items())
            if name.startswith("WLG_")
        },
        "robot_asset": env_cfg.asset.file,
        "control_stiffness": env_cfg.control.stiffness,
        "control_damping": env_cfg.control.damping,
    }
    selected_ids = {row["episode_id"] for row in selected}
    traces = [item for item in recorder.traces if item[0] in selected_ids]
    write_outputs(out_dir, selected, traces, manifest)
    print(f"Spin evaluation written to {out_dir}")


if __name__ == "__main__":
    main()
