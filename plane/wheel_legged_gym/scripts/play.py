# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA
# SPDX-License-Identifier: BSD-3-Clause

import os
import math
import numpy as np
from play_hud import AMBER, CYAN, GREEN, RED, build_hud_lines

import isaacgym
from isaacgym import gymtorch
import torch

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import get_args, export_policy_as_jit, task_registry

try:
    from pynput import keyboard
except ImportError:
    print("Missing dependency: pynput. Please install it with: pip install pynput")
    raise


# --------------------
# Global command state
# --------------------
cmd_x = float(os.getenv("WLG_PLAY_INITIAL_LIN_VEL", "0.0"))
ang_vel = float(os.getenv("WLG_PLAY_INITIAL_YAW", "0.0"))
cmd_height = float(os.getenv("WLG_PLAY_HEIGHT", "0.20"))
PLAY_YAW_RAMP_S = float(os.getenv("WLG_PLAY_YAW_RAMP_S", "0.0"))
if not math.isfinite(PLAY_YAW_RAMP_S) or PLAY_YAW_RAMP_S < 0:
    raise ValueError("WLG_PLAY_YAW_RAMP_S must be finite and nonnegative")
PLAY_LIN_RAMP_S = float(os.getenv("WLG_PLAY_LIN_RAMP_S", "0.0"))
if not math.isfinite(PLAY_LIN_RAMP_S) or PLAY_LIN_RAMP_S < 0:
    raise ValueError("WLG_PLAY_LIN_RAMP_S must be finite and nonnegative")
applied_yaw = 0.0
ramp_start_yaw = 0.0
ramp_target_yaw = 0.0
ramp_elapsed_s = 0.0
applied_lin_vel = cmd_x
lin_ramp_start = cmd_x
lin_ramp_target = cmd_x
lin_ramp_elapsed_s = 0.0
running = True
reset_requested = False
turn_left_pressed = False
turn_right_pressed = False
hud_enabled = os.getenv("WLG_PLAY_HUD", "0").strip().lower() in (
    "1", "true", "yes", "on"
)
hud_toggle_pressed = False
HUD_UPDATE_STEPS = 5
HUD_CAMERA_OFFSET = np.array([-5.0, -6.0, 3.5], dtype=np.float32)
HUD_RIGHT = np.array([6.0, -5.0, 0.0], dtype=np.float32)
HUD_RIGHT /= np.linalg.norm(HUD_RIGHT)

LIN_VEL_CMD = float(os.getenv("WLG_PLAY_LIN_VEL_CMD", "0.5"))
YAW_STEP = float(os.getenv("WLG_PLAY_YAW_STEP", "1.0"))
HEIGHT_STEP = 0.02

# Initial viewer camera. Applied once after the environments are created.
INITIAL_CAMERA_POSITION = [20.0, -20.0, 10.0]
INITIAL_CAMERA_LOOK_AT = [20.0, 40.0, 0.0]
FOLLOW_CAMERA_OFFSET = [-2.5, -3.0, 1.6]
FOLLOW_CAMERA_LOOK_AHEAD = [0.8, 0.0, 0.2]
PLAY_SPAWN_Z = float(
    os.getenv("WLG_PLAY_SPAWN_Z", os.getenv("WLG_SPAWN_Z", "0.12"))
)
PLAY_WITH_RANDOMIZATION = os.getenv(
    "WLG_PLAY_WITH_RANDOMIZATION", "1"
).strip().lower() in ("1", "true", "yes", "on")



def update_yaw_cmd():
    global ang_vel
    if turn_left_pressed and not turn_right_pressed:
        ang_vel = YAW_STEP
    elif turn_right_pressed and not turn_left_pressed:
        ang_vel = -YAW_STEP
    else:
        ang_vel = 0.0


def on_press(key):
    global cmd_x, ang_vel, cmd_height, running
    global reset_requested
    global turn_left_pressed, turn_right_pressed
    global hud_enabled, hud_toggle_pressed

    if key == keyboard.Key.esc:
        running = False
        print("[CMD] quit (ESC)")
        return False

    try:
        k = key.char.lower()
    except Exception:
        return

    if k == "q":
        running = False
        print("[CMD] quit (q)")
        return False
    if k == "r":
        reset_requested = True
        print("[CMD] reset environment")
    elif k == "w":
        cmd_x = LIN_VEL_CMD
        print(f"[CMD] forward: x={cmd_x:.2f}")
    elif k == "s":
        cmd_x = -LIN_VEL_CMD
        print(f"[CMD] backward: x={cmd_x:.2f}")
    elif k == "a":
        if not turn_left_pressed:
            print("[CMD] turn left (hold)")
        turn_left_pressed = True
        update_yaw_cmd()
    elif k == "d":
        if not turn_right_pressed:
            print("[CMD] turn right (hold)")
        turn_right_pressed = True
        update_yaw_cmd()
    elif k == "e":
        cmd_x = 0.0
        turn_left_pressed = False
        turn_right_pressed = False
        update_yaw_cmd()
        print("[CMD] stop")
    elif k == "x":
        cmd_height += HEIGHT_STEP
        print(f"[CMD] height up: h={cmd_height:.2f}")
    elif k == "c":
        cmd_height -= HEIGHT_STEP
        print(f"[CMD] height down: h={cmd_height:.2f}")
    elif k == "h" and not hud_toggle_pressed:
        hud_toggle_pressed = True
        hud_enabled = not hud_enabled
        print(f"[HUD] {'on' if hud_enabled else 'off'}")


def on_release(key):
    global turn_left_pressed, turn_right_pressed, hud_toggle_pressed
    try:
        k = key.char.lower()
    except Exception:
        return

    if k == "a":
        turn_left_pressed = False
        update_yaw_cmd()
    elif k == "d":
        turn_right_pressed = False
        update_yaw_cmd()
    elif k == "h":
        hud_toggle_pressed = False
    return


def apply_manual_commands(env, env_cfg, advance_yaw=True):
    global cmd_x, ang_vel, cmd_height
    global applied_yaw, ramp_start_yaw, ramp_target_yaw, ramp_elapsed_s
    global applied_lin_vel, lin_ramp_start, lin_ramp_target, lin_ramp_elapsed_s

    cmd_x = float(
        np.clip(
            cmd_x,
            env_cfg.commands.ranges.lin_vel_x[0],
            env_cfg.commands.ranges.lin_vel_x[1],
        )
    )
    ang_vel = float(
        np.clip(
            ang_vel,
            env_cfg.commands.ranges.ang_vel_yaw[0],
            env_cfg.commands.ranges.ang_vel_yaw[1],
        )
    )
    cmd_height = float(
        np.clip(
            cmd_height,
            env_cfg.commands.ranges.height[0],
            env_cfg.commands.ranges.height[1],
        )
    )

    if PLAY_LIN_RAMP_S == 0.0:
        applied_lin_vel = cmd_x
        lin_ramp_target = cmd_x
    else:
        if cmd_x != lin_ramp_target:
            lin_ramp_start = applied_lin_vel
            lin_ramp_target = cmd_x
            lin_ramp_elapsed_s = 0.0
        if advance_yaw:
            lin_ramp_elapsed_s = min(
                lin_ramp_elapsed_s + env.dt, PLAY_LIN_RAMP_S
            )
            fraction = lin_ramp_elapsed_s / PLAY_LIN_RAMP_S
            applied_lin_vel = (
                lin_ramp_start
                + (lin_ramp_target - lin_ramp_start) * fraction
            )

    if PLAY_YAW_RAMP_S == 0.0:
        applied_yaw = ang_vel
        ramp_target_yaw = ang_vel
    else:
        if ang_vel != ramp_target_yaw:
            ramp_start_yaw = applied_yaw
            ramp_target_yaw = ang_vel
            ramp_elapsed_s = 0.0
        if advance_yaw:
            ramp_elapsed_s = min(ramp_elapsed_s + env.dt, PLAY_YAW_RAMP_S)
            fraction = ramp_elapsed_s / PLAY_YAW_RAMP_S
            applied_yaw = (
                ramp_start_yaw + (ramp_target_yaw - ramp_start_yaw) * fraction
            )

    env.commands[:, 2] = cmd_height

    jump_ids = getattr(env, "jump_ramp_idx", None)
    if jump_ids is None or len(jump_ids) == 0:
        env.commands[:, 0] = applied_lin_vel
        env.commands[:, 1] = applied_yaw
        return

    manual_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    manual_mask[jump_ids] = False
    manual_ids = manual_mask.nonzero(as_tuple=False).flatten()
    if len(manual_ids) != 0:
        env.commands[manual_ids, 0] = applied_lin_vel
        env.commands[manual_ids, 1] = applied_yaw

    env.commands[jump_ids, 0] = env_cfg.commands.jump_ramp_lin_vel_x
    env.commands[jump_ids, 2] = env_cfg.commands.jump_ramp_height
    env.commands[jump_ids, 3] = env_cfg.commands.jump_ramp_heading


def apply_play_spawn_pose(env):
    """Place every play actor above its terrain before the first physics step."""
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.root_states[env_ids] = env.base_init_state
    env.root_states[env_ids, :3] += env.env_origins[env_ids]
    env.root_states[env_ids, 7:13] = 0.0

    env_ids_int32 = env_ids.to(dtype=torch.int32)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)

    # Avoid interpreting the one-time placement as a large initial velocity.
    if hasattr(env, "last_base_position"):
        env.last_base_position[:] = env.root_states[:, :3]
    if hasattr(env, "last_root_vel"):
        env.last_root_vel[:] = env.root_states[:, 7:13]


def update_follow_camera(env, env_idx):
    """Keep the viewer close to the selected robot in world coordinates."""
    if getattr(env, "viewer", None) is None:
        return

    robot_position = env.root_states[env_idx, :3].detach().cpu().tolist()
    camera_position = [
        robot_position[i] + FOLLOW_CAMERA_OFFSET[i] for i in range(3)
    ]
    camera_look_at = [
        robot_position[i] + FOLLOW_CAMERA_LOOK_AHEAD[i] for i in range(3)
    ]
    env.set_camera(camera_position, camera_look_at)


def _status_color(value, good, caution):
    if value <= good:
        return GREEN
    return AMBER if value <= caution else RED


def draw_play_hud(env, env_idx, left_wheel_idx, right_wheel_idx):
    """Draw live metrics beside the focused robot in the Isaac Gym viewer."""
    robot_position = env.root_states[env_idx, :3].detach().cpu().numpy()
    env_origin = env.env_origins[env_idx].detach().cpu().numpy()
    camera_target = robot_position + HUD_RIGHT * 1.8 + np.array(
        [0.0, 0.0, 1.35], dtype=np.float32
    )
    env.set_camera((camera_target + HUD_CAMERA_OFFSET).tolist(), camera_target.tolist())

    commanded_yaw = env.commands[env_idx, 1].item()
    actual_yaw = env.base_ang_vel[env_idx, 2].item()
    tilt_deg = math.degrees(math.acos(float(np.clip(
        -env.projected_gravity[env_idx, 2].item(), -1.0, 1.0
    ))))
    rp_rate = torch.norm(env.base_ang_vel[env_idx, :2]).item()
    wheel_speed = env.dof_vel[env_idx, [left_wheel_idx, right_wheel_idx]].tolist()
    wheel_speed_limit = env.dof_vel_limits[[left_wheel_idx, right_wheel_idx]].tolist()
    wheel_torque = env.torques[env_idx, [left_wheel_idx, right_wheel_idx]].tolist()
    wheel_limit = env.torque_limits[[left_wheel_idx, right_wheel_idx]].tolist()
    speed_fraction = [
        abs(speed) / max(limit, 1e-6)
        for speed, limit in zip(wheel_speed, wheel_speed_limit)
    ]
    torque_fraction = [
        abs(torque) / max(limit, 1e-6)
        for torque, limit in zip(wheel_torque, wheel_limit)
    ]

    rows = [
        ("SET", f"{ang_vel:+.2f} RAD/S", CYAN),
        ("CMD", f"{commanded_yaw:+.2f} RAD/S", CYAN),
        ("YAW", f"{actual_yaw:+.2f} RAD/S", GREEN),
        ("ERR", f"{abs(commanded_yaw - actual_yaw):.2f} RAD/S",
         _status_color(abs(commanded_yaw - actual_yaw), 0.8, 1.5)),
        ("TILT", f"{tilt_deg:.1f} DEG", _status_color(tilt_deg, 5.0, 12.0)),
        ("RP", f"{rp_rate:.2f} RAD/S", _status_color(rp_rate, 0.5, 1.0)),
        ("WL", f"{wheel_speed[0]:+.1f}/{wheel_speed_limit[0]:.1f}",
         _status_color(speed_fraction[0], 0.7, 0.9)),
        ("WR", f"{wheel_speed[1]:+.1f}/{wheel_speed_limit[1]:.1f}",
         _status_color(speed_fraction[1], 0.7, 0.9)),
        ("TL", f"{abs(wheel_torque[0]):.1f}/{wheel_limit[0]:.1f} NM",
         _status_color(torque_fraction[0], 0.7, 0.9)),
        ("TR", f"{abs(wheel_torque[1]):.1f}/{wheel_limit[1]:.1f} NM",
         _status_color(torque_fraction[1], 0.7, 0.9)),
    ]
    panel_origin = robot_position - env_origin + HUD_RIGHT * 0.95
    panel_origin[2] += 0.35
    vertices, colors = build_hud_lines(panel_origin, HUD_RIGHT, rows)
    env.gym.clear_lines(env.viewer)
    env.gym.add_lines(env.viewer, env.envs[env_idx], len(colors), vertices, colors)


def play(args):
    global running, reset_requested

    print("\n====== Keyboard Control Mode (NO Enter) ======")
    print("w      : forward")
    print("s      : backward")
    print("a      : hold to turn left")
    print("d      : hold to turn right")
    print("e      : stop")
    print("r      : reset focused environment")
    print("x      : height up")
    print("c      : height down")
    print("h      : toggle live HUD")
    print("q/ESC  : quit")
    print("camera : HUD follow view" if hud_enabled else "camera : fixed overview")
    print(f"initial command: vx={cmd_x:.2f}, yaw={ang_vel:.2f}, h={cmd_height:.2f}")
    print(f"yaw command ramp: {PLAY_YAW_RAMP_S:.2f} s")
    print(f"linear command ramp: {PLAY_LIN_RAMP_S:.2f} s")
    print(f"live HUD: {'on' if hud_enabled else 'off'}")
    print("=============================================\n")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.init_state.pos[2] = PLAY_SPAWN_Z
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.env.episode_length_s = 20
    env_cfg.terrain.num_rows = 5
    # Keep all 20 curriculum columns so the custom curb/drop terrain (column
    # 19) is present during visual validation.
    env_cfg.terrain.num_cols = 20
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    if PLAY_WITH_RANDOMIZATION:
        print("[PLAY] training randomization: enabled")
    else:
        print("[PLAY] training randomization: disabled")
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_restitution = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_inertia = False
        env_cfg.domain_rand.randomize_base_com = False
        env_cfg.domain_rand.randomize_Kp = False
        env_cfg.domain_rand.randomize_Kd = False
        env_cfg.domain_rand.randomize_motor_torque = False
        env_cfg.domain_rand.randomize_default_dof_pos = False
        env_cfg.domain_rand.randomize_action_delay = False
        env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.lift_robots = False
    env_cfg.domain_rand.downward_impulse_robots = False
    env_cfg.domain_rand.downward_impulse_interval_s = 3
    env_cfg.domain_rand.downward_impulse_vel_range = [2.4, 2.8]
    # env_cfg.domain_rand.vmc_force_events = True
    env_cfg.terrain.curriculum = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    apply_play_spawn_pose(env)
    print(f"[PLAY] spawn height: z={PLAY_SPAWN_Z:.3f} m above terrain origin")

    focus_env_idx = 0
    focus_terrain = os.getenv("WLG_PLAY_FOCUS_TERRAIN", "custom_drop").lower()
    focus_attr = {
        "custom_drop": "custom_curb_drop_idx",
        "reverse_climb": "custom_reverse_climb_idx",
        "pyramid_climb": "stair_up_idx",
        "smooth_slope": "smooth_slope_idx",
        "rough_slope": "rough_slope_idx",
    }.get(focus_terrain)
    custom_ids = getattr(env, focus_attr, None) if focus_attr is not None else None
    has_selected_focus = custom_ids is not None and len(custom_ids) != 0
    if has_selected_focus:
        focus_env_idx = int(custom_ids[0].item())
        print(f"[PLAY] focusing {focus_terrain} env: {focus_env_idx}")

    if getattr(env, "viewer", None) is not None and not hud_enabled:
        if has_selected_focus:
            update_follow_camera(env, focus_env_idx)
        else:
            env.set_camera(INITIAL_CAMERA_POSITION, INITIAL_CAMERA_LOOK_AT)

    wheel_indices = env.wheel_dof_indices.tolist()
    left_wheel_idx = next(
        index for index in wheel_indices if env.dof_names[index].lower().startswith("l")
    )
    right_wheel_idx = next(
        index for index in wheel_indices if env.dof_names[index].lower().startswith("r")
    )

    apply_manual_commands(env, env_cfg, advance_yaw=False)
    obs, obs_history = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    is_sequence_policy = bool(ppo_runner.alg.actor_critic.is_sequence)

    if EXPORT_POLICY:
        path = os.path.join(
            WHEEL_LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy to:", path)

    i = 0
    hud_was_drawn = False
    try:
        while running and i < 100000:
            if reset_requested:
                reset_requested = False
                env.reset_idx(torch.tensor([focus_env_idx], device=env.device, dtype=torch.long))
                obs, obs_history = env.get_observations()
                apply_manual_commands(env, env_cfg, advance_yaw=False)
                hud_was_drawn = False
                print(f"[RESET] env={focus_env_idx}")
                continue
            apply_manual_commands(env, env_cfg, advance_yaw=True)
            if is_sequence_policy:
                actions, _ = policy(obs, obs_history)
            else:
                actions = policy(obs)

            obs, _, _, _, _, obs_history = env.step(actions)
            apply_manual_commands(env, env_cfg, advance_yaw=False)

            if getattr(env, "viewer", None) is not None:
                if hud_enabled and (i % HUD_UPDATE_STEPS == 0 or not hud_was_drawn):
                    draw_play_hud(env, focus_env_idx, left_wheel_idx, right_wheel_idx)
                    hud_was_drawn = True
                elif not hud_enabled and hud_was_drawn:
                    env.gym.clear_lines(env.viewer)
                    hud_was_drawn = False

            if i % 50 == 0:
                actual_vx = env.base_lin_vel[focus_env_idx, 0].item()
                actual_vy = env.base_lin_vel[focus_env_idx, 1].item()
                vz = env.root_states[focus_env_idx, 9].item()
                yaw_rate = env.base_ang_vel[focus_env_idx, 2].item()
                gx, gy, gz = env.projected_gravity[focus_env_idx].tolist()
                roll_deg = math.degrees(math.atan2(gy, -gz))
                pitch_deg = math.degrees(math.atan2(-gx, math.hypot(gy, gz)))
                actual_height = env.base_height[focus_env_idx].item()
                # left_F = env.vmc_F[0, 0].item()
                # right_F = env.vmc_F[0, 1].item()
                print(
                    f"[{i}] env={focus_env_idx}, "
                    f"cmd_x={env.commands[focus_env_idx, 0].item():.2f}, "
                    f"actual_vx={actual_vx:.2f}, actual_vy={actual_vy:.2f}, "
                    f"vz={vz:.3f}, "
                    f"cmd_yaw={env.commands[focus_env_idx, 1].item():.3f}, "
                    f"real_yaw={yaw_rate:.3f}, "
                    f"h_cmd={env.commands[focus_env_idx, 2].item():.3f}, "
                    f"h_real={actual_height:.3f}, "
                    f"roll={roll_deg:.1f}deg, pitch={pitch_deg:.1f}deg, "
                    # f"F_left={left_F:.2f}, F_right={right_F:.2f}"
                )
            i += 1
    finally:
        try:
            listener.stop()
        except Exception:
            pass


if __name__ == "__main__":
    EXPORT_POLICY = False
    args = get_args()
    play(args)
