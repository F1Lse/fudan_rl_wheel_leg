# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA
# SPDX-License-Identifier: BSD-3-Clause

import os
import numpy as np

import isaacgym
from isaacgym import gymapi
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
cmd_x = float(os.getenv("WLG_PLAY_SPEED", "0.0"))
ang_vel = 0.0
cmd_height = 0.2
running = True
reset_requested = False
turn_left_pressed = False
turn_right_pressed = False

YAW_STEP = 1.0

# Camera follow
ENABLE_CAMERA_FOLLOW = bool(int(os.getenv("WLG_PLAY_CAMERA_FOLLOW", "0")))
CAMERA_DISTANCE = 2.2
CAMERA_HEIGHT = 1.0
CAMERA_LOOK_AT_HEIGHT = 0.35
CAMERA_SIDE_OFFSET = 0.0
CAMERA_SMOOTHING = 0.2
CAMERA_YAW_OFFSET = np.pi  # 180 deg: flip front/back view


def update_yaw_cmd():
    global ang_vel
    if turn_left_pressed and not turn_right_pressed:
        ang_vel = YAW_STEP
    elif turn_right_pressed and not turn_left_pressed:
        ang_vel = -YAW_STEP
    else:
        ang_vel = 0.0


def on_press(key):
    global cmd_x, ang_vel, cmd_height, running, reset_requested
    global turn_left_pressed, turn_right_pressed

    if key == keyboard.Key.esc:
        running = False
        print("[CMD] quit (ESC)")
        return False

    try:
        k = key.char.lower()
    except Exception:
        return

    speed_keys = {
        "1": -1.5,
        "2": -1.0,
        "3": -0.5,
        "4": 0.0,
        "5": 0.5,
        "6": 1.0,
        "7": 1.5,
    }
    if k in speed_keys:
        cmd_x = speed_keys[k]
        print(f"[CMD] linear speed {cmd_x:+.1f} m/s")
        return

    if k == "q":
        running = False
        print("[CMD] quit (q)")
        return False
    if k == "r":
        reset_requested = True
        print("[CMD] reset requested")
        return
    if k == "w":
        cmd_x = -2.5
        print("[CMD] forward")
    elif k == "s":
        cmd_x = 0.0
        print("[CMD] stop")
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
        turn_left_pressed = False
        turn_right_pressed = False
        update_yaw_cmd()
        print("[CMD] stop turning")
    elif k == "x":
        cmd_height += 0.05
        print("[CMD] height up")
    elif k == "c":
        cmd_height -= 0.05
        print("[CMD] height down")


def on_release(key):
    global turn_left_pressed, turn_right_pressed
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
    return


def apply_manual_commands(env, env_cfg):
    env.commands[:, 0] = cmd_x
    env.commands[:, 1] = ang_vel
    env.commands[:, 2] = cmd_height

    env.commands[:, 0] = torch.clamp(
        env.commands[:, 0],
        env_cfg.commands.ranges.lin_vel_x[0],
        env_cfg.commands.ranges.lin_vel_x[1],
    )
    env.commands[:, 1] = torch.clamp(
        env.commands[:, 1],
        env_cfg.commands.ranges.ang_vel_yaw[0],
        env_cfg.commands.ranges.ang_vel_yaw[1],
    )
    env.commands[:, 2] = torch.clamp(
        env.commands[:, 2],
        env_cfg.commands.ranges.height[0],
        env_cfg.commands.ranges.height[1],
    )


def quat_xyzw_to_yaw(quat_xyzw):
    x, y, z, w = quat_xyzw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def update_follow_camera(env, camera_pos):
    if not ENABLE_CAMERA_FOLLOW or getattr(env, "viewer", None) is None:
        return camera_pos

    base_state = env.root_states[0]
    base_pos = base_state[:3].detach().cpu().numpy()
    base_quat = base_state[3:7].detach().cpu().numpy()

    yaw = quat_xyzw_to_yaw(base_quat) + CAMERA_YAW_OFFSET
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
    right = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float64)

    look_at = base_pos + np.array([0.0, 0.0, CAMERA_LOOK_AT_HEIGHT], dtype=np.float64)
    desired_pos = (
        look_at
        - CAMERA_DISTANCE * forward
        + CAMERA_SIDE_OFFSET * right
        + np.array([0.0, 0.0, CAMERA_HEIGHT], dtype=np.float64)
    )

    if camera_pos is None:
        camera_pos = desired_pos
    else:
        camera_pos = (1.0 - CAMERA_SMOOTHING) * camera_pos + CAMERA_SMOOTHING * desired_pos

    env.gym.viewer_camera_look_at(
        env.viewer,
        None,
        gymapi.Vec3(float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])),
        gymapi.Vec3(float(look_at[0]), float(look_at[1]), float(look_at[2])),
    )
    return camera_pos


def play(args):
    global running, reset_requested

    print("\n====== Keyboard Control Mode (NO Enter) ======")
    print("w      : forward")
    print("s      : stop")
    print("a      : hold to turn left")
    print("d      : hold to turn right")
    print("e      : stop turning")
    print("x      : height up")
    print("c      : height down")
    print("r      : reset robot")
    print("q/ESC  : quit")
    print("camera : follow robot (third-person)")
    print(f"initial speed: {cmd_x:+.2f} m/s (set WLG_PLAY_SPEED to override)")
    print("=============================================\n")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 20
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.terrain.curriculum = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

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
    camera_pos = None
    reset_requested = False
    try:
        while running and i < 100000:
            if reset_requested:
                reset_requested = False
                env.reset()
                obs, obs_history = env.get_observations()
                camera_pos = None
                print("[CMD] robot reset")

            # Apply keyboard commands before inference so the current command
            # is included in the observation used for this action.
            apply_manual_commands(env, env_cfg)

            if is_sequence_policy:
                actions, _ = policy(obs, obs_history)
            else:
                actions = policy(obs)

            obs, _, _, _, _, obs_history = env.step(actions)
            camera_pos = update_follow_camera(env, camera_pos)
            if i % 50 == 0:
                actual_vx = env.base_lin_vel[0, 0].item()
                print(
                    f"[STATE] cmd_x={cmd_x:+.2f} m/s, "
                    f"actual_vx={actual_vx:+.2f} m/s, "
                    f"vz={env.root_states[0, 9].item():+.2f}"
                )
            # Keep the terminal quiet during visualization.
            # if i % 50 == 0:
                # vz = env.root_states[0, 9].item()
                # yaw_rate = env.base_ang_vel[0, 2].item()
                # print(
                #     f"[{i}] vz={vz:.3f}, cmd_x={cmd_x:.2f}, "
                #     f"cmd_yaw={env.commands[0, 1].item():.3f}, real_yaw={yaw_rate:.3f}"
                # )
                
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
