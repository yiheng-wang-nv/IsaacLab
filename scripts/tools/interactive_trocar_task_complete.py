# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive GUI runner for the task-complete trocar GR00T policy.

The model is trained with a ``task_complete`` output head (soft label) that
predicts whether the current stage sub-task is finished.  Press 1-5 to select
the stage prompt, then SPACE to run.  The script auto-pauses when the predicted
``task_complete`` value exceeds a configurable threshold.

Controls
--------
* ``1``-``5`` – select stage task prompt
* ``SPACE / P`` – toggle continuous inference
* ``S`` – execute one policy step
* ``G`` – probe task_complete prediction without stepping the environment
* ``R`` – reset the environment
* ``O`` – rotate tray by the configured yaw delta
* ``B`` – rotate tray back to reset yaw
* ``H`` – print this help again
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run an interactive GUI for the task-complete trocar policy.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the Isaac-GR00T checkpoint directory.")
parser.add_argument(
    "--task_id",
    type=str,
    default="Isaac-Assemble-Trocar-G129-Dex3-RLinf-v0",
    help="Gym task id for the trocar environment.",
)
parser.add_argument(
    "--initial_task",
    type=int,
    default=0,
    choices=[0, 1, 2, 3, 4],
    help="Initial stage task index (0-4).",
)
parser.add_argument("--denoising_steps", type=int, default=4, help="GR00T denoising steps.")
parser.add_argument("--step_hz", type=float, default=10.0, help="Continuous inference stepping rate.")
parser.add_argument(
    "--open_loop_steps",
    type=int,
    default=1,
    help="Number of env steps to execute per policy inference call.",
)
parser.add_argument(
    "--task_timeout_steps",
    type=int,
    default=300,
    help="Pause continuous inference when this many steps have elapsed.",
)
parser.add_argument(
    "--task_complete_threshold",
    type=float,
    default=0.5,
    help="Predicted task_complete value above which auto-pause triggers.",
)
parser.add_argument(
    "--enable_task_complete_stop",
    action="store_true",
    default=True,
    help="Auto-pause when task_complete prediction exceeds the threshold.",
)
parser.add_argument(
    "--disable_task_complete_stop",
    action="store_false",
    dest="enable_task_complete_stop",
    help="Disable task_complete auto-pause.",
)
parser.add_argument("--seed", type=int, default=None, help="Optional environment reset seed.")
parser.add_argument(
    "--model_device",
    type=str,
    default=None,
    help="Device for the policy. Defaults to the simulator device when omitted.",
)
parser.add_argument(
    "--initial_tray_yaw_min_deg",
    type=float,
    default=0.0,
    help="Minimum random tray yaw [deg] after every reset.",
)
parser.add_argument(
    "--initial_tray_yaw_max_deg",
    type=float,
    default=5.0,
    help="Maximum random tray yaw [deg] after every reset.",
)
parser.add_argument(
    "--tray_yaw_increment_deg",
    type=float,
    default=-30.0,
    help="Tray yaw delta [deg] applied by the interactive rotate button.",
)
parser.add_argument(
    "--tray_yaw_steps",
    type=int,
    default=5,
    help="Number of env steps used to interpolate each tray rotation.",
)
parser.add_argument(
    "--fixed_initial_state_dataset",
    type=str,
    default=None,
    help="Optional LeRobot dataset directory. If set, uses a fixed start pose from the dataset.",
)
parser.add_argument("--fixed_initial_state_episode", type=int, default=0, help="Episode index for fixed start pose.")
parser.add_argument("--fixed_initial_state_frame", type=int, default=0, help="Frame index for fixed start pose.")
parser.add_argument(
    "--fixed_initial_state_steps",
    type=int,
    default=30,
    help="Number of env steps to command the fixed start pose after every reset.",
)
parser.add_argument(
    "--fixed_initial_state_tolerance",
    type=float,
    default=0.035,
    help="Stop fixed-start warm-up early when max 28-DoF state error is below this value.",
)
parser.add_argument(
    "--tray_carry_xy_margin",
    type=float,
    default=0.10,
    help="Extra XY margin [m] for deciding whether a trocar is still on the tray.",
)
parser.add_argument(
    "--tray_carry_z_tolerance",
    type=float,
    default=0.04,
    help="Z-offset tolerance [m] for deciding whether a trocar is still on the tray.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
args.enable_cameras = True


def _infer_display_from_x11_sockets() -> str | None:
    x11_dir = Path("/tmp/.X11-unix")
    if not x11_dir.exists():
        return None
    display_ids: list[int] = []
    for socket_path in x11_dir.glob("X*"):
        suffix = socket_path.name[1:]
        if suffix.isdigit():
            display_ids.append(int(suffix))
    if not display_ids:
        return None
    return f":{min(display_ids)}"


if not os.environ.get("DISPLAY"):
    inferred_display = _infer_display_from_x11_sockets()
    if inferred_display is not None:
        os.environ["DISPLAY"] = inferred_display
        print(f"[DEBUG] DISPLAY was unset. Inferred DISPLAY={inferred_display} from /tmp/.X11-unix.")
    else:
        print("[DEBUG] DISPLAY is unset and no X11 socket was found under /tmp/.X11-unix.")

if not os.environ.get("XAUTHORITY"):
    default_xauthority = Path.home() / ".Xauthority"
    if default_xauthority.exists():
        os.environ["XAUTHORITY"] = str(default_xauthority)

if getattr(args, "visualizer", None) in (None, []):
    args.visualizer = ["kit"]
    args.visualizer_explicit = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import carb
import carb.input
import gymnasium as gym
import numpy as np
import omni.appwindow
import omni.kit.app
import omni.ui as ui
import torch
import warp as wp

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


TASK_DESCRIPTIONS = [
    "left hand pick up trocar",
    "right hand pick up trocar",
    "align trocars",
    "install trocar",
    "place trocar",
]

CAMERA_KEYS = ["front_camera", "left_wrist_camera", "right_wrist_camera"]

PERM_TO_REF = list(range(28))  # identity — dataset recorded with record_trocar_episodes.py
INV_PERM = np.argsort(PERM_TO_REF)
BODY_JOINT_INDICES = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
DEX3_JOINT_INDICES = [31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38]
SHOULDER_SLICE = (15, 29)
ACTION_PREFIX_PAD = 15
ROBOT_ACTION_DIM = ACTION_PREFIX_PAD + len(PERM_TO_REF)


def _print_controls() -> None:
    print("Interactive trocar task-complete controls:")
    print("  1-5   : select stage task prompt")
    print("  SPACE / P : toggle continuous inference")
    print("  S : run one policy step")
    print("  G : probe task_complete prediction without stepping")
    print("  O : rotate the tray by the configured yaw delta")
    print("  B : rotate the tray back to its reset yaw")
    print("  R : reset the environment")
    print("  H : print this help again")


def _find_isaac_gr00t_root(model_path: Path) -> Path:
    for candidate in [model_path, *model_path.parents]:
        if (candidate / "gr00t_config.py").exists() and (candidate / "gr00t" / "model" / "policy.py").exists():
            return candidate
    env_root = Path("/localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T")
    if (env_root / "gr00t_config.py").exists():
        return env_root
    raise FileNotFoundError("Could not locate Isaac-GR00T root.")


def _load_policy(model_path: str, model_device: str, denoising_steps: int):
    model_path_obj = Path(model_path).expanduser().resolve()
    isaac_gr00t_root = _find_isaac_gr00t_root(model_path_obj)
    if str(isaac_gr00t_root) not in sys.path:
        sys.path.insert(0, str(isaac_gr00t_root))

    from gr00t.model.policy import Gr00tPolicy

    spec = importlib.util.spec_from_file_location(
        "interactive_trocar_gr00t_config",
        str(isaac_gr00t_root / "gr00t_config.py"),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load gr00t_config.py from {isaac_gr00t_root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data_config = module.UnitreeG1SimTaskCompleteInferDataConfig()

    policy = Gr00tPolicy(
        model_path=str(model_path_obj),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=denoising_steps,
        device=model_device,
    )
    return policy


def _create_env(task_id: str, device: str):
    env_cfg = parse_env_cfg(task_id, device=device, num_envs=1)
    for cam_name in CAMERA_KEYS:
        cam_cfg = getattr(env_cfg.scene, cam_name)
        cam_cfg.data_types = ["rgb"]
    env_cfg.recorders = {}
    return gym.make(task_id, cfg=env_cfg).unwrapped


def _flush_observations(env):
    app = omni.kit.app.get_app()
    env.sim.set_setting("/app/player/playSimulations", False)
    app.update()
    env.sim.set_setting("/app/player/playSimulations", True)
    for sensor in env.scene.sensors.values():
        sensor.update(dt=0.0, force_recompute=True)
    return env.observation_manager.compute(update_history=True)


def _reset_env(
    env,
    seed: int | None,
    tray_yaw_deg: float | None = None,
    fixed_raw_action: np.ndarray | None = None,
    fixed_state_ref: np.ndarray | None = None,
):
    obs, _ = env.reset(seed=seed)
    if tray_yaw_deg is not None:
        _set_tray_trocar_yaw_delta(env, tray_yaw_deg)
    if fixed_raw_action is not None and fixed_state_ref is not None:
        info = _apply_fixed_initial_state(
            env, fixed_raw_action, fixed_state_ref,
            args.fixed_initial_state_steps, args.fixed_initial_state_tolerance,
        )
        print(f"[INFO] Fixed initial state warm-start: steps={info['steps']}, max_error={info['max_state_error']:.4f}")
    _flush_observations(env)
    return obs


def _get_joint_states(env) -> np.ndarray:
    joint_pos = wp.to_torch(env.scene["robot"].data.joint_pos)
    device = joint_pos.device
    body_idx = torch.tensor(BODY_JOINT_INDICES, device=device, dtype=torch.long)
    body_pos = joint_pos[:, body_idx]
    shoulder_pos = body_pos[:, SHOULDER_SLICE[0] : SHOULDER_SLICE[1]]
    dex3_idx = torch.tensor(DEX3_JOINT_INDICES, device=device, dtype=torch.long)
    dex3_pos = joint_pos[:, dex3_idx]
    states = torch.cat([shoulder_pos, dex3_pos], dim=-1)
    return states.cpu().numpy().astype(np.float32)


def _write_fixed_initial_state_to_sim(env, target_state_ref: np.ndarray) -> None:
    """Directly teleport the robot's controlled joints to the target state."""
    robot = env.scene["robot"]
    action_term = env.action_manager.get_term(env.action_manager.active_terms[0])
    joint_ids = action_term._joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = torch.arange(
            wp.to_torch(robot.data.joint_pos).shape[1], device=env.device, dtype=torch.long
        )[joint_ids]
    else:
        joint_ids = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
    # Use only the 28 controlled joints (skip prefix pad)
    joint_ids = joint_ids[ACTION_PREFIX_PAD:].to(dtype=torch.int32)

    target_internal = np.asarray(target_state_ref, dtype=np.float32)[INV_PERM]
    target_pos = torch.tensor(target_internal, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    target_vel = torch.zeros_like(target_pos)
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
    robot.write_joint_position_to_sim_index(position=target_pos, joint_ids=joint_ids, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=target_vel, joint_ids=joint_ids, env_ids=env_ids)


def _apply_fixed_initial_state(
    env,
    fixed_raw_action: np.ndarray,
    target_state_ref: np.ndarray,
    steps: int,
    tolerance: float,
) -> dict:
    """Teleport robot to target state, then hold for settle steps."""
    # Teleport first (direct sim write, bypasses termination)
    _write_fixed_initial_state_to_sim(env, target_state_ref)

    # Hold position for settle steps via env.step()
    action_tensor = torch.tensor(
        fixed_raw_action.reshape(1, ROBOT_ACTION_DIM), dtype=torch.float32, device=env.device
    )
    steps_run = 0
    for warm_step in range(max(steps, 0)):
        env.step(action_tensor)
        steps_run = warm_step + 1
        error = float(np.abs(_get_joint_states(env)[:, PERM_TO_REF] - target_state_ref).max())
        if error <= tolerance:
            break

    # Teleport again to ensure exact pose after any auto-resets during settling
    _write_fixed_initial_state_to_sim(env, target_state_ref)
    max_state_error = float(np.abs(_get_joint_states(env)[:, PERM_TO_REF] - target_state_ref).max())
    return {"steps": steps_run, "max_state_error": max_state_error}


def _load_fixed_initial_state_ref(dataset_dir: str, episode_idx: int, frame_idx: int) -> np.ndarray:
    import pandas as pd

    dataset_path = Path(dataset_dir).expanduser()
    episode_path = dataset_path / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    if not episode_path.exists():
        matches = sorted((dataset_path / "data").glob(f"chunk-*/episode_{episode_idx:06d}.parquet"))
        if not matches:
            raise FileNotFoundError(f"Could not find episode_{episode_idx:06d}.parquet under {dataset_path}/data.")
        episode_path = matches[0]

    df = pd.read_parquet(episode_path, columns=["observation.state"])
    if frame_idx < 0 or frame_idx >= len(df):
        raise IndexError(f"Frame {frame_idx} out of range for {episode_path} ({len(df)} frames).")
    state_ref = np.asarray(df["observation.state"].iloc[frame_idx], dtype=np.float32)
    if state_ref.shape != (len(PERM_TO_REF),):
        raise ValueError(f"Expected state shape {(len(PERM_TO_REF),)}, got {state_ref.shape}.")
    return state_ref


def _fixed_initial_state_ref_to_raw_action(env, state_ref: np.ndarray) -> np.ndarray:
    action_term = env.action_manager.get_term(env.action_manager.active_terms[0])
    if action_term.action_dim != ROBOT_ACTION_DIM:
        raise RuntimeError(f"Expected action dim {ROBOT_ACTION_DIM}, got {action_term.action_dim}.")
    target_joint_pos = np.zeros(ROBOT_ACTION_DIM, dtype=np.float32)
    target_joint_pos[ACTION_PREFIX_PAD:] = np.asarray(state_ref, dtype=np.float32)[INV_PERM]
    device = torch.device(env.device)
    target = torch.tensor(target_joint_pos, dtype=torch.float32, device=device)
    scale = action_term._scale
    if isinstance(scale, torch.Tensor):
        scale = scale.to(device=device)
    else:
        scale = torch.full((action_term.action_dim,), float(scale), device=device)
    offset = action_term._offset
    if isinstance(offset, torch.Tensor):
        offset = offset.to(device=device)
    else:
        offset = torch.full((action_term.action_dim,), float(offset), device=device)
    raw_action = ((target - offset) / scale).cpu().numpy().astype(np.float32)
    raw_action[:ACTION_PREFIX_PAD] = 0.0
    return raw_action




def _get_camera_rgb(env, cam_name: str) -> np.ndarray:
    sensor = env.scene.sensors[cam_name]
    imgs = sensor.data.output["rgb"]
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().numpy()
    if imgs.shape[-1] == 4:
        imgs = imgs[..., :3]
    return imgs.astype(np.uint8)


def _sample_initial_tray_yaw_deg(rng: np.random.Generator) -> float:
    yaw_min = min(args.initial_tray_yaw_min_deg, args.initial_tray_yaw_max_deg)
    yaw_max = max(args.initial_tray_yaw_min_deg, args.initial_tray_yaw_max_deg)
    if yaw_max <= yaw_min:
        return float(yaw_min)
    return float(rng.uniform(yaw_min, yaw_max))


def _quat_mul_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def _quat_apply_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[:, :3]
    q_w = q[:, 3:4]
    uv = torch.cross(q_vec, v, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return v + 2.0 * (q_w * uv + uuv)


def _yaw_quat_xyzw(env, yaw_deg: float, count: int) -> torch.Tensor:
    yaw_rad = torch.full((count,), np.deg2rad(yaw_deg), device=env.device, dtype=torch.float32)
    yaw_quat = torch.zeros(count, 4, device=env.device, dtype=torch.float32)
    yaw_quat[:, 2] = torch.sin(yaw_rad / 2.0)
    yaw_quat[:, 3] = torch.cos(yaw_rad / 2.0)
    return yaw_quat


def _trocar_on_tray_mask(
    tray_state: torch.Tensor,
    trocar_state: torch.Tensor,
    tray_default: torch.Tensor,
    trocar_default: torch.Tensor,
) -> torch.Tensor:
    current_xy_dist = torch.linalg.norm(trocar_state[:, :2] - tray_state[:, :2], dim=-1)
    default_xy_dist = torch.linalg.norm(trocar_default[:, :2] - tray_default[:, :2], dim=-1)
    current_z_offset = trocar_state[:, 2] - tray_state[:, 2]
    default_z_offset = trocar_default[:, 2] - tray_default[:, 2]
    xy_close = current_xy_dist <= default_xy_dist + args.tray_carry_xy_margin
    z_close = torch.abs(current_z_offset - default_z_offset) <= args.tray_carry_z_tolerance
    return xy_close & z_close


def _set_tray_trocar_yaw_delta(
    env,
    yaw_deg: float,
    previous_yaw_deg: float | None = None,
    carry_trocars_on_tray_only: bool = False,
) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    tray = env.scene["tray"]
    trocar_1 = env.scene["trocar_1"]
    trocar_2 = env.scene["trocar_2"]

    tray_default = wp.to_torch(tray.data.default_root_state)[env_ids].clone()
    trocar_1_default = wp.to_torch(trocar_1.data.default_root_state)[env_ids].clone()
    trocar_2_default = wp.to_torch(trocar_2.data.default_root_state)[env_ids].clone()

    env_origins = env.scene.env_origins[env_ids]
    tray_default[:, :3] += env_origins
    trocar_1_default[:, :3] += env_origins
    trocar_2_default[:, :3] += env_origins

    target_quat = _yaw_quat_xyzw(env, yaw_deg, len(env_ids))
    tray_center = tray_default[:, :3]

    tray_new = tray_default.clone()
    tray_new[:, 3:7] = _quat_mul_xyzw(target_quat, tray_default[:, 3:7])
    zero_velocity = torch.zeros(len(env_ids), 6, device=env.device)
    tray.write_root_pose_to_sim_index(root_pose=tray_new[:, :7], env_ids=env_ids)
    tray.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_ids)

    if not carry_trocars_on_tray_only or previous_yaw_deg is None:
        for trocar, trocar_default in [(trocar_1, trocar_1_default), (trocar_2, trocar_2_default)]:
            trocar_new = trocar_default.clone()
            trocar_new[:, :3] = tray_center + _quat_apply_xyzw(target_quat, trocar_default[:, :3] - tray_center)
            trocar_new[:, 3:7] = _quat_mul_xyzw(target_quat, trocar_default[:, 3:7])
            trocar.write_root_pose_to_sim_index(root_pose=trocar_new[:, :7], env_ids=env_ids)
            trocar.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_ids)
        return

    tray_current = wp.to_torch(tray.data.root_state_w)[env_ids].clone()
    trocar_1_current = wp.to_torch(trocar_1.data.root_state_w)[env_ids].clone()
    trocar_2_current = wp.to_torch(trocar_2.data.root_state_w)[env_ids].clone()
    step_quat = _yaw_quat_xyzw(env, yaw_deg - previous_yaw_deg, len(env_ids))

    for trocar, trocar_current, trocar_default in [
        (trocar_1, trocar_1_current, trocar_1_default),
        (trocar_2, trocar_2_current, trocar_2_default),
    ]:
        carried = _trocar_on_tray_mask(tray_current, trocar_current, tray_default, trocar_default)
        if not bool(carried.any().item()):
            continue
        trocar_new = trocar_current.clone()
        trocar_new[:, :3] = tray_center + _quat_apply_xyzw(step_quat, trocar_current[:, :3] - tray_center)
        trocar_new[:, 3:7] = _quat_mul_xyzw(step_quat, trocar_current[:, 3:7])
        trocar.write_root_pose_to_sim_index(root_pose=trocar_new[carried, :7], env_ids=env_ids[carried])
        trocar.write_root_velocity_to_sim_index(root_velocity=zero_velocity[carried], env_ids=env_ids[carried])


def _normalize_env_raw_action(action: np.ndarray | torch.Tensor) -> np.ndarray:
    action_np = np.asarray(action, dtype=np.float32).reshape(ROBOT_ACTION_DIM).copy()
    action_np[:ACTION_PREFIX_PAD] = 0.0
    return action_np


def _raw_action_to_env_tensor(action: np.ndarray, device: str) -> torch.Tensor:
    action = _normalize_env_raw_action(action).reshape(1, ROBOT_ACTION_DIM)
    return torch.tensor(action, dtype=torch.float32, device=device)


def _build_policy_obs(env, task_description: str) -> dict:
    states_internal = _get_joint_states(env)[0]
    states_ref = states_internal[PERM_TO_REF]

    obs_dict: dict = {
        "state.left_arm": states_ref[0:7][np.newaxis],
        "state.right_arm": states_ref[7:14][np.newaxis],
        "state.left_hand": states_ref[14:21][np.newaxis],
        "state.right_hand": states_ref[21:28][np.newaxis],
        "annotation.human.task_description": [task_description],
    }
    cam_map = {
        "front_camera": "video.room_view",
        "left_wrist_camera": "video.left_wrist_view",
        "right_wrist_camera": "video.right_wrist_view",
    }
    for cam_key, gr00t_key in cam_map.items():
        obs_dict[gr00t_key] = _get_camera_rgb(env, cam_key)[0][np.newaxis]

    return obs_dict


def _convert_policy_action_chunk_to_env(action_dict: dict) -> list[np.ndarray]:
    parts = []
    for key in ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]:
        value = np.asarray(action_dict[key], dtype=np.float32)
        if value.ndim == 3:
            value = value[0]
        elif value.ndim == 1:
            value = value[np.newaxis, :]
        parts.append(value)

    chunk_len = min(part.shape[0] for part in parts)
    action_chunk: list[np.ndarray] = []
    for step_idx in range(chunk_len):
        action_ref = np.concatenate([part[step_idx] for part in parts], axis=-1)
        action_internal = action_ref[INV_PERM]
        action = np.concatenate([np.zeros(ACTION_PREFIX_PAD, dtype=np.float32), action_internal], axis=0)
        action_chunk.append(_normalize_env_raw_action(action))
    return action_chunk


def _extract_task_complete(action_dict: dict) -> float | None:
    """Return the mean task_complete prediction over the action chunk, or None."""
    value = action_dict.get("action.task_complete")
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        arr = value.float().reshape(-1).cpu().numpy()
    else:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(arr.mean()) if arr.size > 0 else None


class KeyboardInterface:
    _TASK_KEY_MAP = {
        "1": 0, "KEY_1": 0, "NUMPAD_1": 0,
        "2": 1, "KEY_2": 1, "NUMPAD_2": 1,
        "3": 2, "KEY_3": 2, "NUMPAD_3": 2,
        "4": 3, "KEY_4": 3, "NUMPAD_4": 3,
        "5": 4, "KEY_5": 4, "NUMPAD_5": 4,
    }

    def __init__(self, stop_on_task_complete: bool, initial_task_idx: int = 0):
        self.running = False
        self.pending_single_step = False
        self.pending_task_complete_probe = False
        self.pending_reset = False
        self.pending_tray_rotation = False
        self.pending_tray_return = False
        self.selected_task_idx = initial_task_idx
        self.stop_on_task_complete = stop_on_task_complete
        self.last_message = f"Ready. Task {initial_task_idx + 1}: {TASK_DESCRIPTIONS[initial_task_idx]}. Press SPACE."
        self.last_key = "n/a"
        self._input = carb.input.acquire_input_interface()
        self._app_window = omni.appwindow.get_default_app_window()
        self._keyboard = None if self._app_window is None else self._app_window.get_keyboard()
        self._sub_keyboard = None
        if self._keyboard is not None:
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        else:
            self.last_message = "Keyboard unavailable. Use the on-screen buttons."

    def close(self):
        if self._sub_keyboard is not None and self._keyboard is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            self._sub_keyboard = None

    def consume_single_step(self) -> bool:
        v = self.pending_single_step
        self.pending_single_step = False
        return v

    def consume_task_complete_probe(self) -> bool:
        v = self.pending_task_complete_probe
        self.pending_task_complete_probe = False
        return v

    def consume_reset(self) -> bool:
        v = self.pending_reset
        self.pending_reset = False
        return v

    def consume_tray_rotation(self) -> bool:
        v = self.pending_tray_rotation
        self.pending_tray_rotation = False
        return v

    def consume_tray_return(self) -> bool:
        v = self.pending_tray_return
        self.pending_tray_return = False
        return v

    def toggle_running(self) -> None:
        self.running = not self.running
        self.last_message = f"Continuous inference {'running' if self.running else 'paused'}."

    def queue_reset(self) -> None:
        self.running = False
        self.pending_single_step = False
        self.pending_task_complete_probe = False
        self.pending_tray_rotation = False
        self.pending_tray_return = False
        self.pending_reset = True
        self.last_message = "Queued environment reset."

    def print_help(self) -> None:
        _print_controls()
        self.last_message = "Printed controls to the terminal."

    @staticmethod
    def _key_aliases(raw_key: str) -> set[str]:
        key = raw_key.upper()
        aliases = {key}
        for prefix in ("KEY_", "NUMPAD_", "NUMROW_", "KP_"):
            if key.startswith(prefix):
                aliases.add(key[len(prefix):])
        return aliases

    def _on_keyboard_event(self, event):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        raw_key = str(event.input.name)
        aliases = self._key_aliases(raw_key)
        self.last_key = raw_key

        for alias in aliases:
            if alias in self._TASK_KEY_MAP:
                idx = self._TASK_KEY_MAP[alias]
                self.selected_task_idx = idx
                self.last_message = f"Selected task {idx + 1}: {TASK_DESCRIPTIONS[idx]}"
                return True

        if aliases & {"SPACE", "SPACEBAR", "P"}:
            self.toggle_running()
        elif aliases & {"S", "ENTER"}:
            self.pending_single_step = True
            self.last_message = "Queued one policy step."
        elif aliases & {"G"}:
            self.pending_task_complete_probe = True
            self.last_message = "Queued a task_complete probe without env step."
        elif aliases & {"O"}:
            self.pending_tray_rotation = True
            self.pending_tray_return = False
            self.last_message = f"Queued tray rotation: {args.tray_yaw_increment_deg:.1f} deg."
        elif aliases & {"B"}:
            self.pending_tray_return = True
            self.pending_tray_rotation = False
            self.last_message = "Queued tray rotation back to reset yaw."
        elif aliases & {"R"}:
            self.queue_reset()
        elif aliases & {"H", "F1"}:
            self.print_help()
        return True


class StatusPanel:
    def __init__(self, keyboard: KeyboardInterface):
        self._keyboard = keyboard
        self._window = ui.Window(
            "Trocar Task-Complete Debug",
            width=440,
            height=420,
            visible=True,
            dock_preference=ui.DockPreference.DISABLED,
        )
        with self._window.frame:
            with ui.VStack(spacing=6, height=0):
                self._status = ui.Label("", word_wrap=True)
                ui.Spacer(height=8)
                ui.Label("Stage Task (1-5):", word_wrap=True)
                with ui.HStack(spacing=4):
                    for idx, desc in enumerate(TASK_DESCRIPTIONS):
                        ui.Button(
                            str(idx + 1),
                            width=30,
                            clicked_fn=lambda i=idx: setattr(keyboard, "selected_task_idx", i) or setattr(
                                keyboard, "last_message", f"Selected task {i + 1}: {TASK_DESCRIPTIONS[i]}"
                            ),
                            tooltip=desc,
                        )
                ui.Label("Controls:", word_wrap=True)
                with ui.HStack(spacing=4):
                    ui.Button("Start/Pause", width=0, clicked_fn=keyboard.toggle_running)
                    ui.Button("Step", width=0, clicked_fn=lambda: setattr(keyboard, "pending_single_step", True))
                    ui.Button(
                        "Probe TC",
                        width=0,
                        clicked_fn=lambda: setattr(keyboard, "pending_task_complete_probe", True),
                        tooltip="Probe task_complete without stepping (G key)",
                    )
                    ui.Button("Reset", width=0, clicked_fn=keyboard.queue_reset)
                    ui.Button("Help", width=0, clicked_fn=keyboard.print_help)
                with ui.HStack(spacing=4):
                    ui.Button(
                        f"Rotate {args.tray_yaw_increment_deg:.0f}deg",
                        width=0,
                        clicked_fn=lambda: setattr(keyboard, "pending_tray_rotation", True),
                    )
                    ui.Button(
                        "Rotate Back",
                        width=0,
                        clicked_fn=lambda: setattr(keyboard, "pending_tray_return", True),
                    )
                ui.Spacer(height=4)
                ui.Label(
                    "Keys: 1-5 select task  SPACE/P start|pause  S step  G probe TC  O rotate  B back  R reset",
                    word_wrap=True,
                )

    def update(
        self,
        keyboard: KeyboardInterface,
        task_complete_value: float | None,
        step_count: int,
        task_run_step_count: int,
        episode_done: bool,
        open_loop_steps_remaining: int,
        tray_yaw_deg: float,
        tray_rotation_active: bool,
        tray_rotation_progress: int,
        tray_rotation_total: int,
        last_policy_ms: float | None,
        last_env_step_ms: float | None,
    ):
        mode = "RUNNING" if keyboard.running else "PAUSED"
        if episode_done:
            mode += " / EPISODE DONE"
        if tray_rotation_active:
            mode += " / TRAY ROTATING"

        tc_text = "n/a" if task_complete_value is None else f"{task_complete_value:.4f}"
        tc_above = (
            "n/a"
            if task_complete_value is None
            else str(task_complete_value >= args.task_complete_threshold)
        )
        policy_ms = "n/a" if last_policy_ms is None else f"{last_policy_ms:.0f}"
        env_ms = "n/a" if last_env_step_ms is None else f"{last_env_step_ms:.0f}"
        task_desc = TASK_DESCRIPTIONS[keyboard.selected_task_idx]

        self._status.text = (
            f"Mode: {mode}\n"
            f"Task {keyboard.selected_task_idx + 1}: {task_desc}\n"
            f"task_complete: {tc_text}  (>= {args.task_complete_threshold:.2f}: {tc_above})\n"
            f"Auto-stop on TC: {keyboard.stop_on_task_complete}\n"
            f"Task run steps: {task_run_step_count} / {args.task_timeout_steps}\n"
            f"Open-loop steps remaining: {open_loop_steps_remaining} / {args.open_loop_steps}\n"
            f"Tray yaw: {tray_yaw_deg:.2f} deg\n"
            f"Tray rotation steps: {tray_rotation_progress} / {tray_rotation_total}\n"
            f"Timing ms: policy={policy_ms}  env_step={env_ms}\n"
            f"Episode steps: {step_count}\n"
            f"Last event: {keyboard.last_message}"
        )


def main():
    model_device = args.model_device or args.device
    policy = _load_policy(args.model_path, model_device=model_device, denoising_steps=args.denoising_steps)
    env = _create_env(args.task_id, device=args.device)
    rng = np.random.default_rng(args.seed)

    fixed_initial_state_ref = None
    fixed_initial_raw_action = None
    if args.fixed_initial_state_dataset:
        fixed_initial_state_ref = _load_fixed_initial_state_ref(
            args.fixed_initial_state_dataset,
            args.fixed_initial_state_episode,
            args.fixed_initial_state_frame,
        )
        fixed_initial_raw_action = _fixed_initial_state_ref_to_raw_action(env, fixed_initial_state_ref)
        print(
            f"[INFO] Fixed initial state: dataset={args.fixed_initial_state_dataset}, "
            f"episode={args.fixed_initial_state_episode}, frame={args.fixed_initial_state_frame}"
        )

    keyboard = KeyboardInterface(
        stop_on_task_complete=args.enable_task_complete_stop,
        initial_task_idx=args.initial_task,
    )
    panel = StatusPanel(keyboard)
    _print_controls()

    tray_yaw_deg = _sample_initial_tray_yaw_deg(rng)
    _reset_env(
        env, seed=args.seed, tray_yaw_deg=tray_yaw_deg,
        fixed_raw_action=fixed_initial_raw_action, fixed_state_ref=fixed_initial_state_ref,
    )

    step_count = 0
    task_run_step_count = 0
    episode_done = False
    task_complete_value: float | None = None
    previous_task_idx = keyboard.selected_task_idx
    last_policy_ms: float | None = None
    last_env_step_ms: float | None = None
    step_period = 1.0 / max(args.step_hz, 1e-6)
    last_step_time = 0.0
    cached_policy_actions: list[np.ndarray] = []
    open_loop_steps_remaining = 0
    tray_base_yaw_deg = tray_yaw_deg
    tray_rotation_active = False
    tray_rotation_start_yaw = tray_yaw_deg
    tray_rotation_target_yaw = tray_yaw_deg
    tray_rotation_step_count = 0
    tray_rotation_total_steps = max(args.tray_yaw_steps, 1)

    try:
        while simulation_app.is_running():
            # --- Task switch: clear stale task_complete value ---
            if keyboard.selected_task_idx != previous_task_idx:
                previous_task_idx = keyboard.selected_task_idx
                task_complete_value = None
                task_run_step_count = 0
                cached_policy_actions = []
                open_loop_steps_remaining = 0

            # --- Reset ---
            if keyboard.consume_reset():
                tray_yaw_deg = _sample_initial_tray_yaw_deg(rng)
                _reset_env(
                    env, seed=args.seed, tray_yaw_deg=tray_yaw_deg,
                    fixed_raw_action=fixed_initial_raw_action, fixed_state_ref=fixed_initial_state_ref,
                )
                step_count = 0
                task_run_step_count = 0
                episode_done = False
                task_complete_value = None
                cached_policy_actions = []
                open_loop_steps_remaining = 0
                tray_base_yaw_deg = tray_yaw_deg
                tray_rotation_active = False
                tray_rotation_start_yaw = tray_yaw_deg
                tray_rotation_target_yaw = tray_yaw_deg
                tray_rotation_step_count = 0
                keyboard.last_message = f"Reset done. Tray yaw={tray_yaw_deg:.2f} deg."

            # --- Tray rotation ---
            if keyboard.consume_tray_rotation():
                if episode_done:
                    keyboard.last_message = "Episode done. Press R to reset first."
                else:
                    tray_rotation_active = True
                    tray_rotation_start_yaw = tray_yaw_deg
                    tray_rotation_target_yaw = tray_base_yaw_deg + args.tray_yaw_increment_deg
                    tray_rotation_step_count = 0
                    tray_rotation_total_steps = max(args.tray_yaw_steps, 1)
                    keyboard.last_message = (
                        f"Tray rotating to {tray_rotation_target_yaw:.2f} deg "
                        f"over {tray_rotation_total_steps} steps."
                    )

            if keyboard.consume_tray_return():
                if episode_done:
                    keyboard.last_message = "Episode done. Press R to reset first."
                else:
                    tray_rotation_active = True
                    tray_rotation_start_yaw = tray_yaw_deg
                    tray_rotation_target_yaw = tray_base_yaw_deg
                    tray_rotation_step_count = 0
                    tray_rotation_total_steps = max(args.tray_yaw_steps, 1)
                    keyboard.last_message = (
                        f"Tray rotating back to {tray_rotation_target_yaw:.2f} deg "
                        f"over {tray_rotation_total_steps} steps."
                    )

            # --- task_complete probe (no env step) ---
            if keyboard.consume_task_complete_probe():
                if episode_done:
                    keyboard.last_message = "Episode done. Press R to reset first."
                else:
                    task_desc = TASK_DESCRIPTIONS[keyboard.selected_task_idx]
                    with torch.inference_mode():
                        t0 = time.perf_counter()
                        policy_obs = _build_policy_obs(env, task_desc)
                        action_dict = policy.get_action(policy_obs)
                        last_policy_ms = (time.perf_counter() - t0) * 1000.0
                    task_complete_value = _extract_task_complete(action_dict)
                    keyboard.last_message = f"task_complete probe ({task_desc}): {task_complete_value}"

            # --- Decide whether to step ---
            policy_step_requested = False
            if not episode_done and keyboard.consume_single_step():
                policy_step_requested = True
            if not episode_done and keyboard.running:
                now = time.perf_counter()
                if now - last_step_time >= step_period:
                    policy_step_requested = True
                    last_step_time = now

            tray_hold_step_requested = False
            if not episode_done and tray_rotation_active and not policy_step_requested:
                now = time.perf_counter()
                if now - last_step_time >= step_period:
                    tray_hold_step_requested = True
                    last_step_time = now

            should_step = policy_step_requested or tray_hold_step_requested

            # --- Build action ---
            action_tensor: torch.Tensor | None = None
            executing_policy_step = False

            if should_step:
                if policy_step_requested:
                    executing_policy_step = True
                    if not cached_policy_actions:
                        task_desc = TASK_DESCRIPTIONS[keyboard.selected_task_idx]
                        with torch.inference_mode():
                            t0 = time.perf_counter()
                            policy_obs = _build_policy_obs(env, task_desc)
                            action_dict = policy.get_action(policy_obs)
                            last_policy_ms = (time.perf_counter() - t0) * 1000.0
                        task_complete_value = _extract_task_complete(action_dict)
                        action_chunk = _convert_policy_action_chunk_to_env(action_dict)
                        if not action_chunk:
                            raise RuntimeError("Policy returned an empty action chunk.")
                        cached_policy_actions = [a.copy() for a in action_chunk[: max(args.open_loop_steps, 1)]]
                        open_loop_steps_remaining = len(cached_policy_actions)
                    action_np = cached_policy_actions.pop(0)
                    action_tensor = _raw_action_to_env_tensor(action_np, device=env.device)
                else:
                    # tray hold step — hold current joint positions
                    action_term = env.action_manager.get_term(env.action_manager.active_terms[0])
                    joint_pos = wp.to_torch(env.scene["robot"].data.joint_pos)
                    joint_ids = action_term._joint_ids
                    if isinstance(joint_ids, slice):
                        joint_ids = torch.arange(joint_pos.shape[1], device=joint_pos.device)[joint_ids]
                    else:
                        joint_ids = torch.as_tensor(joint_ids, device=joint_pos.device, dtype=torch.long)
                    scale = action_term._scale
                    if isinstance(scale, torch.Tensor):
                        scale = scale.to(device=joint_pos.device)
                    else:
                        scale = torch.full((action_term.action_dim,), float(scale), device=joint_pos.device)
                    offset = action_term._offset
                    if isinstance(offset, torch.Tensor):
                        offset = offset.to(device=joint_pos.device)
                    else:
                        offset = torch.full((action_term.action_dim,), float(offset), device=joint_pos.device)
                    raw = ((joint_pos[0, joint_ids].float() - offset) / scale).cpu().numpy()
                    action_tensor = _raw_action_to_env_tensor(raw, device=env.device)

            # --- Execute step ---
            if should_step:
                if action_tensor is None:
                    raise RuntimeError("Expected an action tensor before stepping.")

                if tray_rotation_active:
                    prev_yaw = tray_yaw_deg
                    tray_rotation_step_count += 1
                    alpha = tray_rotation_step_count / max(tray_rotation_total_steps, 1)
                    tray_yaw_deg = (1.0 - alpha) * tray_rotation_start_yaw + alpha * tray_rotation_target_yaw
                    _set_tray_trocar_yaw_delta(
                        env, tray_yaw_deg, previous_yaw_deg=prev_yaw, carry_trocars_on_tray_only=True
                    )
                    if tray_rotation_step_count >= tray_rotation_total_steps:
                        tray_rotation_active = False
                        keyboard.last_message = f"Tray rotation done. yaw={tray_yaw_deg:.2f} deg."

                t0 = time.perf_counter()
                _, _, terminated, truncated, _ = env.step(action_tensor)
                last_env_step_ms = (time.perf_counter() - t0) * 1000.0
                step_count += 1
                episode_done = bool(terminated[0]) or bool(truncated[0])

                if executing_policy_step:
                    open_loop_steps_remaining = len(cached_policy_actions)
                    if keyboard.running:
                        task_run_step_count += 1

                if episode_done:
                    keyboard.running = False
                    cached_policy_actions = []
                    open_loop_steps_remaining = 0
                    tray_rotation_active = False
                    reason = "terminated" if bool(terminated[0]) else "truncated"
                    keyboard.last_message = f"Episode finished ({reason}). Press R to reset."
                elif (
                    executing_policy_step
                    and keyboard.running
                    and keyboard.stop_on_task_complete
                    and task_complete_value is not None
                    and task_complete_value >= args.task_complete_threshold
                ):
                    keyboard.running = False
                    keyboard.last_message = (
                        f"Paused: task_complete={task_complete_value:.4f} "
                        f">= threshold {args.task_complete_threshold:.2f}."
                    )
                elif keyboard.running and task_run_step_count >= args.task_timeout_steps:
                    keyboard.running = False
                    keyboard.last_message = (
                        f"Paused by timeout after {task_run_step_count} steps "
                        f"(limit {args.task_timeout_steps})."
                    )
            else:
                env.sim.render()
                time.sleep(1.0 / 60.0)

            panel.update(
                keyboard=keyboard,
                task_complete_value=task_complete_value,
                step_count=step_count,
                task_run_step_count=task_run_step_count,
                episode_done=episode_done,
                open_loop_steps_remaining=open_loop_steps_remaining,
                tray_yaw_deg=tray_yaw_deg,
                tray_rotation_active=tray_rotation_active,
                tray_rotation_progress=tray_rotation_step_count,
                tray_rotation_total=tray_rotation_total_steps,
                last_policy_ms=last_policy_ms,
                last_env_step_ms=last_env_step_ms,
            )
    finally:
        keyboard.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
