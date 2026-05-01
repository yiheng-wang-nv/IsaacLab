# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive GUI runner for the multi-task trocar GR00T policy.

This script opens Isaac Sim with the trocar environment and lets the user:

* press ``1``-``5`` to select one of the split-stage task prompts,
* press ``SPACE`` to toggle continuous policy inference,
* press ``S`` to execute a single policy step,
* press ``G`` to probe stage prediction without stepping the environment,
* press ``R`` to reset the environment.

It is intended for debugging multi-task prompt switching and stage transitions
with a finetuned Isaac-GR00T checkpoint. Action inference uses the selected
sub-task prompt; stage prediction can optionally be probed with a fixed global
prompt so the classifier is less directly tied to the selected action prompt.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run an interactive GUI for the multi-task trocar policy.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the Isaac-GR00T checkpoint directory.")
parser.add_argument(
    "--task_id",
    type=str,
    default="Isaac-Assemble-Trocar-G129-Dex3-RLinf-v0",
    help="Gym task id for the trocar environment.",
)
parser.add_argument("--denoising_steps", type=int, default=4, help="GR00T denoising steps.")
parser.add_argument("--step_hz", type=float, default=10.0, help="Continuous inference stepping rate.")
parser.add_argument(
    "--open_loop_steps",
    type=int,
    default=1,
    help="Number of env steps to execute before running policy inference again.",
)
parser.add_argument(
    "--task_timeout_steps",
    type=int,
    default=60,
    help="Pause continuous inference when the selected task has executed this many steps.",
)
parser.add_argument(
    "--model_device",
    type=str,
    default=None,
    help="Device for the policy. Defaults to the simulator device when omitted.",
)
parser.add_argument(
    "--stage_prompt",
    type=str,
    default="",
    help=(
        "Optional fixed prompt used for the stage-classifier probe. Leave empty "
        "to reuse the selected task forward's stage logits."
    ),
)
parser.add_argument(
    "--disable_stage_probe",
    action="store_true",
    help="Use the selected-task action forward's stage logits instead of running a fixed-prompt stage probe.",
)
parser.add_argument(
    "--enable_stage_pred_stop",
    action="store_true",
    default=True,
    help="Start with auto-pause enabled when the stage probe predicts the next selected task stage.",
)
parser.add_argument(
    "--disable_stage_pred_stop",
    action="store_false",
    dest="enable_stage_pred_stop",
    help="Start with auto-pause disabled even if the stage probe predicts the next selected task stage.",
)
parser.add_argument(
    "--stage_pred_min_confidence",
    type=float,
    default=0.9,
    help="Minimum argmax softmax probability required for stage-pred auto-pause.",
)
parser.add_argument("--seed", type=int, default=None, help="Optional environment reset seed.")
parser.add_argument(
    "--fixed_initial_state_dataset",
    type=str,
    default=None,
    help="Optional LeRobot dataset directory. If set, episode/frame observation.state is used as a fixed start pose.",
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
    "--initial_tray_yaw_min_deg",
    type=float,
    default=0.0,
    help="Minimum random tray yaw delta [deg] applied after every reset.",
)
parser.add_argument(
    "--initial_tray_yaw_max_deg",
    type=float,
    default=5.0,
    help="Maximum random tray yaw delta [deg] applied after every reset.",
)
parser.add_argument(
    "--tray_yaw_increment_deg",
    type=float,
    default=-30.0,
    help="Tray yaw delta [deg] applied by the interactive rotate button relative to the reset yaw.",
)
parser.add_argument(
    "--tray_yaw_steps",
    type=int,
    default=5,
    help="Number of env steps used to interpolate each interactive tray rotation.",
)
parser.add_argument(
    "--tray_carry_xy_margin",
    type=float,
    default=0.10,
    help="Extra XY margin [m] for deciding whether a trocar is still on the tray during tray motion.",
)
parser.add_argument(
    "--tray_carry_z_tolerance",
    type=float,
    default=0.04,
    help="Z-offset tolerance [m] for deciding whether a trocar is still on the tray during tray motion.",
)
parser.add_argument(
    "--retry_return_steps",
    type=int,
    default=10,
    help="Number of env steps used to interpolate back to the selected task's recorded start action.",
)
parser.add_argument(
    "--reason2_url",
    type=str,
    default="http://localhost:10086/v1/chat/completions",
    help="OpenAI-compatible Cosmos Reason2 chat completions endpoint.",
)
parser.add_argument("--reason2_model", type=str, default="nvidia/Cosmos-Reason2-2B", help="Reason2 model id.")
parser.add_argument(
    "--reason2_query_dir",
    type=str,
    default="/tmp/cosmos_reason2_queries",
    help="Directory for current-frame PNGs sent to Reason2. Must be allowed by the vLLM server.",
)
parser.add_argument("--reason2_timeout_s", type=float, default=60.0, help="Reason2 HTTP timeout [s].")
parser.add_argument("--reason2_max_tokens", type=int, default=256, help="Max Reason2 output tokens.")
parser.add_argument("--reason2_temperature", type=float, default=0.0, help="Reason2 sampling temperature.")
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
        print(f"[DEBUG] XAUTHORITY was unset. Using {default_xauthority}.")

# IsaacLab's AppLauncher will still force headless unless a Kit visualizer is
# explicitly requested. Default to `kit` unless the user already provided a
# visualizer selection on the CLI.
if getattr(args, "visualizer", None) in (None, []):
    args.visualizer = ["kit"]
    args.visualizer_explicit = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("[DEBUG] DISPLAY =", os.environ.get("DISPLAY"))
print("[DEBUG] visualizer =", getattr(args, "visualizer", None))
print("[DEBUG] AppLauncher._headless =", getattr(app_launcher, "_headless", None))
print("[DEBUG] AppLauncher._sim_experience_file =", getattr(app_launcher, "_sim_experience_file", None))


import carb
import carb.input
import gymnasium as gym
import numpy as np
import omni.appwindow
import omni.kit.app
import omni.ui as ui
import requests
import torch
import warp as wp

import cv2

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

PERM_TO_REF = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    16,
    19,
    20,
    15,
    18,
    14,
    17,
    23,
    26,
    27,
    21,
    24,
    22,
    25,
]
INV_PERM = np.argsort(PERM_TO_REF)
BODY_JOINT_INDICES = [
    0,
    3,
    6,
    9,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
]
DEX3_JOINT_INDICES = [31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38]
SHOULDER_SLICE = (15, 29)
ACTION_PREFIX_PAD = 15
ROBOT_ACTION_DIM = ACTION_PREFIX_PAD + len(INV_PERM)
LEFT_HAND_ACTION_INDICES = ACTION_PREFIX_PAD + np.asarray(PERM_TO_REF[14:21], dtype=np.int64)
RIGHT_HAND_ACTION_INDICES = ACTION_PREFIX_PAD + np.asarray(PERM_TO_REF[21:28], dtype=np.int64)


def _print_controls() -> None:
    print("Interactive trocar controls:")
    print("  1-5 : select task prompt")
    print("  SPACE / P : toggle continuous inference")
    print("  S : run one policy step")
    print("  G : probe stage prediction without stepping")
    print("  C : ask Cosmos Reason2 which hands hold trocar")
    print("  Y : retry the selected task from its recorded start action")
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
    raise FileNotFoundError(
        "Could not locate Isaac-GR00T root. Set the checkpoint path inside the Isaac-GR00T repo "
        "or update `_find_isaac_gr00t_root`."
    )


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
    data_config = module.UnitreeG1SimInferDataConfig()

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
    fixed_raw_action: np.ndarray | None = None,
    fixed_state_ref: np.ndarray | None = None,
    tray_yaw_deg: float | None = None,
):
    obs, _ = env.reset(seed=seed)
    if tray_yaw_deg is not None:
        _set_tray_trocar_yaw_delta(env, tray_yaw_deg)
    if fixed_raw_action is not None and fixed_state_ref is not None:
        obs, fixed_info = _apply_fixed_initial_state(
            env,
            obs,
            fixed_raw_action,
            fixed_state_ref,
            args.fixed_initial_state_steps,
            args.fixed_initial_state_tolerance,
        )
        print(
            "[INFO] Fixed initial state warm-start: "
            f"steps={fixed_info['steps']}, max_state_error={fixed_info['max_state_error']:.4f}"
        )
    obs = _flush_observations(env)
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


def _load_fixed_initial_state_ref(dataset_dir: str, episode_idx: int, frame_idx: int) -> np.ndarray:
    """Load one 28-DoF reference-order state from a LeRobot episode parquet."""
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
        raise IndexError(f"Frame {frame_idx} out of range for {episode_path} with {len(df)} frames.")

    state_ref = np.asarray(df["observation.state"].iloc[frame_idx], dtype=np.float32)
    if state_ref.shape != (len(PERM_TO_REF),):
        raise ValueError(f"Expected observation.state shape {(len(PERM_TO_REF),)}, got {state_ref.shape}.")
    return state_ref


def _get_joint_pos_action_term(env):
    if "joint_pos" in env.action_manager.active_terms:
        return env.action_manager.get_term("joint_pos")
    if len(env.action_manager.active_terms) == 1:
        return env.action_manager.get_term(env.action_manager.active_terms[0])
    raise RuntimeError(f"Could not identify joint position action term from {env.action_manager.active_terms}.")


def _get_action_joint_ids(env) -> torch.Tensor:
    action_term = _get_joint_pos_action_term(env)
    joint_pos = wp.to_torch(env.scene["robot"].data.joint_pos)
    joint_ids = action_term._joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = torch.arange(joint_pos.shape[1], device=joint_pos.device, dtype=torch.long)[joint_ids]
    else:
        joint_ids = torch.as_tensor(joint_ids, device=joint_pos.device, dtype=torch.long)
    if joint_ids.numel() != action_term.action_dim:
        raise RuntimeError(
            f"Action joint id count {joint_ids.numel()} does not match action dim {action_term.action_dim}."
        )
    return joint_ids


def _as_action_vector(value: torch.Tensor | float, action_dim: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        value = value.to(device=device, dtype=torch.float32)
        if value.ndim == 2:
            value = value[0]
        return value.reshape(action_dim)
    return torch.full((action_dim,), float(value), device=device, dtype=torch.float32)


def _get_env_action_state(env) -> np.ndarray:
    """Return the raw env action that holds the robot at current joint positions."""
    action_term = _get_joint_pos_action_term(env)
    joint_pos = wp.to_torch(env.scene["robot"].data.joint_pos)
    joint_ids = _get_action_joint_ids(env)
    joint_pos_action_order = joint_pos[0, joint_ids].to(dtype=torch.float32)

    scale = _as_action_vector(action_term._scale, action_term.action_dim, joint_pos.device)
    offset = _as_action_vector(action_term._offset, action_term.action_dim, joint_pos.device)
    raw_action = (joint_pos_action_order - offset) / scale
    return _normalize_env_raw_action(raw_action.cpu().numpy())


def _fixed_initial_state_ref_to_raw_action(env, state_ref: np.ndarray) -> np.ndarray:
    """Convert a reference-order joint position state to the env raw action space."""
    action_term = _get_joint_pos_action_term(env)
    if action_term.action_dim != ROBOT_ACTION_DIM:
        raise RuntimeError(f"Expected action dim {ROBOT_ACTION_DIM}, got {action_term.action_dim}.")

    target_joint_pos = np.zeros(ROBOT_ACTION_DIM, dtype=np.float32)
    target_joint_pos[ACTION_PREFIX_PAD:] = np.asarray(state_ref, dtype=np.float32)[INV_PERM]

    device = torch.device(env.device)
    target = torch.tensor(target_joint_pos, dtype=torch.float32, device=device)
    scale = _as_action_vector(action_term._scale, action_term.action_dim, device)
    offset = _as_action_vector(action_term._offset, action_term.action_dim, device)
    raw_action = ((target - offset) / scale).cpu().numpy().astype(np.float32)
    raw_action[:ACTION_PREFIX_PAD] = 0.0
    return raw_action


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
    """Heuristically decide whether each trocar is still resting on the tray."""
    current_xy_dist = torch.linalg.norm(trocar_state[:, :2] - tray_state[:, :2], dim=-1)
    default_xy_dist = torch.linalg.norm(trocar_default[:, :2] - tray_default[:, :2], dim=-1)
    current_z_offset = trocar_state[:, 2] - tray_state[:, 2]
    default_z_offset = trocar_default[:, 2] - tray_default[:, 2]
    xy_close = current_xy_dist <= default_xy_dist + args.tray_carry_xy_margin
    z_close = torch.abs(current_z_offset - default_z_offset) <= args.tray_carry_z_tolerance
    return xy_close & z_close


def _event_mask_value(mask: torch.Tensor) -> bool | list[bool]:
    values = [bool(value) for value in mask.detach().cpu().tolist()]
    return values[0] if len(values) == 1 else values


def _write_carried_trocar_pose(
    env_ids: torch.Tensor,
    tray_center: torch.Tensor,
    step_quat: torch.Tensor,
    zero_velocity: torch.Tensor,
    trocar,
    trocar_current: torch.Tensor,
    carried_mask: torch.Tensor,
) -> None:
    if not bool(carried_mask.any().item()):
        return

    trocar_new = trocar_current.clone()
    trocar_new[:, :3] = tray_center + _quat_apply_xyzw(step_quat, trocar_current[:, :3] - tray_center)
    trocar_new[:, 3:7] = _quat_mul_xyzw(step_quat, trocar_current[:, 3:7])

    carried_env_ids = env_ids[carried_mask]
    trocar.write_root_pose_to_sim_index(root_pose=trocar_new[carried_mask, :7], env_ids=carried_env_ids)
    trocar.write_root_velocity_to_sim_index(root_velocity=zero_velocity[carried_mask], env_ids=carried_env_ids)


def _set_tray_trocar_yaw_delta(
    env,
    yaw_deg: float,
    previous_yaw_deg: float | None = None,
    carry_trocars_on_tray_only: bool = False,
) -> dict[str, bool | list[bool]]:
    """Set tray yaw and optionally carry only trocars that are still on the tray."""
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
    tray_current = None
    trocar_1_current = None
    trocar_2_current = None
    if carry_trocars_on_tray_only and previous_yaw_deg is not None:
        tray_current = wp.to_torch(tray.data.root_state_w)[env_ids].clone()
        trocar_1_current = wp.to_torch(trocar_1.data.root_state_w)[env_ids].clone()
        trocar_2_current = wp.to_torch(trocar_2.data.root_state_w)[env_ids].clone()

    tray_new = tray_default.clone()
    tray_new[:, 3:7] = _quat_mul_xyzw(target_quat, tray_default[:, 3:7])

    zero_velocity = torch.zeros(len(env_ids), 6, device=env.device)
    tray.write_root_pose_to_sim_index(root_pose=tray_new[:, :7], env_ids=env_ids)
    tray.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_ids)

    if not carry_trocars_on_tray_only or previous_yaw_deg is None:
        trocar_1_new = trocar_1_default.clone()
        trocar_2_new = trocar_2_default.clone()
        trocar_1_new[:, :3] = tray_center + _quat_apply_xyzw(target_quat, trocar_1_default[:, :3] - tray_center)
        trocar_2_new[:, :3] = tray_center + _quat_apply_xyzw(target_quat, trocar_2_default[:, :3] - tray_center)
        trocar_1_new[:, 3:7] = _quat_mul_xyzw(target_quat, trocar_1_default[:, 3:7])
        trocar_2_new[:, 3:7] = _quat_mul_xyzw(target_quat, trocar_2_default[:, 3:7])
        trocar_1.write_root_pose_to_sim_index(root_pose=trocar_1_new[:, :7], env_ids=env_ids)
        trocar_2.write_root_pose_to_sim_index(root_pose=trocar_2_new[:, :7], env_ids=env_ids)
        trocar_1.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_ids)
        trocar_2.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_ids)
        all_carried = torch.ones(len(env_ids), device=env.device, dtype=torch.bool)
        return {"trocar_1_carried": _event_mask_value(all_carried), "trocar_2_carried": _event_mask_value(all_carried)}

    assert tray_current is not None
    assert trocar_1_current is not None
    assert trocar_2_current is not None
    trocar_1_carried = _trocar_on_tray_mask(tray_current, trocar_1_current, tray_default, trocar_1_default)
    trocar_2_carried = _trocar_on_tray_mask(tray_current, trocar_2_current, tray_default, trocar_2_default)
    step_quat = _yaw_quat_xyzw(env, yaw_deg - previous_yaw_deg, len(env_ids))
    _write_carried_trocar_pose(
        env_ids, tray_center, step_quat, zero_velocity, trocar_1, trocar_1_current, trocar_1_carried
    )
    _write_carried_trocar_pose(
        env_ids, tray_center, step_quat, zero_velocity, trocar_2, trocar_2_current, trocar_2_carried
    )
    return {
        "trocar_1_carried": _event_mask_value(trocar_1_carried),
        "trocar_2_carried": _event_mask_value(trocar_2_carried),
    }


def _apply_retry_hand_hold(
    action: np.ndarray,
    hold_action: np.ndarray,
    hold_left_hand: bool,
    hold_right_hand: bool,
) -> np.ndarray:
    masked_action = _normalize_env_raw_action(action)
    hold_action = _normalize_env_raw_action(hold_action)
    if hold_left_hand:
        masked_action[LEFT_HAND_ACTION_INDICES] = hold_action[LEFT_HAND_ACTION_INDICES]
    if hold_right_hand:
        masked_action[RIGHT_HAND_ACTION_INDICES] = hold_action[RIGHT_HAND_ACTION_INDICES]
    return masked_action


def _fixed_initial_state_error(env, target_state_ref: np.ndarray) -> float:
    states_ref = _get_joint_states(env)[:, PERM_TO_REF]
    error = np.abs(states_ref - target_state_ref[np.newaxis, :])
    return float(error.max()) if error.size else 0.0


def _apply_fixed_initial_state(
    env,
    obs,
    fixed_raw_action: np.ndarray,
    target_state_ref: np.ndarray,
    steps: int,
    tolerance: float,
) -> tuple[dict, dict]:
    action_batch = fixed_raw_action.reshape(1, ROBOT_ACTION_DIM)
    action_tensor = torch.tensor(action_batch, dtype=torch.float32, device=env.device)
    steps_run = 0
    max_state_error = _fixed_initial_state_error(env, target_state_ref)

    for warm_step in range(max(steps, 0)):
        obs, _, _, _, _ = env.step(action_tensor)
        steps_run = warm_step + 1
        max_state_error = _fixed_initial_state_error(env, target_state_ref)
        if max_state_error <= tolerance:
            break

    return obs, {
        "steps": steps_run,
        "max_state_error": max_state_error,
    }


def _get_camera_rgb(env, cam_name: str) -> np.ndarray:
    sensor = env.scene.sensors[cam_name]
    imgs = sensor.data.output["rgb"]
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().numpy()
    if imgs.shape[-1] == 4:
        imgs = imgs[..., :3]
    return imgs.astype(np.uint8)


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"No JSON object found in Reason2 response: {text!r}")
    return json.loads(text[start : end + 1])


def _save_reason2_query_images(env, task_idx: int, step_count: int) -> dict[str, str]:
    query_dir = (
        Path(args.reason2_query_dir).expanduser().resolve()
        / "interactive"
        / f"task_{task_idx + 1}_step_{step_count}"
    )
    query_dir.mkdir(parents=True, exist_ok=True)

    rgb_images: dict[str, np.ndarray] = {}
    image_paths: dict[str, str] = {}
    for cam_name in CAMERA_KEYS:
        rgb = _get_camera_rgb(env, cam_name)[0]
        if cam_name == "left_wrist_camera":
            rgb = cv2.rotate(rgb, cv2.ROTATE_180)
        rgb_images[cam_name] = rgb
        path = query_dir / f"{cam_name}.png"
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        image_paths[cam_name] = str(path)

    query_images = {
        "left_query": np.concatenate([rgb_images["front_camera"], rgb_images["left_wrist_camera"]], axis=1),
        "right_query": np.concatenate([rgb_images["front_camera"], rgb_images["right_wrist_camera"]], axis=1),
    }
    for query_name, query_rgb in query_images.items():
        path = query_dir / f"{query_name}.png"
        cv2.imwrite(str(path), cv2.cvtColor(query_rgb, cv2.COLOR_RGB2BGR))
        image_paths[query_name] = str(path)
    return image_paths


def _build_reason2_single_hand_prompt(hand: str, task_idx: int) -> str:
    return f"""You are judging an Isaac Sim trocar assembly scene from one horizontally concatenated image.

The image has two equal-width panels from left to right:
1. front room camera,
2. {hand} wrist camera attached to the robot {hand} hand.
Current selected task: Task {task_idx + 1} - {TASK_DESCRIPTIONS[task_idx]}.

The target object is called a trocar in this task. Visually, a trocar may look like a purple rod,
a purple pencil-like shaft, or a purple-and-white tool with a white cylindrical handle/collar.
If a robot hand is gripping this purple/white tool, count that hand as holding a trocar even if
you would normally describe the object as a pencil-like rod or white cylinder.

Your goal is NOT to judge whether the grasp is perfect. Your goal is to decide whether the robot
{hand} hand should keep its fingers fixed/closed during rollback to avoid dropping or disturbing a trocar/tool.
Answer true if the {hand} hand is holding, pinching, supporting, touching, or partially grasping any
purple/white trocar/tool such that opening or moving the fingers could drop, release, or disturb the tool.
Answer false only if the {hand} hand is clearly empty, merely far from the tool, or the tool is clearly
resting on the tray without finger contact. Being near a trocar, hovering above it, poised to grasp it,
or about to interact with it is false unless there is visible finger contact or visible support of the
trocar/tool.

Return only valid JSON with this exact schema:
{{
  "{hand}_hand_holding_trocar": true,
  "confidence": 0.0,
  "reason": "short visual explanation"
}}
"""


def _reason2_hand_plan(task_idx: int) -> tuple[dict[str, bool | None], list[str], str]:
    return {"left": None, "right": None}, ["left", "right"], "query_both_hands"


def _query_reason2_hand_status(env, task_idx: int, step_count: int) -> dict:
    hand_priors, query_hands, hand_plan = _reason2_hand_plan(task_idx)
    hand_status = {
        "left_hand_holding_trocar": hand_priors["left"],
        "right_hand_holding_trocar": hand_priors["right"],
    }
    if not query_hands:
        return {
            "ok": True,
            "source": "task_prior",
            "hand_plan": hand_plan,
            "queried_hands": [],
            "left_hand_holding_trocar": bool(hand_status["left_hand_holding_trocar"]),
            "right_hand_holding_trocar": bool(hand_status["right_hand_holding_trocar"]),
            "confidence": 1.0,
            "reason": "Skipped Cosmos query because task prior already marks both hands as protected.",
        }

    image_paths = _save_reason2_query_images(env, task_idx, step_count)
    raw_responses = {}
    confidences = {}
    reasons = {}
    for hand in query_hands:
        query_key = f"{hand}_query"
        content = [
            {"type": "image_url", "image_url": {"url": f"file://{image_paths[query_key]}"}},
            {"type": "text", "text": _build_reason2_single_hand_prompt(hand, task_idx)},
        ]
        payload = {
            "model": args.reason2_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a careful visual state classifier for a robot manipulation retry controller. "
                        "Return only the requested JSON object."
                    ),
                },
                {"role": "user", "content": content},
            ],
            "temperature": args.reason2_temperature,
            "max_tokens": args.reason2_max_tokens,
        }
        response = requests.post(args.reason2_url, json=payload, timeout=args.reason2_timeout_s)
        response.raise_for_status()
        content_text = response.json()["choices"][0]["message"].get("content") or ""
        parsed = _extract_json_object(content_text)
        raw_responses[hand] = content_text
        hand_status[f"{hand}_hand_holding_trocar"] = bool(parsed.get(f"{hand}_hand_holding_trocar", False))
        confidences[hand] = float(parsed.get("confidence", 0.0))
        reasons[hand] = str(parsed.get("reason", ""))

    return {
        "ok": True,
        "source": "cosmos_reason2_single_hand_selective",
        "hand_plan": hand_plan,
        "queried_hands": query_hands,
        "left_hand_holding_trocar": bool(hand_status["left_hand_holding_trocar"]),
        "right_hand_holding_trocar": bool(hand_status["right_hand_holding_trocar"]),
        "confidence": min(confidences.values()) if confidences else 1.0,
        "confidences": confidences,
        "reasons": reasons,
        "image_paths": image_paths,
        "raw_response": raw_responses,
    }


def _build_policy_obs(env, task_description: str) -> dict[str, np.ndarray | list[str]]:
    states_internal = _get_joint_states(env)[0]
    states_ref = states_internal[PERM_TO_REF]

    obs_dict: dict[str, np.ndarray | list[str]] = {
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


def _extract_stage_prediction(action_dict: dict) -> tuple[int | None, list[float] | None]:
    stage_pred = action_dict.get("stage_pred")
    stage_logits = action_dict.get("stage_logits")

    stage_pred_value: int | None = None
    if stage_pred is not None:
        if isinstance(stage_pred, torch.Tensor):
            stage_pred_value = int(stage_pred.reshape(-1)[0].item())
        else:
            stage_pred_value = int(np.asarray(stage_pred).reshape(-1)[0])

    stage_prob_values: list[float] | None = None
    if stage_logits is not None:
        if isinstance(stage_logits, torch.Tensor):
            logits = stage_logits.reshape(-1).float()
        else:
            logits = torch.tensor(np.asarray(stage_logits).reshape(-1), dtype=torch.float32)
        probs = torch.softmax(logits, dim=-1)
        stage_prob_values = [float(x) for x in probs.tolist()]

    return stage_pred_value, stage_prob_values


def _format_stage_probs(stage_probs: list[float] | None) -> str:
    if not stage_probs:
        return "n/a"
    return ", ".join(f"{i + 1}:{prob:.2f}" for i, prob in enumerate(stage_probs))


def _format_stage_prediction(stage_pred: int | None) -> str:
    if stage_pred is None:
        return "n/a"
    if 0 <= stage_pred < len(TASK_DESCRIPTIONS):
        return f"{stage_pred + 1}: {TASK_DESCRIPTIONS[stage_pred]}"
    return str(stage_pred)


def _stage_pred_confidence(stage_pred: int | None, stage_probs: list[float] | None) -> float | None:
    if stage_pred is None or stage_probs is None:
        return None
    if stage_pred < 0 or stage_pred >= len(stage_probs):
        return None
    return stage_probs[stage_pred]


def _stage_pred_reached_next_task(
    selected_task_idx: int,
    stage_pred: int | None,
    stage_probs: list[float] | None,
) -> bool:
    if stage_pred is None or stage_pred < selected_task_idx + 1:
        return False
    confidence = _stage_pred_confidence(stage_pred, stage_probs)
    if confidence is not None and confidence < args.stage_pred_min_confidence:
        return False
    return True


def _format_stage_pred_success(
    selected_task_idx: int,
    stage_pred: int | None,
    stage_probs: list[float] | None,
) -> str:
    reached = _stage_pred_reached_next_task(selected_task_idx, stage_pred, stage_probs)
    confidence = _stage_pred_confidence(stage_pred, stage_probs)
    confidence_text = "n/a" if confidence is None else f"{confidence:.3f}"
    return (
        f"{reached} "
        f"(target >= {selected_task_idx + 2}, confidence={confidence_text}, "
        f"min={args.stage_pred_min_confidence:.3f})"
    )


def _normalize_env_raw_action(action: np.ndarray | torch.Tensor) -> np.ndarray:
    action_np = np.asarray(action, dtype=np.float32).reshape(ROBOT_ACTION_DIM).copy()
    action_np[:ACTION_PREFIX_PAD] = 0.0
    return action_np


def _raw_action_to_env_tensor(action: np.ndarray, device: str) -> torch.Tensor:
    action = _normalize_env_raw_action(action).reshape(1, ROBOT_ACTION_DIM)
    return torch.tensor(action, dtype=torch.float32, device=device)


def _get_stage_probe_prediction(policy, env, task_description: str, action_dict: dict):
    """Return action-prompt and fixed-prompt stage predictions for the current state."""
    action_stage_pred, action_stage_probs = _extract_stage_prediction(action_dict)
    probe_prompt = args.stage_prompt.strip()
    if args.disable_stage_probe or not probe_prompt or probe_prompt == task_description:
        return action_stage_pred, action_stage_probs, action_stage_pred, action_stage_probs, task_description

    stage_obs = _build_policy_obs(env, probe_prompt)
    stage_action_dict = policy.get_action(stage_obs)
    stage_probe_pred, stage_probe_probs = _extract_stage_prediction(stage_action_dict)
    return action_stage_pred, action_stage_probs, stage_probe_pred, stage_probe_probs, probe_prompt


class KeyboardInterface:
    """Keyboard-driven control state for the interactive runner."""

    _TASK_KEY_MAP = {
        "1": 0,
        "KEY_1": 0,
        "NUMPAD_1": 0,
        "2": 1,
        "KEY_2": 1,
        "NUMPAD_2": 1,
        "3": 2,
        "KEY_3": 2,
        "NUMPAD_3": 2,
        "4": 3,
        "KEY_4": 3,
        "NUMPAD_4": 3,
        "5": 4,
        "KEY_5": 4,
        "NUMPAD_5": 4,
    }

    def __init__(self, stop_on_predicted_next_stage: bool):
        self.running = False
        self.pending_single_step = False
        self.pending_stage_probe = False
        self.pending_reason2_probe = False
        self.pending_reset = False
        self.pending_retry_task = False
        self.pending_tray_rotation = False
        self.pending_tray_return = False
        self.selected_task_idx = 0
        self.stop_on_predicted_next_stage = stop_on_predicted_next_stage
        self.last_message = "Ready. Select a task and press SPACE."
        self.last_key = "n/a"
        self._input = carb.input.acquire_input_interface()
        self._app_window = omni.appwindow.get_default_app_window()
        self._keyboard = None if self._app_window is None else self._app_window.get_keyboard()
        self._sub_keyboard = None
        if self._keyboard is not None:
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        else:
            self.last_message = "Keyboard subscription is unavailable. Use the on-screen buttons."

    def close(self):
        if self._sub_keyboard is not None and self._keyboard is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            self._sub_keyboard = None

    def consume_single_step(self) -> bool:
        value = self.pending_single_step
        self.pending_single_step = False
        return value

    def consume_stage_probe(self) -> bool:
        value = self.pending_stage_probe
        self.pending_stage_probe = False
        return value

    def consume_reason2_probe(self) -> bool:
        value = self.pending_reason2_probe
        self.pending_reason2_probe = False
        return value

    def consume_reset(self) -> bool:
        value = self.pending_reset
        self.pending_reset = False
        return value

    def consume_retry_task(self) -> bool:
        value = self.pending_retry_task
        self.pending_retry_task = False
        return value

    def consume_tray_rotation(self) -> bool:
        value = self.pending_tray_rotation
        self.pending_tray_rotation = False
        return value

    def consume_tray_return(self) -> bool:
        value = self.pending_tray_return
        self.pending_tray_return = False
        return value

    def select_task(self, task_idx: int) -> None:
        self.selected_task_idx = max(0, min(task_idx, len(TASK_DESCRIPTIONS) - 1))
        self.last_message = f"Selected task {self.selected_task_idx + 1}: {TASK_DESCRIPTIONS[self.selected_task_idx]}"

    def toggle_running(self) -> None:
        self.running = not self.running
        state = "running" if self.running else "paused"
        self.last_message = f"Continuous inference {state}."

    def queue_single_step(self) -> None:
        self.pending_single_step = True
        self.last_message = "Queued one policy step."

    def queue_stage_probe(self) -> None:
        self.pending_stage_probe = True
        self.last_message = "Queued a stage-pred probe without env step."

    def queue_reason2_probe(self) -> None:
        self.pending_reason2_probe = True
        self.last_message = "Queued a Cosmos Reason2 hand-status probe."

    def queue_reset(self) -> None:
        self.running = False
        self.pending_single_step = False
        self.pending_stage_probe = False
        self.pending_reason2_probe = False
        self.pending_tray_rotation = False
        self.pending_tray_return = False
        self.pending_reset = True
        self.last_message = "Queued environment reset."

    def queue_retry_task(self) -> None:
        self.running = False
        self.pending_single_step = False
        self.pending_stage_probe = False
        self.pending_reason2_probe = False
        self.pending_retry_task = True
        self.last_message = f"Queued a Task {self.selected_task_idx + 1} return-to-start retry motion."

    def queue_tray_rotation(self) -> None:
        self.pending_tray_rotation = True
        self.pending_tray_return = False
        self.last_message = (
            f"Queued tray rotation: {args.tray_yaw_increment_deg:.1f} deg "
            f"over {max(args.tray_yaw_steps, 1)} env steps."
        )

    def queue_tray_return(self) -> None:
        self.pending_tray_return = True
        self.pending_tray_rotation = False
        self.last_message = f"Queued tray rotation back over {max(args.tray_yaw_steps, 1)} env steps."

    def print_help(self) -> None:
        _print_controls()
        self.last_message = "Printed controls to the terminal."

    @staticmethod
    def _key_aliases(raw_key: str) -> set[str]:
        key = raw_key.upper()
        aliases = {key}
        for prefix in ("KEY_", "NUMPAD_", "NUMROW_", "KP_"):
            if key.startswith(prefix):
                aliases.add(key[len(prefix) :])
        return aliases

    def _on_keyboard_event(self, event):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

        raw_key = str(event.input.name)
        aliases = self._key_aliases(raw_key)
        self.last_key = raw_key

        for alias in aliases:
            if alias in self._TASK_KEY_MAP:
                self.select_task(self._TASK_KEY_MAP[alias])
                return True

        if aliases & {"SPACE", "SPACEBAR", "P"}:
            self.toggle_running()
        elif aliases & {"S", "ENTER"}:
            self.queue_single_step()
        elif aliases & {"G"}:
            self.queue_stage_probe()
        elif aliases & {"C"}:
            self.queue_reason2_probe()
        elif aliases & {"Y"}:
            self.queue_retry_task()
        elif aliases & {"O"}:
            self.queue_tray_rotation()
        elif aliases & {"B"}:
            self.queue_tray_return()
        elif aliases & {"R"}:
            self.queue_reset()
        elif aliases & {"H", "F1"}:
            self.print_help()
        return True


class StatusPanel:
    """Simple docked status panel shown inside Isaac Sim."""

    def __init__(self, keyboard: KeyboardInterface):
        self._keyboard = keyboard
        self._window = ui.Window(
            "Trocar Multitask Debug",
            width=420,
            height=460,
            visible=True,
            dock_preference=ui.DockPreference.DISABLED,
        )
        with self._window.frame:
            with ui.VStack(spacing=6, height=0):
                self._status = ui.Label("", word_wrap=True)
                ui.Spacer(height=8)
                ui.Label("Mouse Controls", word_wrap=True)
                with ui.HStack(spacing=4):
                    for task_idx, task_name in enumerate(TASK_DESCRIPTIONS):
                        ui.Button(
                            f"{task_idx + 1}",
                            width=0,
                            clicked_fn=lambda idx=task_idx: self._keyboard.select_task(idx),
                            tooltip=f"Select task {task_idx + 1}: {task_name}",
                        )
                with ui.HStack(spacing=4):
                    ui.Button(
                        "Start/Pause",
                        width=0,
                        clicked_fn=self._keyboard.toggle_running,
                    )
                    ui.Button(
                        "Step",
                        width=0,
                        clicked_fn=self._keyboard.queue_single_step,
                    )
                    ui.Button(
                        "Probe Stage",
                        width=0,
                        clicked_fn=self._keyboard.queue_stage_probe,
                    )
                    ui.Button(
                        "Ask Cosmos",
                        width=0,
                        clicked_fn=self._keyboard.queue_reason2_probe,
                    )
                    ui.Button(
                        "Reset",
                        width=0,
                        clicked_fn=self._keyboard.queue_reset,
                    )
                    ui.Button(
                        "Retry",
                        width=0,
                        clicked_fn=self._keyboard.queue_retry_task,
                    )
                    ui.Button(
                        "Help",
                        width=0,
                        clicked_fn=self._keyboard.print_help,
                    )
                with ui.HStack(spacing=4):
                    ui.Button(
                        "Rotate -30deg",
                        width=0,
                        clicked_fn=self._keyboard.queue_tray_rotation,
                    )
                    ui.Button(
                        "Rotate Back",
                        width=0,
                        clicked_fn=self._keyboard.queue_tray_return,
                    )
                ui.Spacer(height=4)
                self._controls = ui.Label(
                    "Keyboard Controls\n"
                    "1-5: select task\n"
                    "SPACE/P: start or pause\n"
                    "S: single step\n"
                    "G: probe stage without stepping\n"
                    "C: ask Cosmos hand-status judge\n"
                    "Y: retry selected task from recorded start\n"
                    "O: rotate tray by configured yaw delta\n"
                    "B: rotate tray back to reset yaw\n"
                    "R: reset\n"
                    "H: print help\n"
                    "\n"
                    "VNC note: click the viewport or this panel before pressing keys.\n"
                    "If keyboard focus is awkward, use the buttons above.",
                    word_wrap=True,
                )

    def update(
        self,
        keyboard: KeyboardInterface,
        selected_task: str,
        stage_probe_pred: int | None,
        stage_probe_probs: list[float] | None,
        step_count: int,
        episode_done: bool,
        task_run_step_count: int,
        selected_task_history_len: int,
        retry_replay_active: bool,
        retry_replay_progress: int,
        retry_replay_total: int,
        retry_hold_left_hand: bool,
        retry_hold_right_hand: bool,
        open_loop_steps_remaining: int,
        tray_yaw_deg: float,
        tray_rotation_active: bool,
        tray_rotation_progress: int,
        tray_rotation_total: int,
        reason2_result: dict | None,
        last_policy_ms: float | None,
        last_env_step_ms: float | None,
        last_step_interval_ms: float | None,
    ):
        mode = "RUNNING" if keyboard.running else "PAUSED"
        if episode_done:
            mode += " / EPISODE DONE"
        if retry_replay_active:
            mode += " / RETRY RETURN"
        if tray_rotation_active:
            mode += " / TRAY ROTATING"

        if reason2_result is None:
            reason2_text = "n/a"
        elif reason2_result.get("ok"):
            reason2_text = (
                f"L={reason2_result.get('left_hand_holding_trocar')} "
                f"R={reason2_result.get('right_hand_holding_trocar')}"
            )
        else:
            reason2_text = f"ERROR: {reason2_result.get('error')}"

        policy_ms_text = "n/a" if last_policy_ms is None else f"{last_policy_ms:.0f}"
        env_step_ms_text = "n/a" if last_env_step_ms is None else f"{last_env_step_ms:.0f}"
        step_interval_ms_text = "n/a" if last_step_interval_ms is None else f"{last_step_interval_ms:.0f}"

        self._status.text = (
            f"Mode: {mode}\n"
            f"Selected task: {keyboard.selected_task_idx + 1}: {selected_task}\n"
            f"Stage pred: {_format_stage_prediction(stage_probe_pred)}\n"
            f"Stage probs: {_format_stage_probs(stage_probe_probs)}\n"
            f"Predicted next-stage complete: "
            f"{_format_stage_pred_success(keyboard.selected_task_idx, stage_probe_pred, stage_probe_probs)}\n"
            f"Task run steps: {task_run_step_count} / {args.task_timeout_steps}\n"
            f"Open-loop steps remaining: {open_loop_steps_remaining} / {args.open_loop_steps}\n"
            f"Selected task recorded actions: {selected_task_history_len}\n"
            f"Retry return steps: {retry_replay_progress} / {retry_replay_total}\n"
            f"Retry hand hold: L={retry_hold_left_hand} R={retry_hold_right_hand}\n"
            f"Tray yaw: {tray_yaw_deg:.2f} deg\n"
            f"Tray rotation steps: {tray_rotation_progress} / {tray_rotation_total}\n"
            f"Cosmos hand status: {reason2_text}\n"
            f"Timing ms: policy={policy_ms_text} env_step={env_step_ms_text} interval={step_interval_ms_text}\n"
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
            "[INFO] Fixed initial state enabled: "
            f"dataset={args.fixed_initial_state_dataset}, "
            f"episode={args.fixed_initial_state_episode}, frame={args.fixed_initial_state_frame}, "
            f"warmup_steps={args.fixed_initial_state_steps}"
        )
        print(f"[INFO] Fixed initial left hand ref: {fixed_initial_state_ref[14:21].tolist()}")
        print(f"[INFO] Fixed initial right hand ref: {fixed_initial_state_ref[21:28].tolist()}")

    keyboard = KeyboardInterface(stop_on_predicted_next_stage=args.enable_stage_pred_stop)
    panel = StatusPanel(keyboard)
    _print_controls()

    tray_yaw_deg = _sample_initial_tray_yaw_deg(rng)
    obs = _reset_env(
        env,
        seed=args.seed,
        fixed_raw_action=fixed_initial_raw_action,
        fixed_state_ref=fixed_initial_state_ref,
        tray_yaw_deg=tray_yaw_deg,
    )
    del obs

    step_count = 0
    episode_done = False
    action_stage_pred: int | None = None
    action_stage_probs: list[float] | None = None
    stage_probe_pred: int | None = None
    stage_probe_probs: list[float] | None = None
    stage_probe_prompt = args.stage_prompt.strip()
    reason2_result: dict | None = None
    last_step_time = 0.0
    last_env_step_start_time: float | None = None
    last_policy_ms: float | None = None
    last_env_step_ms: float | None = None
    last_step_interval_ms: float | None = None
    step_period = 1.0 / max(args.step_hz, 1e-6)
    task_run_step_count = 0
    previous_running = False
    previous_task_idx = keyboard.selected_task_idx
    task_action_histories: list[list[np.ndarray]] = [[] for _ in TASK_DESCRIPTIONS]
    task_history_locked = [False for _ in TASK_DESCRIPTIONS]
    retry_replay_active = False
    retry_replay_task_idx: int | None = None
    retry_replay_actions: list[np.ndarray] = []
    retry_replay_step_count = 0
    retry_hold_left_hand = False
    retry_hold_right_hand = False
    cached_policy_actions: list[np.ndarray] = []
    open_loop_steps_remaining = 0
    tray_rotation_active = False
    tray_base_yaw_deg = tray_yaw_deg
    tray_rotation_start_yaw = tray_yaw_deg
    tray_rotation_target_yaw = tray_yaw_deg
    tray_rotation_step_count = 0
    tray_rotation_total_steps = max(args.tray_yaw_steps, 1)

    try:
        while simulation_app.is_running():
            if keyboard.selected_task_idx != previous_task_idx:
                if task_action_histories[previous_task_idx] and not task_history_locked[previous_task_idx]:
                    task_history_locked[previous_task_idx] = True
                task_run_step_count = 0
                cached_policy_actions = []
                open_loop_steps_remaining = 0
                previous_task_idx = keyboard.selected_task_idx

            if keyboard.running and not previous_running:
                task_run_step_count = 0
            previous_running = keyboard.running

            if keyboard.consume_reset():
                tray_yaw_deg = _sample_initial_tray_yaw_deg(rng)
                _reset_env(
                    env,
                    seed=args.seed,
                    fixed_raw_action=fixed_initial_raw_action,
                    fixed_state_ref=fixed_initial_state_ref,
                    tray_yaw_deg=tray_yaw_deg,
                )
                step_count = 0
                episode_done = False
                action_stage_pred = None
                action_stage_probs = None
                stage_probe_pred = None
                stage_probe_probs = None
                reason2_result = None
                task_run_step_count = 0
                task_action_histories = [[] for _ in TASK_DESCRIPTIONS]
                task_history_locked = [False for _ in TASK_DESCRIPTIONS]
                retry_replay_active = False
                retry_replay_task_idx = None
                retry_replay_actions = []
                retry_replay_step_count = 0
                retry_hold_left_hand = False
                retry_hold_right_hand = False
                cached_policy_actions = []
                open_loop_steps_remaining = 0
                tray_rotation_active = False
                tray_base_yaw_deg = tray_yaw_deg
                tray_rotation_start_yaw = tray_yaw_deg
                tray_rotation_target_yaw = tray_yaw_deg
                tray_rotation_step_count = 0
                tray_rotation_total_steps = max(args.tray_yaw_steps, 1)
                keyboard.last_message = f"Reset environment. Initial tray yaw={tray_yaw_deg:.2f} deg."

            if keyboard.consume_tray_rotation():
                if episode_done:
                    keyboard.last_message = "Tray rotation requested, but the episode is done. Press R to reset."
                else:
                    tray_rotation_active = True
                    tray_rotation_start_yaw = tray_yaw_deg
                    tray_rotation_target_yaw = tray_base_yaw_deg + args.tray_yaw_increment_deg
                    tray_rotation_step_count = 0
                    tray_rotation_total_steps = max(args.tray_yaw_steps, 1)
                    keyboard.last_message = (
                        f"Started tray rotation from {tray_rotation_start_yaw:.2f} deg "
                        f"to {tray_rotation_target_yaw:.2f} deg over {tray_rotation_total_steps} env steps."
                    )

            if keyboard.consume_tray_return():
                if episode_done:
                    keyboard.last_message = "Tray rotation return requested, but the episode is done. Press R to reset."
                else:
                    tray_rotation_active = True
                    tray_rotation_start_yaw = tray_yaw_deg
                    tray_rotation_target_yaw = tray_base_yaw_deg
                    tray_rotation_step_count = 0
                    tray_rotation_total_steps = max(args.tray_yaw_steps, 1)
                    keyboard.last_message = (
                        f"Started tray rotation back from {tray_rotation_start_yaw:.2f} deg "
                        f"to {tray_rotation_target_yaw:.2f} deg over {tray_rotation_total_steps} env steps."
                    )

            if keyboard.consume_retry_task():
                retry_task_idx = keyboard.selected_task_idx
                retry_history = task_action_histories[retry_task_idx]
                if episode_done:
                    keyboard.last_message = "Retry requested, but the episode is done. Press R to reset."
                elif not retry_history:
                    keyboard.last_message = (
                        f"Retry Task {retry_task_idx + 1} requested, "
                        "but this task has not recorded any policy actions yet."
                    )
                else:
                    try:
                        _flush_observations(env)
                        reason2_result = _query_reason2_hand_status(env, retry_task_idx, step_count)
                        retry_hold_left_hand = bool(reason2_result.get("left_hand_holding_trocar"))
                        retry_hold_right_hand = bool(reason2_result.get("right_hand_holding_trocar"))
                        retry_current_action = _get_env_action_state(env)
                    except Exception as exc:
                        reason2_result = {"ok": False, "error": repr(exc)}
                        retry_hold_left_hand = False
                        retry_hold_right_hand = False
                        keyboard.last_message = f"Retry aborted because Cosmos hand-status probe failed: {exc!r}"
                    else:
                        keyboard.running = False
                        task_history_locked[retry_task_idx] = True
                        retry_replay_active = True
                        retry_replay_task_idx = retry_task_idx
                        retry_start_action = _apply_retry_hand_hold(
                            retry_history[0],
                            retry_current_action,
                            retry_hold_left_hand,
                            retry_hold_right_hand,
                        )
                        retry_return_steps = max(args.retry_return_steps, 1)
                        retry_replay_actions = []
                        for return_step in range(1, retry_return_steps + 1):
                            alpha = return_step / retry_return_steps
                            retry_replay_actions.append(
                                _normalize_env_raw_action(
                                    (1.0 - alpha) * retry_current_action + alpha * retry_start_action
                                )
                            )
                        retry_replay_step_count = 0
                        task_run_step_count = 0
                        action_stage_pred = None
                        action_stage_probs = None
                        stage_probe_pred = None
                        stage_probe_probs = None
                        cached_policy_actions = []
                        open_loop_steps_remaining = 0
                        keyboard.last_message = (
                            f"Started Task {retry_task_idx + 1} return-to-start retry with "
                            f"{len(retry_replay_actions)} interpolated actions. "
                            f"Hold hands: left={retry_hold_left_hand}, right={retry_hold_right_hand}."
                        )

            if keyboard.consume_stage_probe():
                if episode_done:
                    keyboard.last_message = "Stage probe requested, but the episode is done. Press R to reset."
                else:
                    task_description = TASK_DESCRIPTIONS[keyboard.selected_task_idx]
                    with torch.inference_mode():
                        policy_start_time = time.perf_counter()
                        policy_obs = _build_policy_obs(env, task_description)
                        action_dict = policy.get_action(policy_obs)
                        (
                            action_stage_pred,
                            action_stage_probs,
                            stage_probe_pred,
                            stage_probe_probs,
                            stage_probe_prompt,
                        ) = _get_stage_probe_prediction(policy, env, task_description, action_dict)
                        last_policy_ms = (time.perf_counter() - policy_start_time) * 1000.0
                    keyboard.last_message = (
                        "Stage probe updated without env step. "
                        f"Complete={_stage_pred_reached_next_task(keyboard.selected_task_idx, stage_probe_pred, stage_probe_probs)}."
                    )

            if keyboard.consume_reason2_probe():
                if episode_done:
                    keyboard.last_message = "Cosmos probe requested, but the episode is done. Press R to reset."
                else:
                    try:
                        _flush_observations(env)
                        reason2_result = _query_reason2_hand_status(env, keyboard.selected_task_idx, step_count)
                        keyboard.last_message = (
                            "Cosmos Reason2 updated: "
                            f"left={reason2_result.get('left_hand_holding_trocar')}, "
                            f"right={reason2_result.get('right_hand_holding_trocar')}."
                        )
                    except Exception as exc:
                        reason2_result = {"ok": False, "error": repr(exc)}
                        keyboard.last_message = f"Cosmos Reason2 probe failed: {exc!r}"

            policy_step_requested = False
            if not episode_done and keyboard.consume_single_step():
                policy_step_requested = True
            if not episode_done and keyboard.running:
                now = time.perf_counter()
                if now - last_step_time >= step_period:
                    policy_step_requested = True
                    last_step_time = now

            tray_hold_step_requested = False
            if not episode_done and tray_rotation_active and not policy_step_requested and not retry_replay_active:
                now = time.perf_counter()
                if now - last_step_time >= step_period:
                    tray_hold_step_requested = True
                    last_step_time = now

            should_step = policy_step_requested or tray_hold_step_requested

            action_tensor: torch.Tensor | None = None
            executing_retry_replay_step = False
            executing_policy_step = False
            executing_tray_hold_step = False
            executed_policy_action: np.ndarray | None = None
            if retry_replay_active and not episode_done:
                if retry_replay_step_count >= len(retry_replay_actions):
                    retry_replay_active = False
                    finished_task_idx = retry_replay_task_idx
                    retry_replay_task_idx = None
                    retry_replay_actions = []
                    retry_replay_step_count = 0
                    if finished_task_idx is None:
                        keyboard.last_message = "Finished retry return. Press Start to run the selected task again."
                    else:
                        keyboard.last_message = (
                            f"Finished Task {finished_task_idx + 1} retry return. "
                            f"Press Start to run Task {finished_task_idx + 1} again."
                        )
                else:
                    executing_retry_replay_step = True
                    action_tensor = _raw_action_to_env_tensor(
                        retry_replay_actions[retry_replay_step_count], device=env.device
                    )

            if should_step and action_tensor is None:
                if policy_step_requested:
                    executing_policy_step = True
                    if not cached_policy_actions:
                        task_description = TASK_DESCRIPTIONS[keyboard.selected_task_idx]
                        policy_start_time = time.perf_counter()
                        policy_obs = _build_policy_obs(env, task_description)
                        action_dict = policy.get_action(policy_obs)
                        (
                            action_stage_pred,
                            action_stage_probs,
                            stage_probe_pred,
                            stage_probe_probs,
                            stage_probe_prompt,
                        ) = _get_stage_probe_prediction(policy, env, task_description, action_dict)
                        action_chunk = _convert_policy_action_chunk_to_env(action_dict)
                        if not action_chunk:
                            raise RuntimeError("Policy returned an empty action chunk.")
                        cached_policy_actions = [action.copy() for action in action_chunk[: max(args.open_loop_steps, 1)]]
                        open_loop_steps_remaining = len(cached_policy_actions)
                        last_policy_ms = (time.perf_counter() - policy_start_time) * 1000.0
                    executed_policy_action = cached_policy_actions.pop(0)
                    action_tensor = _raw_action_to_env_tensor(executed_policy_action, device=env.device)
                else:
                    executing_tray_hold_step = True
                    action_tensor = _raw_action_to_env_tensor(_get_env_action_state(env), device=env.device)

            if should_step or executing_retry_replay_step:
                if action_tensor is None:
                    raise RuntimeError("Expected an action tensor before stepping the environment.")
                if tray_rotation_active:
                    previous_tray_yaw = tray_yaw_deg
                    tray_rotation_step_count += 1
                    alpha = tray_rotation_step_count / max(tray_rotation_total_steps, 1)
                    tray_yaw_deg = (1.0 - alpha) * tray_rotation_start_yaw + alpha * tray_rotation_target_yaw
                    _set_tray_trocar_yaw_delta(
                        env,
                        tray_yaw_deg,
                        previous_yaw_deg=previous_tray_yaw,
                        carry_trocars_on_tray_only=True,
                    )
                    if tray_rotation_step_count >= tray_rotation_total_steps:
                        tray_rotation_active = False
                        keyboard.last_message = f"Finished tray rotation. Current tray yaw={tray_yaw_deg:.2f} deg."
                env_step_start_time = time.perf_counter()
                if last_env_step_start_time is not None:
                    last_step_interval_ms = (env_step_start_time - last_env_step_start_time) * 1000.0
                last_env_step_start_time = env_step_start_time
                _, _, terminated, truncated, _ = env.step(action_tensor)
                last_env_step_ms = (time.perf_counter() - env_step_start_time) * 1000.0
                step_count += 1
                episode_done = bool(terminated[0]) or bool(truncated[0])
                if executing_retry_replay_step:
                    retry_replay_step_count += 1
                    if retry_replay_step_count >= len(retry_replay_actions):
                        retry_replay_active = False
                        finished_task_idx = retry_replay_task_idx
                        retry_replay_task_idx = None
                        retry_replay_actions = []
                        retry_replay_step_count = 0
                        if finished_task_idx is None:
                            keyboard.last_message = "Finished retry return. Press Start to run the selected task again."
                        else:
                            keyboard.last_message = (
                                f"Finished Task {finished_task_idx + 1} retry return. "
                                f"Press Start to run Task {finished_task_idx + 1} again."
                            )
                elif executing_policy_step:
                    selected_task_idx = keyboard.selected_task_idx
                    if not task_history_locked[selected_task_idx] and executed_policy_action is not None:
                        task_action_histories[selected_task_idx].append(executed_policy_action.copy())
                        if len(task_action_histories[selected_task_idx]) == 1:
                            keyboard.last_message = (
                                f"Started recording Task {selected_task_idx + 1} action history for retry."
                            )
                    open_loop_steps_remaining = len(cached_policy_actions)
                    if keyboard.running:
                        task_run_step_count += 1
                elif executing_tray_hold_step:
                    pass
                elif keyboard.running:
                    task_run_step_count += 1

                if executing_retry_replay_step:
                    pass
                elif (
                    keyboard.stop_on_predicted_next_stage
                    and _stage_pred_reached_next_task(
                        keyboard.selected_task_idx,
                        stage_probe_pred,
                        stage_probe_probs,
                    )
                ):
                    keyboard.running = False
                    stage_pred_confidence = _stage_pred_confidence(stage_probe_pred, stage_probe_probs)
                    confidence_text = "n/a" if stage_pred_confidence is None else f"{stage_pred_confidence:.3f}"
                    keyboard.last_message = (
                        f"Paused by stage_pred: selected task {keyboard.selected_task_idx + 1}, "
                        f"predicted stage {stage_probe_pred + 1}, confidence {confidence_text}."
                    )
                elif keyboard.running and task_run_step_count >= args.task_timeout_steps:
                    keyboard.running = False
                    keyboard.last_message = (
                        f"Paused by step timeout after {task_run_step_count} steps "
                        f"(limit {args.task_timeout_steps})."
                    )
                if episode_done:
                    keyboard.running = False
                    retry_replay_active = False
                    retry_replay_task_idx = None
                    retry_replay_actions = []
                    retry_replay_step_count = 0
                    retry_hold_left_hand = False
                    retry_hold_right_hand = False
                    cached_policy_actions = []
                    open_loop_steps_remaining = 0
                    tray_rotation_active = False
                    tray_rotation_start_yaw = tray_yaw_deg
                    tray_rotation_target_yaw = tray_yaw_deg
                    reason = "success/drop/termination" if bool(terminated[0]) else "timeout/truncation"
                    keyboard.last_message = f"Episode finished: {reason}. Press R to reset."
            else:
                env.sim.render()
                time.sleep(1.0 / 60.0)

            panel.update(
                keyboard=keyboard,
                selected_task=TASK_DESCRIPTIONS[keyboard.selected_task_idx],
                stage_probe_pred=stage_probe_pred,
                stage_probe_probs=stage_probe_probs,
                step_count=step_count,
                episode_done=episode_done,
                task_run_step_count=task_run_step_count,
                selected_task_history_len=len(task_action_histories[keyboard.selected_task_idx]),
                retry_replay_active=retry_replay_active,
                retry_replay_progress=retry_replay_step_count,
                retry_replay_total=len(retry_replay_actions),
                retry_hold_left_hand=retry_hold_left_hand,
                retry_hold_right_hand=retry_hold_right_hand,
                open_loop_steps_remaining=open_loop_steps_remaining,
                tray_yaw_deg=tray_yaw_deg,
                tray_rotation_active=tray_rotation_active,
                tray_rotation_progress=tray_rotation_step_count,
                tray_rotation_total=tray_rotation_total_steps,
                reason2_result=reason2_result,
                last_policy_ms=last_policy_ms,
                last_env_step_ms=last_env_step_ms,
                last_step_interval_ms=last_step_interval_ms,
            )
    finally:
        keyboard.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
