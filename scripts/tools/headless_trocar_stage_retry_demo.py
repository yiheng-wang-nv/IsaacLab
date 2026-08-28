# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless trocar split-stage demo with per-task retry.

Runs task prompts 1 through 5 sequentially. If a task does not reach the next
stage within its step budget, the script commands controlled joints back to that
task's first recorded state and tries again. Each rollout is saved as a 30 FPS
MP4 plus an event log with the corresponding video frame counts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record a headless trocar staged retry demo.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the Isaac-GR00T checkpoint directory.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory where the MP4 and event log are saved.")
parser.add_argument(
    "--task_id",
    type=str,
    default="Isaac-Assemble-Trocar-G129-Dex3-RLinf-v0",
    help="Gym task id for the trocar environment.",
)
parser.add_argument("--denoising_steps", type=int, default=4, help="GR00T denoising steps.")
parser.add_argument("--open_loop_steps", type=int, default=1, help="Action chunk steps to run per inference.")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record.")
parser.add_argument("--task_max_steps", type=int, default=60, help="Max policy steps per task attempt.")
parser.add_argument(
    "--task1_max_steps",
    type=int,
    default=45,
    help="Optional max policy steps for each Task 1 attempt. Defaults to 45.",
)
parser.add_argument(
    "--task2_max_steps",
    type=int,
    default=45,
    help="Optional max policy steps for each Task 2 attempt. Defaults to 45.",
)
parser.add_argument(
    "--task3_max_steps",
    type=int,
    default=None,
    help="Optional max policy steps for each Task 3 attempt. Defaults to --task_max_steps.",
)
parser.add_argument(
    "--task4_max_steps",
    type=int,
    default=None,
    help="Optional max policy steps for each Task 4 attempt. Defaults to --task_max_steps.",
)
parser.add_argument(
    "--task5_max_steps",
    type=int,
    default=None,
    help="Optional max policy steps for Task 5. Defaults to --task_max_steps.",
)
parser.add_argument("--task_max_retries", type=int, default=3, help="Default retries after the first try for each task.")
parser.add_argument("--task3_max_retries", type=int, default=3, help="Number of Task 3 retries after the first try.")
parser.add_argument("--task4_max_retries", type=int, default=3, help="Number of Task 4 retries after the first try.")
parser.add_argument("--task5_max_retries", type=int, default=0, help="Number of Task 5 retries after the first try.")
parser.add_argument(
    "--task_success_source",
    type=str,
    default="stage_pred",
    choices=["stage_pred", "env_oracle"],
    help="Signal used to advance between tasks. stage_pred avoids simulator oracle state.",
)
parser.add_argument(
    "--stage_pred_prompt",
    type=str,
    default="",
    help="Optional prompt used for an extra stage-prediction probe. Empty means reuse the action forward output.",
)
parser.add_argument(
    "--stage_pred_min_confidence",
    type=float,
    default=0.0,
    help="Minimum softmax probability required for a stage_pred task-success transition.",
)
parser.add_argument(
    "--perturb_tray_task",
    type=int,
    default=2,
    help="1-based task/stage whose policy rollout triggers the tray/trocar perturbation.",
)
parser.add_argument(
    "--perturb_tray_after_steps",
    type=int,
    default=0,
    help="Local policy step in --perturb_tray_task at which to start rotating. Use a negative value to disable.",
)
parser.add_argument(
    "--initial_tray_yaw_deg",
    type=float,
    default=0.0,
    help="Initial tray/trocar yaw delta [deg] applied after reset.",
)
parser.add_argument(
    "--perturb_tray_yaw_deg",
    type=float,
    default=10.0,
    help="Target tray/trocar yaw delta [deg] after the scripted perturbation finishes.",
)
parser.add_argument(
    "--perturb_tray_duration_steps",
    type=int,
    default=15,
    help="Env steps used to interpolate tray/trocar yaw. Use 1 for an instant perturbation.",
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
    "--back_to_init_steps",
    type=int,
    default=60,
    help="Max joint-target interpolation steps used to return to the first retryable-task state.",
)
parser.add_argument(
    "--back_to_init_tolerance",
    type=float,
    default=0.035,
    help="Stop returning early when the max controlled-joint error is below this value.",
)
parser.add_argument("--model_device", type=str, default=None, help="Policy device. Defaults to simulator device.")
parser.add_argument("--seed", type=int, default=None, help="Optional environment reset seed.")
parser.add_argument(
    "--camera",
    type=str,
    default="front_camera",
    choices=["front_camera", "left_wrist_camera", "right_wrist_camera"],
    help="Camera to save into the demo MP4.",
)
parser.add_argument("--fps", type=float, default=30.0, help="Output MP4 FPS.")
parser.add_argument("--video_name", type=str, default="trocar_staged_retry_demo.mp4", help="Output MP4 filename.")
parser.add_argument("--final_hold_frames", type=int, default=60, help="Repeated final frames to show a pause.")
parser.add_argument("--no_overlay", action="store_true", help="Disable text overlay on the saved MP4.")
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
    "--use_reason2_retry_judge",
    action="store_true",
    help="Pause before retry and ask Cosmos Reason2 whether each hand is holding a trocar.",
)
parser.add_argument(
    "--reason2_url",
    type=str,
    default="http://localhost:8000/v1/chat/completions",
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
parser.add_argument(
    "--reason2_pause_frames",
    type=int,
    default=15,
    help="Repeated MP4 frames inserted while the robot is paused for a Reason2 retry judgment.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_episodes < 1:
    parser.error("--num_episodes must be >= 1.")
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import cv2
import gymnasium as gym
import numpy as np
import omni.kit.app
import requests
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
ARM_ACTION_SLICE = slice(ACTION_PREFIX_PAD, ACTION_PREFIX_PAD + 14)
HAND_ACTION_SLICE = slice(ACTION_PREFIX_PAD + 14, ROBOT_ACTION_DIM)
ARM_ACTION_INDICES = np.arange(ARM_ACTION_SLICE.start, ARM_ACTION_SLICE.stop, dtype=np.int64)
LEFT_HAND_ACTION_INDICES = np.array(
    [ACTION_PREFIX_PAD + idx for idx, ref_idx in enumerate(PERM_TO_REF) if 14 <= ref_idx < 21],
    dtype=np.int64,
)
RIGHT_HAND_ACTION_INDICES = np.array(
    [ACTION_PREFIX_PAD + idx for idx, ref_idx in enumerate(PERM_TO_REF) if 21 <= ref_idx < 28],
    dtype=np.int64,
)


@dataclass
class TaskResult:
    """Summary of a task attempt."""

    success: bool
    episode_done: bool


class VideoRecorder:
    """Small MP4 writer that records frame counts and optional overlays."""

    def __init__(self, output_path: Path, fps: float, overlay: bool):
        self.output_path = output_path
        self.fps = fps
        self.overlay = overlay
        self.frame_count = 0
        self._writer: cv2.VideoWriter | None = None
        self._last_frame_bgr: np.ndarray | None = None
        self._last_raw_frame_bgr: np.ndarray | None = None

    def write(self, rgb_frame: np.ndarray, lines: list[str]) -> int:
        raw_frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        frame_bgr = raw_frame_bgr.copy()
        if self.overlay:
            self._draw_overlay(frame_bgr, lines)
        if self._writer is None:
            h, w = frame_bgr.shape[:2]
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (w, h))
        self._writer.write(frame_bgr)
        self._last_frame_bgr = frame_bgr.copy()
        self._last_raw_frame_bgr = raw_frame_bgr.copy()
        frame_idx = self.frame_count
        self.frame_count += 1
        return frame_idx

    def hold(self, num_frames: int, lines: list[str]) -> None:
        if self._last_raw_frame_bgr is None:
            return
        for _ in range(max(num_frames, 0)):
            frame_bgr = self._last_raw_frame_bgr.copy()
            if self.overlay:
                self._draw_overlay(frame_bgr, lines)
            self._writer.write(frame_bgr)
            self._last_frame_bgr = frame_bgr.copy()
            self.frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    @staticmethod
    def _draw_overlay(frame_bgr: np.ndarray, lines: list[str]) -> None:
        y = 28
        for line in lines:
            cv2.putText(frame_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
            cv2.putText(frame_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            y += 26


def _find_isaac_gr00t_root(model_path: Path) -> Path:
    for candidate in [model_path, *model_path.parents]:
        if (candidate / "gr00t_config.py").exists() and (candidate / "gr00t" / "model" / "policy.py").exists():
            return candidate
    repo_root = Path(__file__).resolve().parents[2] / "Isaac-GR00T"
    if (repo_root / "gr00t_config.py").exists() and (repo_root / "gr00t" / "model" / "policy.py").exists():
        return repo_root
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
        "headless_trocar_gr00t_config",
        str(isaac_gr00t_root / "gr00t_config.py"),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load gr00t_config.py from {isaac_gr00t_root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data_config = module.UnitreeG1SimInferDataConfig()

    return Gr00tPolicy(
        model_path=str(model_path_obj),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=denoising_steps,
        device=model_device,
    )


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


def _get_env_action_state(env) -> np.ndarray:
    """Return the raw env action that reproduces current joint positions."""
    action_term = _get_joint_pos_action_term(env)
    joint_pos = wp.to_torch(env.scene["robot"].data.joint_pos)
    joint_ids = _get_action_joint_ids(env)
    joint_pos_action_order = joint_pos[0, joint_ids].to(dtype=torch.float32)

    scale = _as_action_vector(action_term._scale, action_term.action_dim, joint_pos.device)
    offset = _as_action_vector(action_term._offset, action_term.action_dim, joint_pos.device)
    raw_action = (joint_pos_action_order - offset) / scale
    return _normalize_env_raw_action(raw_action.cpu().numpy())


def _fixed_initial_state_error(env, target_state_ref: np.ndarray) -> float:
    states_ref = _get_joint_states(env)[:, PERM_TO_REF]
    error = np.abs(states_ref - target_state_ref[np.newaxis, :])
    return float(error.max()) if error.size else 0.0


def _apply_fixed_initial_state(
    env,
    fixed_raw_action: np.ndarray,
    target_state_ref: np.ndarray,
    steps: int,
    tolerance: float,
) -> dict:
    """Command a fixed start pose before demo inference begins."""
    action_tensor = _raw_action_to_env_tensor(fixed_raw_action, device=env.device)
    steps_run = 0
    episode_done = False
    max_state_error = _fixed_initial_state_error(env, target_state_ref)

    with torch.inference_mode():
        for warm_step in range(max(steps, 0)):
            _, _, terminated, truncated, _ = env.step(action_tensor)
            steps_run = warm_step + 1
            max_state_error = _fixed_initial_state_error(env, target_state_ref)
            episode_done = bool(terminated[0]) or bool(truncated[0])
            if episode_done or max_state_error <= tolerance:
                break

    return {
        "steps": steps_run,
        "max_state_error": max_state_error,
        "episode_done": episode_done,
        "target_state_ref": target_state_ref.tolist(),
        "target_left_hand_ref": target_state_ref[14:21].tolist(),
        "target_right_hand_ref": target_state_ref[21:28].tolist(),
    }


def _get_camera_rgb(env, cam_name: str) -> np.ndarray:
    sensor = env.scene.sensors[cam_name]
    imgs = sensor.data.output["rgb"]
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().numpy()
    if imgs.shape[-1] == 4:
        imgs = imgs[..., :3]
    return imgs[0].astype(np.uint8)


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"No JSON object found in Reason2 response: {text!r}")
    return json.loads(text[start : end + 1])


def _save_reason2_retry_images(env, episode_idx: int, task_idx: int, retry_idx: int, step_count: int) -> dict[str, str]:
    query_dir = (
        Path(args.reason2_query_dir).expanduser().resolve()
        / f"episode_{episode_idx + 1:06d}"
        / f"task_{task_idx + 1}_retry_{retry_idx}_step_{step_count}"
    )
    query_dir.mkdir(parents=True, exist_ok=True)

    image_paths: dict[str, str] = {}
    rgb_images: dict[str, np.ndarray] = {}
    for cam_name in CAMERA_KEYS:
        rgb = _get_camera_rgb(env, cam_name)
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


def _build_reason2_single_hand_prompt(hand: str, task_idx: int, retry_idx: int) -> str:
    return f"""You are judging an Isaac Sim trocar assembly scene from one horizontally concatenated image.

The image has two equal-width panels from left to right:
1. front room camera,
2. {hand} wrist camera attached to the robot {hand} hand.
Current task being retried: Task {task_idx + 1} - {TASK_DESCRIPTIONS[task_idx]}.
Upcoming retry attempt index: {retry_idx}.

The target object is called a trocar in this task. Visually, a trocar may look like a purple rod,
a purple pencil-like shaft, or a purple-and-white tool with a white cylindrical handle/collar.
If a robot hand is gripping this purple/white tool, count that hand as holding a trocar even if
you would normally describe the object as a pencil-like rod or white cylinder.

Your goal is NOT to judge whether the grasp is perfect. Your goal is to decide whether each robot
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


def _reason2_retry_hand_plan(task_idx: int) -> tuple[dict[str, bool | None], list[str], str]:
    """Return task-prior hand states and the hands that need visual judgment."""
    if task_idx == 0:
        return {"left": None, "right": False}, ["left"], "task1_query_left_only"
    if task_idx == 1:
        return {"left": True, "right": None}, ["right"], "task2_query_right_left_held"
    if task_idx in (2, 3):
        return {"left": True, "right": True}, [], "task3_task4_hold_both"
    return {"left": False, "right": None}, ["right"], "task5_query_right_left_free"


def _query_reason2_retry_hand_status(
    env,
    recorder: VideoRecorder,
    events: list[dict],
    episode_idx: int,
    task_idx: int,
    retry_idx: int,
    step_count: int,
) -> dict:
    """Ask Reason2 which hands currently hold trocars before retry rollback."""
    if not args.use_reason2_retry_judge:
        return {"enabled": False, "ok": False, "source": "disabled"}

    hand_priors, query_hands, hand_plan = _reason2_retry_hand_plan(task_idx)
    hand_status = {
        "left_hand_holding_trocar": hand_priors["left"],
        "right_hand_holding_trocar": hand_priors["right"],
    }
    if not query_hands:
        return {
            "enabled": True,
            "ok": True,
            "source": "reason2_single_hand_selective",
            "hand_plan": hand_plan,
            "queried_hands": [],
            "left_hand_holding_trocar": bool(hand_status["left_hand_holding_trocar"]),
            "right_hand_holding_trocar": bool(hand_status["right_hand_holding_trocar"]),
            "confidence": 1.0,
            "reason": "Skipped Reason2 query because task prior already marks both hands as protected.",
        }

    _event(
        events,
        "reason2_retry_judge_start",
        recorder,
        step_count,
        task=task_idx + 1,
        retry=retry_idx,
        hand_plan=hand_plan,
        queried_hands=query_hands,
    )
    if args.reason2_pause_frames > 0:
        recorder.hold(
            args.reason2_pause_frames,
            [
                "PAUSED FOR REASON2",
                f"Task {task_idx + 1}: {TASK_DESCRIPTIONS[task_idx]}",
                f"Retry: {retry_idx}",
                f"Question: {', '.join(query_hands)} hand(s) holding?",
            ],
        )

    try:
        _flush_observations(env)
        image_paths = _save_reason2_retry_images(env, episode_idx, task_idx, retry_idx, step_count)
        raw_responses = {}
        confidences = {}
        reasons = {}
        for hand in query_hands:
            query_key = f"{hand}_query"
            content = [
                {"type": "image_url", "image_url": {"url": f"file://{image_paths[query_key]}"}},
                {"type": "text", "text": _build_reason2_single_hand_prompt(hand, task_idx, retry_idx)},
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
            response_json = response.json()
            message = response_json["choices"][0]["message"]
            content_text = message.get("content") or ""
            parsed = _extract_json_object(content_text)
            raw_responses[hand] = content_text
            hand_status[f"{hand}_hand_holding_trocar"] = bool(parsed.get(f"{hand}_hand_holding_trocar", False))
            confidences[hand] = float(parsed.get("confidence", 0.0))
            reasons[hand] = str(parsed.get("reason", ""))

        result = {
            "enabled": True,
            "ok": True,
            "source": "reason2_single_hand_selective",
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
    except Exception as exc:
        result = {
            "enabled": True,
            "ok": False,
            "source": "fallback",
            "hand_plan": hand_plan,
            "queried_hands": query_hands,
            "error": repr(exc),
        }

    _event(
        events,
        "reason2_retry_judge_done",
        recorder,
        step_count,
        task=task_idx + 1,
        retry=retry_idx,
        result=result,
    )
    return result


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
        obs_dict[gr00t_key] = _get_camera_rgb(env, cam_key)[np.newaxis]
    return obs_dict


def _normalize_env_raw_action(action: np.ndarray | torch.Tensor) -> np.ndarray:
    action_np = np.asarray(action, dtype=np.float32).reshape(ROBOT_ACTION_DIM).copy()
    action_np[:ACTION_PREFIX_PAD] = 0.0
    return action_np


def _raw_action_to_env_tensor(action: np.ndarray, device: str) -> torch.Tensor:
    action = _normalize_env_raw_action(action).reshape(1, ROBOT_ACTION_DIM)
    return torch.tensor(action, dtype=torch.float32, device=device)


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


def _stage_pred_confidence(stage_pred: int | None, stage_probs: list[float] | None) -> float | None:
    if stage_pred is None or stage_probs is None:
        return None
    if stage_pred < 0 or stage_pred >= len(stage_probs):
        return None
    return stage_probs[stage_pred]


def _stage_pred_success(
    task_idx: int,
    stage_pred: int | None,
    stage_probs: list[float] | None,
) -> bool:
    if stage_pred is None:
        return False
    confidence = _stage_pred_confidence(stage_pred, stage_probs)
    if confidence is not None and confidence < args.stage_pred_min_confidence:
        return False
    return stage_pred >= task_idx + 1


def _get_stage_info(env) -> tuple[int, int]:
    env_stage = int(env._task_stage[0].item()) if hasattr(env, "_task_stage") else 0
    lift_z = 0.85483 + 0.05
    t1_pos = wp.to_torch(env.scene["trocar_1"].data.root_pos_w)[0]
    t2_pos = wp.to_torch(env.scene["trocar_2"].data.root_pos_w)[0]
    either_lifted = bool((t1_pos[2] > lift_z) or (t2_pos[2] > lift_z))
    dataset_stage = env_stage + 1 if env_stage >= 1 else (1 if either_lifted else 0)
    return env_stage, dataset_stage


def _event(events: list[dict], name: str, recorder: VideoRecorder, step_count: int, **kwargs) -> None:
    record = {
        "event": name,
        "video_frame": recorder.frame_count,
        "env_step": step_count,
    }
    record.update(kwargs)
    events.append(record)
    print(f"[EVENT] {json.dumps(record, ensure_ascii=False)}")


def _task_step_limit(task_idx: int) -> int:
    if task_idx == 0 and args.task1_max_steps is not None:
        return args.task1_max_steps
    if task_idx == 1 and args.task2_max_steps is not None:
        return args.task2_max_steps
    if task_idx == 2 and args.task3_max_steps is not None:
        return args.task3_max_steps
    if task_idx == 3 and args.task4_max_steps is not None:
        return args.task4_max_steps
    if task_idx == 4 and args.task5_max_steps is not None:
        return args.task5_max_steps
    return args.task_max_steps


def _task_max_retries(task_idx: int) -> int:
    if task_idx == 2:
        return args.task3_max_retries
    if task_idx == 3:
        return args.task4_max_retries
    return args.task_max_retries


def _overlay_lines(mode: str, task_idx: int, local_step: int, step_count: int, retry_idx: int | None, env) -> list[str]:
    _, dataset_stage = _get_stage_info(env)
    retry_text = "initial" if retry_idx is None else f"retry {retry_idx}"
    if mode.startswith("BACK TO"):
        step_text = f"Return step: {local_step} / {args.back_to_init_steps}"
    else:
        step_text = f"Task step: {local_step} / {_task_step_limit(task_idx)}"
    return [
        f"Mode: {mode}",
        f"Task {task_idx + 1}: {TASK_DESCRIPTIONS[task_idx]}",
        f"{step_text}   {retry_text}",
        f"Env step: {step_count}   Dataset stage: {dataset_stage}",
    ]


def _write_current_frame(
    env,
    recorder: VideoRecorder,
    mode: str,
    task_idx: int,
    local_step: int,
    step_count: int,
    retry_idx: int | None = None,
) -> int:
    frame = _get_camera_rgb(env, args.camera)
    return recorder.write(frame, _overlay_lines(mode, task_idx, local_step, step_count, retry_idx, env))


def _maybe_apply_tray_perturbation(
    env,
    recorder: VideoRecorder,
    events: list[dict],
    step_count: int,
    perturb_state: dict,
    task_idx: int,
    task_step: int,
) -> None:
    if args.perturb_tray_after_steps < 0 or perturb_state.get("completed", False):
        return
    if task_idx != args.perturb_tray_task - 1:
        return
    if task_step < args.perturb_tray_after_steps:
        return

    duration_steps = max(args.perturb_tray_duration_steps, 1)
    just_started = False
    if not perturb_state.get("started", False):
        perturb_state["started"] = True
        perturb_state["steps_run"] = 0
        just_started = True

    steps_run = int(perturb_state.get("steps_run", 0)) + 1
    alpha = min(steps_run / duration_steps, 1.0)
    yaw_deg = (1.0 - alpha) * args.initial_tray_yaw_deg + alpha * args.perturb_tray_yaw_deg
    previous_yaw_deg = float(perturb_state.get("current_yaw_deg", args.initial_tray_yaw_deg))
    carry_status = _set_tray_trocar_yaw_delta(
        env,
        yaw_deg,
        previous_yaw_deg=previous_yaw_deg,
        carry_trocars_on_tray_only=True,
    )
    _flush_observations(env)
    perturb_state["steps_run"] = steps_run
    perturb_state["current_yaw_deg"] = yaw_deg
    perturb_state["last_carry_status"] = carry_status

    if just_started:
        env_stage, dataset_stage = _get_stage_info(env)
        _event(
            events,
            "tray_trocar_yaw_perturbation_start",
            recorder,
            step_count,
            task=task_idx + 1,
            task_step=task_step,
            from_yaw_deg=args.initial_tray_yaw_deg,
            to_yaw_deg=args.perturb_tray_yaw_deg,
            duration_steps=duration_steps,
            carry_trocars_on_tray_only=True,
            carry_status=carry_status,
            raw_stage=env_stage,
            dataset_stage=dataset_stage,
        )

    if steps_run >= duration_steps:
        env_stage, dataset_stage = _get_stage_info(env)
        perturb_state["completed"] = True
        _event(
            events,
            "tray_trocar_yaw_perturbation_done",
            recorder,
            step_count,
            task=task_idx + 1,
            task_step=task_step,
            yaw_deg=yaw_deg,
            carry_status=carry_status,
            raw_stage=env_stage,
            dataset_stage=dataset_stage,
        )


def _infer_action_chunk(policy, env, task_idx: int) -> tuple[list[np.ndarray], int | None, list[float] | None]:
    task_description = TASK_DESCRIPTIONS[task_idx]
    policy_obs = _build_policy_obs(env, task_description)
    action_dict = policy.get_action(policy_obs)
    action_chunk = _convert_policy_action_chunk_to_env(action_dict)
    if not action_chunk:
        raise RuntimeError("Policy returned an empty action chunk.")
    stage_pred, stage_probs = _extract_stage_prediction(action_dict)

    stage_pred_prompt = args.stage_pred_prompt.strip()
    if stage_pred_prompt:
        stage_obs = _build_policy_obs(env, stage_pred_prompt)
        stage_action_dict = policy.get_action(stage_obs)
        stage_pred, stage_probs = _extract_stage_prediction(stage_action_dict)

    return [action.copy() for action in action_chunk[: max(args.open_loop_steps, 1)]], stage_pred, stage_probs


def _run_task_attempt(
    policy,
    env,
    recorder: VideoRecorder,
    events: list[dict],
    task_idx: int,
    step_count: int,
    perturb_state: dict,
    retry_idx: int | None = None,
) -> tuple[TaskResult, int]:
    target_dataset_stage = task_idx + 1
    step_limit = _task_step_limit(task_idx)
    _, dataset_stage = _get_stage_info(env)
    stage_pred_target = task_idx + 1
    _event(
        events,
        "task_start",
        recorder,
        step_count,
        task=task_idx + 1,
        retry=retry_idx,
        dataset_stage=dataset_stage,
        max_steps=step_limit,
        success_source=args.task_success_source,
        stage_pred_target=stage_pred_target,
    )
    if args.task_success_source == "env_oracle" and dataset_stage >= target_dataset_stage:
        _event(events, "task_already_complete", recorder, step_count, task=task_idx + 1, retry=retry_idx)
        return TaskResult(success=True, episode_done=False), step_count

    cached_actions: list[np.ndarray] = []
    stage_pred: int | None = None
    stage_probs: list[float] | None = None
    episode_done = False
    success = False

    with torch.inference_mode():
        for local_step in range(step_limit):
            if not cached_actions:
                cached_actions, stage_pred, stage_probs = _infer_action_chunk(policy, env, task_idx)
            action = cached_actions.pop(0)

            action_tensor = _raw_action_to_env_tensor(action, device=env.device)
            _, _, terminated, truncated, _ = env.step(action_tensor)
            step_count += 1
            task_step = local_step + 1
            _maybe_apply_tray_perturbation(env, recorder, events, step_count, perturb_state, task_idx, task_step)
            _write_current_frame(env, recorder, "POLICY", task_idx, task_step, step_count, retry_idx)

            env_stage, dataset_stage = _get_stage_info(env)
            episode_done = bool(terminated[0]) or bool(truncated[0])
            stage_pred_confidence = _stage_pred_confidence(stage_pred, stage_probs)
            if args.task_success_source == "env_oracle":
                task_success = dataset_stage >= target_dataset_stage
            else:
                task_success = _stage_pred_success(task_idx, stage_pred, stage_probs)
            if task_success:
                success = True
                _event(
                    events,
                    "task_success",
                    recorder,
                    step_count,
                    task=task_idx + 1,
                    retry=retry_idx,
                    raw_stage=env_stage,
                    dataset_stage=dataset_stage,
                    stage_pred=stage_pred,
                    stage_pred_confidence=stage_pred_confidence,
                    stage_pred_probs=stage_probs,
                    success_source=args.task_success_source,
                    task_steps=local_step + 1,
                )
                break
            if episode_done:
                _event(
                    events,
                    "episode_done",
                    recorder,
                    step_count,
                    task=task_idx + 1,
                    retry=retry_idx,
                    raw_stage=env_stage,
                    dataset_stage=dataset_stage,
                    stage_pred=stage_pred,
                    stage_pred_confidence=stage_pred_confidence,
                    stage_pred_probs=stage_probs,
                    success_source=args.task_success_source,
                    task_steps=local_step + 1,
                )
                break

    if not success and not episode_done:
        env_stage, dataset_stage = _get_stage_info(env)
        stage_pred_confidence = _stage_pred_confidence(stage_pred, stage_probs)
        _event(
            events,
            "task_timeout",
            recorder,
            step_count,
            task=task_idx + 1,
            retry=retry_idx,
            raw_stage=env_stage,
            dataset_stage=dataset_stage,
            stage_pred=stage_pred,
            stage_pred_confidence=stage_pred_confidence,
            stage_pred_probs=stage_probs,
            success_source=args.task_success_source,
            task_steps=step_limit,
        )
    return TaskResult(success=success, episode_done=episode_done), step_count


def _return_joint_groups(
    task_idx: int,
    retry_hand_status: dict | None = None,
) -> tuple[np.ndarray, list[str], list[str], str]:
    """Return action indices controlled during the retry return motion."""
    controlled_indices = [ARM_ACTION_INDICES]
    controlled_groups = ["arms"]
    held_groups: list[str] = []

    reason2_sources = {"reason2", "reason2_single_hand_selective"}
    if retry_hand_status and retry_hand_status.get("ok") and retry_hand_status.get("source") in reason2_sources:
        if bool(retry_hand_status.get("left_hand_holding_trocar", False)):
            held_groups.append("left_hand")
        else:
            controlled_indices.append(LEFT_HAND_ACTION_INDICES)
            controlled_groups.append("left_hand")

        if bool(retry_hand_status.get("right_hand_holding_trocar", False)):
            held_groups.append("right_hand")
        else:
            controlled_indices.append(RIGHT_HAND_ACTION_INDICES)
            controlled_groups.append("right_hand")

        return np.concatenate(controlled_indices), controlled_groups, held_groups, "reason2_hand_status"

    if task_idx == 0:
        controlled_indices.extend([LEFT_HAND_ACTION_INDICES, RIGHT_HAND_ACTION_INDICES])
        controlled_groups.extend(["left_hand", "right_hand"])
    elif task_idx == 1:
        controlled_indices.append(RIGHT_HAND_ACTION_INDICES)
        controlled_groups.append("right_hand")
        held_groups.append("left_hand")
    else:
        held_groups.extend(["left_hand", "right_hand"])

    return np.concatenate(controlled_indices), controlled_groups, held_groups, "task_default"


def _controlled_joint_error(
    env,
    target_action_state: np.ndarray,
    controlled_indices: np.ndarray = ARM_ACTION_INDICES,
) -> float:
    current_action_state = _get_env_action_state(env)
    error = np.abs(current_action_state[controlled_indices] - target_action_state[controlled_indices])
    return float(error.max()) if error.size else 0.0


def _back_to_task_init(
    env,
    recorder: VideoRecorder,
    events: list[dict],
    task_idx: int,
    task_init_state: np.ndarray,
    step_count: int,
    retry_idx: int,
    retry_hand_status: dict | None = None,
) -> tuple[bool, int]:
    start_state = _get_env_action_state(env)
    controlled_indices, controlled_groups, held_groups, rollback_policy_source = _return_joint_groups(
        task_idx, retry_hand_status
    )
    initial_error = _controlled_joint_error(env, task_init_state, controlled_indices)
    initial_arm_error = _controlled_joint_error(env, task_init_state, ARM_ACTION_INDICES)
    _event(
        events,
        "back_to_init_start",
        recorder,
        step_count,
        task=task_idx + 1,
        retry=retry_idx,
        max_return_error=initial_error,
        max_arm_error=initial_arm_error,
        initial_arm_error=initial_arm_error,
        max_steps=args.back_to_init_steps,
        controlled_joint_groups=controlled_groups,
        held_joint_groups=held_groups,
        rollback_policy_source=rollback_policy_source,
        retry_hand_status=retry_hand_status,
    )
    if initial_error <= args.back_to_init_tolerance:
        _event(
            events,
            "back_to_init_already_close",
            recorder,
            step_count,
            task=task_idx + 1,
            retry=retry_idx,
            max_return_error=initial_error,
            max_arm_error=initial_arm_error,
            initial_arm_error=initial_arm_error,
            controlled_joint_groups=controlled_groups,
            held_joint_groups=held_groups,
            rollback_policy_source=rollback_policy_source,
            retry_hand_status=retry_hand_status,
        )
        return False, step_count

    episode_done = False
    with torch.inference_mode():
        for return_step in range(1, args.back_to_init_steps + 1):
            alpha = return_step / max(args.back_to_init_steps, 1)
            target = start_state.copy()
            target[controlled_indices] = (
                (1.0 - alpha) * start_state[controlled_indices] + alpha * task_init_state[controlled_indices]
            )
            action_tensor = _raw_action_to_env_tensor(target, device=env.device)
            _, _, terminated, truncated, _ = env.step(action_tensor)
            step_count += 1
            _write_current_frame(
                env,
                recorder,
                f"BACK TO TASK {task_idx + 1} INIT STATE",
                task_idx,
                return_step,
                step_count,
                retry_idx,
            )
            episode_done = bool(terminated[0]) or bool(truncated[0])
            max_return_error = _controlled_joint_error(env, task_init_state, controlled_indices)
            if episode_done or max_return_error <= args.back_to_init_tolerance:
                break

    env_stage, dataset_stage = _get_stage_info(env)
    max_return_error = _controlled_joint_error(env, task_init_state, controlled_indices)
    max_arm_error = _controlled_joint_error(env, task_init_state, ARM_ACTION_INDICES)
    _event(
        events,
        "back_to_init_done",
        recorder,
        step_count,
        task=task_idx + 1,
        retry=retry_idx,
        raw_stage=env_stage,
        dataset_stage=dataset_stage,
        max_return_error=max_return_error,
        max_arm_error=max_arm_error,
        controlled_joint_groups=controlled_groups,
        held_joint_groups=held_groups,
        rollback_policy_source=rollback_policy_source,
        retry_hand_status=retry_hand_status,
        episode_done=episode_done,
    )
    return episode_done, step_count


def _run_retryable_task(
    policy,
    env,
    recorder: VideoRecorder,
    events: list[dict],
    episode_idx: int,
    task_idx: int,
    step_count: int,
    max_retries: int,
    perturb_state: dict,
) -> tuple[bool, bool, int, dict]:
    task_init_state = _get_env_action_state(env)
    env_stage, dataset_stage = _get_stage_info(env)
    _event(
        events,
        "task_init_state_recorded",
        recorder,
        step_count,
        task=task_idx + 1,
        raw_stage=env_stage,
        dataset_stage=dataset_stage,
    )

    success = False
    episode_done = False
    attempts = 0
    retries_used = 0
    reason2_retry_judgments: list[dict] = []
    for attempt_idx in range(max_retries + 1):
        retry_idx = None if attempt_idx == 0 else attempt_idx
        attempts += 1
        result, step_count = _run_task_attempt(
            policy,
            env,
            recorder,
            events,
            task_idx=task_idx,
            step_count=step_count,
            perturb_state=perturb_state,
            retry_idx=retry_idx,
        )
        success = result.success
        episode_done = result.episode_done
        if success or episode_done:
            break
        if attempt_idx >= max_retries:
            break
        retry_hand_status = _query_reason2_retry_hand_status(
            env,
            recorder,
            events,
            episode_idx,
            task_idx,
            attempt_idx + 1,
            step_count,
        )
        reason2_retry_judgments.append(retry_hand_status)
        episode_done, step_count = _back_to_task_init(
            env,
            recorder,
            events,
            task_idx,
            task_init_state,
            step_count,
            retry_idx=attempt_idx + 1,
            retry_hand_status=retry_hand_status,
        )
        retries_used += 1
        if episode_done:
            break

    stats = {
        "task": task_idx + 1,
        "success": success,
        "attempts": attempts,
        "retries_used": retries_used,
        "max_retries": max_retries,
        "reason2_retry_judgments": reason2_retry_judgments,
    }
    return success, episode_done, step_count, stats


def _episode_output_paths(output_dir: Path, episode_idx: int) -> tuple[Path, Path]:
    video_path = output_dir / args.video_name
    if args.num_episodes == 1:
        return video_path, output_dir / "events.json"

    episode_number = episode_idx + 1
    suffix = video_path.suffix or ".mp4"
    video_path = video_path.with_name(f"{video_path.stem}_episode_{episode_number:06d}{suffix}")
    return video_path, output_dir / f"events_episode_{episode_number:06d}.json"


def _run_episode(
    policy,
    env,
    episode_idx: int,
    video_path: Path,
    events_path: Path,
    fixed_initial_state_ref: np.ndarray | None,
) -> dict:
    recorder = VideoRecorder(video_path, fps=args.fps, overlay=not args.no_overlay)
    events: list[dict] = []
    step_count = 0
    episode_number = episode_idx + 1
    episode_seed = None if args.seed is None else args.seed + episode_idx

    task3_success = False
    task4_success = False
    task5_success = False
    episode_done = False
    first_failed_task: int | None = None
    task_results: list[dict] = []
    task_successes = [False] * len(TASK_DESCRIPTIONS)
    perturb_state = {
        "started": False,
        "completed": False,
        "steps_run": 0,
        "current_yaw_deg": args.initial_tray_yaw_deg,
    }

    try:
        env.reset(seed=episode_seed)
        _set_tray_trocar_yaw_delta(env, args.initial_tray_yaw_deg)
        fixed_initial_info = None
        if fixed_initial_state_ref is not None:
            fixed_initial_raw_action = _fixed_initial_state_ref_to_raw_action(env, fixed_initial_state_ref)
            fixed_initial_info = _apply_fixed_initial_state(
                env,
                fixed_initial_raw_action,
                fixed_initial_state_ref,
                args.fixed_initial_state_steps,
                args.fixed_initial_state_tolerance,
            )
        _flush_observations(env)
        _write_current_frame(env, recorder, "RESET", 0, 0, step_count, None)
        env_stage, dataset_stage = _get_stage_info(env)
        _event(
            events,
            "reset",
            recorder,
            step_count,
            episode_index=episode_number,
            seed=episode_seed,
            raw_stage=env_stage,
            dataset_stage=dataset_stage,
            initial_tray_yaw_deg=args.initial_tray_yaw_deg,
            perturb_tray_task=args.perturb_tray_task,
            perturb_tray_after_steps=args.perturb_tray_after_steps,
            perturb_tray_yaw_deg=args.perturb_tray_yaw_deg,
            perturb_tray_duration_steps=max(args.perturb_tray_duration_steps, 1),
            fixed_initial_state=fixed_initial_info,
        )

        for task_idx in range(len(TASK_DESCRIPTIONS)):
            task_success, episode_done, step_count, task_stats = _run_retryable_task(
                policy,
                env,
                recorder,
                events,
                episode_idx=episode_idx,
                task_idx=task_idx,
                step_count=step_count,
                max_retries=_task_max_retries(task_idx),
                perturb_state=perturb_state,
            )
            task_successes[task_idx] = task_success
            task_results.append(task_stats)
            if not task_success:
                first_failed_task = task_idx + 1
                break
            if episode_done:
                break

        task3_success = task_successes[2]
        task4_success = task_successes[3]
        task5_success = task_successes[4]

        env_stage, dataset_stage = _get_stage_info(env)
        if task5_success:
            final_event = "paused_after_task5_success"
        elif first_failed_task is not None:
            final_event = f"paused_after_task{first_failed_task}_retries"
        else:
            final_event = "paused_after_incomplete_task_sequence"
        if episode_done and not task5_success:
            final_event = "paused_after_episode_done"
        _event(
            events,
            final_event,
            recorder,
            step_count,
            raw_stage=env_stage,
            dataset_stage=dataset_stage,
            task3_success=task3_success,
            task4_success=task4_success,
            task5_success=task5_success,
            first_failed_task=first_failed_task,
            task_successes={str(i + 1): value for i, value in enumerate(task_successes)},
            task_results=task_results,
            tray_perturbation=perturb_state,
        )
        recorder.hold(
            args.final_hold_frames,
            [
                "PAUSED",
                f"First failed task: {first_failed_task}",
                f"Task 1 success: {task_successes[0]}",
                f"Task 2 success: {task_successes[1]}",
                f"Task 3 success: {task3_success}",
                f"Task 4 success: {task4_success}",
                f"Task 5 success: {task5_success}",
                f"Dataset stage: {dataset_stage}",
                f"Video frame: {recorder.frame_count}",
            ],
        )
    finally:
        recorder.close()
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.write_text(
            json.dumps(
                {
                    "episode_index": episode_number,
                    "seed": episode_seed,
                    "video": str(video_path),
                    "events": events,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    summary = {
        "episode_index": episode_number,
        "seed": episode_seed,
        "video": str(video_path),
        "events": str(events_path),
        "video_frames": recorder.frame_count,
        "env_steps": step_count,
        "episode_done": episode_done,
        "first_failed_task": first_failed_task,
        "task3_success": task3_success,
        "task4_success": task4_success,
        "task5_success": task5_success,
        "task_successes": {str(i + 1): value for i, value in enumerate(task_successes)},
        "task_results": task_results,
        "fixed_initial_state": fixed_initial_info,
    }
    episode_label = f"{episode_number}/{args.num_episodes}"
    print(f"[INFO] Episode {episode_label}: saved MP4: {video_path}")
    print(f"[INFO] Episode {episode_label}: saved events: {events_path}")
    print(f"[INFO] Episode {episode_label}: total video frames: {recorder.frame_count}")
    return summary


def main() -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_device = args.model_device or args.device
    policy = _load_policy(args.model_path, model_device=model_device, denoising_steps=args.denoising_steps)
    summaries: list[dict] = []

    fixed_initial_state_ref = None
    if args.fixed_initial_state_dataset:
        fixed_initial_state_ref = _load_fixed_initial_state_ref(
            args.fixed_initial_state_dataset,
            args.fixed_initial_state_episode,
            args.fixed_initial_state_frame,
        )
        print(
            "[INFO] Fixed initial state enabled: "
            f"dataset={args.fixed_initial_state_dataset}, "
            f"episode={args.fixed_initial_state_episode}, frame={args.fixed_initial_state_frame}, "
            f"warmup_steps={args.fixed_initial_state_steps}"
        )
        print(f"[INFO] Fixed initial left hand ref: {fixed_initial_state_ref[14:21].tolist()}")
        print(f"[INFO] Fixed initial right hand ref: {fixed_initial_state_ref[21:28].tolist()}")

    try:
        for episode_idx in range(args.num_episodes):
            episode_number = episode_idx + 1
            video_path, events_path = _episode_output_paths(output_dir, episode_idx)
            print(f"[INFO] Starting episode {episode_number}/{args.num_episodes}")
            env = _create_env(args.task_id, device=args.device)
            try:
                summaries.append(
                    _run_episode(policy, env, episode_idx, video_path, events_path, fixed_initial_state_ref)
                )
            except Exception as exc:
                print(f"[ERROR] Episode {episode_number}/{args.num_episodes} failed: {exc!r}")
                summaries.append(
                    {
                        "episode_index": episode_number,
                        "seed": None if args.seed is None else args.seed + episode_idx,
                        "video": str(video_path),
                        "events": str(events_path),
                        "error": repr(exc),
                    }
                )
            finally:
                env.close()
    finally:
        simulation_app.close()

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"episodes": summaries}, indent=2), encoding="utf-8")
    print(f"[INFO] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
