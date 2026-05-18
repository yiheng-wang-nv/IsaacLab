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
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run an interactive GUI for the task-complete trocar policy.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the Isaac-GR00T checkpoint directory.")
parser.add_argument(
    "--gr00t_root",
    type=str,
    default=None,
    help="Path to the Isaac-GR00T repository that contains gr00t_config.py.",
)
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
    default=60,
    help="Pause continuous inference when this many steps have elapsed.",
)
parser.add_argument(
    "--task_complete_threshold",
    type=float,
    default=0.5,
    help="Predicted task_complete value above which auto-pause triggers.",
)
parser.add_argument(
    "--task3_complete_threshold",
    type=float,
    default=None,
    help="Optional task_complete auto-pause threshold override for task 3.",
)
parser.add_argument(
    "--auto_multistage_direct",
    action="store_true",
    help="Run tasks automatically without keyboard/GUI control, advancing only when progress reaches threshold.",
)
parser.add_argument(
    "--auto_task_indices",
    type=str,
    default="0,1,2,3,4",
    help="Comma-separated zero-based task indices for direct multi-stage inference.",
)
parser.add_argument(
    "--auto_task_thresholds",
    type=str,
    default="0.98,0.98,0.999,0.98,0.98",
    help="Comma-separated thresholds matching --auto_task_indices, or one value per known task.",
)
parser.add_argument(
    "--auto_num_episodes",
    type=int,
    default=1,
    help="Number of environment episodes to run in direct multi-stage inference.",
)
parser.add_argument(
    "--auto_total_steps_per_task",
    type=int,
    default=300,
    help="Maximum policy env steps allowed per task in direct multi-stage inference.",
)
parser.add_argument(
    "--auto_no_retry_last_task",
    action="store_true",
    help="In direct multi-stage inference, run the final configured task for only one attempt.",
)
parser.add_argument(
    "--auto_single_attempt_task_indices",
    type=str,
    default="0,1",
    help="Comma-separated zero-based task indices that should run only one attempt in direct mode.",
)
parser.add_argument(
    "--auto_hold_height_task_indices",
    type=str,
    default="2,3",
    help="Comma-separated zero-based task indices where trocars must stay above table height plus margin.",
)
parser.add_argument(
    "--auto_hold_table_height",
    type=float,
    default=0.85483,
    help="Table height used by direct-mode held-trocar failure checks.",
)
parser.add_argument(
    "--auto_hold_height_margin",
    type=float,
    default=0.05,
    help="Margin above table height required during --auto_hold_height_task_indices.",
)
parser.add_argument(
    "--auto_record_video",
    action="store_true",
    help="Record the three camera views during direct multi-stage inference.",
)
parser.add_argument(
    "--auto_video_dir",
    type=str,
    default="multistage_direct_videos",
    help="Output directory for direct multi-stage camera videos.",
)
parser.add_argument(
    "--auto_video_fps",
    type=float,
    default=30.0,
    help="FPS used when writing direct multi-stage videos.",
)
parser.add_argument(
    "--task_broken_drop_threshold",
    type=float,
    default=0.30,
    help="Pause and suggest recover if current task progress drops this much from the previous prediction.",
)
parser.add_argument(
    "--task_broken_enable_threshold",
    type=float,
    default=0.80,
    help="Only enable broken-task drop detection after current task progress has exceeded this value.",
)
parser.add_argument(
    "--task_stuck_window",
    type=int,
    default=0,
    help="Pause and suggest recover if progress does not improve enough over this many predictions. Disabled when <= 1.",
)
parser.add_argument(
    "--task_stuck_min_progress_delta",
    type=float,
    default=0.0,
    help="Minimum required progress increase over --task_stuck_window predictions. Disabled when <= 0.",
)
parser.add_argument(
    "--task_stuck_enable_threshold",
    type=float,
    default=0.50,
    help="Only enable stuck-task detection after current task progress has exceeded this value.",
)
parser.add_argument(
    "--stage_precondition_threshold",
    type=float,
    default=0.95,
    help="For task N>1, require task N-1 progress to be at least this value before executing task N.",
)
parser.add_argument(
    "--progress_regressor_path",
    type=str,
    default=None,
    help="Optional task-progress regressor checkpoint. If set, use it instead of action.task_complete.",
)
parser.add_argument(
    "--enable_stage_precondition",
    action="store_true",
    default=True,
    help="Gate task N>1 execution on task N-1 progress.",
)
parser.add_argument(
    "--disable_stage_precondition",
    action="store_false",
    dest="enable_stage_precondition",
    help="Disable the previous-task completion gate.",
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
    "--recover_steps",
    type=int,
    default=200,
    help="Maximum env steps used to drive the robot back toward the current task-start pose on recover.",
)
parser.add_argument(
    "--task34_recover_reference_dataset",
    type=str,
    default=None,
    help="Dataset directory for the fixed task 3/4 recover target. Defaults to --fixed_initial_state_dataset.",
)
parser.add_argument(
    "--task34_recover_reference_episode",
    type=int,
    default=2,
    help="Episode index used as the fixed task 3/4 recover target.",
)
parser.add_argument(
    "--task34_recover_reference_frame",
    type=int,
    default=10,
    help="Frame index used as the fixed task 3/4 recover target.",
)
parser.add_argument(
    "--task34_recover_spread_steps",
    type=int,
    default=30,
    help="For task 3/4 recover, first drive arms outward for this many env steps before normal recover.",
)
parser.add_argument(
    "--task34_recover_left_arm_delta",
    type=str,
    default="0.0,0.0,0.0,0.35,0.0,0.0,0.0",
    help="Comma-separated 7-DoF delta added to the current left arm during task 3/4 spread.",
)
parser.add_argument(
    "--task34_recover_right_arm_delta",
    type=str,
    default="0.0,0.0,0.0,0.35,0.0,0.0,0.0",
    help="Comma-separated 7-DoF delta added to the current right arm during task 3/4 spread.",
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
args.headless = bool(args.auto_multistage_direct)
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

if not args.auto_multistage_direct and getattr(args, "visualizer", None) in (None, []):
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

TASK_OVERLAY_LABELS = [
    "Left trocar pickup",
    "Right trocar pickup",
    "Align trocars",
    "Insert trocar",
    "Placement",
]

CAMERA_KEYS = ["front_camera", "left_wrist_camera", "right_wrist_camera"]

PERM_TO_REF = list(range(28))  # identity — dataset recorded with record_trocar_episodes.py
INV_PERM = np.argsort(PERM_TO_REF)
BODY_JOINT_INDICES = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
DEX3_JOINT_INDICES = [31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38]
SHOULDER_SLICE = (15, 29)
ACTION_PREFIX_PAD = 15
ROBOT_ACTION_DIM = ACTION_PREFIX_PAD + len(PERM_TO_REF)
STATE_JOINT_NAMES = [
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
    "kLeftHandThumb0",
    "kLeftHandThumb1",
    "kLeftHandThumb2",
    "kLeftHandMiddle0",
    "kLeftHandMiddle1",
    "kLeftHandIndex0",
    "kLeftHandIndex1",
    "kRightHandThumb0",
    "kRightHandThumb1",
    "kRightHandThumb2",
    "kRightHandIndex0",
    "kRightHandIndex1",
    "kRightHandMiddle0",
    "kRightHandMiddle1",
]


def _format_top_state_errors(current_state_ref: np.ndarray, target_state_ref: np.ndarray, top_k: int = 5) -> str:
    errors = np.abs(np.asarray(current_state_ref, dtype=np.float32) - np.asarray(target_state_ref, dtype=np.float32))
    top_indices = np.argsort(errors)[-top_k:][::-1]
    return ", ".join(
        f"{STATE_JOINT_NAMES[i]}={errors[i]:.4f}"
        for i in top_indices
    )


def _task_complete_threshold(task_idx: int) -> float:
    if task_idx == 2 and args.task3_complete_threshold is not None:
        return args.task3_complete_threshold
    return args.task_complete_threshold


def _parse_index_list(value: str, name: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{name} must contain at least one index.")
    indices = [int(p) for p in parts]
    invalid = [idx for idx in indices if idx < 0 or idx >= len(TASK_DESCRIPTIONS)]
    if invalid:
        raise ValueError(f"{name} contains invalid task indices {invalid}; valid range is 0-{len(TASK_DESCRIPTIONS) - 1}.")
    return indices


def _parse_optional_index_set(value: str, name: str) -> set[int]:
    if not value.strip():
        return set()
    return set(_parse_index_list(value, name))


def _parse_float_list(value: str, name: str) -> list[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{name} must contain at least one float.")
    return [float(p) for p in parts]


def _direct_task_thresholds(task_indices: list[int]) -> dict[int, float]:
    values = _parse_float_list(args.auto_task_thresholds, "--auto_task_thresholds")
    if len(values) == 1:
        return {idx: values[0] for idx in task_indices}
    if len(values) == len(task_indices):
        return {idx: value for idx, value in zip(task_indices, values, strict=True)}
    if len(values) == len(TASK_DESCRIPTIONS):
        return {idx: values[idx] for idx in task_indices}
    raise ValueError(
        "--auto_task_thresholds must contain 1 value, one value per --auto_task_indices, "
        f"or {len(TASK_DESCRIPTIONS)} values for all tasks; got {len(values)}."
    )


def _print_controls() -> None:
    print("Interactive trocar task-complete controls:")
    print("  1-5   : select stage task prompt")
    print("  SPACE / P : toggle continuous inference")
    print("  S : run one policy step")
    print("  G : probe task_complete prediction without stepping")
    print("  T : recover robot toward the current task-start state")
    print("  O : rotate the tray by the configured yaw delta")
    print("  B : rotate the tray back to its reset yaw")
    print("  R : reset the environment")
    print("  H : print this help again")


def _task_overlay_label(task_idx: int, retry: bool = False) -> str:
    label = TASK_OVERLAY_LABELS[task_idx] if 0 <= task_idx < len(TASK_OVERLAY_LABELS) else f"Task {task_idx + 1}"
    return f"{label} RETRY" if retry else label


def _find_isaac_gr00t_root(model_path: Path, gr00t_root: str | None = None) -> Path:
    if gr00t_root is not None:
        candidate = Path(gr00t_root).expanduser().resolve()
        if (candidate / "gr00t_config.py").exists() and (candidate / "gr00t" / "model" / "policy.py").exists():
            return candidate
        raise FileNotFoundError(f"Could not locate Isaac-GR00T root at {candidate}.")
    for candidate in [model_path, *model_path.parents]:
        if (candidate / "gr00t_config.py").exists() and (candidate / "gr00t" / "model" / "policy.py").exists():
            return candidate
    env_root = Path("/localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T")
    if (env_root / "gr00t_config.py").exists():
        return env_root
    raise FileNotFoundError("Could not locate Isaac-GR00T root.")


def _load_policy(model_path: str, model_device: str, denoising_steps: int, progress_regressor_path: str | None = None):
    model_path_obj = Path(model_path).expanduser().resolve()
    isaac_gr00t_root = _find_isaac_gr00t_root(model_path_obj, args.gr00t_root)
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
        progress_regressor_path=progress_regressor_path,
        progress_task_descriptions=TASK_DESCRIPTIONS,
    )
    return policy


def _create_env(task_id: str, device: str):
    env_cfg = parse_env_cfg(task_id, device=device, num_envs=1)
    for cam_name in CAMERA_KEYS:
        cam_cfg = getattr(env_cfg.scene, cam_name)
        cam_cfg.data_types = ["rgb"]
    disabled_terms = []
    terminations_to_disable = ("time_out", "object_drop") if args.auto_multistage_direct else (
        "time_out",
        "task_success",
        "object_drop",
    )
    for term_name in terminations_to_disable:
        if hasattr(env_cfg.terminations, term_name):
            setattr(env_cfg.terminations, term_name, None)
            disabled_terms.append(term_name)
    if disabled_terms:
        mode_text = "direct success is env-native" if args.auto_multistage_direct else "interactive stopping is prediction-driven"
        print(
            "[INFO] Disabled env terminations "
            f"({', '.join(disabled_terms)}); {mode_text}."
        )
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


def _get_current_state_ref(env) -> np.ndarray:
    return _get_joint_states(env)[0, PERM_TO_REF].copy()


def _get_env_task_stage(env) -> int | None:
    stage = getattr(env, "_task_stage", None)
    if stage is None:
        return None
    if isinstance(stage, torch.Tensor):
        return int(stage[0].item())
    return int(stage[0])


def _trocar_dropped(env, drop_height_threshold: float = 0.5) -> bool:
    pos1 = wp.to_torch(env.scene["trocar_1"].data.root_pos_w)
    pos2 = wp.to_torch(env.scene["trocar_2"].data.root_pos_w)
    return bool(((pos1[:, 2] < drop_height_threshold) | (pos2[:, 2] < drop_height_threshold)).any().item())


def _trocar_below_height(env, min_height: float) -> bool:
    pos1 = wp.to_torch(env.scene["trocar_1"].data.root_pos_w)
    pos2 = wp.to_torch(env.scene["trocar_2"].data.root_pos_w)
    return bool(((pos1[:, 2] < min_height) | (pos2[:, 2] < min_height)).any().item())


def _get_robot_joint_pos(env) -> torch.Tensor:
    return wp.to_torch(env.scene["robot"].data.joint_pos).clone()


def _get_controlled_joint_ids(env) -> torch.Tensor:
    robot = env.scene["robot"]
    action_term = env.action_manager.get_term(env.action_manager.active_terms[0])
    joint_ids = action_term._joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = torch.arange(
            wp.to_torch(robot.data.joint_pos).shape[1], device=env.device, dtype=torch.long
        )[joint_ids]
    else:
        joint_ids = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
    return joint_ids[ACTION_PREFIX_PAD:].to(dtype=torch.int32)


def _write_fixed_initial_state_to_sim(env, target_state_ref: np.ndarray) -> None:
    """Directly teleport the robot's controlled joints to the target state."""
    robot = env.scene["robot"]
    joint_ids = _get_controlled_joint_ids(env)

    target_internal = np.asarray(target_state_ref, dtype=np.float32)[INV_PERM]
    target_pos = torch.tensor(target_internal, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    target_vel = torch.zeros_like(target_pos)
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
    robot.write_joint_position_to_sim_index(position=target_pos, joint_ids=joint_ids, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=target_vel, joint_ids=joint_ids, env_ids=env_ids)


def _capture_scene_snapshot(env) -> dict:
    snapshot = {"articulations": {}, "rigid_objects": {}}
    for name, asset in getattr(env.scene, "articulations", {}).items():
        snapshot["articulations"][name] = {
            "root_state": wp.to_torch(asset.data.root_state_w).clone(),
            "joint_pos": wp.to_torch(asset.data.joint_pos).clone(),
            "joint_vel": wp.to_torch(asset.data.joint_vel).clone(),
        }
    for name, asset in getattr(env.scene, "rigid_objects", {}).items():
        snapshot["rigid_objects"][name] = {
            "root_state": wp.to_torch(asset.data.root_state_w).clone(),
        }
    return snapshot


def _restore_scene_snapshot(env, snapshot: dict) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
    for name, state in snapshot.get("articulations", {}).items():
        if name not in env.scene.articulations:
            continue
        asset = env.scene.articulations[name]
        root_state = state["root_state"].to(device=env.device)
        joint_pos = state["joint_pos"].to(device=env.device)
        joint_vel = state["joint_vel"].to(device=env.device)
        asset.write_root_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=root_state[:, 7:13], env_ids=env_ids)
        asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
    for name, state in snapshot.get("rigid_objects", {}).items():
        rigid_objects = getattr(env.scene, "rigid_objects", {})
        if name not in rigid_objects:
            continue
        asset = rigid_objects[name]
        root_state = state["root_state"].to(device=env.device)
        asset.write_root_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=root_state[:, 7:13], env_ids=env_ids)
    _flush_observations(env)


def _make_recover_state_ref(task_idx: int, task_start_state_ref: np.ndarray, current_state_ref: np.ndarray) -> np.ndarray:
    recover_state_ref = np.asarray(task_start_state_ref, dtype=np.float32).copy()
    current_state_ref = np.asarray(current_state_ref, dtype=np.float32)
    if task_idx == 1:
        # Task 2 recover keeps the left hand/fingers at their current hold pose.
        recover_state_ref[14:21] = current_state_ref[14:21]
    elif task_idx >= 3:
        # Later recoveries keep both hands/fingers holding whatever they currently grasp.
        recover_state_ref[14:28] = current_state_ref[14:28]
    return recover_state_ref


def _parse_state_delta(value: str, expected_len: int, name: str) -> np.ndarray:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} comma-separated floats, got {len(parts)}: {value}")
    return np.asarray([float(p) for p in parts], dtype=np.float32)


def _make_task34_recover_spread_state_ref(current_state_ref: np.ndarray) -> np.ndarray:
    spread_state_ref = np.asarray(current_state_ref, dtype=np.float32).copy()
    spread_state_ref[0:7] += _parse_state_delta(args.task34_recover_left_arm_delta, 7, "--task34_recover_left_arm_delta")
    spread_state_ref[7:14] += _parse_state_delta(
        args.task34_recover_right_arm_delta, 7, "--task34_recover_right_arm_delta"
    )
    return spread_state_ref


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


def _state_ref_to_policy_style_action(state_ref: np.ndarray) -> np.ndarray:
    """Build an env action with the same layout used for GR00T policy outputs."""
    action_internal = np.asarray(state_ref, dtype=np.float32)[INV_PERM]
    action = np.concatenate([np.zeros(ACTION_PREFIX_PAD, dtype=np.float32), action_internal], axis=0)
    return _normalize_env_raw_action(action)


def _recover_state_ref_to_action(state_ref: np.ndarray) -> np.ndarray:
    action = _state_ref_to_policy_style_action(state_ref)
    # The env action term applies a -0.3 offset to both elbows. Compensate only
    # those two recover targets; changing the full action mapping drives other
    # joints toward zero in this interactive setup.
    action[ACTION_PREFIX_PAD + 3] += 0.3
    action[ACTION_PREFIX_PAD + 10] += 0.3
    return action


def _drive_state_ref_target(
    env,
    target_state_ref: np.ndarray,
    step_limit: int,
    tolerance: float,
    on_step=None,
) -> dict:
    steps_run = 0
    last_env_step_ms = None
    start_state_ref = _get_current_state_ref(env).copy()
    target_state_ref = np.asarray(target_state_ref, dtype=np.float32)
    max_error = float(np.abs(_get_current_state_ref(env) - target_state_ref).max())
    if max_error <= tolerance:
        return {"steps": steps_run, "max_error": max_error, "last_env_step_ms": last_env_step_ms}
    step_count = max(step_limit, 1)
    for step_idx in range(step_count):
        alpha = float(step_idx + 1) / float(step_count)
        step_target_state_ref = start_state_ref + alpha * (target_state_ref - start_state_ref)
        raw_action = _recover_state_ref_to_action(step_target_state_ref)
        action = torch.tensor(raw_action.reshape(1, ROBOT_ACTION_DIM), dtype=torch.float32, device=env.device)
        t0 = time.perf_counter()
        env.step(action)
        last_env_step_ms = (time.perf_counter() - t0) * 1000.0
        steps_run = step_idx + 1
        if on_step is not None:
            on_step()
        max_error = float(np.abs(_get_current_state_ref(env) - target_state_ref).max())
        if max_error <= tolerance:
            break
    return {"steps": steps_run, "max_error": max_error, "last_env_step_ms": last_env_step_ms}


def _drive_reverse_state_history(env, history: list[np.ndarray], stop_index: int, step_limit: int) -> dict:
    """Replay recorded robot states backward, one env step per recorded state."""
    if not history:
        return {"steps": 0, "max_error": 0.0, "last_env_step_ms": None, "target_index": None}
    stop_index = max(0, min(stop_index, len(history) - 1))
    indices = list(range(len(history) - 1, stop_index - 1, -1))[: max(step_limit, 1)]
    steps_run = 0
    last_env_step_ms = None
    target_index = indices[-1] if indices else len(history) - 1
    max_error = float(np.abs(_get_current_state_ref(env) - history[target_index]).max())
    for steps_run, state_idx in enumerate(indices, start=1):
        raw_action = _recover_state_ref_to_action(history[state_idx])
        action = torch.tensor(raw_action.reshape(1, ROBOT_ACTION_DIM), dtype=torch.float32, device=env.device)
        t0 = time.perf_counter()
        env.step(action)
        last_env_step_ms = (time.perf_counter() - t0) * 1000.0
        target_index = state_idx
        max_error = float(np.abs(_get_current_state_ref(env) - history[target_index]).max())
    return {
        "steps": steps_run,
        "max_error": max_error,
        "last_env_step_ms": last_env_step_ms,
        "target_index": target_index,
    }




def _get_camera_rgb(env, cam_name: str) -> np.ndarray:
    sensor = env.scene.sensors[cam_name]
    imgs = sensor.data.output["rgb"]
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().numpy()
    if imgs.shape[-1] == 4:
        imgs = imgs[..., :3]
    return imgs.astype(np.uint8)


class MultiViewVideoRecorder:
    def __init__(self, output_dir: str | Path, fps: float):
        import cv2

        self._cv2 = cv2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self._writers = {}
        self.frame_count = 0
        self.overlay_text = ""

    def set_overlay(self, text: str) -> None:
        self.overlay_text = text

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        if not self.overlay_text:
            return frame
        frame = frame.copy()
        text = self.overlay_text
        font = self._cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        x, y = 12, 24
        (text_w, text_h), baseline = self._cv2.getTextSize(text, font, font_scale, thickness)
        self._cv2.rectangle(
            frame,
            (x - 6, y - text_h - 6),
            (x + text_w + 6, y + baseline + 6),
            (0, 0, 0),
            -1,
        )
        self._cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            self._cv2.LINE_AA,
        )
        return frame

    def capture(self, env, overlay_text: str | None = None) -> None:
        if overlay_text is not None:
            self.set_overlay(overlay_text)
        for cam_name in CAMERA_KEYS:
            frame = self._draw_overlay(_get_camera_rgb(env, cam_name)[0])
            writer = self._writers.get(cam_name)
            if writer is None:
                height, width = frame.shape[:2]
                path = self.output_dir / f"{cam_name}.mp4"
                writer = self._cv2.VideoWriter(
                    str(path),
                    self._cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {path}")
                self._writers[cam_name] = writer
            writer.write(self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR))
        self.frame_count += 1

    def close(self) -> None:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()

    @property
    def paths(self) -> list[Path]:
        return [self.output_dir / f"{cam_name}.mp4" for cam_name in CAMERA_KEYS]


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
    value = action_dict.get("action.task_progress", action_dict.get("action.task_complete"))
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        arr = value.float().reshape(-1).cpu().numpy()
    else:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(arr.mean()) if arr.size > 0 else None


def _predict_task_progress(action_dict: dict) -> float | None:
    return _extract_task_complete(action_dict)


def _probe_task_progress(policy, env, task_idx: int):
    task_desc = TASK_DESCRIPTIONS[task_idx]
    t0 = time.perf_counter()
    policy_obs = _build_policy_obs(env, task_desc)
    action_dict = policy.get_action(policy_obs)
    policy_ms = (time.perf_counter() - t0) * 1000.0
    progress = _predict_task_progress(action_dict)
    return action_dict, progress, policy_ms


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
        self.pending_recover = False
        self.pending_recover_task_idx: int | None = None
        self.pending_tray_rotation = False
        self.pending_tray_return = False
        self.selected_task_idx = initial_task_idx
        self.task_selection_serial = 0
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

    def consume_recover(self) -> tuple[bool, int | None]:
        v = self.pending_recover
        task_idx = self.pending_recover_task_idx
        self.pending_recover = False
        self.pending_recover_task_idx = None
        return v, task_idx

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

    def select_task(self, idx: int) -> None:
        self.selected_task_idx = idx
        self.task_selection_serial += 1
        self.last_message = f"Selected task {idx + 1}: {TASK_DESCRIPTIONS[idx]}"

    def queue_recover(self, task_idx: int | None = None) -> None:
        self.running = False
        self.pending_recover = True
        self.pending_recover_task_idx = task_idx
        self.pending_single_step = False
        self.pending_task_complete_probe = False
        if task_idx is None:
            self.last_message = "Queued current task recover."
        else:
            self.last_message = f"Queued task {task_idx + 1} recover."

    def queue_reset(self) -> None:
        self.running = False
        self.pending_single_step = False
        self.pending_task_complete_probe = False
        self.pending_tray_rotation = False
        self.pending_tray_return = False
        self.pending_recover = False
        self.pending_recover_task_idx = None
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
                self.select_task(self._TASK_KEY_MAP[alias])
                return True

        if aliases & {"SPACE", "SPACEBAR", "P"}:
            self.toggle_running()
        elif aliases & {"S", "ENTER"}:
            self.pending_single_step = True
            self.last_message = "Queued one policy step."
        elif aliases & {"G"}:
            self.pending_task_complete_probe = True
            self.last_message = "Queued a task_complete probe without env step."
        elif aliases & {"T"}:
            self.queue_recover()
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
                            clicked_fn=lambda i=idx: keyboard.select_task(i),
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
                    ui.Button(
                        "Recover",
                        width=0,
                        clicked_fn=lambda: keyboard.queue_recover(),
                        tooltip="Drive robot toward the current task-start state; preserves hand holds for later stages.",
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
                    "Keys: 1-5 select task  SPACE/P start|pause  S step  G probe TC  T recover current  O rotate  B back  R reset",
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
        tc_threshold = _task_complete_threshold(keyboard.selected_task_idx)
        tc_above = (
            "n/a"
            if task_complete_value is None
            else str(task_complete_value >= tc_threshold)
        )
        policy_ms = "n/a" if last_policy_ms is None else f"{last_policy_ms:.0f}"
        env_ms = "n/a" if last_env_step_ms is None else f"{last_env_step_ms:.0f}"
        task_desc = TASK_DESCRIPTIONS[keyboard.selected_task_idx]

        self._status.text = (
            f"Mode: {mode}\n"
            f"Task {keyboard.selected_task_idx + 1}: {task_desc}\n"
            f"task_complete: {tc_text}  (>= {tc_threshold:.2f}: {tc_above})\n"
            f"Auto-stop on TC: {keyboard.stop_on_task_complete}\n"
            f"Task run steps: {task_run_step_count} / {args.task_timeout_steps}\n"
            f"Open-loop steps remaining: {open_loop_steps_remaining} / {args.open_loop_steps}\n"
            f"Tray yaw: {tray_yaw_deg:.2f} deg\n"
            f"Tray rotation steps: {tray_rotation_progress} / {tray_rotation_total}\n"
            f"Timing ms: policy={policy_ms}  env_step={env_ms}\n"
            f"Episode steps: {step_count}\n"
            f"Last event: {keyboard.last_message}"
        )


def _auto_recover_task34_reference(
    env,
    task_idx: int,
    task34_recover_state_ref: np.ndarray,
    video_recorder: MultiViewVideoRecorder | None = None,
) -> dict:
    current_state_ref = _get_current_state_ref(env)
    spread_steps_run = 0
    spread_error = None
    if args.task34_recover_spread_steps > 0:
        if video_recorder is not None:
            video_recorder.set_overlay(_task_overlay_label(task_idx, retry=True))
        spread_state_ref = _make_task34_recover_spread_state_ref(current_state_ref)
        spread_info = _drive_state_ref_target(
            env,
            spread_state_ref,
            args.task34_recover_spread_steps,
            args.fixed_initial_state_tolerance,
            on_step=None if video_recorder is None else lambda: video_recorder.capture(env),
        )
        spread_steps_run = int(spread_info["steps"])
        spread_error = float(spread_info["max_error"])

    if video_recorder is not None:
        video_recorder.set_overlay(_task_overlay_label(task_idx, retry=True))
    recover_info = _drive_state_ref_target(
        env,
        task34_recover_state_ref.copy(),
        max(args.recover_steps, 1),
        args.fixed_initial_state_tolerance,
        on_step=None if video_recorder is None else lambda: video_recorder.capture(env),
    )
    recover_error = float(recover_info["max_error"])
    top_errors = _format_top_state_errors(_get_current_state_ref(env), task34_recover_state_ref)
    _flush_observations(env)
    print(
        f"[AUTO] Recovered task {task_idx + 1} to reference episode "
        f"{args.task34_recover_reference_episode} frame {args.task34_recover_reference_frame}: "
        f"spread={spread_steps_run}/{args.task34_recover_spread_steps}"
        + ("" if spread_error is None else f" spread_error={spread_error:.4f}")
        + f", recover={int(recover_info['steps'])}/{max(args.recover_steps, 1)} "
        f"max_error={recover_error:.4f}."
    )
    print(f"[AUTO] Recover top joint errors: {top_errors}")
    return {
        "spread_steps": spread_steps_run,
        "recover_steps": int(recover_info["steps"]),
        "recover_error": recover_error,
    }


def _run_direct_multistage(
    policy,
    env,
    task34_recover_state_ref: np.ndarray | None,
    video_recorder: MultiViewVideoRecorder | None = None,
) -> dict:
    task_indices = _parse_index_list(args.auto_task_indices, "--auto_task_indices")
    single_attempt_task_indices = _parse_optional_index_set(
        args.auto_single_attempt_task_indices,
        "--auto_single_attempt_task_indices",
    )
    hold_height_task_indices = _parse_optional_index_set(
        args.auto_hold_height_task_indices,
        "--auto_hold_height_task_indices",
    )
    min_hold_height = args.auto_hold_table_height + args.auto_hold_height_margin
    thresholds = _direct_task_thresholds(task_indices)
    attempt_step_limit = max(args.task_timeout_steps, 1)
    total_step_limit = max(args.auto_total_steps_per_task, attempt_step_limit)
    max_attempts = max(1, (total_step_limit + attempt_step_limit - 1) // attempt_step_limit)
    total_env_steps = 0
    result = {
        "success": False,
        "success_source": None,
        "tasks": [],
        "total_policy_steps": 0,
        "recover_count": 0,
        "failed_task_idx": None,
    }
    task_attempt_counts = {idx: 0 for idx in task_indices}
    task_policy_step_counts = {idx: 0 for idx in task_indices}
    task_start_state_refs: dict[int, np.ndarray] = {}

    print(
        "[AUTO] Direct multi-stage inference starting: "
        f"tasks={[idx + 1 for idx in task_indices]}, "
        f"thresholds={[thresholds[idx] for idx in task_indices]}, "
        f"attempt_steps={attempt_step_limit}, total_steps_per_task={total_step_limit}, "
        f"max_attempts={max_attempts}."
    )
    if video_recorder is not None:
        video_recorder.capture(env, "Episode start")

    task_order_idx = 0
    while task_order_idx < len(task_indices):
        task_idx = task_indices[task_order_idx]
        threshold = thresholds[task_idx]
        is_final_task = task_order_idx == len(task_indices) - 1
        task_max_attempts = (
            1
            if task_idx in single_attempt_task_indices or (args.auto_no_retry_last_task and is_final_task)
            else max_attempts
        )
        task_total_step_limit = attempt_step_limit if task_max_attempts == 1 else total_step_limit
        task_success = False
        task_start_state_refs.setdefault(task_idx, _get_current_state_ref(env).copy())
        task_result = {
            "task_idx": task_idx,
            "task_number": task_idx + 1,
            "threshold": threshold,
            "success": False,
            "attempts": task_attempt_counts[task_idx],
            "policy_steps": task_policy_step_counts[task_idx],
            "recover_count": 0,
            "final_progress": None,
            "env_stage": _get_env_task_stage(env),
        }
        print(
            f"[AUTO] Task {task_idx + 1} start: {TASK_DESCRIPTIONS[task_idx]}, "
            f"threshold={threshold:.4f}, max_attempts={task_max_attempts}"
            + (" (final success uses env-native task_success)." if is_final_task else ".")
        )

        while task_attempt_counts[task_idx] < task_max_attempts:
            if task_policy_step_counts[task_idx] >= task_total_step_limit:
                break
            task_attempt_counts[task_idx] += 1
            attempt_idx = task_attempt_counts[task_idx]
            task_result["attempts"] = attempt_idx
            cached_policy_actions: list[np.ndarray] = []
            attempt_steps = 0
            last_progress: float | None = None
            overlay_text = _task_overlay_label(
                task_idx,
                retry=attempt_idx > 1 or task_result["recover_count"] > 0,
            )
            if video_recorder is not None:
                video_recorder.set_overlay(overlay_text)
            print(
                f"[AUTO] Task {task_idx + 1} attempt {attempt_idx}/{task_max_attempts} "
                f"starting at task_steps={task_policy_step_counts[task_idx]}/{task_total_step_limit}."
            )

            while simulation_app.is_running() and attempt_steps < attempt_step_limit:
                if task_policy_step_counts[task_idx] >= task_total_step_limit:
                    break
                if not cached_policy_actions:
                    with torch.inference_mode():
                        action_dict, last_progress, policy_ms = _probe_task_progress(policy, env, task_idx)
                    if not is_final_task and last_progress is not None and last_progress >= threshold:
                        task_success = True
                        print(
                            f"[AUTO] Task {task_idx + 1} reached threshold before step: "
                            f"progress={last_progress:.4f} >= {threshold:.4f}."
                        )
                        task_result["final_progress"] = last_progress
                        break
                    action_chunk = _convert_policy_action_chunk_to_env(action_dict)
                    if not action_chunk:
                        raise RuntimeError("Policy returned an empty action chunk.")
                    cached_policy_actions = [a.copy() for a in action_chunk[: max(args.open_loop_steps, 1)]]
                    print(
                        f"[AUTO] Task {task_idx + 1} attempt {attempt_idx}: "
                        f"progress={last_progress} policy_ms={policy_ms:.0f} "
                        f"steps={attempt_steps}/{attempt_step_limit}."
                    )

                action_np = cached_policy_actions.pop(0)
                action_tensor = _raw_action_to_env_tensor(action_np, device=env.device)
                t0 = time.perf_counter()
                _, _, terminated, truncated, _ = env.step(action_tensor)
                env_step_ms = (time.perf_counter() - t0) * 1000.0
                total_env_steps += 1
                task_policy_step_counts[task_idx] += 1
                attempt_steps += 1
                if video_recorder is not None:
                    video_recorder.capture(env, overlay_text)

                if _trocar_dropped(env):
                    print(
                        f"[AUTO] Episode failed: trocar dropped during task {task_idx + 1} "
                        f"attempt {attempt_idx}."
                    )
                    task_result["success"] = False
                    task_result["policy_steps"] = task_policy_step_counts[task_idx]
                    task_result["final_progress"] = last_progress
                    task_result["env_stage"] = _get_env_task_stage(env)
                    result["tasks"].append(task_result)
                    result["success_source"] = "trocar_dropped"
                    result["failed_task_idx"] = task_idx
                    result["total_policy_steps"] = total_env_steps
                    return result

                if task_idx in hold_height_task_indices and _trocar_below_height(env, min_hold_height):
                    print(
                        f"[AUTO] Episode failed: trocar below held-height threshold during task {task_idx + 1} "
                        f"(min_height={min_hold_height:.4f})."
                    )
                    task_result["success"] = False
                    task_result["policy_steps"] = task_policy_step_counts[task_idx]
                    task_result["final_progress"] = last_progress
                    task_result["env_stage"] = _get_env_task_stage(env)
                    result["tasks"].append(task_result)
                    result["success_source"] = "trocar_below_hold_height"
                    result["failed_task_idx"] = task_idx
                    result["total_policy_steps"] = total_env_steps
                    return result

                if bool(terminated[0]) or bool(truncated[0]):
                    reason = "terminated" if bool(terminated[0]) else "truncated"
                    print(
                        f"[AUTO] Episode finished by env-native {reason} during task {task_idx + 1} "
                        f"attempt {attempt_idx} ({reason}) after env_step_ms={env_step_ms:.0f}."
                    )
                    task_success = True
                    task_result["success"] = True
                    task_result["policy_steps"] = task_policy_step_counts[task_idx]
                    task_result["final_progress"] = last_progress
                    task_result["env_stage"] = _get_env_task_stage(env)
                    result["tasks"].append(task_result)
                    result["success"] = bool(terminated[0])
                    result["success_source"] = "env_task_success" if bool(terminated[0]) else "env_truncated"
                    result["failed_task_idx"] = None if bool(terminated[0]) else task_idx
                    result["total_policy_steps"] = total_env_steps
                    return result

            if task_success:
                break

            with torch.inference_mode():
                _, last_progress, policy_ms = _probe_task_progress(policy, env, task_idx)
            if not is_final_task and last_progress is not None and last_progress >= threshold:
                task_success = True
                print(
                    f"[AUTO] Task {task_idx + 1} reached threshold after attempt {attempt_idx}: "
                    f"progress={last_progress:.4f} >= {threshold:.4f}."
                )
                task_result["final_progress"] = last_progress
                break
            task_result["final_progress"] = last_progress

            print(
                f"[AUTO] Task {task_idx + 1} attempt {attempt_idx} ended at "
                f"progress={last_progress} after {attempt_steps}/{attempt_step_limit} steps "
                f"(task_steps={task_policy_step_counts[task_idx]}/{task_total_step_limit})."
            )

            if (
                task_idx == 2
                and task_policy_step_counts[task_idx] < task_total_step_limit
                and attempt_idx < task_max_attempts
                and task34_recover_state_ref is not None
            ):
                print("[AUTO] Task 3 timed out below threshold; running recover before the next attempt.")
                _auto_recover_task34_reference(env, task_idx, task34_recover_state_ref, video_recorder)
                task_result["recover_count"] += 1
                result["recover_count"] += 1
                if _trocar_dropped(env):
                    print("[AUTO] Episode failed: trocar dropped during task 3 recover.")
                    task_result["success"] = False
                    task_result["policy_steps"] = task_policy_step_counts[task_idx]
                    task_result["env_stage"] = _get_env_task_stage(env)
                    result["tasks"].append(task_result)
                    result["success_source"] = "trocar_dropped"
                    result["failed_task_idx"] = task_idx
                    result["total_policy_steps"] = total_env_steps
                    return result
            elif task_idx == 2 and task34_recover_state_ref is None:
                print("[AUTO] Task 3 recover target is unavailable; retrying without recover.")

            if (
                task_idx == 3
                and task_policy_step_counts[task_idx] < task_total_step_limit
                and attempt_idx < task_max_attempts
                and 2 in task_start_state_refs
            ):
                print("[AUTO] Task 4 attempt failed; returning robot to task 3 start and rerunning task 3.")
                if video_recorder is not None:
                    video_recorder.set_overlay(_task_overlay_label(3, retry=True))
                _drive_state_ref_target(
                    env,
                    task_start_state_refs[2],
                    max(args.recover_steps, 1),
                    args.fixed_initial_state_tolerance,
                    on_step=None if video_recorder is None else lambda: video_recorder.capture(env),
                )
                if _trocar_dropped(env):
                    print("[AUTO] Episode failed: trocar dropped while returning to task 3 start.")
                    task_result["success"] = False
                    task_result["policy_steps"] = task_policy_step_counts[task_idx]
                    task_result["env_stage"] = _get_env_task_stage(env)
                    result["tasks"].append(task_result)
                    result["success_source"] = "trocar_dropped"
                    result["failed_task_idx"] = task_idx
                    result["total_policy_steps"] = total_env_steps
                    return result
                task_result["success"] = False
                task_result["policy_steps"] = task_policy_step_counts[task_idx]
                task_result["env_stage"] = _get_env_task_stage(env)
                result["tasks"].append(task_result)
                task_order_idx = task_indices.index(2)
                break

        task_result["success"] = task_success
        task_result["policy_steps"] = task_policy_step_counts[task_idx]
        task_result["env_stage"] = _get_env_task_stage(env)
        if task_success and task_result["final_progress"] is None:
            with torch.inference_mode():
                _, final_progress, _ = _probe_task_progress(policy, env, task_idx)
            task_result["final_progress"] = final_progress
        if not result["tasks"] or result["tasks"][-1] is not task_result:
            result["tasks"].append(task_result)

        if not task_success:
            if task_idx == 3 and task_order_idx == task_indices.index(2):
                continue
            print(
                f"[AUTO] Task {task_idx + 1} failed to reach threshold {threshold:.4f} "
                f"within {task_policy_step_counts[task_idx]}/{task_total_step_limit} policy steps."
            )
            result["failed_task_idx"] = task_idx
            result["total_policy_steps"] = total_env_steps
            return result

        print(
            f"[AUTO] Task {task_idx + 1} complete after {task_policy_step_counts[task_idx]} policy steps. "
            "Advancing to next task."
        )
        task_order_idx += 1

    print(f"[AUTO] Direct multi-stage inference finished successfully after {total_env_steps} policy env steps.")
    result["success"] = True
    result["total_policy_steps"] = total_env_steps
    return result


def _run_direct_multistage_episodes(
    policy,
    env,
    rng: np.random.Generator,
    fixed_initial_raw_action: np.ndarray | None,
    fixed_initial_state_ref: np.ndarray | None,
    task34_recover_state_ref: np.ndarray | None,
) -> bool:
    episode_count = max(args.auto_num_episodes, 1)
    all_success = True
    episode_results = []
    for episode_idx in range(episode_count):
        tray_yaw_deg = _sample_initial_tray_yaw_deg(rng)
        print(
            f"[AUTO] ===== Episode {episode_idx + 1}/{episode_count} start "
            f"(tray_yaw={tray_yaw_deg:.2f} deg) ====="
        )
        _reset_env(
            env,
            seed=None if args.seed is None else args.seed + episode_idx,
            tray_yaw_deg=tray_yaw_deg,
            fixed_raw_action=fixed_initial_raw_action,
            fixed_state_ref=fixed_initial_state_ref,
        )

        video_recorder = None
        if args.auto_record_video:
            video_dir = Path(args.auto_video_dir) / f"episode_{episode_idx:06d}"
            video_recorder = MultiViewVideoRecorder(video_dir, args.auto_video_fps)

        try:
            episode_result = _run_direct_multistage(policy, env, task34_recover_state_ref, video_recorder)
            episode_result["episode_idx"] = episode_idx
            episode_result["tray_yaw_deg"] = tray_yaw_deg
            episode_results.append(episode_result)
            success = bool(episode_result["success"])
            all_success = all_success and success
            _write_direct_success_status(episode_results)
            print(
                f"[AUTO] ===== Episode {episode_idx + 1}/{episode_count} "
                f"{'succeeded' if success else 'failed'} ====="
            )
        finally:
            if video_recorder is not None:
                video_recorder.close()
                print(
                    f"[AUTO] Saved episode {episode_idx + 1} video "
                    f"({video_recorder.frame_count} frames): "
                    f"{', '.join(str(path) for path in video_recorder.paths)}"
                )

    _write_direct_multistage_results(episode_results)
    success_count = sum(1 for item in episode_results if item["success"])
    success_rate = success_count / max(len(episode_results), 1)
    task_indices = _parse_index_list(args.auto_task_indices, "--auto_task_indices")
    print(
        f"[AUTO] Success rate: {success_count}/{len(episode_results)} episodes "
        f"({success_rate * 100.0:.1f}%)."
    )
    for task_idx in task_indices:
        task_results = [
            task
            for episode in episode_results
            for task in episode["tasks"]
            if task["task_idx"] == task_idx
        ]
        task_success_count = sum(1 for task in task_results if task["success"])
        task_rate = task_success_count / max(len(task_results), 1)
        print(
            f"[AUTO] Task {task_idx + 1} success rate: "
            f"{task_success_count}/{len(task_results)} ({task_rate * 100.0:.1f}%)."
        )
    print(
        f"[AUTO] Completed {episode_count} direct episode(s); "
        f"all_success={all_success}."
    )
    return all_success


def _write_direct_multistage_results(episode_results: list[dict]) -> None:
    video_dir = Path(args.auto_video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    summary_path = video_dir / "summary.json"
    success_count = sum(1 for item in episode_results if item["success"])
    summary = {
        "num_episodes": len(episode_results),
        "success_count": success_count,
        "success_rate": success_count / max(len(episode_results), 1),
        "total_recover_count": sum(int(item.get("recover_count", 0)) for item in episode_results),
        "episodes": episode_results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[AUTO] Wrote direct multi-stage summary: {summary_path}")


def _write_direct_success_status(episode_results: list[dict]) -> None:
    video_dir = Path(args.auto_video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    success_path = video_dir / "success.json"
    success_count = sum(1 for item in episode_results if item["success"])
    payload = {
        "completed_episodes": len(episode_results),
        "success_count": success_count,
        "failure_count": len(episode_results) - success_count,
        "success_rate": success_count / max(len(episode_results), 1),
        "episodes": [
            {
                "episode_idx": item["episode_idx"],
                "success": item["success"],
                "success_source": item.get("success_source"),
                "failed_task_idx": item.get("failed_task_idx"),
                "failed_task_number": (
                    None
                    if item.get("failed_task_idx") is None
                    else int(item["failed_task_idx"]) + 1
                ),
                "total_policy_steps": item.get("total_policy_steps"),
                "recover_count": item.get("recover_count", 0),
            }
            for item in episode_results
        ],
    }
    tmp_path = success_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(success_path)


def main():
    model_device = args.model_device or args.device
    policy = _load_policy(
        args.model_path,
        model_device=model_device,
        denoising_steps=args.denoising_steps,
        progress_regressor_path=args.progress_regressor_path,
    )
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

    task34_recover_state_ref = None
    task34_recover_dataset = args.task34_recover_reference_dataset or args.fixed_initial_state_dataset
    if task34_recover_dataset:
        task34_recover_state_ref = _load_fixed_initial_state_ref(
            task34_recover_dataset,
            args.task34_recover_reference_episode,
            args.task34_recover_reference_frame,
        )
        print(
            f"[INFO] Task 3/4 recover target: dataset={task34_recover_dataset}, "
            f"episode={args.task34_recover_reference_episode}, frame={args.task34_recover_reference_frame}"
        )

    if args.auto_multistage_direct:
        try:
            success = _run_direct_multistage_episodes(
                policy,
                env,
                rng,
                fixed_initial_raw_action,
                fixed_initial_state_ref,
                task34_recover_state_ref,
            )
            if not success:
                raise RuntimeError("At least one direct multi-stage episode did not reach all requested thresholds.")
        finally:
            env.close()
            simulation_app.close()
        return

    tray_yaw_deg = _sample_initial_tray_yaw_deg(rng)
    _reset_env(
        env, seed=args.seed, tray_yaw_deg=tray_yaw_deg,
        fixed_raw_action=fixed_initial_raw_action, fixed_state_ref=fixed_initial_state_ref,
    )

    keyboard = KeyboardInterface(
        stop_on_task_complete=args.enable_task_complete_stop,
        initial_task_idx=args.initial_task,
    )
    panel = StatusPanel(keyboard)
    _print_controls()

    step_count = 0
    task_run_step_count = 0
    episode_done = False
    task_complete_value: float | None = None
    previous_task_progress_value: float | None = None
    task_broken_detection_enabled = False
    task_stuck_detection_enabled = False
    task_progress_history: list[float] = []
    task_start_state_ref: np.ndarray | None = None
    task_start_state_refs: list[np.ndarray | None] = [None] * len(TASK_DESCRIPTIONS)
    task_state_histories: list[list[np.ndarray]] = [[] for _ in TASK_DESCRIPTIONS]
    previous_task_idx = keyboard.selected_task_idx
    previous_task_selection_serial = keyboard.task_selection_serial
    previous_running = keyboard.running
    recovered_task_indices: set[int] = set()
    stage_precondition_checked_for_task_idx: int | None = None
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
            # --- Task switch/selection: clear stale task state, even when re-selecting the same task ---
            if (
                keyboard.selected_task_idx != previous_task_idx
                or keyboard.task_selection_serial != previous_task_selection_serial
            ):
                previous_task_idx = keyboard.selected_task_idx
                previous_task_selection_serial = keyboard.task_selection_serial
                task_complete_value = None
                previous_task_progress_value = None
                task_broken_detection_enabled = False
                task_stuck_detection_enabled = False
                task_progress_history = []
                task_start_state_ref = None
                task_state_histories[keyboard.selected_task_idx] = []
                task_run_step_count = 0
                cached_policy_actions = []
                open_loop_steps_remaining = 0
                stage_precondition_checked_for_task_idx = None

            # Starting a new attempt should not inherit broken/stuck statistics from a previous run.
            if keyboard.running and not previous_running:
                task_complete_value = None
                previous_task_progress_value = None
                task_broken_detection_enabled = False
                task_stuck_detection_enabled = False
                task_progress_history = []
                task_run_step_count = 0
                cached_policy_actions = []
                open_loop_steps_remaining = 0
                stage_precondition_checked_for_task_idx = None
                print(f"[INFO] Starting task {keyboard.selected_task_idx + 1}: cleared progress monitor history.")
            previous_running = keyboard.running

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
                previous_task_progress_value = None
                task_broken_detection_enabled = False
                task_stuck_detection_enabled = False
                task_progress_history = []
                task_start_state_ref = None
                task_start_state_refs = [None] * len(TASK_DESCRIPTIONS)
                task_state_histories = [[] for _ in TASK_DESCRIPTIONS]
                cached_policy_actions = []
                open_loop_steps_remaining = 0
                stage_precondition_checked_for_task_idx = None
                tray_base_yaw_deg = tray_yaw_deg
                tray_rotation_active = False
                tray_rotation_start_yaw = tray_yaw_deg
                tray_rotation_target_yaw = tray_yaw_deg
                tray_rotation_step_count = 0
                recovered_task_indices.clear()
                keyboard.last_message = f"Reset done. Tray yaw={tray_yaw_deg:.2f} deg."

            # --- Recover a task by driving only the robot toward that task-start pose ---
            recover_requested, recover_task_idx = keyboard.consume_recover()
            if recover_requested:
                keyboard.running = False
                cached_policy_actions = []
                open_loop_steps_remaining = 0
                tray_rotation_active = False
                target_task_idx = keyboard.selected_task_idx if recover_task_idx is None else recover_task_idx
                target_task_start_state_ref = task_start_state_refs[target_task_idx]
                use_task34_reference = target_task_idx in (2, 3) and task34_recover_state_ref is not None
                if episode_done:
                    keyboard.last_message = "Episode done. Press R to reset first."
                elif target_task_start_state_ref is None and not use_task34_reference:
                    keyboard.last_message = (
                        f"No task {target_task_idx + 1} start robot state yet. "
                        f"Run task {target_task_idx + 1} once before recover."
                    )
                else:
                    keyboard.selected_task_idx = target_task_idx
                    previous_task_idx = target_task_idx
                    current_state_ref = _get_current_state_ref(env)
                    recover_step_limit = max(args.recover_steps, 1)
                    spread_steps_run = 0
                    spread_error = None
                    if use_task34_reference:
                        if args.task34_recover_spread_steps > 0:
                            spread_state_ref = _make_task34_recover_spread_state_ref(current_state_ref)
                            spread_info = _drive_state_ref_target(
                                env,
                                spread_state_ref,
                                args.task34_recover_spread_steps,
                                args.fixed_initial_state_tolerance,
                            )
                            spread_steps_run = int(spread_info["steps"])
                            spread_error = float(spread_info["max_error"])
                            last_env_step_ms = spread_info["last_env_step_ms"]
                            step_count += spread_steps_run
                        recover_state_ref = task34_recover_state_ref.copy()
                        recover_mode_text = (
                            f" to reference episode {args.task34_recover_reference_episode} "
                            f"frame {args.task34_recover_reference_frame}"
                        )
                    else:
                        recover_state_ref = _make_recover_state_ref(
                            target_task_idx, target_task_start_state_ref, current_state_ref
                        )
                        recover_mode_text = " by direct target"
                    recover_info = _drive_state_ref_target(
                        env,
                        recover_state_ref,
                        recover_step_limit,
                        args.fixed_initial_state_tolerance,
                    )
                    recover_steps_run = int(recover_info["steps"])
                    recover_error = float(recover_info["max_error"])
                    last_env_step_ms = recover_info["last_env_step_ms"]
                    step_count += recover_steps_run
                    top_recover_errors = _format_top_state_errors(_get_current_state_ref(env), recover_state_ref)
                    _flush_observations(env)
                    episode_done = False
                    task_complete_value = None
                    previous_task_progress_value = None
                    task_broken_detection_enabled = False
                    task_stuck_detection_enabled = False
                    task_progress_history = []
                    task_run_step_count = 0
                    stage_precondition_checked_for_task_idx = None
                    next_task_idx = 2 if target_task_idx == 3 and use_task34_reference else target_task_idx
                    keyboard.selected_task_idx = next_task_idx
                    previous_task_idx = next_task_idx
                    task_start_state_ref = task_start_state_refs[next_task_idx]
                    recovered_task_indices.add(next_task_idx)
                    spread_text = (
                        ""
                        if spread_steps_run == 0
                        else f" after spread {spread_steps_run}/{args.task34_recover_spread_steps} steps "
                        f"(spread_error={spread_error:.4f})"
                    )
                    next_step_text = (
                        " Press Start to redo task 3 before retrying task 4."
                        if target_task_idx == 3 and next_task_idx == 2
                        else " Press Start to run task again."
                    )
                    keyboard.last_message = (
                        f"Recovered task {target_task_idx + 1}{recover_mode_text}: "
                        f"drove robot {spread_text}for {recover_steps_run}/{recover_step_limit} steps "
                        f"(max_error={recover_error:.4f}) without resetting scene.{next_step_text}"
                    )
                    print(f"[INFO] {keyboard.last_message}")
                    print(f"[INFO] Recover top joint errors: {top_recover_errors}")

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
                        _, task_complete_value, last_policy_ms = _probe_task_progress(
                            policy, env, keyboard.selected_task_idx
                        )
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
                        selected_task_idx = keyboard.selected_task_idx
                        if (
                            args.enable_stage_precondition
                            and selected_task_idx > 0
                            and selected_task_idx not in recovered_task_indices
                            and stage_precondition_checked_for_task_idx != selected_task_idx
                        ):
                            required_prev_task_idx = selected_task_idx - 1
                            with torch.inference_mode():
                                _, previous_progress, last_policy_ms = _probe_task_progress(
                                    policy, env, required_prev_task_idx
                                )
                            if previous_progress is None or previous_progress < args.stage_precondition_threshold:
                                keyboard.running = False
                                policy_step_requested = False
                                executing_policy_step = False
                                should_step = False
                                cached_policy_actions = []
                                open_loop_steps_remaining = 0
                                keyboard.last_message = (
                                    f"Blocked task {selected_task_idx + 1}: task {required_prev_task_idx + 1} "
                                    f"progress={previous_progress} < {args.stage_precondition_threshold:.2f}. "
                                    f"Select task {required_prev_task_idx + 1} and press Start before retrying "
                                    f"task {selected_task_idx + 1}."
                                )
                                print(f"[WARN] {keyboard.last_message}")
                            else:
                                stage_precondition_checked_for_task_idx = selected_task_idx
                        if should_step:
                            task_desc = TASK_DESCRIPTIONS[keyboard.selected_task_idx]
                            if task_start_state_ref is None:
                                task_start_state_ref = _get_current_state_ref(env)
                                task_start_state_refs[keyboard.selected_task_idx] = task_start_state_ref
                                task_state_histories[keyboard.selected_task_idx] = [task_start_state_ref.copy()]
                                previous_task_progress_value = None
                                task_broken_detection_enabled = False
                                task_stuck_detection_enabled = False
                                task_progress_history = []
                                print(
                                    f"[INFO] Captured task {keyboard.selected_task_idx + 1} "
                                    "start robot state for recover."
                                )
                            with torch.inference_mode():
                                action_dict, task_complete_value, last_policy_ms = _probe_task_progress(
                                    policy, env, keyboard.selected_task_idx
                                )
                            if task_complete_value is not None:
                                if task_complete_value > args.task_broken_enable_threshold:
                                    task_broken_detection_enabled = True
                                if (
                                    args.task_stuck_window > 1
                                    and args.task_stuck_min_progress_delta > 0.0
                                    and
                                    not task_stuck_detection_enabled
                                    and task_complete_value > args.task_stuck_enable_threshold
                                ):
                                    task_stuck_detection_enabled = True
                                    task_progress_history = []
                                progress_drop = (
                                    0.0
                                    if previous_task_progress_value is None
                                    else previous_task_progress_value - task_complete_value
                                )
                                broken_due_to_drop = (
                                    task_broken_detection_enabled
                                    and args.task_broken_drop_threshold > 0.0
                                    and progress_drop >= args.task_broken_drop_threshold
                                )
                                if task_stuck_detection_enabled:
                                    task_progress_history.append(task_complete_value)
                                if len(task_progress_history) > max(args.task_stuck_window, 1):
                                    task_progress_history = task_progress_history[-max(args.task_stuck_window, 1):]
                                stuck_delta = 0.0
                                broken_due_to_stuck = False
                                if (
                                    task_stuck_detection_enabled
                                    and
                                    args.task_stuck_window > 1
                                    and args.task_stuck_min_progress_delta > 0.0
                                    and len(task_progress_history) >= args.task_stuck_window
                                ):
                                    stuck_delta = task_progress_history[-1] - task_progress_history[0]
                                    broken_due_to_stuck = stuck_delta < args.task_stuck_min_progress_delta
                                if broken_due_to_drop or broken_due_to_stuck:
                                    keyboard.running = False
                                    policy_step_requested = False
                                    executing_policy_step = False
                                    should_step = False
                                    cached_policy_actions = []
                                    open_loop_steps_remaining = 0
                                    if broken_due_to_drop:
                                        keyboard.last_message = (
                                            "task broken, suggest recover "
                                            f"(progress dropped {progress_drop:.4f} from previous "
                                            f"{previous_task_progress_value:.4f})."
                                        )
                                    else:
                                        keyboard.last_message = (
                                            "task broken, suggest recover "
                                            f"(progress increased only {stuck_delta:.4f} over "
                                            f"{args.task_stuck_window} predictions)."
                                        )
                                    print(f"[WARN] {keyboard.last_message}")
                                else:
                                    previous_task_progress_value = task_complete_value
                            if should_step:
                                action_chunk = _convert_policy_action_chunk_to_env(action_dict)
                                if not action_chunk:
                                    raise RuntimeError("Policy returned an empty action chunk.")
                                cached_policy_actions = [a.copy() for a in action_chunk[: max(args.open_loop_steps, 1)]]
                                open_loop_steps_remaining = len(cached_policy_actions)
                    if should_step:
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
                    task_state_histories[keyboard.selected_task_idx].append(_get_current_state_ref(env))
                    if keyboard.running:
                        task_run_step_count += 1

                if episode_done:
                    keyboard.running = False
                    cached_policy_actions = []
                    open_loop_steps_remaining = 0
                    tray_rotation_active = False
                    reason = "terminated" if bool(terminated[0]) else "truncated"
                    keyboard.last_message = f"Episode finished ({reason}). Press R to reset."
                selected_task_threshold = _task_complete_threshold(keyboard.selected_task_idx)
                direct_complete = task_complete_value is not None and task_complete_value >= selected_task_threshold
                if (
                    executing_policy_step
                    and keyboard.running
                    and keyboard.stop_on_task_complete
                    and direct_complete
                ):
                    keyboard.running = False
                    keyboard.last_message = (
                        f"Paused task {keyboard.selected_task_idx + 1}: task_complete={task_complete_value:.4f} "
                        f">= threshold {selected_task_threshold:.2f}."
                    )
                    print(f"[INFO] {keyboard.last_message}")
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
