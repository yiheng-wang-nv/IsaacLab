# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Record trocar assembly episodes with GR00T model inference.

Records joint states, actions, 480x640 RGB videos, and semantic segmentation masks
from 3 cameras. Outputs data in LeRobot format.

Usage:
    ./isaaclab.sh -p scripts/tools/record_trocar_episodes.py \
        --model_path /localhome/local-vennw/models/orca_rlinf_weights/rlinf/actor/model_state_dict \
        --output_dir /localhome/local-vennw/data/trocar_recorded \
        --num_episodes 10
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record trocar episodes with GR00T inference.")
parser.add_argument("--model_path", type=str, required=True, help="Path to GR00T model checkpoint.")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for recorded data.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to record.")
parser.add_argument("--max_steps", type=int, default=256, help="Max steps per episode.")
parser.add_argument("--denoising_steps", type=int, default=4, help="GR00T denoising steps.")
parser.add_argument(
    "--open_loop_steps",
    type=int,
    default=None,
    help="Number of action chunk steps to execute before running GR00T inference again."
    " Defaults to 8 for --use_gr00t_policy and 1 for RLinf checkpoints.",
)
parser.add_argument(
    "--rlinf_base_config_path",
    type=str,
    default="/localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T/sft_4gpu_256bs_50ksteps_success_lt7/checkpoint-50000",
    help="When the RLinf checkpoint contains only full_weights.pt (no config.json), "
    "load model architecture from this base SFT checkpoint and then load RLinf weights on top.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs for recording.")
parser.add_argument("--model_device", type=str, default="cuda:0", help="Device for model inference.")
parser.add_argument("--seed", type=int, default=None, help="Random seed for env reproducibility.")
parser.add_argument("--skip_first_n", type=int, default=3, help="Skip recording the first N frames (physics settling).")
parser.add_argument("--skip_last_n", type=int, default=1, help="Skip recording the last N frames.")
parser.add_argument("--randomize_lighting", action="store_true", default=False, help="Randomize lighting per episode (surgical room range).")
parser.add_argument("--tray_yaw_min_deg", type=float, default=0.0, help="Minimum reset tray yaw randomization [deg].")
parser.add_argument("--tray_yaw_max_deg", type=float, default=10.0, help="Maximum reset tray yaw randomization [deg].")
parser.add_argument("--use_gr00t_policy", action="store_true", default=False,
                    help="Use Gr00tPolicy directly (SFT checkpoint) instead of GR00T_N1_5_ForRLActionPrediction (RLinf).")
parser.add_argument("--no_mask", action="store_true", default=False,
                    help="Skip segmentation mask recording (faster; only saves RGB videos + parquet).")
parser.add_argument(
    "--scene",
    type=str,
    default="lightwheel",
    choices=("lightwheel", "orca", "factory", "surgical_room"),
    help="Background scene variant. 'lightwheel' is the default scene03.usd.",
)
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
    "--fixed_initial_state_mode",
    type=str,
    choices=("command", "teleport", "teleport_settle"),
    default="command",
    help=(
        "How to reach the fixed start pose. 'command' sends joint-position actions until tolerance; "
        "'teleport' writes the target robot joints to sim, holds for fixed_initial_state_steps, then writes again; "
        "'teleport_settle' writes once, then holds for fixed_initial_state_steps without a final write."
    ),
)
parser.add_argument(
    "--fixed_initial_state_tolerance",
    type=float,
    default=0.035,
    help="Stop fixed-start warm-up early when max 28-DoF state error is below this value.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- Rest of imports after AppLauncher ----
import json
import os
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import warp as wp

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401 — register tasks
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# GR00T imports
from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from rlinf.models.embodiment.gr00t.gr00t_action_model import GR00T_N1_5_ForRLActionPrediction

# ---------------------------------------------------------------------------
# Constants — must match YAML config in assemble_trocar
# ---------------------------------------------------------------------------
TASK_ID = "Isaac-Assemble-Trocar-G129-Dex3-RLinf-v0"
NUCLEUS_SERVER = "isaac-dev.ov.nvidia.com"
NUCLEUS_HEALTHCARE_BASE = f"omniverse://{NUCLEUS_SERVER}/Library/IsaacHealthcare/0.5.0"
SCENE1MX2 = f"{NUCLEUS_HEALTHCARE_BASE}/Props/OrcaScenes/Scene1MX2"
TASK_DESCRIPTION = "install trocar from box"
FPS = 30.0  # video recording fps (matches reference data format)

CAMERA_KEYS = ["front_camera", "left_wrist_camera", "right_wrist_camera"]
# Match reference dataset naming (cam_room / cam_left_wrist / cam_right_wrist)
CAMERA_LEROBOT_NAMES = {
    "front_camera": "observation.images.cam_room",
    "left_wrist_camera": "observation.images.cam_left_wrist",
    "right_wrist_camera": "observation.images.cam_right_wrist",
}

# Current env action order after the 15-DoF prefix already matches reference order:
# left arm, right arm, left hand thumb/middle/index, right hand thumb/middle/index.
PERM_TO_REF = list(range(28))

# Reference joint names (kCamelCase)
REF_JOINT_NAMES = [
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
    "kLeftHandMiddle0", "kLeftHandMiddle1",
    "kLeftHandIndex0", "kLeftHandIndex1",
    "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1",
    "kRightHandMiddle0", "kRightHandMiddle1",
]

# Joint indices for state extraction (from observations.py)
BODY_JOINT_INDICES = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
DEX3_JOINT_INDICES = [31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38]

# State slicing for GR00T: robot_joint_state[15:29] (14 shoulder) + dex3 (14)
SHOULDER_SLICE = (15, 29)  # from the 29-element body joint pos, take indices 15..29

# Action mapping
ACTION_PREFIX_PAD = 15  # zero-pad for uncontrolled body joints
ROBOT_ACTION_DIM = ACTION_PREFIX_PAD + len(PERM_TO_REF)


# ---------------------------------------------------------------------------
# GR00T data config (full pipeline with VideoCrop + VideoResize for 480x640 input)
# ---------------------------------------------------------------------------
VIDEO_KEYS = ["video.left_wrist_view", "video.right_wrist_view", "video.room_view"]
STATE_KEYS = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"]
ACTION_KEYS = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]


def build_modality_config():
    observation_indices = [0]
    action_indices = list(range(16))
    return {
        "video": ModalityConfig(delta_indices=observation_indices, modality_keys=VIDEO_KEYS),
        "state": ModalityConfig(delta_indices=observation_indices, modality_keys=STATE_KEYS),
        "action": ModalityConfig(delta_indices=action_indices, modality_keys=ACTION_KEYS),
        "language": ModalityConfig(
            delta_indices=observation_indices, modality_keys=["annotation.human.task_description"]
        ),
    }


def build_modality_transform():
    """Full transform pipeline for 480x640 input (includes crop+resize to 224x224)."""
    transforms = [
        VideoToTensor(apply_to=VIDEO_KEYS),
        VideoCrop(apply_to=VIDEO_KEYS, scale=0.95),
        VideoResize(apply_to=VIDEO_KEYS, height=224, width=224, interpolation="linear"),
        VideoToNumpy(apply_to=VIDEO_KEYS),
        StateActionToTensor(apply_to=STATE_KEYS),
        StateActionSinCosTransform(apply_to=STATE_KEYS),
        StateActionToTensor(apply_to=ACTION_KEYS),
        StateActionTransform(
            apply_to=ACTION_KEYS,
            normalization_modes={key: "min_max" for key in ACTION_KEYS},
        ),
        ConcatTransform(
            video_concat_order=VIDEO_KEYS,
            state_concat_order=STATE_KEYS,
            action_concat_order=ACTION_KEYS,
        ),
        GR00TTransform(state_horizon=1, action_horizon=16, max_state_dim=64, max_action_dim=32),
    ]
    return ComposedModalityTransform(transforms=transforms)


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------
# Merged category mapping: instance prim path pattern → category ID
# 0=background, 1=ground, 2=robot, 3=trocar_1, 4=trocar_2, 5=tray, 6=cart, 7=instrument_trolley
CATEGORY_NAMES = {
    0: "background",
    1: "ground",
    2: "robot",
    3: "trocar_1",
    4: "trocar_2",
    5: "tray",
    6: "cart",
    7: "instrument_trolley",
}


def _capture_episode_init_state(env) -> dict:
    """Capture randomized initial state for replay/verification.

    Returns dict with: tray_yaw_deg, tray_pose, trocar_1_pose, trocar_2_pose, light_params.
    """
    info = {}
    # Tray pose (world coordinates)
    if "tray" in env.scene.keys():
        tray_pose = wp.to_torch(env.scene["tray"].data.root_state_w)[0]  # (13,) pos+quat+vel
        info["tray_pos"] = tray_pose[0:3].cpu().numpy().tolist()
        info["tray_quat"] = tray_pose[3:7].cpu().numpy().tolist()
        # Compute yaw rotation relative to default
        default_quat = wp.to_torch(env.scene["tray"].data.default_root_state)[0, 3:7]
        # IsaacLab quat convention here is (x, y, z, w). Yaw around Z axis: 2*atan2(z, w)
        from math import atan2, degrees
        cur_x, cur_y, cur_z, cur_w = info["tray_quat"]
        def_x, def_y, def_z, def_w = default_quat.cpu().numpy().tolist()
        cur_yaw = degrees(2 * atan2(cur_z, cur_w))
        def_yaw = degrees(2 * atan2(def_z, def_w))
        info["tray_yaw_deg"] = cur_yaw - def_yaw

    if "trocar_1" in env.scene.keys():
        p = wp.to_torch(env.scene["trocar_1"].data.root_state_w)[0]
        info["trocar_1_pos"] = p[0:3].cpu().numpy().tolist()
        info["trocar_1_quat"] = p[3:7].cpu().numpy().tolist()
    if "trocar_2" in env.scene.keys():
        p = wp.to_torch(env.scene["trocar_2"].data.root_state_w)[0]
        info["trocar_2_pos"] = p[0:3].cpu().numpy().tolist()
        info["trocar_2_quat"] = p[3:7].cpu().numpy().tolist()
    if "robot" in env.scene.keys():
        robot_state_internal = get_joint_states_batch(env)[0]
        robot_state_ref = robot_state_internal[PERM_TO_REF]
        info["robot_state_ref"] = robot_state_ref.tolist()
        info["robot_left_hand_ref"] = robot_state_ref[14:21].tolist()
        info["robot_right_hand_ref"] = robot_state_ref[21:28].tolist()
    return info


def _capture_surgical_lighting_baseline(env) -> list[dict]:
    """Capture original surgical-room light attributes used as randomization baseline.

    Args:
        env: Isaac Lab environment containing the USD stage.

    Returns:
        A list of light attribute records. Intensities are in USD light intensity units.
    """
    baseline = []
    stage = env.sim.stage
    for prim in stage.Traverse():
        tname = str(prim.GetTypeName())
        if "Light" not in tname:
            continue

        intensity_attr_name = None
        intensity = None
        for attr_name in ["inputs:intensity", "intensity"]:
            attr = prim.GetAttribute(attr_name)
            if attr.IsValid() and attr.Get() is not None:
                intensity_attr_name = attr_name
                intensity = float(attr.Get())
                break

        color_attr_name = None
        color = None
        for attr_name in ["inputs:color", "color"]:
            attr = prim.GetAttribute(attr_name)
            if attr.IsValid() and attr.Get() is not None:
                color_attr_name = attr_name
                color_value = attr.Get()
                color = [float(color_value[0]), float(color_value[1]), float(color_value[2])]
                break

        if intensity_attr_name is None and color_attr_name is None:
            continue

        baseline.append({
            "path": str(prim.GetPath()),
            "intensity_attr": intensity_attr_name,
            "intensity": intensity,
            "color_attr": color_attr_name,
            "color": color,
        })

    return baseline


def _randomize_surgical_lighting(env, rng: np.random.RandomState, lighting_baseline: list[dict]) -> dict:
    """Randomize ALL lights within realistic surgical-room range.

    Real OR lighting is highly standardized (4000-5000K neutral-cool white,
    strict color accuracy). We randomize:
      - Intensity: ±30% variation (shadow/position differences)
      - Color temperature: very subtle cool-to-neutral shift

    Args:
        env: Isaac Lab environment containing the USD stage.
        rng: Numpy random state for deterministic per-episode sampling.
        lighting_baseline: Original light attributes captured before randomization.

    Returns:
        Dict with the applied params for replay/logging.
    """
    intensity_scale = float(rng.uniform(0.7, 1.3))
    temp_shift = float(rng.uniform(-0.05, 0.05))
    color = (
        max(0.0, 1.0 - temp_shift),
        1.0,
        min(1.0, 1.0 + temp_shift),
    )
    light_info = {
        "intensity_scale": intensity_scale,
        "color": list(color),
        "temp_shift": temp_shift,
    }
    from pxr import Gf
    stage = env.sim.stage
    for light in lighting_baseline:
        prim = stage.GetPrimAtPath(light["path"])
        if not prim.IsValid():
            continue

        if light["intensity_attr"] is not None and light["intensity"] is not None:
            attr = prim.GetAttribute(light["intensity_attr"])
            if attr.IsValid():
                attr.Set(float(light["intensity"]) * intensity_scale)

        if light["color_attr"] is not None:
            attr = prim.GetAttribute(light["color_attr"])
            if attr.IsValid():
                attr.Set(Gf.Vec3f(*color))
    light_info["num_lights"] = len(lighting_baseline)
    return light_info


def _instance_id_key_to_int(key) -> int:
    """Convert Isaac Sim instance-id map keys to the integer stored in the mask."""
    if isinstance(key, tuple):
        r, g, b, a = (int(x) for x in key)
        return r | (g << 8) | (b << 16) | (a << 24)
    return int(key)


def _label_entry_to_prim_path(label_entry) -> str:
    """Extract a prim path string from Isaac Sim idToLabels entries."""
    if isinstance(label_entry, dict):
        for key in ("primPath", "prim_path", "path", "class"):
            value = label_entry.get(key)
            if value:
                return str(value)
    return str(label_entry)


def _build_instance_to_category(info: dict) -> dict[int, int]:
    """Build mapping from instance IDs to merged category IDs using prim path patterns."""
    id_to_labels = info.get("instance_id_segmentation_fast", {}).get("idToLabels", {})
    mapping = {}
    for inst_id, prim_path in id_to_labels.items():
        inst_id = _instance_id_key_to_int(inst_id)
        prim_path = _label_entry_to_prim_path(prim_path)
        if "Robot/" in prim_path:
            mapping[inst_id] = 2  # robot
        elif "trocar_1" in prim_path:
            mapping[inst_id] = 3  # trocar_1
        elif "trocar_2" in prim_path:
            mapping[inst_id] = 4  # trocar_2
        elif "surgical_tray/" in prim_path:
            mapping[inst_id] = 5  # tray
        elif "Cart001" in prim_path:
            mapping[inst_id] = 6  # cart
        elif "InstrumentTrolley" in prim_path:
            mapping[inst_id] = 7  # instrument_trolley
        elif "FlatGrid" in prim_path or "GroundPlane" in prim_path:
            mapping[inst_id] = 1  # ground
        else:
            mapping[inst_id] = 0  # background / unknown
    return mapping


def apply_scene_variant(env_cfg, variant: str):
    """Patch env_cfg.scene in-place to use the requested background scene."""
    if variant == "lightwheel":
        return  # default, no changes needed

    env_cfg.scene.env_spacing = 50.0

    # Props shared by all non-lightwheel variants (matches i4h-workflows-internal exactly).
    # Rotations below are in IsaacLab 3.0 xyzw convention.
    # Original sim5 values were wxyz; converted via (w,x,y,z) -> (x,y,z,w).
    cart_cfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/cart001",
        spawn=UsdFileCfg(usd_path=f"{NUCLEUS_HEALTHCARE_BASE}/Props/LightWheel/Assets/Cart001/Cart001.usd"),
        # sim5 wxyz (1,0,0,0) = identity -> sim6 xyzw (0,0,0,1)
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.48242, 2.03195, 0.00279), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    trolley_cfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/instrument_trolley002",
        spawn=UsdFileCfg(
            usd_path=f"{NUCLEUS_HEALTHCARE_BASE}/Props/LightWheel/Assets/InstrumentTrolley001/InstrumentTrolley002.usd",
            scale=(1.05, 1.05, 1.05),
        ),
        # sim5 wxyz (0,0,0,1) = 180° around Z -> sim6 xyzw (0,0,1,0)
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.52131, 1.4862, 0.0), rot=(0.0, 0.0, 1.0, 0.0)),
    )


    if variant == "orca":
        env_cfg.scene.scene = AssetBaseCfg(
            prim_path="/World/envs/env_.*/Scene",
            spawn=UsdFileCfg(usd_path=f"{SCENE1MX2}/main_new_light.usd"),
            # sim5 wxyz (1,0,0,0) = identity -> sim6 xyzw (0,0,0,1)
            init_state=AssetBaseCfg.InitialStateCfg(pos=(4.0, -5.5, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
        )
        env_cfg.scene.light = None  # scene has embedded lights; matches i4h (dome commented out)
        setattr(env_cfg.scene, "cart001", cart_cfg)
        setattr(env_cfg.scene, "instrument_trolley002", trolley_cfg)

    elif variant == "factory":
        env_cfg.scene.scene = AssetBaseCfg(
            prim_path="/World/envs/env_.*/Scene",
            spawn=UsdFileCfg(usd_path=f"{SCENE1MX2}/rlinf_scenes/factory.usd"),
            # sim5 wxyz (0,0,0,1) = 180° around Z -> sim6 xyzw (0,0,1,0)
            init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 3.0, 0.0), rot=(0.0, 0.0, 1.0, 0.0)),
        )
        env_cfg.scene.light = None  # scene has embedded lights; matches i4h (dome commented out)
        setattr(env_cfg.scene, "cart001", cart_cfg)
        setattr(env_cfg.scene, "instrument_trolley002", trolley_cfg)

    elif variant == "surgical_room":
        env_cfg.scene.scene = AssetBaseCfg(
            prim_path="/World/envs/env_.*/Scene",
            spawn=UsdFileCfg(
                usd_path=f"{SCENE1MX2}/push-cart-OR-scenes/main.usd",
                scale=(0.008, 0.008, 0.008),
            ),
            # sim5 wxyz (0.707,0,0,-0.707) = -90° around Z -> sim6 xyzw (0,0,-0.707,0.707)
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-3.8, 5.3, 0.0), rot=(0.0, 0.0, -0.70710678, 0.70710678)
            ),
        )
        env_cfg.scene.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
            # sim5 wxyz (1,0,0,0) = identity -> sim6 xyzw (0,0,0,1)
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-3.8, 5.3, 2.0), rot=(0.0, 0.0, 0.0, 1.0)),
        )
        setattr(env_cfg.scene, "cart001", cart_cfg)
        setattr(env_cfg.scene, "instrument_trolley002", trolley_cfg)


def create_env():
    """Create IsaacLab env with instance segmentation cameras."""
    env_cfg = parse_env_cfg(TASK_ID, device=args.model_device, num_envs=args.num_envs)
    apply_scene_variant(env_cfg, args.scene)
    if hasattr(env_cfg.events, "reset_tray_random_rotation"):
        env_cfg.events.reset_tray_random_rotation.params["rotation_range"] = [
            args.tray_yaw_min_deg,
            args.tray_yaw_max_deg,
        ]

    for cam_name in CAMERA_KEYS:
        cam_cfg = getattr(env_cfg.scene, cam_name)
        if args.no_mask:
            cam_cfg.data_types = ["rgb"]
        else:
            cam_cfg.data_types = ["rgb", "instance_id_segmentation_fast"]
            cam_cfg.colorize_instance_id_segmentation = False

    # Keep terminations (success / drop / timeout) so episodes end naturally
    # Only disable recorders
    env_cfg.recorders = {}

    env = gym.make(TASK_ID, cfg=env_cfg).unwrapped
    return env


# ---------------------------------------------------------------------------
# Observation / action conversion
# ---------------------------------------------------------------------------
def get_joint_states_batch(env) -> np.ndarray:
    """Extract joint positions for all envs: (B, 28) — shoulder(14) + dex3(14)."""
    joint_pos = wp.to_torch(env.scene["robot"].data.joint_pos)  # (B, num_joints)
    device = joint_pos.device

    body_idx = torch.tensor(BODY_JOINT_INDICES, device=device, dtype=torch.long)
    body_pos = joint_pos[:, body_idx]
    shoulder_pos = body_pos[:, SHOULDER_SLICE[0] : SHOULDER_SLICE[1]]  # (B, 14)

    dex3_idx = torch.tensor(DEX3_JOINT_INDICES, device=device, dtype=torch.long)
    dex3_pos = joint_pos[:, dex3_idx]  # (B, 14)

    states = torch.cat([shoulder_pos, dex3_pos], dim=-1)  # (B, 28)
    return states.cpu().numpy().astype(np.float32)


def wrap_obs_rlinf(obs: dict, num_envs: int) -> dict:
    """Wrap observations exactly like extension.py _wrap_obs does.

    Uses the imported functions from extension.py directly.
    """
    from isaaclab_contrib.rl.rlinf.extension import _gpu_resize_images, _get_isaaclab_cfg

    policy_obs = obs.get("policy", obs)
    camera_obs = obs.get("camera_images", {})
    cfg = _get_isaaclab_cfg()
    task_desc = cfg.get("task_description", TASK_DESCRIPTION)

    rlinf_obs = {"task_descriptions": [task_desc] * num_envs}

    target_h = cfg.get("gpu_resize_height", 224)
    target_w = cfg.get("gpu_resize_width", 224)
    crop_scale = cfg.get("gpu_crop_scale", 0.05)
    # Use center crop for eval (not random)
    random_crop = False

    main_key = cfg.get("main_images", "front_camera")
    if main_key and main_key in camera_obs:
        rlinf_obs["main_images"] = _gpu_resize_images(
            camera_obs[main_key], target_h, target_w, crop_scale, random_crop
        )

    extra_keys = cfg.get("extra_view_images", [])
    if extra_keys:
        if isinstance(extra_keys, str):
            extra_keys = [extra_keys]
        extra_imgs = []
        for k in extra_keys:
            if k in camera_obs:
                extra_imgs.append(
                    _gpu_resize_images(camera_obs[k], target_h, target_w, crop_scale, random_crop)
                )
        if extra_imgs:
            rlinf_obs["extra_view_images"] = torch.stack(extra_imgs, dim=1)

    state_specs = cfg.get("states", [])
    if state_specs:
        state_parts = []
        for spec in state_specs:
            if isinstance(spec, str):
                state = policy_obs.get(spec)
                if state is not None:
                    state_parts.append(state)
            elif isinstance(spec, dict):
                state = policy_obs.get(spec.get("key"))
                if state is not None:
                    slice_range = spec.get("slice")
                    if slice_range:
                        state = state[:, slice_range[0] : slice_range[1]]
                    state_parts.append(state)
        if state_parts:
            rlinf_obs["states"] = torch.cat(state_parts, dim=-1)

    return rlinf_obs


INV_PERM = np.argsort(PERM_TO_REF)  # ref order → internal order


def load_fixed_initial_state_ref(dataset_dir: str, episode_idx: int, frame_idx: int) -> np.ndarray:
    """Load one 28-DoF reference-order state from a LeRobot episode parquet."""
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


def _as_action_vector(value: torch.Tensor | float, action_dim: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        value = value.to(device=device, dtype=torch.float32)
        if value.ndim == 2:
            value = value[0]
        return value.reshape(action_dim)
    return torch.full((action_dim,), float(value), device=device, dtype=torch.float32)


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


def fixed_initial_state_ref_to_raw_action(env, state_ref: np.ndarray) -> np.ndarray:
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


def _fixed_initial_state_error(env, target_state_ref: np.ndarray) -> float:
    states_ref = get_joint_states_batch(env)[:, PERM_TO_REF]
    error = np.abs(states_ref - target_state_ref[np.newaxis, :])
    return float(error.max()) if error.size else 0.0


def _write_fixed_initial_state_to_sim(env, target_state_ref: np.ndarray) -> None:
    """Directly write the fixed start pose to the controlled robot joints."""
    robot = env.scene["robot"]
    joint_ids = _get_action_joint_ids(env)[ACTION_PREFIX_PAD:].to(dtype=torch.int32)
    target_internal = np.asarray(target_state_ref, dtype=np.float32)[INV_PERM]
    target_pos = torch.tensor(target_internal, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    target_vel = torch.zeros_like(target_pos)
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
    robot.write_joint_position_to_sim_index(position=target_pos, joint_ids=joint_ids, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=target_vel, joint_ids=joint_ids, env_ids=env_ids)


def apply_fixed_initial_state(
    env,
    obs: dict,
    fixed_raw_action: np.ndarray | None,
    target_state_ref: np.ndarray | None,
    steps: int,
    tolerance: float,
    mode: str = "command",
) -> tuple[dict, dict | None]:
    """Command a fixed start pose for a few env steps before policy inference."""
    if fixed_raw_action is None or target_state_ref is None:
        return obs, None

    if mode in ("teleport", "teleport_settle"):
        _write_fixed_initial_state_to_sim(env, target_state_ref)

    action_batch = np.repeat(fixed_raw_action[np.newaxis, :], env.num_envs, axis=0)
    action_tensor = torch.tensor(action_batch, dtype=torch.float32, device=env.device)
    steps_run = 0
    max_state_error = _fixed_initial_state_error(env, target_state_ref)

    for warm_step in range(max(steps, 0)):
        obs, _, _, _, _ = env.step(action_tensor)
        steps_run = warm_step + 1
        max_state_error = _fixed_initial_state_error(env, target_state_ref)
        if mode == "command" and max_state_error <= tolerance:
            break
    if mode == "teleport":
        # Set the exact requested start pose after the settling steps so policy inference
        # starts from a fixed robot state while objects have already settled.
        _write_fixed_initial_state_to_sim(env, target_state_ref)
        max_state_error = _fixed_initial_state_error(env, target_state_ref)

    return obs, {
        "mode": mode,
        "steps": steps_run,
        "max_state_error": max_state_error,
        "target_state_ref": target_state_ref.tolist(),
        "target_left_hand_ref": target_state_ref[14:21].tolist(),
        "target_right_hand_ref": target_state_ref[21:28].tolist(),
    }


def wrap_obs_gr00t(env) -> dict:
    """Build observation dict for Gr00tPolicy.get_action() (single env only).

    Returns raw camera images and joint states in the format expected by
    UnitreeG1SimDataConfig — the transform pipeline inside Gr00tPolicy
    handles resizing, sin/cos encoding, and normalization.
    """
    # Camera images: (H, W, 3) uint8, add batch dim → (1, H, W, 3)
    cam_map = {
        "front_camera":       "video.room_view",
        "left_wrist_camera":  "video.left_wrist_view",
        "right_wrist_camera": "video.right_wrist_view",
    }
    obs_dict = {}
    for cam_key, gr00t_key in cam_map.items():
        img = get_camera_rgb_batch(env, cam_key)[0]       # (H, W, 3)
        obs_dict[gr00t_key] = img[np.newaxis]             # (1, H, W, 3)

    # Joint states in ref order (same as training data)
    states_ref = get_joint_states_batch(env)[0]           # (28,) ref order
    obs_dict["state.left_arm"]   = states_ref[0:7][np.newaxis]   # (1, 7)
    obs_dict["state.right_arm"]  = states_ref[7:14][np.newaxis]  # (1, 7)
    obs_dict["state.left_hand"]  = states_ref[14:21][np.newaxis] # (1, 7)
    obs_dict["state.right_hand"] = states_ref[21:28][np.newaxis] # (1, 7)

    obs_dict["annotation.human.task_description"] = [TASK_DESCRIPTION]
    return obs_dict


def gr00t_action_chunk_to_isaaclab(action_dict: dict) -> np.ndarray:
    """Convert Gr00tPolicy action dict → Isaac Lab action chunk.

    Concatenates the four action keys in ref order, then prepends 15 zero-padded
    body joints that the env action space requires.

    Returns:
        Array with shape ``(T, 43)`` where ``T`` is the available action horizon.
    """
    parts = []
    for key in ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]:
        value = np.asarray(action_dict[key], dtype=np.float32)
        if value.ndim == 3:
            value = value[0]
        elif value.ndim == 1:
            value = value[np.newaxis, :]
        parts.append(value)

    chunk_len = min(part.shape[0] for part in parts)
    action_chunk = []
    for step_idx in range(chunk_len):
        action_ref = np.concatenate([part[step_idx] for part in parts], axis=-1)
        action_internal = action_ref[INV_PERM]
        action_43 = np.concatenate(
            [np.zeros(ACTION_PREFIX_PAD, dtype=np.float32), action_internal],
            axis=0,
        )
        action_chunk.append(action_43.astype(np.float32, copy=False))
    return np.stack(action_chunk, axis=0)


def rlinf_action_chunk_to_isaaclab(raw_action: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert RLinf GR00T action output into a batch action chunk.

    Args:
        raw_action: RLinf action output with shape ``(B, T, 43)`` or ``(B, 43)``.

    Returns:
        Array with shape ``(B, T, 43)``.
    """
    if isinstance(raw_action, torch.Tensor):
        raw_action = raw_action.detach().cpu().numpy()
    else:
        raw_action = np.asarray(raw_action)

    if raw_action.ndim == 2:
        raw_action = raw_action[:, np.newaxis, :]
    if raw_action.ndim != 3:
        raise ValueError(f"Expected RLinf action chunk with 2 or 3 dims, got shape {raw_action.shape}.")

    return raw_action.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Camera data extraction
# ---------------------------------------------------------------------------
def get_camera_rgb_batch(env, cam_name: str) -> np.ndarray:
    """Get raw 480x640 RGB images for all envs. Returns (B, H, W, 3) uint8."""
    sensor = env.scene.sensors[cam_name]
    imgs = sensor.data.output["rgb"]  # (B, H, W, 4) or (B, H, W, 3)
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().numpy()
    if imgs.shape[-1] == 4:
        imgs = imgs[..., :3]
    return imgs.astype(np.uint8)


def get_camera_segmentation_batch(env, cam_name: str, inst_to_cat: dict) -> np.ndarray:
    """Get instance segmentation masks merged into category IDs. Returns (B, H, W) uint8."""
    sensor = env.scene.sensors[cam_name]
    seg = sensor.data.output["instance_id_segmentation_fast"]  # (B, H, W) int32
    if isinstance(seg, torch.Tensor):
        seg = seg.cpu().numpy()
    if seg.ndim == 4 and seg.shape[-1] == 1:
        seg = seg[..., 0]
    elif seg.ndim == 4 and seg.shape[-1] == 4:
        seg = seg.astype(np.uint32)
        seg = seg[..., 0] | (seg[..., 1] << 8) | (seg[..., 2] << 16) | (seg[..., 3] << 24)
    # Merge instance IDs into category IDs
    merged = np.zeros_like(seg, dtype=np.uint8)
    for inst_id, cat_id in inst_to_cat.items():
        merged[seg == inst_id] = cat_id
    return merged


# ---------------------------------------------------------------------------
# Data saving (LeRobot format)
# ---------------------------------------------------------------------------
def save_episode_data(
    output_dir: Path,
    episode_idx: int,
    timestamps: list,
    states: list,
    actions: list,
    frames: dict,  # {cam_name: [list of (H,W,3) uint8]}
    masks: dict,   # {cam_name: [list of (H,W) uint8]}
    skip_first_n: int = 0,
    skip_last_n: int = 0,
    stages: list = None,
):
    """Save one episode in LeRobot format. Optionally skip first/last N frames."""
    chunk_dir = "chunk-000"
    end_idx = -skip_last_n if skip_last_n > 0 else None
    if skip_first_n > 0 or skip_last_n > 0:
        timestamps = timestamps[skip_first_n:end_idx]
        states = states[skip_first_n:end_idx]
        actions = actions[skip_first_n:end_idx]
        frames = {k: v[skip_first_n:end_idx] for k, v in frames.items()}
        masks = {k: v[skip_first_n:end_idx] for k, v in masks.items()}
        if stages is not None:
            stages = stages[skip_first_n:end_idx]
    T = len(timestamps)
    if T > 0 and skip_first_n > 0:
        t0 = timestamps[0]
        timestamps = [t - t0 for t in timestamps]

    # ---- data/chunk-000/episode_XXXXXX.parquet ----
    data_dir = output_dir / "data" / chunk_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # States/actions are already in reference dataset order.
    perm = np.array(PERM_TO_REF)
    states_ref = [np.asarray(s, dtype=np.float32)[perm] for s in states]
    actions_ref = [np.asarray(a, dtype=np.float32)[perm] for a in actions]

    # Match reference column order and dtypes exactly
    df_dict = {
        "observation.state": states_ref,
        "action": actions_ref,
        "timestamp": np.array(timestamps, dtype=np.float32),
        "frame_index": np.arange(T, dtype=np.int64),
        "episode_index": np.full(T, episode_idx, dtype=np.int64),
        "index": np.arange(T, dtype=np.int64),
        "task_index": np.zeros(T, dtype=np.int64),
    }
    # Additional annotation columns (not in reference dataset):
    #   stage: 0=no-op, 1=both lifted, 2=tips aligned, 3=inserted, 4=placed
    #   first_trocar_lifted: bool (True once either trocar crossed lift threshold)
    if stages is not None and len(stages) == T:
        df_dict["stage"] = np.array(stages, dtype=np.int64)
    df = pd.DataFrame(df_dict)
    df.to_parquet(data_dir / f"episode_{episode_idx:06d}.parquet", index=False)

    # ---- videos/chunk-000/<camera_key>/episode_XXXXXX.mp4 ----
    for cam_name, cam_frames in frames.items():
        lerobot_name = CAMERA_LEROBOT_NAMES[cam_name]
        vid_dir = output_dir / "videos" / chunk_dir / lerobot_name
        vid_dir.mkdir(parents=True, exist_ok=True)
        vid_path = vid_dir / f"episode_{episode_idx:06d}.mp4"

        h, w = cam_frames[0].shape[:2]
        writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        for frame in cam_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    # ---- masks/chunk-000/<camera_key>/episode_XXXXXX_masks.npz ----
    for cam_name, cam_masks in masks.items():
        if not cam_masks:
            continue
        lerobot_name = CAMERA_LEROBOT_NAMES[cam_name]
        mask_dir = output_dir / "masks" / chunk_dir / lerobot_name
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"episode_{episode_idx:06d}_masks.npz"

        mask_array = np.stack(cam_masks, axis=0)  # (T, H, W)
        np.savez_compressed(str(mask_path), mask_array)


def save_metadata(output_dir: Path, episode_lengths: list, joint_names: list):
    """Generate meta/ directory matching reference dataset format."""
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    num_episodes = len(episode_lengths)
    total_frames = sum(episode_lengths)

    state_dim = len(REF_JOINT_NAMES)
    camera_lerobot_names = list(CAMERA_LEROBOT_NAMES.values())

    # ---- info.json ----
    info = {
        "codebase_version": "v2.1",
        "robot_type": "Unitree_G1_Dex3_3camera",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": num_episodes * len(CAMERA_KEYS),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": int(FPS),
        "splits": {"train": f"0:{num_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "mask_path": "masks/chunk-{episode_chunk:03d}/{mask_key}/episode_{episode_index:06d}_masks.npz",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": [REF_JOINT_NAMES],  # nested list to match reference
            },
            "action": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": [REF_JOINT_NAMES],  # nested list to match reference
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }

    # Add camera features
    for cam_lerobot_name in camera_lerobot_names:
        info["features"][cam_lerobot_name] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": int(FPS),
                "video.channels": 3,
                "has_audio": False,
            },
        }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # ---- modality.json ----
    cam_short = {
        "observation.images.cam_room": "room_view",
        "observation.images.cam_left_wrist": "left_wrist_view",
        "observation.images.cam_right_wrist": "right_wrist_view",
    }
    modality = {
        "state": {
            "left_arm": {"start": 0, "end": 7},
            "right_arm": {"start": 7, "end": 14},
            "left_hand": {"start": 14, "end": 21},
            "right_hand": {"start": 21, "end": 28},
        },
        "action": {
            "left_arm": {"start": 0, "end": 7},
            "right_arm": {"start": 7, "end": 14},
            "left_hand": {"start": 14, "end": 21},
            "right_hand": {"start": 21, "end": 28},
        },
        "video": {
            cam_short[cam]: {"original_key": cam}
            for cam in camera_lerobot_names
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
        "mask": {
            cam_name.split(".")[-1]: {"original_key": cam_name}
            for cam_name in camera_lerobot_names
        },
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    # ---- episodes.jsonl ----
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for i, length in enumerate(episode_lengths):
            f.write(json.dumps({"episode_index": i, "tasks": [TASK_DESCRIPTION], "length": length}) + "\n")

    # ---- tasks.jsonl ----
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK_DESCRIPTION}) + "\n")

    # ---- stats.json + episodes_stats.jsonl ----
    # Compute per-episode and global stats for state and action
    data_dir = output_dir / "data" / "chunk-000"
    if not data_dir.exists():
        return

    all_states = []
    all_actions = []
    per_ep_stats = []
    for i in range(num_episodes):
        ep_path = data_dir / f"episode_{i:06d}.parquet"
        if not ep_path.exists():
            continue
        df = pd.read_parquet(ep_path)
        states_arr = np.array([np.asarray(s) for s in df["observation.state"]], dtype=np.float32)
        actions_arr = np.array([np.asarray(a) for a in df["action"]], dtype=np.float32)
        all_states.append(states_arr)
        all_actions.append(actions_arr)

        ep_stat = {
            "episode_index": i,
            "stats": {
                "observation.state": _compute_field_stats(states_arr),
                "action": _compute_field_stats(actions_arr),
            },
        }
        per_ep_stats.append(ep_stat)

    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for s in per_ep_stats:
            f.write(json.dumps(s) + "\n")

    if all_states:
        global_states = np.concatenate(all_states, axis=0)
        global_actions = np.concatenate(all_actions, axis=0)
        global_stats = {
            "observation.state": _compute_field_stats(global_states),
            "action": _compute_field_stats(global_actions),
        }
        with open(meta_dir / "stats.json", "w") as f:
            json.dump(global_stats, f, indent=2)


def save_episode_results(output_dir: Path, episode_results: list[dict]):
    """Save episode-level success metadata.

    Args:
        output_dir: Directory to write the metadata file to.
        episode_results: Episode result dictionaries accumulated so far.
    """
    tmp_path = output_dir / "episode_results.json.tmp"
    final_path = output_dir / "episode_results.json"
    with open(tmp_path, "w") as f:
        json.dump({
            "total_episodes": len(episode_results),
            "success_count": sum(1 for r in episode_results if r["success"]),
            "fail_count": sum(1 for r in episode_results if not r["success"]),
            "episodes": sorted(episode_results, key=lambda x: x["episode_index"]),
        }, f, indent=2)
    os.replace(tmp_path, final_path)


def _compute_field_stats(arr: np.ndarray) -> dict:
    """Compute min/max/mean/std/q01/q99 for each dim of a (T, D) array."""
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Nucleus authentication
# ---------------------------------------------------------------------------
_NUCLEUS_AUTH_REG = None  # keep reference alive to prevent GC


def _setup_nucleus_auth():
    global _NUCLEUS_AUTH_REG
    token = os.environ.get("OMNI_PASS") or os.environ.get("OMNI_API_TOKEN", "")
    if not token:
        print(
            "[WARN] No Nucleus API token found (OMNI_PASS / OMNI_API_TOKEN). "
            "Remote omniverse:// assets may fail to load.",
            flush=True,
        )
        return
    import omni.client

    def _auth_callback(url_prefix: str):
        return ("$omni-api-token", token)

    _NUCLEUS_AUTH_REG = omni.client.register_authentication_callback(_auth_callback)
    print(f"[INFO] Nucleus auth registered for {NUCLEUS_SERVER}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _setup_nucleus_auth()
    print("[INFO] Creating environment...")
    env = create_env()

    fixed_initial_state_ref = None
    fixed_initial_raw_action = None
    if args.fixed_initial_state_dataset:
        if args.num_envs != 1:
            raise ValueError("--fixed_initial_state_dataset currently requires --num_envs 1 to avoid perturbing active envs.")
        fixed_initial_state_ref = load_fixed_initial_state_ref(
            args.fixed_initial_state_dataset,
            args.fixed_initial_state_episode,
            args.fixed_initial_state_frame,
        )
        fixed_initial_raw_action = fixed_initial_state_ref_to_raw_action(env, fixed_initial_state_ref)
        print(
            "[INFO] Fixed initial state enabled: "
            f"dataset={args.fixed_initial_state_dataset}, "
            f"episode={args.fixed_initial_state_episode}, frame={args.fixed_initial_state_frame}, "
            f"warmup_steps={args.fixed_initial_state_steps}"
        )
        print(f"[INFO] Fixed initial left hand ref: {fixed_initial_state_ref[14:21].tolist()}")
        print(f"[INFO] Fixed initial right hand ref: {fixed_initial_state_ref[21:28].tolist()}")

    print(f"[INFO] Loading GR00T model from {args.model_path}...")
    # Prevent any HuggingFace Hub network requests — model must be local.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    config_dir = str(
        Path(__file__).resolve().parents[2]
        / "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config"
    )
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)

    if args.use_gr00t_policy:
        # ---- SFT model: Gr00tPolicy (no RLinf) ----
        # Import UnitreeG1SimDataConfig from Isaac-GR00T root (not the local isaaclab config)
        isaac_gr00t_dir = str(Path("/localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T"))
        if isaac_gr00t_dir not in sys.path:
            sys.path.insert(0, isaac_gr00t_dir)
        from gr00t.model.policy import Gr00tPolicy
        # Import UnitreeG1SimDataConfig without shadowing IsaacLabDataConfig
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "gr00t_config_sft",
            str(Path(isaac_gr00t_dir) / "gr00t_config.py"),
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        data_config = _mod.UnitreeG1SimInferDataConfig()
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag="new_embodiment",
            denoising_steps=args.denoising_steps,
            device=args.model_device,
        )
        print("[INFO] Using Gr00tPolicy (SFT mode)")
    else:
        # ---- RLinf model: GR00T_N1_5_ForRLActionPrediction ----
        yaml_path = str(Path(config_dir) / "isaaclab_ppo_gr00t_assemble_trocar.yaml")
        os.environ["RLINF_CONFIG_FILE"] = yaml_path
        os.environ.setdefault("RLINF_EXT_MODULE", "isaaclab_contrib.rl.rlinf.extension")

        from isaaclab_contrib.rl.rlinf.extension import _register_gr00t_converters, _patch_embodiment_tags
        isaaclab_cfg = {
            "obs_converter_type": "dex3",
            "embodiment_tag": "new_embodiment",
            "embodiment_tag_id": 31,
            "task_description": TASK_DESCRIPTION,
            "gr00t_mapping": {
                "video": {
                    "main_images": "video.room_view",
                    "extra_view_images": ["video.left_wrist_view", "video.right_wrist_view"],
                },
                "state": [
                    {"gr00t_key": "state.left_arm", "slice": [0, 7]},
                    {"gr00t_key": "state.right_arm", "slice": [7, 14]},
                    {"gr00t_key": "state.left_hand", "slice": [14, 21]},
                    {"gr00t_key": "state.right_hand", "slice": [21, 28]},
                ],
            },
            "action_mapping": {"prefix_pad": 15, "suffix_pad": 0},
        }
        _register_gr00t_converters(isaaclab_cfg)
        _patch_embodiment_tags(isaaclab_cfg)

        from gr00t_config import IsaacLabDataConfig
        data_config = IsaacLabDataConfig()
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        from omegaconf import OmegaConf
        rl_head_config = OmegaConf.create({
            "joint_logprob": False, "noise_method": "flow_sde", "ignore_last": False,
            "safe_get_logprob": False, "noise_anneal": False, "noise_params": [0.7, 0.3, 400],
            "noise_level": 0.3, "add_value_head": True, "chunk_critic_input": False,
            "detach_critic_input": True, "disable_dropout": True,
            "use_vlm_value": False, "value_vlm_mode": "mean_token", "padding_value": 850,
        })

        # Detect bare RLinf checkpoint: dir has full_weights.pt but no config.json.
        bare_rlinf_ckpt = (
            os.path.isfile(os.path.join(args.model_path, "full_weights.pt"))
            and not os.path.isfile(os.path.join(args.model_path, "config.json"))
        )
        if bare_rlinf_ckpt:
            print(
                f"[INFO] Bare RLinf checkpoint detected (only full_weights.pt). "
                f"Building model architecture from {args.rlinf_base_config_path}, then loading RLinf weights."
            )
            config_source = args.rlinf_base_config_path
        else:
            config_source = args.model_path

        policy = GR00T_N1_5_ForRLActionPrediction.from_pretrained(
            config_source,
            torch_dtype=torch.bfloat16,
            embodiment_tag="new_embodiment",
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=args.denoising_steps,
            output_action_chunks=max(args.open_loop_steps or 1, 1),
            obs_converter_type="dex3",
            tune_visual=False,
            tune_llm=False,
            rl_head_config=rl_head_config,
        )

        if bare_rlinf_ckpt:
            weights_path = os.path.join(args.model_path, "full_weights.pt")
            state_dict = torch.load(weights_path, map_location="cpu")
            missing, unexpected = policy.load_state_dict(state_dict, strict=False)
            print(
                f"[INFO] Loaded RLinf weights from {weights_path}: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

        policy.to(torch.bfloat16)
        policy.eval()
        policy.to(args.model_device)
        print("[INFO] Using GR00T_N1_5_ForRLActionPrediction (RLinf mode)")

    # Get joint names for metadata
    joint_names = [
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
        "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
        "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
        "left_hand_index_0", "left_hand_middle_0", "left_hand_thumb_0",
        "left_hand_index_1", "left_hand_middle_1", "left_hand_thumb_1", "left_hand_thumb_2",
        "right_hand_index_0", "right_hand_middle_0", "right_hand_thumb_0",
        "right_hand_index_1", "right_hand_middle_1", "right_hand_thumb_1", "right_hand_thumb_2",
    ]

    episode_lengths = []
    episode_results = []

    inst_to_cat = {}
    num_envs = args.num_envs
    total_needed = args.num_episodes
    open_loop_steps = args.open_loop_steps if args.open_loop_steps is not None else (8 if args.use_gr00t_policy else 1)
    cached_policy_actions: list[np.ndarray] = []

    # Per-env episode buffers
    def _new_buffer():
        return {
            "timestamps": [], "states": [], "actions": [], "stages": [],
            "frames": {cam: [] for cam in CAMERA_KEYS},
            "masks": {cam: [] for cam in CAMERA_KEYS},
            "step_count": 0,
        }

    print(f"[INFO] Recording {total_needed} episodes with {num_envs} parallel envs "
          f"(max {args.max_steps} steps each)...")
    policy_mode = "Gr00tPolicy" if args.use_gr00t_policy else "RLinf GR00T"
    print(f"[INFO] {policy_mode} open-loop steps per inference: {open_loop_steps}")
    if args.use_gr00t_policy:
        if num_envs != 1:
            print("[WARN] --use_gr00t_policy currently builds observations from env 0 and tiles the action to all envs.")

    # Per-episode lighting RNG (use args.seed as base)
    light_rng_base = args.seed if args.seed is not None else 0
    lighting_baseline = _capture_surgical_lighting_baseline(env) if args.randomize_lighting else []

    # Per-env init state captured at episode start
    env_init_states = [None] * args.num_envs

    with torch.inference_mode():
        obs, _ = env.reset(seed=args.seed)
        cached_policy_actions = []
        light_info = None
        if args.randomize_lighting:
            light_info = _randomize_surgical_lighting(
                env,
                np.random.RandomState(light_rng_base),
                lighting_baseline,
            )
        fixed_initial_info = None
        obs, fixed_initial_info = apply_fixed_initial_state(
            env,
            obs,
            fixed_initial_raw_action,
            fixed_initial_state_ref,
            args.fixed_initial_state_steps,
            args.fixed_initial_state_tolerance,
            args.fixed_initial_state_mode,
        )
        # Flush camera: same method as RLinf _patched_reset
        import omni.kit.app
        _app = omni.kit.app.get_app()
        env.sim.set_setting("/app/player/playSimulations", False)
        _app.update()
        env.sim.set_setting("/app/player/playSimulations", True)
        for sensor in env.scene.sensors.values():
            sensor.update(dt=0.0, force_recompute=True)
        obs = env.observation_manager.compute(update_history=True)

        if not args.no_mask:
            print("[INFO] Building instance-to-category mapping...")
            first_sensor = env.scene.sensors[CAMERA_KEYS[0]]
            cam_info = first_sensor.data.info
            if isinstance(cam_info, list):
                info_dict = cam_info[0] if cam_info else {}
            else:
                info_dict = cam_info
            if isinstance(info_dict, str):
                import ast
                info_dict = ast.literal_eval(info_dict)
            inst_to_cat = _build_instance_to_category(info_dict)
            with open(output_dir / "category_mapping.json", "w") as f:
                json.dump(
                    {
                        "categories": CATEGORY_NAMES,
                        "instance_to_category": {str(k): v for k, v in inst_to_cat.items()},
                    },
                    f,
                    indent=2,
                )
            print(f"  Mapped {len(inst_to_cat)} instance IDs to {len(set(inst_to_cat.values()))} categories")

        # Capture init state for env 0 (assigned to first episode)
        init_state = _capture_episode_init_state(env)
        if light_info is not None:
            init_state["lighting"] = light_info
        if fixed_initial_info is not None:
            init_state["fixed_initial_state"] = fixed_initial_info
        init_state["seed"] = args.seed
        env_init_states[0] = init_state

        buffers = [_new_buffer() for _ in range(num_envs)]
        next_ep_idx = 0  # next episode index to assign

        # Assign episode indices to each env
        env_ep_idx = list(range(min(num_envs, total_needed)))
        next_ep_idx = len(env_ep_idx)
        # Pad if fewer episodes than envs
        while len(env_ep_idx) < num_envs:
            env_ep_idx.append(-1)  # -1 = inactive

        while len(episode_results) < total_needed:
            # Inference first (using current obs from reset or previous step)
            if args.use_gr00t_policy:
                if not cached_policy_actions:
                    gr00t_obs = wrap_obs_gr00t(env)
                    with torch.no_grad():
                        action_dict = policy.get_action(gr00t_obs)
                    action_chunk = gr00t_action_chunk_to_isaaclab(action_dict)
                    if len(action_chunk) == 0:
                        raise RuntimeError("GR00T policy returned an empty action chunk.")
                    cached_policy_actions = [
                        action.copy() for action in action_chunk[: max(open_loop_steps, 1)]
                    ]
                raw_action = cached_policy_actions.pop(0)[np.newaxis, :]  # (1, 43)
                raw_action = np.tile(raw_action, (num_envs, 1))          # (B, 43)
            else:
                if not cached_policy_actions:
                    rlinf_obs = wrap_obs_rlinf(obs, num_envs)
                    with torch.no_grad():
                        raw_action_chunk, _ = policy.predict_action_batch(rlinf_obs, mode="eval")
                    action_chunk = rlinf_action_chunk_to_isaaclab(raw_action_chunk)
                    cached_policy_actions = [
                        action_chunk[:, step_idx, :].copy()
                        for step_idx in range(min(action_chunk.shape[1], max(open_loop_steps, 1)))
                    ]
                raw_action = cached_policy_actions.pop(0)  # (B, 43)
            action_tensor = torch.tensor(raw_action, dtype=torch.float32, device=env.device)

            # Step all envs — after this, camera images are fresh
            obs, reward, terminated, truncated, infos = env.step(action_tensor)

            # NOW collect data (images are guaranteed fresh after step)
            states_batch = get_joint_states_batch(env)
            # Compute stage 0-5 per env:
            #   0: initial
            #   1: first trocar lifted (but not both)
            #   2: both trocars lifted
            #   3: tips aligned
            #   4: inserted
            #   5: placed (success)
            env_stage = env._task_stage.cpu().numpy() if hasattr(env, "_task_stage") else np.zeros(num_envs, dtype=np.int32)
            LIFT_Z = 0.85483 + 0.05
            t1_pos = wp.to_torch(env.scene["trocar_1"].data.root_pos_w)
            t2_pos = wp.to_torch(env.scene["trocar_2"].data.root_pos_w)
            either_lifted = ((t1_pos[:, 2] > LIFT_Z) | (t2_pos[:, 2] > LIFT_Z)).cpu().numpy()
            # env_stage 0 + either lifted → our stage 1
            # env_stage 1-4 → our stage 2-5
            stage_batch = np.where(
                env_stage >= 1,
                env_stage + 1,
                np.where(either_lifted, 1, 0)
            ).astype(np.int32)
            rgb_batch = {}
            seg_batch = {}
            for cam_name in CAMERA_KEYS:
                rgb_batch[cam_name] = get_camera_rgb_batch(env, cam_name)
                if not args.no_mask:
                    seg_batch[cam_name] = get_camera_segmentation_batch(env, cam_name, inst_to_cat)

            # Store per-env data
            for i in range(num_envs):
                if env_ep_idx[i] < 0:
                    continue
                buf = buffers[i]
                buf["timestamps"].append(buf["step_count"] / FPS)
                buf["states"].append(states_batch[i])
                buf["stages"].append(int(stage_batch[i]))
                for cam_name in CAMERA_KEYS:
                    buf["frames"][cam_name].append(rgb_batch[cam_name][i])
                    if not args.no_mask:
                        buf["masks"][cam_name].append(seg_batch[cam_name][i])
                action_28 = action_tensor[i, ACTION_PREFIX_PAD:ACTION_PREFIX_PAD + 28].cpu().numpy().astype(np.float32)
                buffers[i]["actions"].append(action_28)
                buffers[i]["step_count"] += 1

            # Check which envs finished
            done = terminated | truncated  # (B,)
            for i in range(num_envs):
                ep_idx = env_ep_idx[i]
                if ep_idx < 0:
                    continue

                is_done = bool(done[i]) or buffers[i]["step_count"] >= args.max_steps
                if not is_done:
                    continue

                # Save this episode
                buf = buffers[i]
                ep_length = len(buf["timestamps"])
                episode_lengths.append(ep_length)

                is_success = bool(terminated[i]) and not bool(truncated[i]) and ep_length < args.max_steps
                ep_result = {
                    "episode_index": ep_idx,
                    "length": ep_length,
                    "success": is_success,
                    "terminated": bool(terminated[i]),
                    "truncated": bool(truncated[i]),
                    "max_steps_reached": ep_length >= args.max_steps,
                }
                episode_results.append(ep_result)

                status = "SUCCESS" if is_success else "FAIL"
                reason = "success" if is_success else ("timeout" if ep_length >= args.max_steps else "terminated")
                n_done = len(episode_results)
                n_success = sum(1 for r in episode_results if r["success"])
                sr = n_success / n_done * 100
                print(f"  Episode {ep_idx}: {ep_length} steps — {status} ({reason})  "
                      f"[{n_done}/{total_needed}]  success rate: {n_success}/{n_done} ({sr:.1f}%)",
                      flush=True)

                save_episode_data(output_dir, ep_idx, buf["timestamps"], buf["states"],
                                  buf["actions"], buf["frames"], buf["masks"],
                                  skip_first_n=args.skip_first_n,
                                  skip_last_n=args.skip_last_n,
                                  stages=buf.get("stages"))
                ep_length = ep_length - args.skip_first_n - args.skip_last_n
                if ep_length < 0:
                    ep_length = 0
                episode_lengths[-1] = ep_length
                ep_result["length"] = ep_length
                ep_result["max_steps_reached"] = ep_length >= (args.max_steps - args.skip_first_n - args.skip_last_n)

                # Save init state for this finished episode
                if env_init_states[i] is not None:
                    ep_result["init_state"] = env_init_states[i]
                    episode_results[-1]["init_state"] = env_init_states[i]

                save_episode_results(output_dir, episode_results)

                # Reset buffer and assign next episode
                buffers[i] = _new_buffer()
                cached_policy_actions = []
                if next_ep_idx < total_needed:
                    env_ep_idx[i] = next_ep_idx
                    next_ep_idx += 1
                    # Force a clean reset for this env so next episode starts from initial state
                    env_ids = torch.tensor([i], device=env.device)
                    obs, _ = env.reset(env_ids=env_ids)
                    next_light_info = None
                    if args.randomize_lighting:
                        light_seed = light_rng_base + env_ep_idx[i] * 7919
                        next_light_info = _randomize_surgical_lighting(
                            env,
                            np.random.RandomState(light_seed),
                            lighting_baseline,
                        )
                    fixed_initial_info = None
                    obs, fixed_initial_info = apply_fixed_initial_state(
                        env,
                        obs,
                        fixed_initial_raw_action,
                        fixed_initial_state_ref,
                        args.fixed_initial_state_steps,
                        args.fixed_initial_state_tolerance,
                        args.fixed_initial_state_mode,
                    )
                    # Flush camera: same method as RLinf _patched_reset
                    env.sim.set_setting("/app/player/playSimulations", False)
                    _app.update()
                    env.sim.set_setting("/app/player/playSimulations", True)
                    for sensor in env.scene.sensors.values():
                        sensor.update(dt=0.0, force_recompute=True)
                    obs = env.observation_manager.compute(update_history=True)
                    # Capture init state for the new episode
                    init_state = _capture_episode_init_state(env)
                    if next_light_info is not None:
                        init_state["lighting"] = next_light_info
                    if fixed_initial_info is not None:
                        init_state["fixed_initial_state"] = fixed_initial_info
                    init_state["seed"] = light_seed if args.randomize_lighting else None
                    env_init_states[i] = init_state
                else:
                    env_ep_idx[i] = -1
                    env_init_states[i] = None

    # Save episode results
    save_episode_results(output_dir, episode_results)

    # Save metadata
    save_metadata(output_dir, episode_lengths, joint_names)

    success_count = sum(1 for r in episode_results if r["success"])
    print(f"\n[INFO] Done! Recorded {len(episode_lengths)} episodes to {output_dir}", flush=True)
    print(f"  Total frames: {sum(episode_lengths)}", flush=True)
    print(f"  Success: {success_count}/{len(episode_results)}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
