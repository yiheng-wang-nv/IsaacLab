# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay recorded trocar episodes for verification.

Reads recorded actions + init state, reproduces the initial scene
(tray pose, trocar positions, lighting), then steps the env with
the recorded action sequence. Saves replayed RGB videos for visual
comparison against the originals.

Usage:
    ./isaaclab.sh -p scripts/tools/replay_trocar_episodes.py \
        --dataset_dir /localhome/local-vennw/data/trocar_parallel/merged \
        --episode_idx 0 \
        --output_dir /localhome/local-vennw/data/trocar_replay
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay recorded trocar episodes.")
parser.add_argument("--dataset_dir", type=str, required=True,
                    help="Directory containing the recorded data (with data/, videos/, episode_results.json)")
parser.add_argument("--episode_idx", type=int, required=True, help="Episode index to replay.")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for replayed videos.")
parser.add_argument("--randomize_lighting", action="store_true", default=False,
                    help="Apply recorded lighting params during replay.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app = AppLauncher(args).app

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

import isaaclab_tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# ---------------------------------------------------------------------------
# Reuse constants and helpers from recording script
# ---------------------------------------------------------------------------
TASK_ID = "Isaac-Assemble-Trocar-G129-Dex3-RLinf-v0"
CAMERA_KEYS = ["front_camera", "left_wrist_camera", "right_wrist_camera"]
CAMERA_LEROBOT_NAMES = {
    "front_camera": "observation.images.cam_room",
    "left_wrist_camera": "observation.images.cam_left_wrist",
    "right_wrist_camera": "observation.images.cam_right_wrist",
}
ACTION_PREFIX_PAD = 15
FPS = 30.0

# Inverse permutation: ref order → our internal order (for action input to env)
# Forward perm: PERM_TO_REF[ref_idx] = our_idx   (used during recording)
# Inverse:      INV_PERM[our_idx] = ref_idx      (recover index in REF storage)
PERM_TO_REF = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    16, 19, 20, 15, 18, 14, 17,
    23, 26, 27, 21, 24, 22, 25,
]
INV_PERM = np.argsort(PERM_TO_REF)


def apply_init_state(env, init_state: dict):
    """Restore the saved init pose for tray + trocars + lighting."""
    if "tray_pos" in init_state:
        for name, p_key, q_key in [
            ("tray", "tray_pos", "tray_quat"),
            ("trocar_1", "trocar_1_pos", "trocar_1_quat"),
            ("trocar_2", "trocar_2_pos", "trocar_2_quat"),
        ]:
            if p_key not in init_state or name not in env.scene.keys():
                continue
            asset = env.scene[name]
            pose = torch.zeros(1, 7, device=env.device)
            pose[0, 0:3] = torch.tensor(init_state[p_key], device=env.device)
            pose[0, 3:7] = torch.tensor(init_state[q_key], device=env.device)
            asset.write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device=env.device))
            zero_vel = torch.zeros(1, 6, device=env.device)
            asset.write_root_velocity_to_sim(zero_vel, env_ids=torch.tensor([0], device=env.device))

    # Apply lighting if recorded
    if args.randomize_lighting and "lighting" in init_state:
        from pxr import Gf
        light = init_state["lighting"]
        intensity_scale = light["intensity_scale"]
        color = tuple(light["color"])
        stage = env.sim.stage
        for prim in stage.Traverse():
            if "Light" not in str(prim.GetTypeName()):
                continue
            for attr_name in ["inputs:intensity", "intensity"]:
                attr = prim.GetAttribute(attr_name)
                if attr.IsValid() and attr.Get() is not None:
                    base = float(attr.Get())
                    attr.Set(base * intensity_scale)
                    break
            for attr_name in ["inputs:color", "color"]:
                attr = prim.GetAttribute(attr_name)
                if attr.IsValid() and attr.Get() is not None:
                    attr.Set(Gf.Vec3f(*color))
                    break


def get_camera_rgb(env, cam_name: str) -> np.ndarray:
    sensor = env.scene.sensors[cam_name]
    img = sensor.data.output["rgb"][0]
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img.astype(np.uint8)


def main():
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ep_idx = args.episode_idx

    # Load episode data
    parquet_path = dataset_dir / f"data/chunk-000/episode_{ep_idx:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode {ep_idx} not found at {parquet_path}")
    df = pd.read_parquet(parquet_path)
    actions_ref = np.array([np.asarray(a) for a in df["action"]], dtype=np.float32)  # (T, 28) in REF order
    print(f"[INFO] Loaded {len(actions_ref)} actions from episode {ep_idx}")

    # Load episode init state
    results_path = dataset_dir / "episode_results.json"
    init_state = None
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        for ep in results["episodes"]:
            if ep["episode_index"] == ep_idx:
                init_state = ep.get("init_state")
                break
    if init_state:
        print(f"[INFO] Found init state: tray_yaw={init_state.get('tray_yaw_deg', 'N/A'):.2f}°")

    # Convert REF action order → internal env order using inverse permutation
    actions_internal = actions_ref[:, INV_PERM]  # (T, 28)
    # Pad to 43-dim env action (15 zero prefix)
    actions_43 = np.concatenate([
        np.zeros((len(actions_internal), ACTION_PREFIX_PAD), dtype=np.float32),
        actions_internal,
    ], axis=1)

    # Create env
    env_cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=1)
    for cam_name in CAMERA_KEYS:
        cam_cfg = getattr(env_cfg.scene, cam_name)
        cam_cfg.data_types = ["rgb"]
    env_cfg.recorders = {}
    # Don't randomize tray on reset — we'll set it manually
    env = gym.make(TASK_ID, cfg=env_cfg).unwrapped

    # Reset and apply saved init state
    seed = init_state.get("seed") if init_state else None
    obs, _ = env.reset(seed=seed)
    if init_state is not None:
        apply_init_state(env, init_state)

    # Flush
    import omni.kit.app
    _app = omni.kit.app.get_app()
    env.sim.set_setting("/app/player/playSimulations", False)
    _app.update()
    env.sim.set_setting("/app/player/playSimulations", True)
    for sensor in env.scene.sensors.values():
        sensor.update(dt=0.0, force_recompute=True)
    obs = env.observation_manager.compute(update_history=True)

    # No warmup — matches record runs with skip_first_n=0 for exact round-trip.

    # Replay loop
    print(f"[INFO] Replaying {len(actions_43)} steps...")
    frames = {cam: [] for cam in CAMERA_KEYS}
    states_replay = []

    with torch.inference_mode():
        for t, action in enumerate(actions_43):
            # Step first, then capture (matches recording flow — skip initial frame)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=env.device).unsqueeze(0)
            obs, _, terminated, truncated, _ = env.step(action_tensor)
            for cam_name in CAMERA_KEYS:
                frames[cam_name].append(get_camera_rgb(env, cam_name))
            if terminated.any() or truncated.any():
                print(f"  Episode terminated at step {t}")
                break

    # Save videos
    for cam_name, cam_frames in frames.items():
        lerobot_name = CAMERA_LEROBOT_NAMES[cam_name]
        vid_dir = output_dir / "videos" / lerobot_name
        vid_dir.mkdir(parents=True, exist_ok=True)
        vid_path = vid_dir / f"episode_{ep_idx:06d}_replay.mp4"
        h, w = cam_frames[0].shape[:2]
        writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        for f in cam_frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  Saved: {vid_path}")

    # Side-by-side comparison video (front camera)
    orig_path = dataset_dir / "videos/chunk-000/observation.images.cam_room" / f"episode_{ep_idx:06d}.mp4"
    if orig_path.exists():
        cap = cv2.VideoCapture(str(orig_path))
        orig_frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            orig_frames.append(f)
        cap.release()

        T = min(len(orig_frames), len(frames["front_camera"]))
        H, W = orig_frames[0].shape[:2]
        compare_path = output_dir / f"compare_episode_{ep_idx:06d}.mp4"
        writer = cv2.VideoWriter(str(compare_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W * 2, H))
        for t in range(T):
            canvas = np.zeros((H, W * 2, 3), dtype=np.uint8)
            canvas[:, 0:W] = orig_frames[t]
            replay_bgr = cv2.cvtColor(frames["front_camera"][t], cv2.COLOR_RGB2BGR)
            canvas[:, W:2*W] = replay_bgr
            cv2.putText(canvas, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(canvas, "REPLAY", (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            writer.write(canvas)
        writer.release()
        print(f"  Side-by-side: {compare_path}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
