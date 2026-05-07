#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extract GR00T backbone features for task-complete classifier training.

Usage:
    python extract_task_complete_features.py \
        --checkpoint /path/to/checkpoint-100000 \
        --dataset /path/to/trocar_success_lt_7s_split_by_stage_task_complete \
        --output_dir /path/to/task_complete_features \
        --device cuda:0

Output:
    {output_dir}/features.npy    -- (N, hidden_dim) float32
    {output_dir}/labels.npy      -- (N,) int32  0=incomplete 1=complete
    {output_dir}/task_index.npy  -- (N,) int32  0-4
    {output_dir}/meta.json       -- dataset stats
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    default="/localhome/local-vennw/code/cosmos_gr00t/Isaac-GR00T"
            "/sft_4gpu_256bs_100ksteps_split_stage_task_complete_soft8_no_bgaug"
            "/checkpoint-100000",
)
parser.add_argument(
    "--dataset",
    default="/localhome/local-vennw/code/trocar_success_lt_7s_split_by_stage_task_complete",
)
parser.add_argument("--output_dir", default="/localhome/local-vennw/code/task_complete_features")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument(
    "--label_threshold",
    type=float,
    default=0.5,
    help="Frames with soft task_complete > this value are labelled as positive (1).",
)
parser.add_argument("--max_episodes", type=int, default=None, help="Cap episodes for quick testing.")
args = parser.parse_args()

GROOT_ROOT = Path(args.checkpoint).parents[1]
sys.path.insert(0, str(GROOT_ROOT))

# ---------------------------------------------------------------------------
# Load policy / backbone
# ---------------------------------------------------------------------------
print(f"[INFO] Loading GR00T from {args.checkpoint}")
spec = importlib.util.spec_from_file_location("gr00t_config", GROOT_ROOT / "gr00t_config.py")
gr00t_cfg_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gr00t_cfg_module)

from gr00t.model.policy import Gr00tPolicy  # noqa: E402

data_config = gr00t_cfg_module.UnitreeG1SimTaskCompleteInferDataConfig()

policy = Gr00tPolicy(
    model_path=args.checkpoint,
    modality_config=data_config.modality_config(),
    modality_transform=data_config.transform(),
    embodiment_tag="new_embodiment",
    denoising_steps=1,  # minimal steps — we only need backbone features
    device=args.device,
)
policy.model.eval()

# ---------------------------------------------------------------------------
# Hook: capture backbone_features after backbone forward
# ---------------------------------------------------------------------------
_backbone_out: dict = {}


def _backbone_hook(module, input, output):
    # output is a BatchFeature; backbone_features shape: (B, n_tokens, hidden)
    _backbone_out["features"] = output["backbone_features"].detach().float()


policy.model.backbone.register_forward_hook(_backbone_hook)

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
TASK_DESCRIPTIONS = [
    "left hand pick up trocar",
    "right hand pick up trocar",
    "align trocars",
    "install trocar",
    "place trocar",
]

VIDEO_KEYS = ["video.left_wrist_view", "video.right_wrist_view", "video.room_view"]
VIDEO_DIR_KEYS = {
    "video.left_wrist_view": "observation.images.cam_left_wrist",
    "video.right_wrist_view": "observation.images.cam_right_wrist",
    "video.room_view": "observation.images.cam_room",
}
STATE_KEYS = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"]
STATE_SLICES = [(0, 7), (7, 14), (14, 21), (21, 28)]


def _load_video_frames(video_path: Path) -> list[np.ndarray]:
    """Return list of RGB uint8 frames (H, W, 3)."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _build_obs(
    images: dict[str, np.ndarray],  # video_key → (H, W, 3) uint8
    state: np.ndarray,              # (28,) float32, internal order
    task_description: str,
) -> dict:
    obs: dict = {"annotation.human.task_description": [task_description]}
    for s_key, (lo, hi) in zip(STATE_KEYS, STATE_SLICES):
        obs[s_key] = state[lo:hi][np.newaxis]   # (1, 7)
    for v_key in VIDEO_KEYS:
        obs[v_key] = images[v_key][np.newaxis]  # (1, H, W, 3)
    return obs


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------
dataset_path = Path(args.dataset)

# Read tasks.jsonl for task index → description mapping
tasks_jsonl = dataset_path / "meta" / "tasks.jsonl"
task_idx_to_desc: dict[int, str] = {}
with open(tasks_jsonl) as f:
    for line in f:
        entry = json.loads(line)
        task_idx_to_desc[entry["task_index"]] = entry["task"]

# Find all episode parquets
import pandas as pd  # noqa: E402

parquet_files = sorted(dataset_path.glob("data/chunk-*/episode_*.parquet"))
if args.max_episodes:
    parquet_files = parquet_files[: args.max_episodes]

all_features: list[np.ndarray] = []
all_labels: list[int] = []
all_task_indices: list[int] = []

for parquet_path in tqdm(parquet_files, desc="Episodes"):
    df = pd.read_parquet(parquet_path)
    n_frames = len(df)
    if n_frames == 0:
        continue

    # Task index (constant per episode)
    task_idx = int(df["task_index"].iloc[0])
    task_desc = task_idx_to_desc.get(task_idx, TASK_DESCRIPTIONS[task_idx])

    # Soft task_complete label (last column of action)
    actions = np.array(df["action"].tolist(), dtype=np.float32)   # (T, 29)
    soft_tc = actions[:, -1]                                        # (T,)
    hard_labels = (soft_tc > args.label_threshold).astype(np.int32)

    # States — (T, 28) float32
    states = np.array(df["observation.state"].tolist(), dtype=np.float32)  # (T, 28)

    # Load video frames per camera
    video_frames: dict[str, list[np.ndarray]] = {}
    ok = True
    for v_key in VIDEO_KEYS:
        chunk_name = parquet_path.parent.name           # e.g. chunk-000
        ep_name = parquet_path.stem + ".mp4"            # e.g. episode_000000.mp4
        vid_path = dataset_path / "videos" / chunk_name / VIDEO_DIR_KEYS[v_key] / ep_name
        if not vid_path.exists():
            print(f"[WARN] Missing video {vid_path}, skipping episode.")
            ok = False
            break
        frames = _load_video_frames(vid_path)
        if len(frames) != n_frames:
            print(f"[WARN] Frame count mismatch ({len(frames)} vs {n_frames}) in {vid_path}, skipping.")
            ok = False
            break
        video_frames[v_key] = frames
    if not ok:
        continue

    # Process frame by frame — GR00T transforms are single-sample only.
    # The hook on policy.model.backbone captures backbone_features after each call.
    for i in range(n_frames):
        images = {v_key: video_frames[v_key][i] for v_key in VIDEO_KEYS}
        obs = _build_obs(images, states[i], task_desc)
        with torch.no_grad():
            policy.get_action(obs)

        feat = _backbone_out["features"]               # (1, n_tokens, hidden)
        feat_pooled = feat.mean(dim=1).cpu().numpy()   # (1, hidden)
        all_features.append(feat_pooled)
        all_labels.append(int(hard_labels[i]))
        all_task_indices.append(task_idx)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs(args.output_dir, exist_ok=True)

features_np = np.concatenate(all_features, axis=0).astype(np.float32)
labels_np = np.array(all_labels, dtype=np.int32)
task_np = np.array(all_task_indices, dtype=np.int32)

np.save(os.path.join(args.output_dir, "features.npy"), features_np)
np.save(os.path.join(args.output_dir, "labels.npy"), labels_np)
np.save(os.path.join(args.output_dir, "task_index.npy"), task_np)

meta = {
    "n_samples": int(len(labels_np)),
    "n_positive": int(labels_np.sum()),
    "n_negative": int((labels_np == 0).sum()),
    "feature_dim": int(features_np.shape[1]),
    "label_threshold": args.label_threshold,
    "checkpoint": args.checkpoint,
    "dataset": args.dataset,
    "per_task": {
        str(i): {
            "n": int((task_np == i).sum()),
            "n_pos": int(((task_np == i) & (labels_np == 1)).sum()),
        }
        for i in range(5)
    },
}
with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n[DONE] Saved {len(labels_np)} samples to {args.output_dir}")
print(f"  Positive: {labels_np.sum()}  Negative: {(labels_np == 0).sum()}")
print(f"  Feature dim: {features_np.shape[1]}")
for i, desc in enumerate(TASK_DESCRIPTIONS):
    n = (task_np == i).sum()
    p = ((task_np == i) & (labels_np == 1)).sum()
    print(f"  Task {i+1} ({desc}): {n} samples, {p} positive")
