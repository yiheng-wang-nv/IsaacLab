#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build ordered-stage progress features by incrementally filling missing boundary prompts.

This script reuses an existing cross-prompt feature directory and only extracts
the adjacent boundary samples that were skipped by the old neighbor-exclusion
rule. Labels are rewritten with ordered-stage semantics:

* source_task < prompt_task: 0.0  (the requested task has not started yet)
* source_task = prompt_task: episode progress in [0, 1]
* source_task > prompt_task: 1.0  (the requested task is already complete)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from tqdm import tqdm


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
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _build_obs(images: dict[str, np.ndarray], state: np.ndarray, task_description: str) -> dict:
    obs: dict = {"annotation.human.task_description": [task_description]}
    for s_key, (lo, hi) in zip(STATE_KEYS, STATE_SLICES):
        obs[s_key] = state[lo:hi][np.newaxis]
    for v_key in VIDEO_KEYS:
        obs[v_key] = images[v_key][np.newaxis]
    return obs


def _ordered_labels(labels: np.ndarray, prompt_task: np.ndarray, source_task: np.ndarray) -> np.ndarray:
    ordered = np.empty(len(labels), dtype=np.float32)
    ordered[source_task < prompt_task] = 0.0
    ordered[source_task > prompt_task] = 1.0
    same_task = source_task == prompt_task
    ordered[same_task] = labels[same_task].astype(np.float32)
    return ordered


def _copy_array(src: np.ndarray, dst: np.ndarray, chunk_size: int, desc: str) -> None:
    for start in tqdm(range(0, len(src), chunk_size), desc=desc):
        end = min(start + chunk_size, len(src))
        dst[start:end] = src[start:end]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing_dir", required=True, help="Existing cross-prompt feature directory.")
    parser.add_argument("--output_dir", required=True, help="Output directory for full ordered-stage features.")
    parser.add_argument("--checkpoint", required=True, help="Path to the Isaac-GR00T checkpoint directory.")
    parser.add_argument("--gr00t_root", required=True, help="Path to the Isaac-GR00T repository.")
    parser.add_argument("--dataset", required=True, help="LeRobot dataset directory.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--neighbor_exclusion_frac", type=float, default=None)
    parser.add_argument("--copy_chunk_size", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    existing_dir = Path(args.existing_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    dataset_path = Path(args.dataset).expanduser()
    gr00t_root = Path(args.gr00t_root).expanduser()

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(existing_dir / "meta.json") as f:
        existing_meta = json.load(f)
    exclusion_frac = (
        float(args.neighbor_exclusion_frac)
        if args.neighbor_exclusion_frac is not None
        else float(existing_meta.get("neighbor_exclusion_frac", 0.10))
    )

    features_in = np.load(existing_dir / "features.npy", mmap_mode="r")
    labels_in = np.load(existing_dir / "labels.npy", mmap_mode="r")
    task_in = np.load(existing_dir / "task_index.npy", mmap_mode="r")
    source_in = np.load(existing_dir / "source_task_index.npy", mmap_mode="r")
    episode_in = np.load(existing_dir / "episode_index.npy", mmap_mode="r")

    feat_dim = int(features_in.shape[1])
    existing_n = int(features_in.shape[0])

    parquet_files = sorted(dataset_path.glob("data/chunk-*/episode_*.parquet"))
    missing_by_episode: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for episode_idx, parquet_path in enumerate(tqdm(parquet_files, desc="Find missing boundary prompts")):
        df = pd.read_parquet(parquet_path, columns=["task_index"])
        n_frames = len(df)
        if n_frames == 0:
            continue
        source_task_idx = int(df["task_index"].iloc[0])
        if n_frames == 1:
            progress_labels = np.ones(1, dtype=np.float32)
        else:
            progress_labels = np.arange(n_frames, dtype=np.float32) / float(n_frames - 1)

        for frame_idx, progress in enumerate(progress_labels):
            progress = float(progress)
            if source_task_idx < len(TASK_DESCRIPTIONS) - 1 and progress > 1.0 - exclusion_frac:
                missing_by_episode[episode_idx].append((frame_idx, source_task_idx + 1, 0.0))
            if source_task_idx > 0 and progress < exclusion_frac:
                missing_by_episode[episode_idx].append((frame_idx, source_task_idx - 1, 1.0))

    missing_n = sum(len(items) for items in missing_by_episode.values())
    total_n = existing_n + missing_n
    print(f"[INFO] Existing samples: {existing_n}")
    print(f"[INFO] Missing boundary samples to extract: {missing_n}")
    print(f"[INFO] Output samples: {total_n}")

    features_out = open_memmap(output_dir / "features.npy", mode="w+", dtype=np.float32, shape=(total_n, feat_dim))
    labels_out = open_memmap(output_dir / "labels.npy", mode="w+", dtype=np.float32, shape=(total_n,))
    task_out = open_memmap(output_dir / "task_index.npy", mode="w+", dtype=np.int32, shape=(total_n,))
    source_out = open_memmap(output_dir / "source_task_index.npy", mode="w+", dtype=np.int32, shape=(total_n,))
    episode_out = open_memmap(output_dir / "episode_index.npy", mode="w+", dtype=np.int32, shape=(total_n,))

    _copy_array(features_in, features_out[:existing_n], args.copy_chunk_size, "Copy features.npy")
    task_out[:existing_n] = task_in[:]
    source_out[:existing_n] = source_in[:]
    episode_out[:existing_n] = episode_in[:]
    labels_out[:existing_n] = _ordered_labels(labels_in[:], task_in[:], source_in[:])

    # Load GR00T only after metadata copying, so a dry failure happens before GPU work.
    sys.path.insert(0, str(gr00t_root))
    spec = importlib.util.spec_from_file_location("ordered_progress_gr00t_config", gr00t_root / "gr00t_config.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load gr00t_config.py from {gr00t_root}")
    gr00t_cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gr00t_cfg_module)

    from gr00t.model.policy import Gr00tPolicy

    data_config = gr00t_cfg_module.UnitreeG1SimTaskCompleteInferDataConfig()
    policy = Gr00tPolicy(
        model_path=args.checkpoint,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=1,
        device=args.device,
    )
    policy.model.eval()

    backbone_out: dict[str, torch.Tensor] = {}

    def _backbone_hook(module, input, output):
        backbone_out["features"] = output["backbone_features"].detach().float()

    policy.model.backbone.register_forward_hook(_backbone_hook)

    tasks_jsonl = dataset_path / "meta" / "tasks.jsonl"
    task_idx_to_desc = {i: desc for i, desc in enumerate(TASK_DESCRIPTIONS)}
    if tasks_jsonl.exists():
        with open(tasks_jsonl) as f:
            for line in f:
                entry = json.loads(line)
                task_idx_to_desc[int(entry["task_index"])] = entry["task"]

    write_idx = existing_n
    for episode_idx, missing_items in tqdm(missing_by_episode.items(), desc="Extract missing features"):
        parquet_path = parquet_files[episode_idx]
        df = pd.read_parquet(parquet_path)
        source_task_idx = int(df["task_index"].iloc[0])
        states = np.array(df["observation.state"].tolist(), dtype=np.float32)
        n_frames = len(df)

        video_frames = {}
        for v_key in VIDEO_KEYS:
            chunk_name = parquet_path.parent.name
            ep_name = parquet_path.stem + ".mp4"
            vid_path = dataset_path / "videos" / chunk_name / VIDEO_DIR_KEYS[v_key] / ep_name
            frames = _load_video_frames(vid_path)
            if len(frames) != n_frames:
                raise RuntimeError(f"Frame mismatch for {vid_path}: {len(frames)} vs {n_frames}")
            video_frames[v_key] = frames

        for frame_idx, prompt_task_idx, label in missing_items:
            images = {v_key: video_frames[v_key][frame_idx] for v_key in VIDEO_KEYS}
            prompt_desc = task_idx_to_desc.get(prompt_task_idx, TASK_DESCRIPTIONS[prompt_task_idx])
            obs = _build_obs(images, states[frame_idx], prompt_desc)
            with torch.no_grad():
                policy.get_action(obs)

            feat = backbone_out["features"].mean(dim=1).cpu().numpy().astype(np.float32)
            features_out[write_idx] = feat[0]
            labels_out[write_idx] = float(label)
            task_out[write_idx] = int(prompt_task_idx)
            source_out[write_idx] = int(source_task_idx)
            episode_out[write_idx] = int(episode_idx)
            write_idx += 1

    if write_idx != total_n:
        raise RuntimeError(f"Expected to write {total_n} samples, wrote {write_idx}.")

    # Flush memmaps before computing final stats from disk.
    del features_out, labels_out, task_out, source_out, episode_out

    labels_np = np.load(output_dir / "labels.npy", mmap_mode="r")
    task_np = np.load(output_dir / "task_index.npy", mmap_mode="r")
    source_np = np.load(output_dir / "source_task_index.npy", mmap_mode="r")
    episode_np = np.load(output_dir / "episode_index.npy", mmap_mode="r")
    source_prompt_mask = task_np == source_np
    cross_prompt_mask = task_np != source_np

    meta = {
        "n_samples": int(total_n),
        "n_existing_samples": int(existing_n),
        "n_augmented_boundary_samples": int(missing_n),
        "n_episodes": int(len(np.unique(episode_np))),
        "feature_dim": int(feat_dim),
        "label_type": "ordered_stage_progress",
        "ordered_stage_rule": "source<prompt -> 0, source==prompt -> episode progress, source>prompt -> 1",
        "label_min": float(labels_np.min()),
        "label_max": float(labels_np.max()),
        "label_mean": float(labels_np.mean()),
        "existing_dir": str(existing_dir),
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "per_task": {
            str(i): {
                "n": int((task_np == i).sum()),
                "n_episodes": int(len(np.unique(episode_np[task_np == i]))),
                "n_source_prompt": int(((task_np == i) & source_prompt_mask).sum()),
                "n_cross_prompt": int(((task_np == i) & cross_prompt_mask).sum()),
                "label_mean": float(labels_np[task_np == i].mean()) if (task_np == i).any() else 0.0,
            }
            for i in range(len(TASK_DESCRIPTIONS))
        },
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE] Saved ordered-stage features to {output_dir}")
    print(f"  Samples: {total_n} = {existing_n} existing + {missing_n} augmented")
    print(f"  Label: min={labels_np.min():.4f} mean={labels_np.mean():.4f} max={labels_np.max():.4f}")
    for i, desc in enumerate(TASK_DESCRIPTIONS):
        mask = task_np == i
        mean_label = float(labels_np[mask].mean()) if mask.any() else 0.0
        print(
            f"  Task {i + 1} ({desc}): n={int(mask.sum())}, "
            f"mean_label={mean_label:.4f}"
        )


if __name__ == "__main__":
    main()
