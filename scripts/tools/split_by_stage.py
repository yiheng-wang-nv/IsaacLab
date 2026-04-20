# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Split each episode into sub-episodes by task stage.

Each source episode becomes up to 5 sub-episodes based on the `stage` column:
  stage 0        → left_hand_pick_trocar
  stage 1        → right_hand_pick_trocar
  stage 2        → align_trocars
  stage 3        → install_trocar
  stage 4, 5     → place_trocar

Regenerates all meta files (info.json, episodes.jsonl, tasks.jsonl,
stats.json, episodes_stats.jsonl) for the new sub-episode layout.

Usage:
    python scripts/tools/split_by_stage.py \
        --input_dir /path/to/dataset \
        --output_dir /path/to/split_dataset
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


TASK_GROUPS = [
    ({0}, "left hand pick up trocar"),
    ({1}, "right hand pick up trocar"),
    ({2}, "align trocars"),
    ({3}, "install trocar"),
    ({4, 5}, "place trocar"),
]
TASKS = [name for _, name in TASK_GROUPS]


def _field_stats(arr: np.ndarray) -> dict:
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def _find_stage_ranges(stages: np.ndarray):
    """Return list of (task_idx, start, end_exclusive) for each task group with frames present."""
    out = []
    for task_idx, (stage_set, _) in enumerate(TASK_GROUPS):
        mask = np.isin(stages, list(stage_set))
        if not mask.any():
            continue
        idxs = np.where(mask)[0]
        # Since stages are monotonically increasing, indices should be contiguous.
        start, end = int(idxs.min()), int(idxs.max()) + 1
        out.append((task_idx, start, end))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    chunk = "chunk-000"
    data_src = inp / "data" / chunk
    videos_src = inp / "videos" / chunk
    masks_src = inp / "masks" / chunk

    data_dst = out / "data" / chunk
    videos_dst = out / "videos" / chunk
    masks_dst = out / "masks" / chunk
    data_dst.mkdir(parents=True, exist_ok=True)

    # Camera dirs
    camera_names = []
    if videos_src.exists():
        camera_names = [d.name for d in sorted(videos_src.glob("*")) if d.is_dir()]
        for cam in camera_names:
            (videos_dst / cam).mkdir(parents=True, exist_ok=True)
            if masks_src.exists():
                (masks_dst / cam).mkdir(parents=True, exist_ok=True)

    # Iterate source episodes
    sub_idx = 0
    total_frames = 0
    ep_lengths = []
    per_ep_stats = []
    task_indices_per_ep = []  # for episodes.jsonl
    all_states, all_actions = [], []

    for src_parquet in sorted(data_src.glob("episode_*.parquet")):
        old_idx = int(src_parquet.stem.split("_")[1])
        df = pd.read_parquet(src_parquet)
        if "stage" not in df.columns:
            raise RuntimeError(f"{src_parquet} missing 'stage' column")
        stages = df["stage"].to_numpy()
        ranges = _find_stage_ranges(stages)

        # Preload videos and masks for this episode
        video_frames = {}
        mask_arrays = {}
        for cam in camera_names:
            vp = videos_src / cam / f"episode_{old_idx:06d}.mp4"
            mp_ = masks_src / cam / f"episode_{old_idx:06d}_masks.npz"
            if vp.exists():
                cap = cv2.VideoCapture(str(vp))
                frames = []
                while True:
                    ret, f = cap.read()
                    if not ret:
                        break
                    frames.append(f)
                cap.release()
                video_frames[cam] = frames
            if mp_.exists():
                mask_arrays[cam] = np.load(mp_)["arr_0"]

        for task_idx, start, end in ranges:
            sub_df = df.iloc[start:end].copy().reset_index(drop=True)
            T = len(sub_df)
            if T == 0:
                continue
            # Renumber frame_index / index within sub-episode
            sub_df["frame_index"] = np.arange(T, dtype=np.int64)
            sub_df["index"] = np.arange(T, dtype=np.int64)
            sub_df["episode_index"] = np.full(T, sub_idx, dtype=np.int64)
            sub_df["task_index"] = np.full(T, task_idx, dtype=np.int64)
            # Re-base timestamp to start at 0
            t0 = float(sub_df["timestamp"].iloc[0])
            sub_df["timestamp"] = (sub_df["timestamp"].to_numpy(np.float32) - t0).astype(np.float32)

            sub_df.to_parquet(data_dst / f"episode_{sub_idx:06d}.parquet", index=False)

            # Slice videos
            fps = 30.0
            for cam in camera_names:
                if cam in video_frames and len(video_frames[cam]) >= end:
                    frames_slice = video_frames[cam][start:end]
                    if frames_slice:
                        h, w = frames_slice[0].shape[:2]
                        vp = videos_dst / cam / f"episode_{sub_idx:06d}.mp4"
                        writer = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        for frame in frames_slice:
                            writer.write(frame)
                        writer.release()
                if cam in mask_arrays and len(mask_arrays[cam]) >= end:
                    mask_slice = mask_arrays[cam][start:end]
                    mp_out = masks_dst / cam / f"episode_{sub_idx:06d}_masks.npz"
                    np.savez_compressed(str(mp_out), mask_slice)

            # Stats
            s = np.array([np.asarray(x) for x in sub_df["observation.state"]], dtype=np.float32)
            a = np.array([np.asarray(x) for x in sub_df["action"]], dtype=np.float32)
            per_ep_stats.append({
                "episode_index": sub_idx,
                "stats": {"observation.state": _field_stats(s), "action": _field_stats(a)},
            })
            all_states.append(s)
            all_actions.append(a)
            ep_lengths.append(T)
            task_indices_per_ep.append(task_idx)
            total_frames += T
            sub_idx += 1

    # Meta
    meta_dir = out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Copy info.json/modality.json from source, then patch totals
    for fname in ["info.json", "modality.json"]:
        src = inp / "meta" / fname
        if src.exists():
            shutil.copy2(src, meta_dir / fname)

    info_path = meta_dir / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        info["total_episodes"] = sub_idx
        info["total_frames"] = total_frames
        info["total_videos"] = sub_idx * len(camera_names)
        info["total_tasks"] = len(TASKS)
        info["splits"] = {"train": f"0:{sub_idx}"}
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for i, name in enumerate(TASKS):
            f.write(json.dumps({"task_index": i, "task": name}) + "\n")

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for i, (length, task_idx) in enumerate(zip(ep_lengths, task_indices_per_ep)):
            f.write(json.dumps({
                "episode_index": i,
                "tasks": [TASKS[task_idx]],
                "length": length,
            }) + "\n")

    # episodes_stats.jsonl
    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for s in per_ep_stats:
            f.write(json.dumps(s) + "\n")

    # stats.json (global)
    if all_states:
        with open(meta_dir / "stats.json", "w") as f:
            json.dump({
                "observation.state": _field_stats(np.concatenate(all_states, axis=0)),
                "action": _field_stats(np.concatenate(all_actions, axis=0)),
            }, f, indent=2)

    # Copy category_mapping.json
    cat_src = inp / "category_mapping.json"
    if cat_src.exists():
        shutil.copy2(cat_src, out / "category_mapping.json")

    # Summary
    task_counts = [0] * len(TASKS)
    for ti in task_indices_per_ep:
        task_counts[ti] += 1
    print(f"Split into {sub_idx} sub-episodes ({total_frames} frames) from {len(list(data_src.glob('*.parquet')))} source episodes.")
    for i, name in enumerate(TASKS):
        print(f"  task {i} ({name}): {task_counts[i]} sub-episodes")


if __name__ == "__main__":
    main()
