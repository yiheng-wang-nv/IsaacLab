# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Filter a recorded dataset to keep only successful episodes.

Renumbers episodes contiguously (0, 1, 2, ...) and regenerates all metadata.

Usage:
    python scripts/tools/filter_success_episodes.py \
        --input_dir /path/to/dataset \
        --output_dir /path/to/success_only_dataset
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _field_stats(arr: np.ndarray) -> dict:
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


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

    # Load episode_results.json
    results_path = inp / "episode_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} not found")
    with open(results_path) as f:
        results = json.load(f)

    success_episodes = [e for e in results["episodes"] if e.get("success")]
    if not success_episodes:
        print("No successful episodes found.")
        return
    old_indices = [e["episode_index"] for e in success_episodes]
    print(f"Filtering {len(success_episodes)} success episodes from {results['total_episodes']} total")

    # Map: old_idx -> new_idx (0, 1, 2, ...)
    idx_map = {old: new for new, old in enumerate(old_indices)}

    chunk_dir = "chunk-000"

    # Copy parquet files with renaming
    (out / "data" / chunk_dir).mkdir(parents=True, exist_ok=True)
    for old, new in idx_map.items():
        src = inp / "data" / chunk_dir / f"episode_{old:06d}.parquet"
        dst = out / "data" / chunk_dir / f"episode_{new:06d}.parquet"
        df = pd.read_parquet(src)
        df["episode_index"] = np.full(len(df), new, dtype=np.int64)
        df.to_parquet(dst, index=False)

    # Copy videos
    videos_src = inp / "videos" / chunk_dir
    if videos_src.exists():
        for cam_dir in videos_src.glob("*"):
            out_cam = out / "videos" / chunk_dir / cam_dir.name
            out_cam.mkdir(parents=True, exist_ok=True)
            for old, new in idx_map.items():
                src = cam_dir / f"episode_{old:06d}.mp4"
                if src.exists():
                    shutil.copy2(src, out_cam / f"episode_{new:06d}.mp4")

    # Copy masks
    masks_src = inp / "masks" / chunk_dir
    if masks_src.exists():
        for cam_dir in masks_src.glob("*"):
            out_cam = out / "masks" / chunk_dir / cam_dir.name
            out_cam.mkdir(parents=True, exist_ok=True)
            for old, new in idx_map.items():
                src = cam_dir / f"episode_{old:06d}_masks.npz"
                if src.exists():
                    shutil.copy2(src, out_cam / f"episode_{new:06d}_masks.npz")

    # Save filtered episode_results.json
    new_results = []
    for new, old in enumerate(old_indices):
        for e in success_episodes:
            if e["episode_index"] == old:
                e_copy = e.copy()
                e_copy["episode_index"] = new
                new_results.append(e_copy)
                break
    with open(out / "episode_results.json", "w") as f:
        json.dump({
            "total_episodes": len(new_results),
            "success_count": len(new_results),
            "fail_count": 0,
            "episodes": new_results,
        }, f, indent=2)

    # Copy category_mapping.json if present
    cat_src = inp / "category_mapping.json"
    if cat_src.exists():
        shutil.copy2(cat_src, out / "category_mapping.json")

    # Regenerate meta/
    meta_dir = out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    inp_meta = inp / "meta"

    # Copy static meta files
    for fname in ["info.json", "modality.json", "tasks.jsonl"]:
        src = inp_meta / fname
        if src.exists():
            shutil.copy2(src, meta_dir / fname)

    # Regenerate episodes.jsonl + stats
    data_dir = out / "data" / chunk_dir
    total_frames = 0
    all_states, all_actions, per_ep_stats = [], [], []
    task_desc = "install trocar from box"
    # Try to get task from tasks.jsonl
    tasks_path = meta_dir / "tasks.jsonl"
    if tasks_path.exists():
        with open(tasks_path) as f:
            task_desc = json.loads(f.readline()).get("task", task_desc)

    with open(meta_dir / "episodes.jsonl", "w") as f_ep:
        for ep_file in sorted(data_dir.glob("episode_*.parquet")):
            ep_i = int(ep_file.stem.split("_")[1])
            df = pd.read_parquet(ep_file)
            length = len(df)
            total_frames += length
            f_ep.write(json.dumps({"episode_index": ep_i, "tasks": [task_desc], "length": length}) + "\n")

            s = np.array([np.asarray(x) for x in df["observation.state"]], dtype=np.float32)
            a = np.array([np.asarray(x) for x in df["action"]], dtype=np.float32)
            all_states.append(s)
            all_actions.append(a)
            per_ep_stats.append({
                "episode_index": ep_i,
                "stats": {"observation.state": _field_stats(s), "action": _field_stats(a)},
            })

    # Update info.json
    info_path = meta_dir / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        info["total_episodes"] = len(new_results)
        info["total_frames"] = total_frames
        info["total_videos"] = len(new_results) * 3
        info["splits"] = {"train": f"0:{len(new_results)}"}
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for s in per_ep_stats:
            f.write(json.dumps(s) + "\n")

    if all_states:
        with open(meta_dir / "stats.json", "w") as f:
            json.dump({
                "observation.state": _field_stats(np.concatenate(all_states, axis=0)),
                "action": _field_stats(np.concatenate(all_actions, axis=0)),
            }, f, indent=2)

    print(f"Saved {len(new_results)} success episodes ({total_frames} frames) to {out}")


if __name__ == "__main__":
    main()
