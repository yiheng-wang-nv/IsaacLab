# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Append a task-complete action dimension to a split Trocar dataset.

The input dataset is expected to contain one sub-task per episode, such as the
output of ``split_by_stage.py``. The script appends a scalar
``action.task_complete`` to the end of the existing ``action`` vector. By
default the final eight frames of each sub-episode are labeled with a soft ramp
ending at 1.0. GR00T's future action horizon will then observe a gradual
completion signal when a predicted chunk approaches the terminal frame.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


TASK_COMPLETE_KEY = "task_complete"


def _copy_or_link(src: str, dst: str) -> None:
    """Hard-link files when possible, falling back to a real copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _field_stats(arr: np.ndarray) -> dict:
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _dump_json(path: Path, data: dict) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def _episode_index_from_path(path: Path) -> int:
    """Return the episode index encoded in ``episode_XXXXXX.parquet``."""
    return int(path.stem.split("_")[1])


def _episode_parquet_paths(dataset_dir: Path) -> list[Path]:
    """Return one parquet path per episode, following the dataset chunk metadata."""
    info = _load_json(dataset_dir / "meta" / "info.json")
    chunk_size = int(info.get("chunks_size", 1000))
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )

    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        episode_indices = []
        with episodes_path.open() as f:
            for line in f:
                if line.strip():
                    episode_indices.append(json.loads(line)["episode_index"])
    else:
        episode_indices = sorted(
            {
                _episode_index_from_path(path)
                for path in (dataset_dir / "data").glob("chunk-*/episode_*.parquet")
            }
        )

    parquet_paths = []
    for episode_index in sorted(episode_indices):
        episode_chunk = episode_index // chunk_size
        path = dataset_dir / data_path.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        if not path.exists():
            matches = sorted(
                (dataset_dir / "data").glob(f"chunk-*/episode_{episode_index:06d}.parquet")
            )
            path = matches[0] if matches else path
        parquet_paths.append(path)
    return parquet_paths


def _append_done_to_action(
    action_values: pd.Series,
    done_last_n: int,
    soft: bool,
) -> tuple[list[np.ndarray], np.ndarray]:
    actions = np.stack(action_values.to_numpy()).astype(np.float32, copy=False)
    if actions.ndim != 2:
        raise ValueError(f"Expected action array with shape [T, D], got {actions.shape}.")
    done = np.zeros((actions.shape[0], 1), dtype=np.float32)
    n = min(done_last_n, actions.shape[0])
    done[-n:, 0] = np.linspace(1.0 / n, 1.0, n, dtype=np.float32) if soft else 1.0
    return [row for row in np.concatenate([actions, done], axis=1)], done


def _patch_modality(meta_dir: Path, action_dim: int) -> None:
    path = meta_dir / "modality.json"
    modality = _load_json(path)
    action_meta = modality.setdefault("action", {})
    if TASK_COMPLETE_KEY in action_meta:
        raise ValueError(f"{path} already contains action.{TASK_COMPLETE_KEY}.")
    action_meta[TASK_COMPLETE_KEY] = {"start": action_dim - 1, "end": action_dim}
    _dump_json(path, modality)


def _patch_info(meta_dir: Path, action_dim: int) -> None:
    path = meta_dir / "info.json"
    info = _load_json(path)
    action_feature = info["features"]["action"]
    action_feature["shape"] = [action_dim]
    names = action_feature.get("names")
    if names and isinstance(names, list) and names and isinstance(names[0], list):
        names[0] = [*names[0], TASK_COMPLETE_KEY]
    _dump_json(path, info)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--done_last_n",
        type=int,
        default=8,
        help="Number of final frames per sub-episode labeled as task completion ramp.",
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Use hard labels for the final done_last_n frames instead of a soft ramp.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_dir if it exists.")
    args = parser.parse_args()

    if args.done_last_n < 1:
        raise ValueError("--done_last_n must be >= 1.")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} already exists; pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)

    shutil.copytree(args.input_dir, args.output_dir, copy_function=_copy_or_link)

    meta_dir = args.output_dir / "meta"
    parquet_paths = _episode_parquet_paths(args.output_dir)
    if not parquet_paths:
        raise RuntimeError(f"No parquet files found in {args.output_dir / 'data'}.")

    all_actions = []
    per_episode_stats = []
    action_dim = None

    for parquet_path in parquet_paths:
        episode_index = int(parquet_path.stem.split("_")[1])
        df = pd.read_parquet(parquet_path)
        if "action" not in df.columns:
            raise RuntimeError(f"{parquet_path} missing action column.")
        if "stage" not in df.columns:
            raise RuntimeError(f"{parquet_path} missing stage column; use split_by_stage.py output.")

        df["action"], _ = _append_done_to_action(
            df["action"],
            done_last_n=args.done_last_n,
            soft=not args.hard,
        )
        actions = np.stack(df["action"].to_numpy()).astype(np.float32, copy=False)
        action_dim = actions.shape[1]
        all_actions.append(actions)
        df.to_parquet(parquet_path, index=False)

        per_episode_stats.append(
            {"episode_index": episode_index, "stats": {"action": _field_stats(actions)}}
        )

    assert action_dim is not None
    _patch_modality(meta_dir, action_dim)
    _patch_info(meta_dir, action_dim)

    stats_path = meta_dir / "stats.json"
    stats = _load_json(stats_path)
    stats["action"] = _field_stats(np.concatenate(all_actions, axis=0))
    _dump_json(stats_path, stats)

    episode_stats_path = meta_dir / "episodes_stats.jsonl"
    existing = {}
    if episode_stats_path.exists():
        with episode_stats_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                existing[item["episode_index"]] = item
    for item in per_episode_stats:
        existing.setdefault(item["episode_index"], {"episode_index": item["episode_index"], "stats": {}})
        existing[item["episode_index"]]["stats"]["action"] = item["stats"]["action"]
    with episode_stats_path.open("w") as f:
        for episode_index in sorted(existing):
            f.write(json.dumps(existing[episode_index]) + "\n")

    print(
        f"Added action.{TASK_COMPLETE_KEY} to {len(parquet_paths)} episodes. "
        f"Action dim: {action_dim - 1} -> {action_dim}. "
        f"Label mode: {'hard' if args.hard else 'soft'}, last_n={args.done_last_n}. "
        f"Output: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
