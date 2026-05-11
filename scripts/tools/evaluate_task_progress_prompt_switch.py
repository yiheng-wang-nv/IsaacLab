#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate whether the task-progress regressor is prompt-conditioned.

For each sampled episode, this script keeps the image/state fixed and swaps
the task prompt before extracting GR00T backbone features. This checks the
multi-stage behavior we want: when task 1 is complete and the prompt switches
to task 2, predicted task-2 progress should drop back near 0.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


class TaskProgressRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x).squeeze(-1))


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


def _find_gr00t_root(checkpoint: Path, gr00t_root: str | None) -> Path:
    if gr00t_root is not None:
        root = Path(gr00t_root).expanduser().resolve()
        if (root / "gr00t_config.py").exists():
            return root
        raise FileNotFoundError(f"Could not find gr00t_config.py under {root}.")
    for candidate in [checkpoint, *checkpoint.parents]:
        if (candidate / "gr00t_config.py").exists():
            return candidate
    raise FileNotFoundError("Could not infer Isaac-GR00T root. Pass --gr00t_root.")


def _load_policy(checkpoint: str, gr00t_root: str | None, device: str):
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    root = _find_gr00t_root(checkpoint_path, gr00t_root)
    sys.path.insert(0, str(root))

    spec = importlib.util.spec_from_file_location("prompt_switch_gr00t_config", root / "gr00t_config.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load gr00t_config.py from {root}")
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)

    from gr00t.model.policy import Gr00tPolicy

    data_config = cfg_module.UnitreeG1SimTaskCompleteInferDataConfig()
    policy = Gr00tPolicy(
        model_path=str(checkpoint_path),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=1,
        device=device,
    )
    policy.model.eval()
    return policy


def _load_regressor(path: str, device: str):
    ckpt = torch.load(Path(path).expanduser(), map_location=device)
    model = TaskProgressRegressor(ckpt["input_dim"], ckpt["hidden_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, int(ckpt["feat_dim"]), int(ckpt["n_tasks"])


def _select_frame_indices(n_frames: int, frame_kinds: list[str]) -> dict[str, int]:
    indices = {}
    for kind in frame_kinds:
        if kind == "first":
            indices[kind] = 0
        elif kind == "middle":
            indices[kind] = n_frames // 2
        elif kind == "last":
            indices[kind] = n_frames - 1
        else:
            raise ValueError(f"Unsupported frame kind: {kind}")
    return indices


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _print_matrix(title: str, results: dict[tuple[int, int], list[float]], n_tasks: int) -> None:
    print(f"\n{title}")
    print("rows=prompt task, cols=image/source task")
    print("          " + " ".join(f"img{j + 1:>7d}" for j in range(n_tasks)))
    for prompt_idx in range(n_tasks):
        row = []
        for image_idx in range(n_tasks):
            row.append(f"{_mean_or_nan(results[(prompt_idx, image_idx)]):8.4f}")
        print(f"prompt{prompt_idx + 1:<3d} " + " ".join(row))


def _write_matrix_csv(path: Path, results: dict[tuple[int, int], list[float]], n_tasks: int) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_task", *[f"img_task_{i + 1}" for i in range(n_tasks)]])
        for prompt_idx in range(n_tasks):
            writer.writerow(
                [f"prompt{prompt_idx + 1}"]
                + [f"{_mean_or_nan(results[(prompt_idx, image_idx)]):.4f}" for image_idx in range(n_tasks)]
            )
    print(f"[INFO] Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to the Isaac-GR00T checkpoint directory.")
    parser.add_argument("--gr00t_root", default=None, help="Path to the Isaac-GR00T repository.")
    parser.add_argument("--dataset", required=True, help="LeRobot dataset directory.")
    parser.add_argument(
        "--features_dir",
        default="/home/nvidia/workspace/yiheng/datasets/task_complete_features_cross",
        help="Feature directory containing episode_index.npy; used to reproduce the train/val split.",
    )
    parser.add_argument(
        "--regressor_path",
        default="/home/nvidia/workspace/yiheng/models/task_progress_regressor/best_model.pt",
        help="Path to trained task-progress regressor checkpoint.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_episodes_per_task", type=int, default=10)
    parser.add_argument("--frame_kinds", default="first,last", help="Comma-separated: first,middle,last.")
    parser.add_argument("--split", choices=["all", "train", "val"], default="val")
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser()
    frame_kinds = [item.strip() for item in args.frame_kinds.split(",") if item.strip()]

    print(f"[INFO] Loading GR00T policy from {args.checkpoint}")
    policy = _load_policy(args.checkpoint, args.gr00t_root, args.device)

    print(f"[INFO] Loading progress regressor from {args.regressor_path}")
    regressor, feat_dim, n_tasks = _load_regressor(args.regressor_path, args.device)

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

    parquet_files = sorted(dataset_path.glob("data/chunk-*/episode_*.parquet"))
    allowed_episode_ids: set[int] | None = None
    if args.split != "all":
        episode_index_path = Path(args.features_dir).expanduser() / "episode_index.npy"
        if not episode_index_path.exists():
            raise FileNotFoundError(f"Cannot use --split {args.split}: missing {episode_index_path}")
        episode_ids = np.unique(np.load(episode_index_path))
        rng = np.random.RandomState(args.seed)
        shuffled_episode_ids = episode_ids.copy()
        rng.shuffle(shuffled_episode_ids)
        n_val_eps = max(1, int(len(shuffled_episode_ids) * args.val_frac))
        val_ids = set(int(ep) for ep in shuffled_episode_ids[:n_val_eps])
        train_ids = set(int(ep) for ep in shuffled_episode_ids[n_val_eps:])
        allowed_episode_ids = val_ids if args.split == "val" else train_ids
        print(
            f"[INFO] Using {args.split} split: {len(allowed_episode_ids)} episodes "
            f"(seed={args.seed}, val_frac={args.val_frac})."
        )

    selected: list[tuple[int, Path]] = []
    per_task_counts = defaultdict(int)
    for episode_idx, parquet_path in enumerate(parquet_files):
        if allowed_episode_ids is not None and episode_idx not in allowed_episode_ids:
            continue
        df_head = pd.read_parquet(parquet_path, columns=["task_index"])
        if len(df_head) == 0:
            continue
        task_idx = int(df_head["task_index"].iloc[0])
        if task_idx >= n_tasks or per_task_counts[task_idx] >= args.max_episodes_per_task:
            continue
        selected.append((task_idx, parquet_path))
        per_task_counts[task_idx] += 1
        if all(per_task_counts[i] >= args.max_episodes_per_task for i in range(n_tasks)):
            break

    print(f"[INFO] Selected episodes per task: {dict(sorted(per_task_counts.items()))}")

    results_by_frame: dict[str, dict[tuple[int, int], list[float]]] = {
        kind: defaultdict(list) for kind in frame_kinds
    }

    with torch.inference_mode():
        for source_task_idx, parquet_path in tqdm(selected, desc="Episodes"):
            df = pd.read_parquet(parquet_path)
            n_frames = len(df)
            if n_frames == 0:
                continue

            states = np.array(df["observation.state"].tolist(), dtype=np.float32)
            frame_indices = _select_frame_indices(n_frames, frame_kinds)

            video_frames = {}
            ok = True
            for v_key in VIDEO_KEYS:
                chunk_name = parquet_path.parent.name
                ep_name = parquet_path.stem + ".mp4"
                vid_path = dataset_path / "videos" / chunk_name / VIDEO_DIR_KEYS[v_key] / ep_name
                if not vid_path.exists():
                    print(f"[WARN] Missing video {vid_path}, skipping {parquet_path}.")
                    ok = False
                    break
                frames = _load_video_frames(vid_path)
                if len(frames) != n_frames:
                    print(f"[WARN] Frame mismatch {vid_path}: {len(frames)} vs {n_frames}, skipping.")
                    ok = False
                    break
                video_frames[v_key] = frames
            if not ok:
                continue

            for frame_kind, frame_idx in frame_indices.items():
                images = {v_key: video_frames[v_key][frame_idx] for v_key in VIDEO_KEYS}
                state = states[frame_idx]
                for prompt_task_idx in range(n_tasks):
                    obs = _build_obs(images, state, task_idx_to_desc[prompt_task_idx])
                    policy.get_action(obs)
                    feat = backbone_out["features"].mean(dim=1)
                    if feat.shape[-1] != feat_dim:
                        raise RuntimeError(f"Expected feature dim {feat_dim}, got {feat.shape[-1]}.")
                    task_onehot = torch.zeros((1, n_tasks), dtype=torch.float32, device=args.device)
                    task_onehot[0, prompt_task_idx] = 1.0
                    model_input = torch.cat([feat.to(args.device), task_onehot], dim=-1)
                    pred = float(regressor(model_input).item())
                    results_by_frame[frame_kind][(prompt_task_idx, source_task_idx)].append(pred)

    for frame_kind in frame_kinds:
        _print_matrix(f"{frame_kind.upper()} frame predicted progress", results_by_frame[frame_kind], n_tasks)
        _write_matrix_csv(Path(f"prompt_switch_{frame_kind}_frame.csv"), results_by_frame[frame_kind], n_tasks)

    if "last" in results_by_frame:
        print("\nStage-switch check on LAST frames")
        print("completed task i with prompt i should be high; same frame with prompt i+1 should be low.")
        last_results = results_by_frame["last"]
        for task_idx in range(n_tasks - 1):
            self_progress = _mean_or_nan(last_results[(task_idx, task_idx)])
            next_progress = _mean_or_nan(last_results[(task_idx + 1, task_idx)])
            print(
                f"  image task {task_idx + 1} done: "
                f"prompt {task_idx + 1}={self_progress:.4f}, "
                f"prompt {task_idx + 2}={next_progress:.4f}, "
                f"drop={self_progress - next_progress:.4f}"
            )


if __name__ == "__main__":
    main()
