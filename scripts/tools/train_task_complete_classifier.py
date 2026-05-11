#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a task-progress MLP regressor on GR00T backbone features.

Usage:
    python train_task_complete_classifier.py \
        --features_dir /localhome/local-vennw/code/task_complete_features \
        --output_dir   /localhome/local-vennw/code/task_complete_regressor \
        --epochs 50

The regressor takes (backbone_feature, task_one_hot) as input and outputs
episode progress in [0, 1]. At inference, feed current GR00T backbone
features + task idx.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument("--features_dir", default="/localhome/local-vennw/code/task_complete_features")
parser.add_argument("--output_dir", default="/localhome/local-vennw/code/task_complete_regressor")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--val_frac", type=float, default=0.15)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(f"[INFO] Loading features from {args.features_dir}")
features = np.load(os.path.join(args.features_dir, "features.npy"))   # (N, D)
labels   = np.load(os.path.join(args.features_dir, "labels.npy"))     # (N,) progress in [0, 1]
task_idx = np.load(os.path.join(args.features_dir, "task_index.npy")) # (N,)
source_task_index_path = os.path.join(args.features_dir, "source_task_index.npy")
source_task_idx = np.load(source_task_index_path) if os.path.exists(source_task_index_path) else task_idx
episode_index_path = os.path.join(args.features_dir, "episode_index.npy")
episode_idx = np.load(episode_index_path) if os.path.exists(episode_index_path) else None

with open(os.path.join(args.features_dir, "meta.json")) as f:
    meta = json.load(f)

N, feat_dim = features.shape
n_tasks = 5
print(f"  Samples: {N}  feature_dim: {feat_dim}")
print(f"  Progress: min={labels.min():.4f} mean={labels.mean():.4f} max={labels.max():.4f}")
print(f"  Source-prompt samples: {(task_idx == source_task_idx).sum()}  Cross-prompt samples: {(task_idx != source_task_idx).sum()}")
for i in range(n_tasks):
    mask = task_idx == i
    label_mean = labels[mask].mean() if mask.any() else 0.0
    print(f"  Task {i}: {mask.sum()} samples, mean_progress={label_mean:.4f}")

# ---------------------------------------------------------------------------
# Train / val split — split by episode block to avoid data leakage.
# Frames within the same episode are sequential; use the saved episode_index
# from feature extraction when available.
# ---------------------------------------------------------------------------
if episode_idx is not None:
    episode_ids = episode_idx
else:
    frames_per_ep = int(np.round(N / meta.get("n_episodes", 2300))) if meta.get("n_episodes", 0) else 30
    frames_per_ep = max(frames_per_ep, 1)
    episode_ids = np.arange(N) // frames_per_ep

unique_eps = np.unique(episode_ids)
np.random.shuffle(unique_eps)
n_val_eps = max(1, int(len(unique_eps) * args.val_frac))
val_eps = set(unique_eps[:n_val_eps].tolist())

val_mask  = np.array([ep in val_eps for ep in episode_ids])
train_mask = ~val_mask

X_train = torch.tensor(features[train_mask], dtype=torch.float32)
y_train = torch.tensor(labels[train_mask],   dtype=torch.float32)
t_train = torch.tensor(task_idx[train_mask], dtype=torch.long)
s_train = torch.tensor(source_task_idx[train_mask], dtype=torch.long)

X_val = torch.tensor(features[val_mask], dtype=torch.float32)
y_val = torch.tensor(labels[val_mask],   dtype=torch.float32)
t_val = torch.tensor(task_idx[val_mask], dtype=torch.long)
s_val = torch.tensor(source_task_idx[val_mask], dtype=torch.long)

print(f"\n  Train: {len(X_train)} samples, mean_progress={y_train.mean().item():.4f}")
print(f"  Val:   {len(X_val)} samples, mean_progress={y_val.mean().item():.4f}")

# One-hot encode task
def make_onehot(task_ids: torch.Tensor, n: int) -> torch.Tensor:
    return torch.zeros(len(task_ids), n).scatter_(1, task_ids.unsqueeze(1), 1.0)

oh_train = make_onehot(t_train, n_tasks)
oh_val   = make_onehot(t_val,   n_tasks)

X_train_full = torch.cat([X_train, oh_train], dim=-1)  # (N, D+5)
X_val_full   = torch.cat([X_val,   oh_val],   dim=-1)

def make_sample_weights(prompt_task: torch.Tensor, source_task: torch.Tensor) -> torch.Tensor:
    """Balance source-prompt progress samples and cross-prompt ordered samples."""
    is_cross = prompt_task != source_task
    n_total = len(prompt_task)
    n_cross = int(is_cross.sum().item())
    n_source = n_total - n_cross
    weights = torch.ones(n_total, dtype=torch.float32)
    if n_source > 0 and n_cross > 0:
        weights[~is_cross] = n_total / (2.0 * n_source)
        weights[is_cross] = n_total / (2.0 * n_cross)
    return weights


w_train = make_sample_weights(t_train, s_train)

train_ds = TensorDataset(X_train_full, y_train, w_train)
val_ds   = TensorDataset(X_val_full,   y_val, t_val, s_val)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
input_dim = feat_dim + n_tasks

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
        return torch.sigmoid(self.net(x).squeeze(-1))  # (B,) progress in [0, 1]

model = TaskProgressRegressor(input_dim, args.hidden_dim).to(args.device)
print(f"\n[INFO] Model: input_dim={input_dim}, hidden={args.hidden_dim}")
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.SmoothL1Loss(beta=0.05, reduction="none")
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def evaluate(loader):
    model.eval()
    total_loss = total_abs = total_sq = total = 0
    source_abs = source_total = cross_abs = cross_total = 0
    with torch.no_grad():
        for xb, yb, tb, sb in loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            preds = model(xb)
            loss = criterion(preds, yb).mean()
            total_loss += loss.item() * len(yb)
            err = preds - yb
            total_abs += err.abs().sum().item()
            total_sq += (err ** 2).sum().item()
            total += len(yb)
            is_cross = tb != sb
            if (~is_cross).any():
                source_abs += err[~is_cross.to(err.device)].abs().sum().item()
                source_total += int((~is_cross).sum().item())
            if is_cross.any():
                cross_abs += err[is_cross.to(err.device)].abs().sum().item()
                cross_total += int(is_cross.sum().item())
    mae = total_abs / total
    rmse = (total_sq / total) ** 0.5
    source_mae = source_abs / source_total if source_total > 0 else mae
    cross_mae = cross_abs / cross_total if cross_total > 0 else mae
    balanced_mae = 0.5 * (source_mae + cross_mae)
    return total_loss / total, mae, rmse, source_mae, cross_mae, balanced_mae

best_mae = float("inf")
best_epoch = 0

print(f"\n{'Epoch':>5} {'train_loss':>10} {'val_loss':>8} {'mae':>8} {'rmse':>8} {'src_mae':>8} {'cross_mae':>9} {'bal_mae':>8}")
print("-" * 78)

for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0.0
    for xb, yb, wb in train_loader:
        xb, yb, wb = xb.to(args.device), yb.to(args.device), wb.to(args.device)
        optimizer.zero_grad()
        loss = (criterion(model(xb), yb) * wb).mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(yb)
    train_loss /= len(train_ds)
    scheduler.step()

    val_loss, mae, rmse, source_mae, cross_mae, balanced_mae = evaluate(val_loader)
    print(
        f"{epoch:>5d} {train_loss:>10.4f} {val_loss:>8.4f} {mae:>8.4f} {rmse:>8.4f} "
        f"{source_mae:>8.4f} {cross_mae:>9.4f} {balanced_mae:>8.4f}"
    )

    if balanced_mae < best_mae:
        best_mae = balanced_mae
        best_epoch = epoch
        torch.save({
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "feat_dim": feat_dim,
            "hidden_dim": args.hidden_dim,
            "n_tasks": n_tasks,
            "best_balanced_mae": best_mae,
            "val_mae": mae,
            "val_source_mae": source_mae,
            "val_cross_prompt_mae": cross_mae,
            "epoch": epoch,
            "output_kind": "episode_progress",
        }, os.path.join(args.output_dir, "best_model.pt"))

print(f"\n[DONE] Best balanced MAE={best_mae:.4f} at epoch {best_epoch}")
print(f"       Saved to {args.output_dir}/best_model.pt")

# Per-task val stats
print("\nPer-task validation:")
model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"))["model_state"])
model.eval()
with torch.no_grad():
    all_preds = model(X_val_full.to(args.device)).cpu()

TASK_NAMES = ["left hand pick up", "right hand pick up", "align trocars", "install trocar", "place trocar"]
for i in range(n_tasks):
    mask = t_val == i
    n_task = int(mask.sum().item())
    if n_task == 0:
        continue
    yb = y_val[mask]
    pb = all_preds[mask]
    err = pb - yb
    mae = err.abs().mean().item()
    rmse = torch.sqrt((err ** 2).mean()).item()
    print(
        f"  Task {i+1} ({TASK_NAMES[i]:<20}): n={n_task:4d} "
        f"MAE={mae:.4f} RMSE={rmse:.4f} pred_mean={pb.mean().item():.4f}"
    )
