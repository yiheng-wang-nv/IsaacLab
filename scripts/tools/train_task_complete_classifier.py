#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a task-complete MLP classifier on GR00T backbone features.

Usage:
    python train_task_complete_classifier.py \
        --features_dir /localhome/local-vennw/code/task_complete_features \
        --output_dir   /localhome/local-vennw/code/task_complete_classifier \
        --epochs 50

The classifier takes (backbone_feature, task_one_hot) as input and outputs
P(task_complete). At inference, feed current GR00T backbone features + task idx.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

parser = argparse.ArgumentParser()
parser.add_argument("--features_dir", default="/localhome/local-vennw/code/task_complete_features")
parser.add_argument("--output_dir", default="/localhome/local-vennw/code/task_complete_classifier")
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
labels   = np.load(os.path.join(args.features_dir, "labels.npy"))     # (N,)
task_idx = np.load(os.path.join(args.features_dir, "task_index.npy")) # (N,)

with open(os.path.join(args.features_dir, "meta.json")) as f:
    meta = json.load(f)

N, feat_dim = features.shape
n_tasks = 5
print(f"  Samples: {N}  feature_dim: {feat_dim}")
print(f"  Positive: {labels.sum()}  Negative: {(labels == 0).sum()}")
for i in range(n_tasks):
    mask = task_idx == i
    print(f"  Task {i}: {mask.sum()} samples, {(labels[mask] == 1).sum()} positive")

# ---------------------------------------------------------------------------
# Train / val split — split by episode block to avoid data leakage.
# Frames within the same episode are sequential; we split the first val_frac
# of unique episode boundaries off as validation.
# ---------------------------------------------------------------------------
# Approximate: each episode is ~30 frames. Group by contiguous blocks.
# Simple approach: split frame indices rather than episode indices since we
# don't have episode_id saved. Use a random 85/15 split at the episode level
# approximated by frame blocks.
frames_per_ep = int(np.round(N / meta["n_samples"] * N / 2300)) if meta["n_samples"] > 0 else 30
frames_per_ep = max(frames_per_ep, 1)

n_episodes_approx = N // frames_per_ep + 1
episode_ids = np.arange(N) // frames_per_ep    # coarse episode grouping

unique_eps = np.unique(episode_ids)
np.random.shuffle(unique_eps)
n_val_eps = max(1, int(len(unique_eps) * args.val_frac))
val_eps = set(unique_eps[:n_val_eps].tolist())

val_mask  = np.array([ep in val_eps for ep in episode_ids])
train_mask = ~val_mask

X_train = torch.tensor(features[train_mask], dtype=torch.float32)
y_train = torch.tensor(labels[train_mask],   dtype=torch.float32)
t_train = torch.tensor(task_idx[train_mask], dtype=torch.long)

X_val = torch.tensor(features[val_mask], dtype=torch.float32)
y_val = torch.tensor(labels[val_mask],   dtype=torch.float32)
t_val = torch.tensor(task_idx[val_mask], dtype=torch.long)

print(f"\n  Train: {len(X_train)} samples ({y_train.sum().int()} pos)")
print(f"  Val:   {len(X_val)} samples ({y_val.sum().int()} pos)")

# One-hot encode task
def make_onehot(task_ids: torch.Tensor, n: int) -> torch.Tensor:
    return torch.zeros(len(task_ids), n).scatter_(1, task_ids.unsqueeze(1), 1.0)

oh_train = make_onehot(t_train, n_tasks)
oh_val   = make_onehot(t_val,   n_tasks)

X_train_full = torch.cat([X_train, oh_train], dim=-1)  # (N, D+5)
X_val_full   = torch.cat([X_val,   oh_val],   dim=-1)

# Weighted sampler to handle class imbalance
pos_count = y_train.sum().item()
neg_count = len(y_train) - pos_count
sample_weights = torch.where(y_train == 1,
                              torch.tensor(neg_count / pos_count),
                              torch.tensor(1.0))
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_ds = TensorDataset(X_train_full, y_train)
val_ds   = TensorDataset(X_val_full,   y_val)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
input_dim = feat_dim + n_tasks

class TaskCompleteClassifier(nn.Module):
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
        return self.net(x).squeeze(-1)  # (B,) logits

model = TaskCompleteClassifier(input_dim, args.hidden_dim).to(args.device)
print(f"\n[INFO] Model: input_dim={input_dim}, hidden={args.hidden_dim}")
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

pos_weight = torch.tensor(neg_count / pos_count, device=args.device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def evaluate(loader):
    model.eval()
    total_loss = correct = total = tp = fp = fn = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            preds = (logits.sigmoid() >= 0.5).float()
            correct += (preds == yb).sum().item()
            total += len(yb)
            tp += ((preds == 1) & (yb == 1)).sum().item()
            fp += ((preds == 1) & (yb == 0)).sum().item()
            fn += ((preds == 0) & (yb == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return total_loss / total, correct / total, precision, recall, f1

best_f1 = 0.0
best_epoch = 0

print(f"\n{'Epoch':>5} {'train_loss':>10} {'val_loss':>8} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}")
print("-" * 55)

for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(args.device), yb.to(args.device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(yb)
    train_loss /= len(train_ds)
    scheduler.step()

    val_loss, acc, prec, rec, f1 = evaluate(val_loader)
    print(f"{epoch:>5d} {train_loss:>10.4f} {val_loss:>8.4f} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")

    if f1 > best_f1:
        best_f1 = f1
        best_epoch = epoch
        torch.save({
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "feat_dim": feat_dim,
            "hidden_dim": args.hidden_dim,
            "n_tasks": n_tasks,
            "best_f1": best_f1,
            "epoch": epoch,
        }, os.path.join(args.output_dir, "best_model.pt"))

print(f"\n[DONE] Best F1={best_f1:.4f} at epoch {best_epoch}")
print(f"       Saved to {args.output_dir}/best_model.pt")

# Per-task val stats
print("\nPer-task validation (best threshold=0.5):")
model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"))["model_state"])
model.eval()
with torch.no_grad():
    all_logits = model(X_val_full.to(args.device)).cpu()
    all_preds = (all_logits.sigmoid() >= 0.5).float()

TASK_NAMES = ["left hand pick up", "right hand pick up", "align trocars", "install trocar", "place trocar"]
for i in range(n_tasks):
    mask = t_val == i
    if mask.sum() == 0:
        continue
    yb = y_val[mask]
    pb = all_preds[mask]
    tp = ((pb == 1) & (yb == 1)).sum().item()
    fp = ((pb == 1) & (yb == 0)).sum().item()
    fn = ((pb == 0) & (yb == 1)).sum().item()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    print(f"  Task {i+1} ({TASK_NAMES[i]:<20}): n={mask.sum():4d} pos={yb.sum().int():3d}  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
