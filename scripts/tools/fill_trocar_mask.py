# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Fill the transparent trocar shaft in OmniGlass segmentation masks.

OmniGlass renders the glass shaft transparent, so instance_id_segmentation
only captures the opaque tip and handle. For one trocar this yields two
disconnected components per frame: a large one (tip + handle) and a small
roughly-triangular one near where the shaft meets the tip. This script
infers the missing shaft as a rectangle between the two components.

Bridge algorithm (per label, per frame):
  1. Find the two largest connected components.
  2. Project the small component's contour onto the small -> big axis.
     The edge of the small triangle facing the big component is the set of
     contour points in the top fraction of that projection.
  3. The perpendicular spread of those facing points = rectangle width
     (i.e. the length of the facing edge).
  4. Rectangle length runs from the small region's facing edge to the
     big region's centroid.
  5. Fill the rectangle with the label color.

Input:  <dataset_dir>/masks/chunk-000/<cam>/episode_*.npz
Output: <dataset_dir>/masks_postprocessed/chunk-000/<cam>/episode_*.npz
        (original masks/ directory is untouched)

This script is self-contained: only depends on numpy, opencv-python, and
matplotlib. It has no isaaclab / isaacsim imports and runs in any Python 3.10+
environment with those three packages installed.

Usage:
    # Preview 9 random frames as RGB / original mask / filled mask.
    python fill_trocar_mask.py --dataset_dir /path/to/dataset --preview

    # Preview with background/ground replaced by a random image from a dir.
    python fill_trocar_mask.py --dataset_dir /path/to/dataset --preview \
        --backdrop_dir /path/to/backdrops

    # Write filled masks into masks_postprocessed/.
    python fill_trocar_mask.py --dataset_dir /path/to/dataset --apply
"""

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _bridge_rect(labels_img: np.ndarray, big_idx: int, small_idx: int,
                 centroids: np.ndarray, facing_frac: float = 0.3) -> np.ndarray | None:
    """Compute the 4 polygon corners that bridge the small triangle to the big component."""
    big_c = centroids[big_idx]
    small_c = centroids[small_idx]
    d = big_c - small_c
    dist = float(np.linalg.norm(d))
    if dist < 1.0:
        return None
    d = d / dist
    perp = np.array([-d[1], d[0]])

    small_mask = (labels_img == small_idx).astype(np.uint8)
    contours, _ = cv2.findContours(small_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(contour) < 3:
        return None

    rel = contour - small_c
    proj_d = rel @ d
    max_d = float(proj_d.max())
    if max_d <= 0:
        return None

    facing = contour[proj_d > max_d * facing_frac]
    if len(facing) < 2:
        return None
    proj_p = (facing - small_c) @ perp
    u_lo, u_hi = float(proj_p.min()), float(proj_p.max())
    if u_hi - u_lo < 2:
        return None

    edge_center = small_c + d * max_d
    return np.array([
        edge_center + perp * u_lo,
        edge_center + perp * u_hi,
        big_c + perp * u_hi,
        big_c + perp * u_lo,
    ], dtype=np.int32)


def _fill_label(mask: np.ndarray, label: int) -> np.ndarray:
    m = (mask == label).astype(np.uint8)
    num, labels_img, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num < 3:
        return mask
    comps = sorted(range(1, num), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    if len(comps) < 2:
        return mask
    big_idx, small_idx = comps[0], comps[1]
    rect = _bridge_rect(labels_img, big_idx, small_idx, centroids)
    if rect is None:
        return mask
    out = mask.copy()
    cv2.fillPoly(out, [rect], color=int(label))
    return out


def _fill_frame(frame: np.ndarray, labels: list[int]) -> np.ndarray:
    out = frame.copy()
    for lbl in labels:
        out = _fill_label(out, lbl)
    return out


PALETTE = np.array([
    [0, 0, 0],
    [120, 120, 120],
    [0, 200, 0],
    [255, 80, 80],
    [80, 80, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
], dtype=np.uint8)


def _colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, c in enumerate(PALETTE):
        out[mask == i] = c
    out[mask >= len(PALETTE)] = (255, 255, 255)
    return out


def _overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Black out background (0) and ground (1); keep RGB elsewhere."""
    out = rgb.copy()
    black_out = (mask == 0) | (mask == 1)
    out[black_out] = 0
    return out


def _composite_with_backdrop(rgb: np.ndarray, mask: np.ndarray, backdrop: np.ndarray) -> np.ndarray:
    """Replace background (0) and ground (1) with the backdrop image; keep RGB elsewhere."""
    h, w = rgb.shape[:2]
    bh, bw = backdrop.shape[:2]
    if (bh, bw) != (h, w):
        backdrop = cv2.resize(backdrop, (w, h), interpolation=cv2.INTER_AREA)
    out = rgb.copy()
    replace = (mask == 0) | (mask == 1)
    out[replace] = backdrop[replace]
    return out


def _read_video_frame(mp4_path: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _load_npz_array(path: Path) -> tuple[str, np.ndarray]:
    data = dict(np.load(path))
    assert len(data) == 1, f"Expected 1 array in {path}, got {list(data.keys())}"
    key = list(data.keys())[0]
    return key, data[key]


def preview(dataset_dir: Path, out_path: Path, labels: list[int], n: int = 9, seed: int = 0,
            backdrop_dir: Path | None = None) -> None:
    rng = random.Random(seed)
    backdrop_files: list[Path] = []
    if backdrop_dir is not None:
        backdrop_files = sorted([p for p in backdrop_dir.iterdir()
                                 if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not backdrop_files:
            raise RuntimeError(f"No image files in {backdrop_dir}")
    masks_root = dataset_dir / "masks" / "chunk-000"
    videos_root = dataset_dir / "videos" / "chunk-000"
    if not masks_root.exists():
        raise FileNotFoundError(f"{masks_root} not found")
    cams = [d.name for d in sorted(masks_root.glob("*")) if d.is_dir()]
    if not cams:
        raise RuntimeError(f"No camera subdirs under {masks_root}")

    candidates = []
    for cam in cams:
        for npz in sorted((masks_root / cam).glob("*.npz")):
            candidates.append((cam, npz))
    if not candidates:
        raise RuntimeError("No npz files found.")
    rng.shuffle(candidates)

    samples = []
    for cam, npz_path in candidates:
        if len(samples) >= n:
            break
        _, arr = _load_npz_array(npz_path)
        if arr.ndim != 3 or arr.shape[0] == 0:
            continue
        for _ in range(5):
            t = rng.randrange(arr.shape[0])
            before = arr[t]
            after = _fill_frame(before, labels)
            if np.array_equal(before, after):
                continue
            ep_stem = npz_path.stem.removesuffix("_masks")
            mp4_path = videos_root / cam / f"{ep_stem}.mp4"
            rgb = _read_video_frame(mp4_path, t)
            if rgb is None:
                continue
            samples.append({
                "cam": cam, "episode": ep_stem, "frame": t,
                "rgb": rgb, "before": before, "after": after,
            })
            break

    if not samples:
        print("No frames with fillable components — nothing to preview.")
        return

    rows = len(samples)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]
    for i, s in enumerate(samples):
        axes[i, 0].imshow(s["rgb"])
        axes[i, 0].set_title(f"{s['cam']}\n{s['episode']} t={s['frame']}", fontsize=8)
        if backdrop_files:
            bd_path = backdrop_files[rng.randrange(len(backdrop_files))]
            bd = cv2.imread(str(bd_path))
            if bd is None:
                raise RuntimeError(f"Failed to read backdrop {bd_path}")
            bd = cv2.cvtColor(bd, cv2.COLOR_BGR2RGB)
            axes[i, 1].imshow(_composite_with_backdrop(s["rgb"], s["before"], bd))
            axes[i, 1].set_title("original mask (bg replaced)", fontsize=8)
            axes[i, 2].imshow(_composite_with_backdrop(s["rgb"], s["after"], bd))
            axes[i, 2].set_title("filled mask (bg replaced)", fontsize=8)
        else:
            axes[i, 1].imshow(_colorize(s["before"]))
            axes[i, 1].set_title("original mask", fontsize=8)
            axes[i, 2].imshow(_colorize(s["after"]))
            axes[i, 2].set_title("filled mask", fontsize=8)
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote preview figure with {len(samples)} samples to {out_path}")


def apply(dataset_dir: Path, labels: list[int]) -> None:
    src_root = dataset_dir / "masks" / "chunk-000"
    dst_root = dataset_dir / "masks_postprocessed" / "chunk-000"
    if not src_root.exists():
        raise FileNotFoundError(f"{src_root} not found")

    total = 0
    for cam_dir in sorted(src_root.glob("*")):
        if not cam_dir.is_dir():
            continue
        out_cam = dst_root / cam_dir.name
        out_cam.mkdir(parents=True, exist_ok=True)
        npzs = sorted(cam_dir.glob("*.npz"))
        print(f"  {cam_dir.name}: {len(npzs)} files")
        for npz in npzs:
            _, arr = _load_npz_array(npz)
            if arr.ndim == 3:
                for t in range(arr.shape[0]):
                    arr[t] = _fill_frame(arr[t], labels)
            np.savez_compressed(out_cam / npz.name, arr)
            total += 1
    print(f"Wrote {total} processed npz files under {dst_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--labels", type=int, nargs="+", default=[3, 4],
                        help="Label IDs to fill (default: 3=trocar_1, 4=trocar_2).")
    parser.add_argument("--preview", action="store_true",
                        help="Generate an RGB / before / after comparison figure.")
    parser.add_argument("--preview_out", type=str, default=None,
                        help="Output figure path (default: <dataset_dir>/mask_fill_preview.png).")
    parser.add_argument("--n", type=int, default=9, help="Number of samples in preview.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--apply", action="store_true",
                        help="Write filled masks into masks_postprocessed/.")
    parser.add_argument("--backdrop_dir", type=str, default=None,
                        help="If set, preview replaces background+ground with a random image from this dir.")
    args = parser.parse_args()

    ds = Path(args.dataset_dir)
    if args.preview:
        out = Path(args.preview_out) if args.preview_out else ds / "mask_fill_preview.png"
        bd = Path(args.backdrop_dir) if args.backdrop_dir else None
        preview(ds, out, args.labels, n=args.n, seed=args.seed, backdrop_dir=bd)
    if args.apply:
        apply(ds, args.labels)
    if not (args.preview or args.apply):
        parser.error("Specify --preview and/or --apply.")


if __name__ == "__main__":
    main()
