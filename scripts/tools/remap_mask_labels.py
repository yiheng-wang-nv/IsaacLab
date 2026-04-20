# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Remap mask category labels in-place or to a new directory.

Old mapping:                     New mapping:
  0 background                     0 background
  1 robot                          2 robot
  2 trocar_1                       3 trocar_1
  3 trocar_2                       4 trocar_2
  4 tray                           5 tray
  5 cart                           6 cart
  6 instrument_trolley             7 instrument_trolley
  7 ground                         1 ground   <-- moved

Updates:
  - masks/**/*.npz (every pixel value remapped)
  - category_mapping.json (categories + instance_to_category)

Usage:
  python remap_mask_labels.py --dataset_dir /path/to/dataset
  # multiple at once:
  python remap_mask_labels.py --dataset_dir /path/a /path/b /path/c
"""

import argparse
import json
from pathlib import Path

import numpy as np

# Old -> new mapping
REMAP = {
    0: 0,  # background
    7: 1,  # ground (moved to 1)
    1: 2,  # robot
    2: 3,  # trocar_1
    3: 4,  # trocar_2
    4: 5,  # tray
    5: 6,  # cart
    6: 7,  # instrument_trolley
}

NEW_CATEGORIES = {
    "0": "background",
    "1": "ground",
    "2": "robot",
    "3": "trocar_1",
    "4": "trocar_2",
    "5": "tray",
    "6": "cart",
    "7": "instrument_trolley",
}


def _build_lut():
    """Build 256-element LUT for fast pixel remap."""
    lut = np.arange(256, dtype=np.uint8)
    for old, new in REMAP.items():
        lut[old] = new
    return lut


def remap_masks(mask_dir: Path, lut: np.ndarray) -> int:
    """Remap every .npz in mask_dir subtree. Returns file count."""
    count = 0
    for npz_path in sorted(mask_dir.rglob("*.npz")):
        data = dict(np.load(npz_path))
        # The NPZ has a single array (key varies)
        assert len(data) == 1, f"Expected 1 array in {npz_path}, got {list(data.keys())}"
        key = list(data.keys())[0]
        arr = data[key]
        # Apply LUT
        arr_new = lut[arr]
        np.savez_compressed(npz_path, arr_new)
        count += 1
    return count


def update_category_mapping(path: Path) -> None:
    """Rewrite category_mapping.json with new label scheme."""
    if not path.exists():
        return
    with open(path) as f:
        data = json.load(f)
    data["categories"] = NEW_CATEGORIES
    # Remap instance_to_category values
    old_to_new = {old: new for old, new in REMAP.items()}
    i2c = data.get("instance_to_category", {})
    new_i2c = {k: old_to_new[v] for k, v in i2c.items()}
    data["instance_to_category"] = new_i2c
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def process_dataset(dataset_dir: Path):
    print(f"=== {dataset_dir} ===")
    if not dataset_dir.exists():
        print("  skip (not found)")
        return
    mask_dir = dataset_dir / "masks"
    lut = _build_lut()
    if mask_dir.exists():
        n = remap_masks(mask_dir, lut)
        print(f"  remapped {n} mask files")
    else:
        print("  no masks/ dir")
    update_category_mapping(dataset_dir / "category_mapping.json")
    print("  updated category_mapping.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, nargs="+", required=True,
                        help="One or more dataset root directories.")
    args = parser.parse_args()
    for d in args.dataset_dir:
        process_dataset(Path(d))


if __name__ == "__main__":
    main()
