# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Test whether primvars:doNotCastShadows = False fixes OmniGlass segmentation.

OmniGlass material renders the glass trocar shaft as transparent, which causes
the segmentation pass to skip it. The "Cast Shadows" checkbox in the Omniverse
Properties panel sets primvars:doNotCastShadows on the Mesh prim; when False,
the mesh participates in shadow/secondary-ray passes (including segmentation).

This script:
  1. Resets the env, captures baseline RGB + segmentation.
  2. Sets primvars:doNotCastShadows = False on every Mesh that has an OmniGlass
     GeomSubset child (i.e. the trocar visual meshes).
  3. Force-rerenders, captures the updated segmentation.
  4. Saves a 3-column figure: RGB / before / after.

Usage:
    conda activate isaaclab_develop_6.0
    python scripts/tools/test_glass_seg.py --output_dir /tmp/glass_seg_test
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="/tmp/glass_seg_test")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pxr import Sdf, Usd, UsdShade

import isaaclab_tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK_ID = "Isaac-Assemble-Trocar-G129-Dex3-RLinf-v0"
CAM_KEY = "front_camera"
GLASS_MDL = "OmniGlass"

PALETTE = np.array([
    [0,   0,   0  ],  # 0 background
    [120, 120, 120],  # 1 ground
    [0,   200, 0  ],  # 2 robot
    [255, 80,  80 ],  # 3 trocar_1
    [80,  80,  255],  # 4 trocar_2
    [255, 255, 0  ],  # 5 tray
    [255, 0,   255],  # 6 cart
    [0,   255, 255],  # 7 instrument_trolley
], dtype=np.uint8)


def _colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, c in enumerate(PALETTE):
        out[mask == i] = c
    out[mask >= len(PALETTE)] = 200
    return out


def _get_rgb(env) -> np.ndarray:
    img = env.scene.sensors[CAM_KEY].data.output["rgb"][0]
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    return img[..., :3].astype(np.uint8)


def _prim_path_to_cat(prim_path: str) -> int:
    if "Robot/" in prim_path:           return 2
    if "trocar_1/" in prim_path:        return 3
    if "trocar_2/" in prim_path:        return 4
    if "surgical_tray/" in prim_path:   return 5
    if "Cart001" in prim_path:          return 6
    if "InstrumentTrolley" in prim_path: return 7
    if "FlatGrid" in prim_path or "GroundPlane" in prim_path: return 1
    return 0


def _build_inst_to_cat(sensor) -> dict[int, int]:
    """Build instance_id → label mapping by matching prim path patterns."""
    raw = sensor.data.info
    info_dict = raw[0] if isinstance(raw, list) else raw
    if isinstance(info_dict, str):
        import ast
        info_dict = ast.literal_eval(info_dict)
    id_to_labels = info_dict.get("instance_id_segmentation_fast", {}).get("idToLabels", {})
    # Key may be string, int, or RGBA tuple depending on Isaac Sim version
    first_key = next(iter(id_to_labels), None)
    print(f"  idToLabels: {len(id_to_labels)} entries, key type = {type(first_key)}, example = {first_key!r}")
    mapping = {}
    for key, prim_path in id_to_labels.items():
        if isinstance(key, tuple):
            # RGBA tuple → pack into int32 as R + G*256 + B*65536
            r, g, b, a = (int(x) for x in key)
            inst_id = r | (g << 8) | (b << 16) | (a << 24)
        else:
            inst_id = int(key)
        mapping[inst_id] = _prim_path_to_cat(str(prim_path))
    return mapping


def _segmentation_to_instance_ids(seg: np.ndarray) -> np.ndarray:
    """Convert segmentation output to an integer instance-id image."""
    if seg.ndim == 3 and seg.shape[-1] == 1:
        return seg[..., 0]
    if seg.ndim == 3 and seg.shape[-1] == 4:
        seg = seg.astype(np.uint32)
        return seg[..., 0] | (seg[..., 1] << 8) | (seg[..., 2] << 16) | (seg[..., 3] << 24)
    return seg


def _get_seg(env, inst_to_cat: dict[int, int]) -> np.ndarray:
    seg = env.scene.sensors[CAM_KEY].data.output["instance_id_segmentation_fast"][0]
    if isinstance(seg, torch.Tensor):
        seg = seg.cpu().numpy()
    seg = _segmentation_to_instance_ids(seg)
    h, w = seg.shape[:2]
    label_img = np.zeros((h, w), dtype=np.uint8)
    for inst_id, cat in inst_to_cat.items():
        label_img[seg == inst_id] = cat
    return label_img


def _force_rerender(env) -> None:
    import omni.kit.app
    omni.kit.app.get_app().update()
    for s in env.scene.sensors.values():
        s.update(dt=0.0, force_recompute=True)


def _find_mesh_prims_with_glass(stage: Usd.Stage) -> list[str]:
    """Return paths of Mesh prims that contain an OmniGlass GeomSubset child."""
    found = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        for child in prim.GetChildren():
            if child.GetTypeName() != "GeomSubset":
                continue
            binding = UsdShade.MaterialBindingAPI(child).GetDirectBinding()
            mat_prim = stage.GetPrimAtPath(str(binding.GetMaterialPath()) + "/Shader")
            if not mat_prim.IsValid():
                continue
            src = mat_prim.GetAttribute("info:mdl:sourceAsset")
            if src.IsValid() and GLASS_MDL in str(src.Get()):
                found.append(str(prim.GetPath()))
                break
    return found


def _set_cast_shadows(stage: Usd.Stage, mesh_paths: list[str], do_not_cast: bool) -> None:
    for path in mesh_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        attr = prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool)
        attr.Set(do_not_cast)
        print(f"  primvars:doNotCastShadows={do_not_cast}  →  {path}")


def main():
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=1)
    cam_cfg = getattr(env_cfg.scene, CAM_KEY)
    cam_cfg.data_types = ["rgb", "instance_id_segmentation_fast"]
    cam_cfg.colorize_instance_id_segmentation = False
    env_cfg.recorders = {}
    import gymnasium as gym
    env = gym.make(TASK_ID, cfg=env_cfg).unwrapped

    env.reset()
    _force_rerender(env)

    inst_to_cat = _build_inst_to_cat(env.scene.sensors[CAM_KEY])
    print(f"Instance→category entries: {len(inst_to_cat)}")

    rgb = _get_rgb(env)
    seg_before = _get_seg(env, inst_to_cat)
    print("Before — unique labels:", np.unique(seg_before).tolist())

    stage = env.sim.stage
    glass_meshes = _find_mesh_prims_with_glass(stage)
    print(f"\nMesh prims with OmniGlass children ({len(glass_meshes)}):")
    for p in glass_meshes:
        print(" ", p)

    if not glass_meshes:
        print("\nNo glass mesh prims found. Saving baseline only.")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(_colorize(seg_before)); ax.set_title("Baseline seg"); ax.axis("off")
        fig.savefig(out_dir / "seg_before_after.png", dpi=120, bbox_inches="tight")
        env.close(); app.close(); return

    # Apply the fix: set doNotCastShadows=False (i.e. DO cast shadows → visible in seg)
    print()
    _set_cast_shadows(stage, glass_meshes, do_not_cast=False)
    _force_rerender(env)

    # Rebuild mapping in case new prims registered after attribute change
    inst_to_cat = _build_inst_to_cat(env.scene.sensors[CAM_KEY])
    seg_after = _get_seg(env, inst_to_cat)
    print("\nAfter — unique labels:", np.unique(seg_after).tolist())
    changed = int((seg_before != seg_after).sum())
    total = seg_before.size
    print(f"Changed pixels: {changed} / {total}  ({100*changed/total:.2f}%)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb);                axes[0].set_title("RGB");                       axes[0].axis("off")
    axes[1].imshow(_colorize(seg_before)); axes[1].set_title("Segmentation BEFORE");    axes[1].axis("off")
    axes[2].imshow(_colorize(seg_after));  axes[2].set_title("Segmentation AFTER\n(primvars:doNotCastShadows=False)"); axes[2].axis("off")
    fig.tight_layout()
    out_path = out_dir / "seg_before_after.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
