#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the fixed-base Unitree G1 upper body behind an MHS-style read/write bridge.

This is the device server. It owns the Isaac Sim process, adds cameras to the pick-and-place scene,
publishes every robot and camera quantity as a described channel, and serves reads and writes over
HTTP on the simulation thread.

.. code-block:: bash

    # on the machine with the GPU and the monitor
    ./isaaclab.sh -p scripts/mhs/run_g1_bridge.py --port 8765

    # from anywhere else
    ssh -N -L 8765:127.0.0.1:8765 nvidia@<host> &
    python scripts/mhs/mhs_cli.py describe
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Serve the G1 upper body over the MHS bridge.")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-G1-InspireFTP-Abs-v0", help="Task id to load.")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Interface to bind; keep it on loopback.")
parser.add_argument("--port", type=int, default=8765, help="TCP port to bind.")
parser.add_argument("--head-link", type=str, default="", help="Robot link to mount the head camera on.")
parser.add_argument("--camera-width", type=int, default=640, help="Camera width in pixels.")
parser.add_argument("--camera-height", type=int, default=480, help="Camera height in pixels.")
parser.add_argument("--no-cameras", action="store_true", help="Skip the cameras (much faster to start).")
parser.add_argument("--reference-out", type=str, default="", help="Also write the reference file here on startup.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.no_cameras:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402

import gymnasium as gym  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402  -- registers task ids
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401, E402  -- surfaces import errors
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from mhs_bridge.core import Registry  # noqa: E402
from mhs_bridge.devices.g1_upper_body import G1UpperBody  # noqa: E402
from mhs_bridge.server import Bridge  # noqa: E402

# Links that different G1 USD variants use for the head, most specific first.
HEAD_LINK_CANDIDATES = ("head_link", "d435_link", "logo_link", "torso_link", "waist_yaw_link", "pelvis")


def pick_head_link(env_cfg, requested: str) -> str:
    """Choose a robot link to mount the head camera on.

    The camera prim path has to be valid before the stage is populated, so the link is resolved from
    the USD file rather than from a live articulation.
    """
    if requested:
        return requested
    from pxr import Usd  # noqa: PLC0415

    from isaaclab.utils.assets import retrieve_file_path  # noqa: PLC0415

    usd_path = retrieve_file_path(env_cfg.scene.robot.spawn.usd_path)
    stage = Usd.Stage.Open(usd_path)
    names = {prim.GetName() for prim in stage.Traverse()}
    for candidate in HEAD_LINK_CANDIDATES:
        if candidate in names:
            return candidate
    raise RuntimeError(f"none of {HEAD_LINK_CANDIDATES} exist in {usd_path}")


def add_cameras(env_cfg, head_link: str, width: int, height: int) -> tuple[str, ...]:
    """Attach an egocentric camera to the robot and a fixed camera in front of the workcell."""
    common = dict(
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)
        ),
    )
    # Egocentric view. The ROS convention points +Z along the optical axis, so the identity rotation
    # already looks along the link's forward axis.
    env_cfg.scene.head_cam = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{head_link}/head_cam",
        offset=CameraCfg.OffsetCfg(pos=(0.08, 0.0, 0.12), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        **common,
    )
    # Third-person view of the table, useful for an agent that wants to check what actually happened.
    env_cfg.scene.front_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/front_cam",
        offset=CameraCfg.OffsetCfg(pos=(1.6, 0.0, 1.6), rot=(0.35, -0.61, 0.61, -0.35), convention="ros"),
        **common,
    )
    return ("head_cam", "front_cam")


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)

    camera_names: tuple[str, ...] = ()
    if not args_cli.no_cameras:
        head_link = pick_head_link(env_cfg, args_cli.head_link)
        print(f"[mhs] mounting the head camera on '{head_link}'")
        camera_names = add_cameras(env_cfg, head_link, args_cli.camera_width, args_cli.camera_height)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot = G1UpperBody(env, camera_names=camera_names)
    registry = Registry(
        name="isaac-sim-g1-workcell",
        description=(
            "Fixed-base Unitree G1 upper body in an Isaac Sim pick-and-place scene. Read the robot and"
            " camera channels to observe, write the wrist target poses and the hand closures to act, then"
            " step the simulation so the controller can track the new targets."
        ),
    )
    robot.register(registry)

    if args_cli.reference_out:
        with open(args_cli.reference_out, "w") as handle:
            json.dump(registry.reference(), handle, indent=2)
        print(f"[mhs] wrote the reference file to {args_cli.reference_out}")

    bridge = Bridge(registry, host=args_cli.host, port=args_cli.port, step_fn=robot.step)
    bridge.start()

    try:
        # Free-running loop: the held action vector is re-applied every step, so the arms keep tracking
        # whatever target was last written while requests are serviced in between.
        while simulation_app.is_running():
            bridge.pump()
            robot.step(1)
    except KeyboardInterrupt:
        print("[mhs] interrupted")
    finally:
        bridge.stop()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
