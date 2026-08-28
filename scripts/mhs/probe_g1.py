# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dump the joint/body layout of the fixed-base Unitree G1 pick-and-place environment.

The MHS device description needs the exact joint names, joint limits and link names of the robot
so that every channel can be labelled and bounded. Run this once and keep the JSON around:

.. code-block:: bash

    ./isaaclab.sh -p scripts/mhs/probe_g1.py --out /tmp/g1_layout.json
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe the G1 upper-body environment layout.")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-G1-InspireFTP-Abs-v0", help="Task id to probe.")
parser.add_argument("--out", type=str, default="/tmp/g1_layout.json", help="Where to write the layout JSON.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# `--headless` stays under the caller's control: on this workstation the GUI is the point.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json

import gymnasium as gym

import isaaclab_tasks  # noqa: F401  -- registers the task ids

# `isaaclab_tasks` swallows import errors while auto-registering; import the package we care about
# explicitly so a missing dependency surfaces as a traceback instead of a "task doesn't exist".
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    action_term = unwrapped.action_manager.get_term(unwrapped.action_manager.active_terms[0])

    layout = {
        "task": args_cli.task,
        "action_dim": int(unwrapped.action_manager.total_action_dim),
        "action_term": unwrapped.action_manager.active_terms[0],
        "num_joints": len(robot.data.joint_names),
        "joint_names": list(robot.data.joint_names),
        "joint_limits": robot.data.joint_pos_limits[0].cpu().tolist(),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "body_names": list(robot.data.body_names),
        "controlled_joint_names": list(getattr(action_term, "_controlled_joint_names", [])),
        "hand_joint_names": list(getattr(action_term, "_hand_joint_names", [])),
        "pink_joint_names": list(getattr(action_term, "_isaaclab_controlled_joint_names", [])),
        "scene_entities": list(unwrapped.scene.keys()) if hasattr(unwrapped.scene, "keys") else [],
        "physics_dt": float(unwrapped.physics_dt),
        "step_dt": float(unwrapped.step_dt),
    }

    with open(args_cli.out, "w") as handle:
        json.dump(layout, handle, indent=2)
    print(f"[probe] wrote {args_cli.out}")
    print(json.dumps({k: v for k, v in layout.items() if k not in ("joint_limits", "default_joint_pos")}, indent=2))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
