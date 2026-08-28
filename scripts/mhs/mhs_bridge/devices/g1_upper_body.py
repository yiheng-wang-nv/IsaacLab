# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MHS devices for the fixed-base Unitree G1 upper body.

The underlying environment is ``Isaac-PickPlace-G1-InspireFTP-Abs-v0``: a 29-DoF G1 with Inspire
hands whose root link is fixed and whose gravity is disabled, so the legs stay put and only the
torso, arms and hands move. Its action term is a Pink IK controller, which means the natural
"write" primitive is a Cartesian wrist pose rather than a joint vector -- exactly the level an
agent wants to work at.

The 38-dimensional action vector is laid out as::

    [ 0: 3]  left wrist target position   (x, y, z) in the env-origin frame
    [ 3: 7]  left wrist target orientation (w, x, y, z)
    [ 7:10]  right wrist target position
    [10:14]  right wrist target orientation
    [14:38]  24 Inspire hand joint targets

The bridge keeps that vector as persistent state: a write updates one slice of it, and every
simulation step re-applies the whole vector. So an agent sets a target and the robot holds it,
which is how a real motion controller behaves.
"""

from __future__ import annotations

import torch

from ..core import READ, READWRITE, WRITE, Channel, Device, Registry

# Slices of the Pink IK action vector.
LEFT_POSE = slice(0, 7)
RIGHT_POSE = slice(7, 14)
HANDS = slice(14, 38)

# Conservative Cartesian box for wrist targets, in the environment-origin frame. The G1 pelvis sits
# at z = 1.0 m, so this keeps the wrists within roughly arm's reach and above the floor.
WRIST_POSITION_MIN = [-0.9, -0.9, 0.3]
WRIST_POSITION_MAX = [0.9, 0.9, 1.9]


class G1UpperBody:
    """Exposes one running Isaac Lab environment as a set of MHS devices."""

    def __init__(self, env, camera_names: tuple[str, ...] = ()):
        """Wrap an already-created environment.

        Args:
            env: The ``gym.make``-created environment (wrapped or unwrapped).
            camera_names: Names of :class:`~isaaclab.sensors.Camera` entities present in the scene.
                Each becomes its own MHS device.
        """
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.robot = self.env.scene["robot"]
        self.device = self.env.device
        self.camera_names = camera_names

        self.action_term = self.env.action_manager.get_term(self.env.action_manager.active_terms[0])
        self.action_dim = int(self.env.action_manager.total_action_dim)
        if self.action_dim != 38:
            raise RuntimeError(f"expected a 38-dim Pink IK action vector, got {self.action_dim}")

        self.joint_names: list[str] = list(self.robot.data.joint_names)
        self.hand_joint_names: list[str] = list(self.action_term._hand_joint_names)
        self.arm_joint_names: list[str] = list(self.action_term._isaaclab_controlled_joint_names)
        self.hand_joint_ids = [self.joint_names.index(name) for name in self.hand_joint_names]
        self.arm_joint_ids = [self.joint_names.index(name) for name in self.arm_joint_names]
        self.left_hand_slots = [i for i, name in enumerate(self.hand_joint_names) if name.startswith("L_")]
        self.right_hand_slots = [i for i, name in enumerate(self.hand_joint_names) if name.startswith("R_")]

        limits = self.robot.data.joint_pos_limits[0]
        self.hand_lower = limits[self.hand_joint_ids, 0].cpu().tolist()
        self.hand_upper = limits[self.hand_joint_ids, 1].cpu().tolist()

        self.left_link = self.env.cfg.actions.pink_ik_cfg.target_eef_link_names["left_wrist"]
        self.right_link = self.env.cfg.actions.pink_ik_cfg.target_eef_link_names["right_wrist"]

        self.action = torch.zeros(self.env.num_envs, self.action_dim, device=self.device)
        self.step_count = 0
        self.last_terminated = False
        self.last_truncated = False
        self.sync_targets_to_measured()

    # -- simulation ------------------------------------------------------------------------------

    def sync_targets_to_measured(self) -> None:
        """Snap the held targets onto the measured state so nothing jumps after a reset."""
        self.action[:, LEFT_POSE] = self._link_pose(self.left_link)
        self.action[:, RIGHT_POSE] = self._link_pose(self.right_link)
        self.action[:, HANDS] = self.robot.data.joint_pos[:, self.hand_joint_ids]

    def step(self, steps: int = 1) -> None:
        """Re-apply the held action vector for ``steps`` environment steps."""
        for _ in range(max(1, steps)):
            _, _, terminated, truncated, _ = self.env.step(self.action)
            self.last_terminated = bool(terminated[0])
            self.last_truncated = bool(truncated[0])
            self.step_count += 1

    def reset(self) -> None:
        """Reset the scene and re-seed the held targets."""
        self.env.reset()
        self.step_count = 0
        self.last_terminated = False
        self.last_truncated = False
        self.sync_targets_to_measured()

    # -- helpers ---------------------------------------------------------------------------------

    def _link_pose(self, link_name: str) -> torch.Tensor:
        """Pose of a robot link as ``(x, y, z, qw, qx, qy, qz)`` in the env-origin frame."""
        index = self.robot.data.body_names.index(link_name)
        position = self.robot.data.body_pos_w[:, index] - self.env.scene.env_origins
        quaternion = self.robot.data.body_quat_w[:, index]
        return torch.cat([position, quaternion], dim=-1)

    def _write_pose(self, target: slice, value: list[float]) -> None:
        self.action[:, target] = torch.tensor(value, dtype=torch.float32, device=self.device)

    def _write_closure(self, slots: list[int], closure: float) -> None:
        """Map a scalar in ``[0, 1]`` onto each finger joint's own travel range."""
        for slot in slots:
            lower, upper = self.hand_lower[slot], self.hand_upper[slot]
            self.action[:, HANDS.start + slot] = lower + closure * (upper - lower)

    def _read_closure(self, slots: list[int]) -> float:
        """Average normalized position of the commanded finger joints."""
        total = 0.0
        for slot in slots:
            lower, upper = self.hand_lower[slot], self.hand_upper[slot]
            span = upper - lower
            total += (float(self.action[0, HANDS.start + slot]) - lower) / span if span > 1e-6 else 0.0
        return total / max(1, len(slots))

    # -- device description ----------------------------------------------------------------------

    def register(self, registry: Registry) -> Registry:
        """Describe the robot, the workcell and the cameras as MHS devices."""
        registry.add(self._robot_device())
        registry.add(self._workcell_device())
        for name in self.camera_names:
            registry.add(self._camera_device(name))
        return registry

    def _robot_device(self) -> Device:
        device = Device(
            name="g1",
            vendor="Unitree G1 29-DoF + Inspire FTP hands (simulated in Isaac Sim)",
            description=(
                "Fixed-base humanoid upper body. The pelvis is welded in place and gravity is off for"
                " the robot, so the legs never move; the arms are driven by a Pink inverse-kinematics"
                " controller that tracks Cartesian wrist targets, and the two 12-joint Inspire hands are"
                " driven in joint space. Positions are expressed in the environment-origin frame and"
                " orientations as (w, x, y, z) quaternions."
            ),
        )

        device.add(
            Channel(
                name="joint_names",
                access=READ,
                description="Names of all robot joints, in the order used by joint_positions.",
                dtype="str",
                shape=(len(self.joint_names),),
                in_state=False,
                getter=lambda: self.joint_names,
            )
        )
        device.add(
            Channel(
                name="joint_positions",
                access=READ,
                description="Measured position of every joint, including the frozen leg joints.",
                shape=(len(self.joint_names),),
                unit="rad",
                labels=self.joint_names,
                getter=lambda: self.robot.data.joint_pos[0],
            )
        )
        device.add(
            Channel(
                name="joint_velocities",
                access=READ,
                description="Measured velocity of every joint.",
                shape=(len(self.joint_names),),
                unit="rad/s",
                labels=self.joint_names,
                in_state=False,
                getter=lambda: self.robot.data.joint_vel[0],
            )
        )
        device.add(
            Channel(
                name="arm_joint_positions",
                access=READ,
                description="Measured position of the 14 arm joints the IK controller drives.",
                shape=(len(self.arm_joint_names),),
                unit="rad",
                labels=self.arm_joint_names,
                getter=lambda: self.robot.data.joint_pos[0, self.arm_joint_ids],
            )
        )
        device.add(
            Channel(
                name="hand_joint_positions",
                access=READ,
                description="Measured position of the 24 commanded Inspire hand joints.",
                shape=(len(self.hand_joint_names),),
                unit="rad",
                labels=self.hand_joint_names,
                getter=lambda: self.robot.data.joint_pos[0, self.hand_joint_ids],
            )
        )
        device.add(
            Channel(
                name="left_wrist_pose",
                access=READ,
                description=f"Measured pose of '{self.left_link}' as (x, y, z, qw, qx, qy, qz).",
                shape=(7,),
                unit="m and unit quaternion",
                getter=lambda: self._link_pose(self.left_link)[0],
            )
        )
        device.add(
            Channel(
                name="right_wrist_pose",
                access=READ,
                description=f"Measured pose of '{self.right_link}' as (x, y, z, qw, qx, qy, qz).",
                shape=(7,),
                unit="m and unit quaternion",
                getter=lambda: self._link_pose(self.right_link)[0],
            )
        )

        pose_limits = {
            "min": WRIST_POSITION_MIN + [-1.0] * 4,
            "max": WRIST_POSITION_MAX + [1.0] * 4,
        }
        pose_safety = (
            "The IK solver tracks this target continuously and will drive the arm at full speed towards"
            " it. Move in steps of at most ~10 cm and step the simulation between writes; a large jump"
            " makes the solver saturate joint limits and can fling the held object."
        )
        device.add(
            Channel(
                name="left_wrist_target_pose",
                access=READWRITE,
                description=(
                    "Cartesian target for the left wrist as (x, y, z, qw, qx, qy, qz). Reading it returns"
                    " the currently held command, not the measured pose."
                ),
                shape=(7,),
                unit="m and unit quaternion",
                limits=pose_limits,
                safety=pose_safety,
                getter=lambda: self.action[0, LEFT_POSE],
                setter=lambda value: self._write_pose(LEFT_POSE, value),
            )
        )
        device.add(
            Channel(
                name="right_wrist_target_pose",
                access=READWRITE,
                description=(
                    "Cartesian target for the right wrist as (x, y, z, qw, qx, qy, qz). Reading it returns"
                    " the currently held command, not the measured pose."
                ),
                shape=(7,),
                unit="m and unit quaternion",
                limits=pose_limits,
                safety=pose_safety,
                getter=lambda: self.action[0, RIGHT_POSE],
                setter=lambda value: self._write_pose(RIGHT_POSE, value),
            )
        )
        device.add(
            Channel(
                name="hand_joint_targets",
                access=READWRITE,
                description="Raw position targets for the 24 commanded Inspire hand joints.",
                shape=(len(self.hand_joint_names),),
                unit="rad",
                labels=self.hand_joint_names,
                limits={"min": self.hand_lower, "max": self.hand_upper},
                safety="Values are clamped to the URDF joint limits; a full close on an object squeezes it.",
                getter=lambda: self.action[0, HANDS],
                setter=lambda value: self.action[0, HANDS].copy_(
                    torch.tensor(value, dtype=torch.float32, device=self.device)
                ),
            )
        )
        for side, slots in (("left", self.left_hand_slots), ("right", self.right_hand_slots)):
            device.add(
                Channel(
                    name=f"{side}_hand_closure",
                    access=READWRITE,
                    description=(
                        f"Convenience scalar driving all {len(slots)} {side}-hand joints together."
                        " 0.0 puts every finger joint at its lower limit, 1.0 at its upper limit."
                    ),
                    shape=None,
                    unit="normalized",
                    limits={"min": 0.0, "max": 1.0},
                    safety="Closing on a rigid object at 1.0 applies the full grip effort.",
                    getter=(lambda s=slots: self._read_closure(s)),
                    setter=(lambda value, s=slots: self._write_closure(s, float(value))),
                )
            )
        device.add(
            Channel(
                name="reset",
                access=WRITE,
                description="Write any value to reset the scene and re-seed the held targets.",
                dtype="bool",
                shape=None,
                safety="Teleports the robot and the object back to their initial poses.",
                setter=lambda _value: self.reset(),
            )
        )
        device.add(
            Channel(
                name="step_count",
                access=READ,
                description="Environment steps taken since the last reset.",
                dtype="int32",
                shape=None,
                unit="steps",
                getter=lambda: self.step_count,
            )
        )
        return device

    def _workcell_device(self) -> Device:
        device = Device(
            name="workcell",
            vendor="Isaac Sim scene",
            description="The table, the manipulated object and the task outcome flags.",
        )
        obj = self.env.scene["object"]
        device.add(
            Channel(
                name="object_pose",
                access=READ,
                description="Pose of the manipulated object as (x, y, z, qw, qx, qy, qz).",
                shape=(7,),
                unit="m and unit quaternion",
                getter=lambda: torch.cat(
                    [obj.data.root_pos_w[0] - self.env.scene.env_origins[0], obj.data.root_quat_w[0]]
                ),
            )
        )
        device.add(
            Channel(
                name="task_succeeded",
                access=READ,
                description="True when the environment's success termination fired on the last step.",
                dtype="bool",
                shape=None,
                getter=lambda: self.last_terminated,
            )
        )
        device.add(
            Channel(
                name="episode_timed_out",
                access=READ,
                description="True when the episode length was reached on the last step.",
                dtype="bool",
                shape=None,
                getter=lambda: self.last_truncated,
            )
        )
        return device

    def _camera_device(self, name: str) -> Device:
        camera = self.env.scene[name]
        device = Device(
            name=name,
            vendor="Isaac Sim pinhole camera",
            description=f"RGB-D camera '{name}' rendered by Isaac Sim.",
        )
        device.add(
            Channel(
                name="rgb",
                access=READ,
                description="Colour image. Fetch it as a PNG through the /image route rather than as JSON.",
                dtype="uint8",
                shape=(-1, -1, 3),
                unit="8-bit sRGB",
                in_state=False,
                getter=lambda: camera.data.output["rgb"][0],
            )
        )
        if "distance_to_image_plane" in camera.cfg.data_types:
            device.add(
                Channel(
                    name="depth",
                    access=READ,
                    description="Per-pixel distance to the image plane; +inf where nothing was hit.",
                    dtype="float32",
                    shape=(-1, -1),
                    unit="m",
                    in_state=False,
                    getter=lambda: camera.data.output["distance_to_image_plane"][0],
                )
            )
        device.add(
            Channel(
                name="resolution",
                access=READ,
                description="Image size as (width, height).",
                dtype="int32",
                shape=(2,),
                unit="pixels",
                getter=lambda: [int(camera.image_shape[1]), int(camera.image_shape[0])],
            )
        )
        device.add(
            Channel(
                name="intrinsics",
                access=READ,
                description="3x3 pinhole intrinsic matrix.",
                shape=(3, 3),
                unit="pixels",
                in_state=False,
                getter=lambda: camera.data.intrinsic_matrices[0],
            )
        )
        device.add(
            Channel(
                name="pose",
                access=READ,
                description="Camera pose in the world frame as (x, y, z, qw, qx, qy, qz).",
                shape=(7,),
                unit="m and unit quaternion",
                getter=lambda: torch.cat([camera.data.pos_w[0], camera.data.quat_w_world[0]]),
            )
        )
        return device
