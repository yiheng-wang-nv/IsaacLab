#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MCP front-end for the MHS bridge.

The Model Hardware Standard exposes the same driver through three doors: an MCP interface, a CLI
and plain code. This module is the first door -- it turns the bridge's read/write primitives into
MCP tools so any agent harness can drive the robot without knowing anything about Isaac Sim.

It runs wherever the agent runs and reaches the simulation over an SSH tunnel:

.. code-block:: bash

    pip install mcp
    ssh -N -L 8765:127.0.0.1:8765 nvidia@<sim-host> &
    claude mcp add isaac-g1 -- python /path/to/scripts/mhs/mcp_server.py

Set ``MHS_URL`` to point at a bridge on a different host or port.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP, Image  # noqa: E402

from mhs_bridge.client import BridgeClient, BridgeError  # noqa: E402

client = BridgeClient(os.environ.get("MHS_URL", "http://127.0.0.1:8765"), timeout=300.0)
mcp = FastMCP("isaac-mhs-bridge")


def _guard(fn, *args, **kwargs) -> str:
    """Run a bridge call and turn failures into readable text instead of protocol errors."""
    try:
        return fn(*args, **kwargs)
    except BridgeError as exc:
        return f"ERROR: {exc}"


@mcp.tool()
def describe_devices() -> str:
    """List every connected device and the channels it exposes, with units, limits and safety notes.

    Call this first: it is the reference file for the hardware and tells you exactly which addresses
    `read_channel` and `write_channel` accept.
    """
    return _guard(client.describe_markdown)


@mcp.tool()
def read_state(device: str = "") -> str:
    """Read the state dictionary -- a snapshot of every cheap readable channel.

    Args:
        device: Optional device name to restrict the snapshot to, e.g. "g1".
    """
    try:
        state = client.state()
    except BridgeError as exc:
        return f"ERROR: {exc}"
    if device:
        state = state.get(device, {})
    return json.dumps(state, indent=2)


@mcp.tool()
def read_channel(path: str) -> str:
    """Read one channel.

    Args:
        path: Channel address as "device.channel", e.g. "g1.right_wrist_pose".
    """
    try:
        return json.dumps(client.read(path), indent=2)
    except BridgeError as exc:
        return f"ERROR: {exc}"


@mcp.tool()
def write_channel(path: str, value: str, settle_steps: int = 30) -> str:
    """Write one channel and optionally let the simulation settle afterwards.

    Args:
        path: Channel address as "device.channel", e.g. "g1.right_wrist_target_pose".
        value: JSON-encoded value, e.g. "[0.25, -0.2, 1.1, 0.5, 0.5, 0.5, 0.5]" or "0.8".
        settle_steps: Environment steps to run after the write so the controller can converge.
    """
    try:
        client.write(path, json.loads(value), settle_steps=settle_steps)
    except json.JSONDecodeError as exc:
        return f"ERROR: 'value' must be JSON: {exc}"
    except BridgeError as exc:
        return f"ERROR: {exc}"
    return f"wrote {path}"


@mcp.tool()
def step_simulation(steps: int = 30) -> str:
    """Advance the simulation while holding the current targets.

    Args:
        steps: Number of environment steps (roughly 1/30 s each).
    """
    result = _guard(client.step, steps)
    return result if isinstance(result, str) else f"stepped {steps}"


@mcp.tool()
def run_program(ops: str) -> str:
    """Run a chain of reads, writes and steps in a single round trip.

    Use this for a whole motion: the timing between the commands is then set by the simulator rather
    than by network latency, which is what makes multi-step manipulation repeatable.

    Args:
        ops: JSON list of operations, e.g.
            '[{"op":"write","path":"g1.right_hand_closure","value":0.0},
              {"op":"step","steps":30},
              {"op":"read","path":"g1.right_wrist_pose"}]'
    """
    try:
        return json.dumps(client.program(json.loads(ops)), indent=2)
    except json.JSONDecodeError as exc:
        return f"ERROR: 'ops' must be JSON: {exc}"
    except BridgeError as exc:
        return f"ERROR: {exc}"


@mcp.tool()
def capture_image(path: str = "front_cam.rgb") -> Image:
    """Capture a camera channel and return it as an image you can look at.

    Args:
        path: Image channel address, e.g. "head_cam.rgb" or "front_cam.rgb".
    """
    return Image(data=client.image(path), format="png")


if __name__ == "__main__":
    mcp.run()
