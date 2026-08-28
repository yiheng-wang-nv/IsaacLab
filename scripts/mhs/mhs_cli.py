#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command line front-end for the MHS bridge.

This is the "operator console" of the three access paths in the Model Hardware Standard: an agent
or a human drives the same read/write primitives that the MCP server exposes, without any extra
runtime. It only needs the python standard library, so it runs from a laptop against an SSH tunnel:

.. code-block:: bash

    ssh -N -L 8765:127.0.0.1:8765 nvidia@10.19.224.59 &

    python scripts/mhs/mhs_cli.py describe
    python scripts/mhs/mhs_cli.py state
    python scripts/mhs/mhs_cli.py read g1.right_wrist_pose
    python scripts/mhs/mhs_cli.py write g1.right_wrist_target_pose '[0.25,-0.2,1.1,0.5,0.5,0.5,0.5]' --settle 60
    python scripts/mhs/mhs_cli.py image head_cam.rgb /tmp/head.png
    python scripts/mhs/mhs_cli.py program motion.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mhs_bridge.client import BridgeClient, BridgeError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Talk to a running MHS bridge.")
    parser.add_argument("--url", default=os.environ.get("MHS_URL", "http://127.0.0.1:8765"), help="Bridge URL.")
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout in seconds.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="Check that the bridge is reachable.")

    p_describe = sub.add_parser("describe", help="Print the reference file for every device.")
    p_describe.add_argument("--json", action="store_true", help="Print raw JSON instead of markdown.")

    p_state = sub.add_parser("state", help="Print the state dictionary.")
    p_state.add_argument("--device", help="Limit the snapshot to one device.")
    p_state.add_argument("--precision", type=int, default=4, help="Round floats to this many decimals.")

    p_read = sub.add_parser("read", help="Read one channel.")
    p_read.add_argument("path", help="Address of the channel, e.g. g1.joint_positions")

    p_write = sub.add_parser("write", help="Write one channel.")
    p_write.add_argument("path", help="Address of the channel, e.g. g1.right_wrist_target_pose")
    p_write.add_argument("value", help="JSON-encoded value, e.g. '[0.2,-0.3,1.0,0.5,0.5,0.5,0.5]'")
    p_write.add_argument("--settle", type=int, default=0, help="Step the simulation N times after writing.")

    p_step = sub.add_parser("step", help="Advance the simulation.")
    p_step.add_argument("steps", type=int, nargs="?", default=1)

    p_image = sub.add_parser("image", help="Save an image channel to a PNG file.")
    p_image.add_argument("path", help="Address of the image channel, e.g. head_cam.rgb")
    p_image.add_argument("out", help="Destination PNG file.")

    p_program = sub.add_parser("program", help="Run a chain of ops from a JSON file (or '-' for stdin).")
    p_program.add_argument("file", help="JSON file holding a list of ops, or an object with an 'ops' key.")

    return parser


def _round(value, precision: int):
    """Round nested floats so a state dump stays readable."""
    if isinstance(value, float):
        return round(value, precision)
    if isinstance(value, list):
        return [_round(v, precision) for v in value]
    if isinstance(value, dict):
        return {k: _round(v, precision) for k, v in value.items()}
    return value


def main() -> int:
    args = build_parser().parse_args()
    client = BridgeClient(args.url, timeout=args.timeout)

    try:
        if args.command == "health":
            print(json.dumps(client.health(), indent=2))
        elif args.command == "describe":
            print(json.dumps(client.describe(), indent=2) if args.json else client.describe_markdown())
        elif args.command == "state":
            state = client.state()
            if args.device:
                state = state.get(args.device, {})
            print(json.dumps(_round(state, args.precision), indent=2))
        elif args.command == "read":
            print(json.dumps(client.read(args.path), indent=2))
        elif args.command == "write":
            client.write(args.path, json.loads(args.value), settle_steps=args.settle)
            print(f"wrote {args.path}")
        elif args.command == "step":
            client.step(args.steps)
            print(f"stepped {args.steps}")
        elif args.command == "image":
            print(client.save_image(args.path, args.out))
        elif args.command == "program":
            raw = sys.stdin.read() if args.file == "-" else open(args.file).read()
            parsed = json.loads(raw)
            ops = parsed["ops"] if isinstance(parsed, dict) else parsed
            print(json.dumps(client.program(ops), indent=2))
    except BridgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
