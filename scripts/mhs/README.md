# MHS bridge for Isaac Sim

An implementation of the read/write device pattern from Anthropic's
[Model Hardware Standard](https://www.anthropic.com/news/model-hardware-standard-research-preview)
research preview, applied to a fixed-base Unitree G1 upper body running in Isaac Sim.

MHS itself is not public yet — it is a research preview you have to apply for, with no SDK, no
schema and no repository published. What *is* public is its shape, and that part is reproducible:

| MHS concept | Here |
| --- | --- |
| Standardized driver with `read` / `write` primitives | `mhs_bridge/core.py` — `Channel`, `Device`, `Registry` |
| Device described once, with metadata tags for units, limits and safety | `Channel` fields, rendered by `Registry.reference()` |
| State dictionary of current device conditions | `Registry.state()`, served at `GET /state` |
| Auto-generated reference file | `Registry.reference_markdown()`, served at `GET /describe.md` |
| MCP interface | `mcp_server.py` |
| Command line for direct operator control | `mhs_cli.py` |
| Chaining driver commands in a code file for long-running work | `POST /program` |

Swapping the simulated G1 for a real one means writing a new module next to
`mhs_bridge/devices/g1_upper_body.py`. Nothing above the device layer changes.

## Architecture

```
   your machine                             the GPU workstation
┌──────────────────┐                   ┌──────────────────────────────────┐
│  agent / MCP     │                   │  Isaac Sim process               │
│  mcp_server.py   │   HTTP over an    │  ┌────────────────────────────┐  │
│       or         │───SSH tunnel────▶ │  │ Bridge (background thread) │  │
│  mhs_cli.py      │   :8765           │  └────────────┬───────────────┘  │
└──────────────────┘                   │      queue    │ futures          │
                                       │  ┌────────────▼───────────────┐  │
                                       │  │ simulation loop (main)     │  │
                                       │  │  pump() → env.step()       │  │
                                       │  └────────────┬───────────────┘  │
                                       │      G1 articulation + cameras   │
                                       └──────────────────────────────────┘
```

Isaac Sim owns the main thread, so the HTTP server never touches the simulation directly. It
queues a closure and blocks; the simulation loop drains the queue between two physics steps and
fulfils the future. Every read is therefore consistent with a single simulation state, and every
write lands at a well-defined point in the step.

The action vector is held state, not a one-shot command: the loop re-applies it every step, so a
written wrist target keeps being tracked by the IK controller until it is overwritten. That is how
a real motion controller behaves, and it is what makes `write` a meaningful primitive.

## Running it

On the workstation (the GUI window opens on the attached monitor):

```bash
cd /home/nvidia/workspace/yiheng/IsaacLab
conda activate isaaclab_develop_6.0
DISPLAY=:1 python scripts/mhs/run_g1_bridge.py --port 8765
```

From anywhere else:

```bash
ssh -N -L 8765:127.0.0.1:8765 nvidia@<host> &

python scripts/mhs/mhs_cli.py describe                       # the reference file
python scripts/mhs/mhs_cli.py state --device g1               # the state dictionary
python scripts/mhs/mhs_cli.py read g1.right_wrist_pose
python scripts/mhs/mhs_cli.py write g1.right_hand_closure 0.9 --settle 40
python scripts/mhs/mhs_cli.py image head_cam.rgb /tmp/head.png
python scripts/mhs/mhs_cli.py program scripts/mhs/programs/wave.json
```

As an MCP server:

```bash
pip install mcp
claude mcp add isaac-g1 -- python $PWD/scripts/mhs/mcp_server.py
```

## The robot

`Isaac-PickPlace-G1-InspireFTP-Abs-v0` is a 29-DoF Unitree G1 with Inspire hands, spawned with
`fix_root_link=True` and gravity disabled, so the lower body is frozen and only the torso, arms and
hands move. Its action term is a Pink IK controller, so the write primitives are Cartesian wrist
poses plus hand joint targets rather than raw joint commands.

Two cameras are added by this script and are not part of the stock task: `head_cam`, mounted on the
robot, and `front_cam`, a fixed third-person view of the table.

## Files

| File | Role |
| --- | --- |
| `mhs_bridge/core.py` | Channel/device/registry abstractions. No Isaac Sim import. |
| `mhs_bridge/server.py` | HTTP transport, sim-thread dispatch, PNG encoding. |
| `mhs_bridge/client.py` | Standard-library client. |
| `mhs_bridge/devices/g1_upper_body.py` | The G1 device description. |
| `run_g1_bridge.py` | Device server: launches Isaac Sim and serves the bridge. |
| `mhs_cli.py` | Operator console. |
| `mcp_server.py` | MCP front-end. |
| `probe_g1.py` | Dumps the robot's joint/link layout. |
| `sync.sh` | rsync this directory to the simulation machine. |
