# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Core abstractions for the MHS-style hardware bridge.

The design follows the primitives described by Anthropic's Model Hardware Standard research
preview: every device is described *once* with rich metadata, and every interaction with it is
either a **read** ("get me a value") or a **write** ("set this value"). Everything else -- the
state dictionary, the auto-generated reference file and the MCP/CLI front-ends -- is derived from
that single description.

Nothing in this module imports Isaac Sim, so it can be unit-tested and reused for a real robot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

# Access modes for a channel.
READ = "read"
WRITE = "write"
READWRITE = "readwrite"


@dataclass
class Channel:
    """A single readable and/or writable quantity exposed by a device.

    The metadata carried here is what an agent sees *before* it touches the hardware. It is
    deliberately verbose: units, physical limits and free-form safety notes are the only thing
    standing between a language model and a real actuator.
    """

    name: str
    """Channel name, unique within its device (e.g. ``joint_positions``)."""

    access: str
    """One of :data:`READ`, :data:`WRITE` or :data:`READWRITE`."""

    description: str
    """Natural-language description of the quantity, written for a model to read."""

    dtype: str = "float32"
    """Element type of the value (``float32``, ``int32``, ``bool``, ``uint8``, ``str``)."""

    shape: tuple[int, ...] | None = None
    """Shape of the value. ``None`` means scalar, ``-1`` marks a dynamic dimension."""

    unit: str = "dimensionless"
    """Physical unit, spelled out (``rad``, ``m``, ``m/s``, ``N``, ``normalized``)."""

    limits: dict[str, Any] | None = None
    """Optional ``{"min": ..., "max": ...}``. Writes outside the range are rejected."""

    safety: str | None = None
    """Free-form note describing what can go wrong when writing this channel."""

    labels: list[str] | None = None
    """Optional per-element names (e.g. joint names) so an agent can index by meaning."""

    in_state: bool = True
    """Whether the channel is cheap enough to include in the polled state dictionary."""

    getter: Callable[[], Any] | None = field(default=None, repr=False)
    setter: Callable[[Any], None] | None = field(default=None, repr=False)

    def reference(self) -> dict[str, Any]:
        """Serializable description of the channel, used to build the reference file."""
        ref = {
            "name": self.name,
            "access": self.access,
            "description": self.description,
            "dtype": self.dtype,
            "shape": list(self.shape) if self.shape is not None else None,
            "unit": self.unit,
        }
        if self.limits is not None:
            ref["limits"] = self.limits
        if self.safety is not None:
            ref["safety"] = self.safety
        if self.labels is not None:
            ref["labels"] = self.labels
        return ref

    def validate(self, value: Any) -> Any:
        """Check a value against the declared shape and limits, returning it unchanged.

        Raises:
            ValueError: if the value has the wrong length or falls outside ``limits``.
        """
        if self.access == READ:
            raise ValueError(f"channel '{self.name}' is read-only")
        flat = _flatten(value)
        if self.shape is not None:
            expected = 1
            for dim in self.shape:
                expected *= dim
            if expected > 0 and len(flat) != expected:
                raise ValueError(
                    f"channel '{self.name}' expects {expected} element(s) with shape {list(self.shape)},"
                    f" got {len(flat)}"
                )
        if self.limits is not None:
            lo, hi = self.limits.get("min"), self.limits.get("max")
            for i, item in enumerate(flat):
                if not isinstance(item, (int, float)):
                    continue
                lo_i = _limit_at(lo, i)
                hi_i = _limit_at(hi, i)
                if lo_i is not None and item < lo_i:
                    raise ValueError(
                        f"channel '{self.name}' element {i} = {item} is below the safe minimum {lo_i}"
                        + (f" ({self.safety})" if self.safety else "")
                    )
                if hi_i is not None and item > hi_i:
                    raise ValueError(
                        f"channel '{self.name}' element {i} = {item} is above the safe maximum {hi_i}"
                        + (f" ({self.safety})" if self.safety else "")
                    )
        return value


def _flatten(value: Any) -> list[Any]:
    """Flatten nested lists/tuples into a flat python list."""
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]


def _limit_at(limit: Any, index: int) -> float | None:
    """Resolve a limit that may be a scalar or a per-element sequence."""
    if limit is None:
        return None
    if isinstance(limit, (list, tuple)):
        return limit[index] if index < len(limit) else None
    return limit


class Device:
    """A collection of channels backed by one physical or simulated piece of hardware."""

    def __init__(self, name: str, description: str, vendor: str = "simulated"):
        self.name = name
        self.description = description
        self.vendor = vendor
        self.channels: dict[str, Channel] = {}

    def add(self, channel: Channel) -> Channel:
        """Register a channel on this device."""
        if channel.name in self.channels:
            raise ValueError(f"device '{self.name}' already has a channel named '{channel.name}'")
        self.channels[channel.name] = channel
        return channel

    def read(self, channel: str) -> Any:
        """Read one channel."""
        ch = self._channel(channel)
        if ch.getter is None:
            raise ValueError(f"channel '{self.name}.{channel}' is not readable")
        return ch.getter()

    def write(self, channel: str, value: Any) -> None:
        """Validate and write one channel."""
        ch = self._channel(channel)
        if ch.setter is None:
            raise ValueError(f"channel '{self.name}.{channel}' is not writable")
        ch.setter(ch.validate(value))

    def state(self) -> dict[str, Any]:
        """Snapshot of every cheap readable channel on this device."""
        snap: dict[str, Any] = {}
        for name, ch in self.channels.items():
            if not ch.in_state or ch.getter is None:
                continue
            try:
                snap[name] = ch.getter()
            except Exception as exc:  # a broken channel must not take down the whole snapshot
                snap[name] = {"error": str(exc)}
        return snap

    def reference(self) -> dict[str, Any]:
        """Serializable description of the device and all of its channels."""
        return {
            "name": self.name,
            "description": self.description,
            "vendor": self.vendor,
            "channels": [ch.reference() for ch in self.channels.values()],
        }

    def _channel(self, channel: str) -> Channel:
        if channel not in self.channels:
            known = ", ".join(sorted(self.channels)) or "<none>"
            raise KeyError(f"device '{self.name}' has no channel '{channel}'. Known channels: {known}")
        return self.channels[channel]


class Registry:
    """All devices reachable through one bridge, addressed as ``device.channel``."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.devices: dict[str, Device] = {}
        self.started_at = time.time()

    def add(self, device: Device) -> Device:
        """Register a device."""
        if device.name in self.devices:
            raise ValueError(f"device '{device.name}' is already registered")
        self.devices[device.name] = device
        return device

    def resolve(self, path: str) -> tuple[Device, str]:
        """Split a ``device.channel`` address into its parts."""
        if "." not in path:
            raise KeyError(f"'{path}' is not a valid address; expected 'device.channel'")
        device_name, channel = path.split(".", 1)
        if device_name not in self.devices:
            known = ", ".join(sorted(self.devices)) or "<none>"
            raise KeyError(f"no device named '{device_name}'. Known devices: {known}")
        return self.devices[device_name], channel

    def read(self, path: str) -> Any:
        """Read ``device.channel``."""
        device, channel = self.resolve(path)
        return device.read(channel)

    def write(self, path: str, value: Any) -> None:
        """Write ``device.channel``."""
        device, channel = self.resolve(path)
        device.write(channel, value)

    def state(self) -> dict[str, Any]:
        """The state dictionary: one nested snapshot per device."""
        return {name: device.state() for name, device in self.devices.items()}

    def reference(self) -> dict[str, Any]:
        """The reference file: everything an agent needs to know before acting."""
        return {
            "bridge": self.name,
            "description": self.description,
            "devices": [device.reference() for device in self.devices.values()],
        }

    def reference_markdown(self) -> str:
        """Human/model-readable rendering of :meth:`reference`."""
        lines = [f"# {self.name}", "", self.description, ""]
        for device in self.devices.values():
            lines += [f"## Device `{device.name}`", "", device.description, ""]
            for ch in device.channels.values():
                shape = "scalar" if ch.shape is None else "x".join(str(d) for d in ch.shape)
                lines.append(f"### `{device.name}.{ch.name}`  ({ch.access})")
                lines.append("")
                lines.append(f"- {ch.description}")
                lines.append(f"- type: `{ch.dtype}` shape: `{shape}` unit: `{ch.unit}`")
                if ch.limits:
                    lines.append(f"- limits: `{ch.limits}`")
                if ch.labels:
                    lines.append(f"- labels: `{ch.labels}`")
                if ch.safety:
                    lines.append(f"- **safety**: {ch.safety}")
                lines.append("")
        return "\n".join(lines)
