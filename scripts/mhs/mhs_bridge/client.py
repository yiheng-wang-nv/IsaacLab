# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standard-library client for a running :class:`~mhs_bridge.server.Bridge`.

The client has no dependencies so it runs anywhere: on the simulation machine, on a laptop through
an SSH tunnel, or inside an MCP server process.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class BridgeError(RuntimeError):
    """Raised when the bridge reports a failed read/write."""


class BridgeClient:
    """Thin HTTP client speaking the bridge's read/write protocol."""

    def __init__(self, url: str = "http://127.0.0.1:8765", timeout: float = 120.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    # -- transport ----------------------------------------------------------------------------

    def _request(self, method: str, path: str, payload: Any = None, raw: bool = False) -> Any:
        data = None if payload is None else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"} if data else {}
        request = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()
            try:
                message = json.loads(body).get("error", body.decode(errors="replace"))
            except json.JSONDecodeError:
                message = body.decode(errors="replace")
            raise BridgeError(message) from None
        except urllib.error.URLError as exc:
            raise BridgeError(f"cannot reach the bridge at {self.url}: {exc.reason}") from None
        if raw:
            return body
        return json.loads(body)

    # -- primitives ---------------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Check that the bridge is up."""
        return self._request("GET", "/health")

    def describe(self) -> dict[str, Any]:
        """Fetch the reference file describing every device and channel."""
        return self._request("GET", "/describe")["reference"]

    def describe_markdown(self) -> str:
        """Fetch the reference file rendered as markdown."""
        return self._request("GET", "/describe.md", raw=True).decode()

    def state(self) -> dict[str, Any]:
        """Fetch the state dictionary."""
        return self._request("GET", "/state")["state"]

    def read(self, path: str) -> Any:
        """Read ``device.channel``."""
        return self._request("POST", "/read", {"path": path})["value"]

    def write(self, path: str, value: Any, settle_steps: int = 0) -> None:
        """Write ``device.channel``, optionally stepping the sim afterwards to let it settle."""
        self._request("POST", "/write", {"path": path, "value": value, "settle_steps": settle_steps})

    def step(self, steps: int = 1) -> None:
        """Advance the simulation."""
        self._request("POST", "/step", {"steps": steps})

    def program(self, ops: list[dict[str, Any]], timeout_s: float | None = None) -> list[Any]:
        """Run a chain of ops in one round trip; returns the result of each ``read``/``step``."""
        payload: dict[str, Any] = {"ops": ops}
        if timeout_s is not None:
            payload["timeout_s"] = timeout_s
        return self._request("POST", "/program", payload)["results"]

    def image(self, path: str) -> bytes:
        """Fetch an image channel as PNG bytes."""
        query = urllib.parse.urlencode({"path": path})
        return self._request("GET", f"/image?{query}", raw=True)

    def save_image(self, path: str, filename: str) -> str:
        """Fetch an image channel and write it to ``filename``."""
        with open(filename, "wb") as handle:
            handle.write(self.image(path))
        return filename
