# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HTTP transport for the MHS bridge, designed to live inside the Isaac Sim process.

Isaac Sim owns the main thread: physics, rendering and every USD/PhysX read has to happen on it.
The server therefore runs on a background thread and does no simulation work itself. It pushes
closures onto a queue; the simulation loop drains that queue once per step via :meth:`Bridge.pump`
and fulfils the futures. From the caller's point of view a ``read``/``write`` is a blocking HTTP
request; from the simulator's point of view it is a callback that happens at a safe point between
two physics steps.

Only the python standard library is used, so the server adds no dependency to the Isaac Sim
environment.
"""

from __future__ import annotations

import json
import queue
import struct
import threading
import time
import traceback
import zlib
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from .core import Registry

# A request waits this long for the simulation thread to service it before giving up.
DEFAULT_TIMEOUT_S = 60.0


class Bridge:
    """Owns the registry, the command queue and the HTTP server."""

    def __init__(
        self,
        registry: Registry,
        host: str = "127.0.0.1",
        port: int = 8765,
        step_fn: Callable[[int], None] | None = None,
    ):
        """Initialize the bridge.

        Args:
            registry: The devices to expose.
            host: Interface to bind. Keep it on loopback and reach it over an SSH tunnel.
            port: TCP port to bind.
            step_fn: Callable advancing the simulation by ``n`` steps. Used by the ``step``
                operation so an agent can let the scene settle before reading it back.
        """
        self.registry = registry
        self.host = host
        self.port = port
        self.step_fn = step_fn
        self._queue: queue.Queue[tuple[Callable[[], Any], Future]] = queue.Queue()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.request_count = 0

    # -- lifecycle ----------------------------------------------------------------------------

    def start(self) -> None:
        """Start serving on a daemon thread."""
        handler = _make_handler(self)
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True, name="mhs-http")
        self._thread.start()
        print(f"[mhs] bridge listening on http://{self.host}:{self.port}")
        print(f"[mhs] devices: {', '.join(sorted(self.registry.devices)) or '<none>'}")

    def stop(self) -> None:
        """Stop serving."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    # -- simulation-thread side ---------------------------------------------------------------

    def submit(self, fn: Callable[[], Any], timeout: float = DEFAULT_TIMEOUT_S) -> Any:
        """Queue ``fn`` for the simulation thread and block until it has run."""
        future: Future = Future()
        self._queue.put((fn, future))
        return future.result(timeout=timeout)

    def pump(self, max_ops: int = 64) -> int:
        """Run queued work on the calling (simulation) thread. Returns the number of ops run."""
        ran = 0
        while ran < max_ops:
            try:
                fn, future = self._queue.get_nowait()
            except queue.Empty:
                break
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(fn())
                except Exception as exc:  # surface the traceback to the caller, keep the sim alive
                    future.set_exception(exc)
            ran += 1
        return ran

    # -- operations ---------------------------------------------------------------------------

    def run_program(self, ops: list[dict[str, Any]]) -> list[Any]:
        """Execute a list of operations back-to-back on the simulation thread.

        This is the "chain driver commands" path: one round trip carries a whole motion, so the
        timing between the writes is set by the simulator rather than by network latency.

        Supported ops: ``{"op": "read", "path": ...}``, ``{"op": "write", "path": ..., "value": ...}``,
        ``{"op": "step", "steps": n}``, ``{"op": "sleep", "seconds": s}``.
        """
        results = []
        for index, op in enumerate(ops):
            kind = op.get("op")
            try:
                if kind == "read":
                    results.append(_jsonable(self.registry.read(op["path"])))
                elif kind == "write":
                    self.registry.write(op["path"], op["value"])
                    results.append(None)
                elif kind == "step":
                    steps = int(op.get("steps", 1))
                    if self.step_fn is None:
                        raise RuntimeError("this bridge was started without a step function")
                    self.step_fn(steps)
                    results.append(steps)
                elif kind == "sleep":
                    time.sleep(float(op.get("seconds", 0.0)))
                    results.append(None)
                else:
                    raise ValueError(f"unknown op '{kind}'")
            except Exception as exc:
                raise RuntimeError(f"op {index} ({kind}) failed: {exc}") from exc
        return results


def _jsonable(value: Any) -> Any:
    """Convert torch tensors / numpy arrays into plain JSON-serializable python."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    tolist = getattr(value, "tolist", None)
    if tolist is not None:
        detach = getattr(value, "detach", None)
        if detach is not None:
            value = detach().cpu()
            tolist = value.tolist
        return tolist()
    return str(value)


def encode_png(rgb: Any) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 array as PNG.

    Pillow is used when available (it ships with Isaac Sim); otherwise a small zlib-based encoder
    keeps this module dependency-free.
    """
    try:
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(_as_uint8_array(rgb)).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        pass

    array = _as_uint8_array(rgb)
    height, width = array.shape[0], array.shape[1]
    raw = bytearray()
    for row in array:
        raw.append(0)  # PNG filter type 0 (none)
        raw.extend(row.tobytes())

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def _as_uint8_array(rgb: Any):
    """Coerce a tensor/array/list into a contiguous ``(H, W, 3)`` uint8 numpy array."""
    import numpy as np

    detach = getattr(rgb, "detach", None)
    if detach is not None:
        rgb = detach().cpu().numpy()
    array = np.asarray(rgb)
    if array.ndim == 4:  # a batched (N, H, W, C) camera output -- take the first environment
        array = array[0]
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0 if array.max() <= 1.0 else array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array[..., :3])


def _make_handler(bridge: Bridge):
    """Build a request handler bound to ``bridge``."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: A002 - silence the default stderr spam
            pass

        # -- helpers ---------------------------------------------------------------------------

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Any, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _fail(self, exc: Exception, status: int = 400) -> None:
            self._send_json({"ok": False, "error": str(exc), "traceback": traceback.format_exc()}, status)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length) or b"{}")

        # -- routes ----------------------------------------------------------------------------

        def do_GET(self):  # noqa: N802 - required by BaseHTTPRequestHandler
            url = urlparse(self.path)
            query = parse_qs(url.query)
            bridge.request_count += 1
            try:
                if url.path == "/health":
                    self._send_json({"ok": True, "uptime_s": time.time() - bridge.registry.started_at})
                elif url.path == "/describe":
                    self._send_json({"ok": True, "reference": bridge.registry.reference()})
                elif url.path == "/describe.md":
                    self._send(200, bridge.registry.reference_markdown().encode(), "text/markdown")
                elif url.path == "/state":
                    state = bridge.submit(lambda: _jsonable(bridge.registry.state()))
                    self._send_json({"ok": True, "state": state})
                elif url.path == "/image":
                    path = query.get("path", [""])[0]
                    png = bridge.submit(lambda: encode_png(bridge.registry.read(path)))
                    self._send(200, png, "image/png")
                else:
                    self._send_json({"ok": False, "error": f"unknown route {url.path}"}, 404)
            except Exception as exc:
                self._fail(exc)

        def do_POST(self):  # noqa: N802 - required by BaseHTTPRequestHandler
            url = urlparse(self.path)
            bridge.request_count += 1
            try:
                body = self._body()
                if url.path == "/read":
                    path = body["path"]
                    value = bridge.submit(lambda: _jsonable(bridge.registry.read(path)))
                    self._send_json({"ok": True, "path": path, "value": value})
                elif url.path == "/write":
                    path, value = body["path"], body["value"]
                    settle = int(body.get("settle_steps", 0))
                    ops = [{"op": "write", "path": path, "value": value}]
                    if settle:
                        ops.append({"op": "step", "steps": settle})
                    bridge.submit(lambda: bridge.run_program(ops))
                    self._send_json({"ok": True, "path": path})
                elif url.path == "/step":
                    steps = int(body.get("steps", 1))
                    if bridge.step_fn is None:
                        raise RuntimeError("this bridge was started without a step function")
                    bridge.submit(lambda: bridge.step_fn(steps), timeout=max(DEFAULT_TIMEOUT_S, steps * 0.5))
                    self._send_json({"ok": True, "steps": steps})
                elif url.path == "/program":
                    ops = body["ops"]
                    timeout = float(body.get("timeout_s", max(DEFAULT_TIMEOUT_S, 0.5 * len(ops))))
                    results = bridge.submit(lambda: bridge.run_program(ops), timeout=timeout)
                    self._send_json({"ok": True, "results": results})
                else:
                    self._send_json({"ok": False, "error": f"unknown route {url.path}"}, 404)
            except Exception as exc:
                self._fail(exc)

    return Handler
