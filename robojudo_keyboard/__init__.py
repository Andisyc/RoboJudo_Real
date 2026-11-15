"""Standalone keyboard transport: no RoboJuDo, policy, SDK or robot imports."""
from __future__ import annotations

import json
import math
import secrets
import socket
import threading
import time

KEY_AXES = {"w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1), "q": (2, 1), "e": (2, -1)}
_MAX_LINE = 4096


def velocity_command(keys, magnitudes):
    """Opposite keys cancel; release changes only its own axis."""
    if len(magnitudes) != 3 or not all(math.isfinite(float(v)) and v > 0 for v in magnitudes):
        raise ValueError("expected three finite positive command magnitudes")
    signs = [0, 0, 0]
    for key in set(keys):
        axis, sign = KEY_AXES[key]
        signs[axis] += sign
    return tuple(sign * float(mag) for sign, mag in zip(signs, magnitudes))


class KeyState:
    """F8 arms only with motion keys released; Space is a held deadman."""

    def __init__(self):
        self._lock = threading.Lock()
        self.keys = set()
        self.armed = self.deadman = self.shutdown = False
        self._f8_down = False

    def press(self, key):
        with self._lock:
            if key in KEY_AXES:
                self.keys.add(key)
            elif key == "space":
                self.deadman = True
            elif key == "f8" and not self._f8_down:
                self._f8_down = True
                self.armed = not self.armed if not self.keys and not self.deadman else False
            elif key in ("x", "esc"):
                self.armed = False
                self.shutdown = key == "esc"

    def release(self, key):
        with self._lock:
            self.keys.discard(key)
            if key == "space":
                self.deadman = False
            elif key == "f8":
                self._f8_down = False

    def disarm(self):
        with self._lock:
            self.armed = self.deadman = False

    def snapshot(self):
        with self._lock:
            return {"keys": sorted(self.keys), "armed": self.armed,
                    "deadman": self.deadman, "shutdown": self.shutdown}


def _line(stream):
    raw = stream.readline(_MAX_LINE + 1)
    if not raw or len(raw) > _MAX_LINE or not raw.endswith(b"\n"):
        raise ValueError("closed connection or oversized keyboard message")
    packet = json.loads(raw)
    if not isinstance(packet, dict):
        raise ValueError("keyboard message must be an object")
    return packet


def _send(sock, packet):
    sock.sendall(json.dumps(packet, separators=(",", ":")).encode() + b"\n")


class KeyboardReceiver:
    """Localhost-only, expiring challenge/reply leases; clients never supply a clock.

    Every challenge expires on the receiver's monotonic clock. Delayed TCP
    packets cannot renew a lease merely by arriving. A connection/reset requires
    a disarmed state followed by an armed, released state before movement.
    """

    def __init__(self, port=8765, timeout_s=0.3):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._server.bind(("127.0.0.1", port))
            self._server.listen(1)
            self._server.settimeout(0.1)
        except BaseException:
            self._server.close()
            raise
        self.port = self._server.getsockname()[1]
        self._connection = None
        self._keys = ()
        self._deadline = 0.0
        self._armed = False
        self._needs_disarm = True
        self._shutdown = False
        self._thread = threading.Thread(target=self._serve, name="keyboard-receiver", daemon=True)
        self._thread.start()

    def reset(self):
        with self._lock:
            self._keys, self._deadline = (), 0.0
            self._armed, self._needs_disarm, self._shutdown = False, True, False

    def snapshot(self):
        with self._lock:
            fresh = time.monotonic() < self._deadline
            if not fresh and self._armed:
                self._keys = ()
                self._armed, self._needs_disarm = False, True
            result = {"keys": self._keys if fresh and self._armed else (),
                      "fresh": fresh, "armed": fresh and self._armed, "shutdown": self._shutdown}
            self._shutdown = False
            return result

    def _serve(self):
        while not self._stop.is_set():
            try:
                connection, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with self._lock:
                self._connection = connection
            try:
                self._receive(connection)
            except (OSError, ValueError, TypeError, KeyError):
                pass  # Invalid/lost input is represented as an expired zero command below.
            finally:
                connection.close()
                with self._lock:
                    self._connection = None
                    self._keys, self._deadline = (), 0.0
                    self._armed, self._needs_disarm = False, True

    def _receive(self, connection):
        connection.settimeout(self.timeout_s)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        session = secrets.token_hex(16)
        with connection.makefile("rb") as stream:
            sequence = 0
            while not self._stop.is_set():
                issued = time.monotonic()
                with self._lock:
                    if self._deadline and issued >= self._deadline:
                        raise ValueError("keyboard input lease expired")
                _send(connection, {"session": session, "challenge": sequence})
                reply = _line(stream)
                if (reply.get("session") != session or type(reply.get("challenge")) is not int
                        or reply["challenge"] != sequence or time.monotonic() >= issued + self.timeout_s):
                    raise ValueError("stale keyboard reply")
                keys = reply.get("keys")
                if (not isinstance(keys, list) or not all(isinstance(k, str) and k in KEY_AXES for k in keys)
                        or any(type(reply.get(k)) is not bool for k in ("armed", "deadman", "shutdown"))):
                    raise ValueError("invalid keyboard state")
                with self._lock:
                    if not reply["armed"]:
                        self._needs_disarm = self._armed = False
                    elif not self._needs_disarm and not keys and not reply["deadman"]:
                        self._armed = True
                    self._shutdown |= reply["shutdown"]
                    if reply["shutdown"]:
                        self._armed = False
                    self._keys = tuple(set(keys)) if self._armed and reply["deadman"] else ()
                    self._deadline = issued + self.timeout_s
                sequence += 1
                self._stop.wait(0.02)

    def close(self):
        self._stop.set()
        with self._lock:
            connection = self._connection
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._server.close()
        self._thread.join(timeout=self.timeout_s + 0.2)
        self.reset()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def run_client(state, *, port=8765, timeout_s=1.0, input_alive=lambda: True):
    """Reply with current state only; disconnects disarm and require a new run."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout_s) as connection:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with connection.makefile("rb") as stream:
                shutdown_sent = False
                while True:
                    try:
                        challenge = _line(stream)
                    except ValueError:
                        if shutdown_sent:
                            return
                        raise
                    if shutdown_sent:
                        return  # The next challenge acknowledges the shutdown request.
                    if not input_alive():
                        raise RuntimeError("keyboard input listener stopped")
                    reply = state.snapshot()
                    reply.update(session=challenge["session"], challenge=challenge["challenge"])
                    _send(connection, reply)
                    shutdown_sent = reply["shutdown"]
    finally:
        state.disarm()
