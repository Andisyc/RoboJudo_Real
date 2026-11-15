"""Only localhost sockets and key state: no model, SDK, environment or robot."""
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


from robojudo_keyboard import KeyState, KeyboardReceiver, run_client, velocity_command


class Peer:
    def __init__(self, receiver):
        self.socket = socket.create_connection(("127.0.0.1", receiver.port), timeout=1)
        self.stream = self.socket.makefile("rb")
        self.challenge = json.loads(self.stream.readline())

    def reply(self, state, *, challenge=None):
        packet = {**(self.challenge if challenge is None else challenge), **state}
        self.socket.sendall(json.dumps(packet).encode() + b"\n")
        self.challenge = json.loads(self.stream.readline())

    def close(self):
        self.stream.close()
        self.socket.close()


def state(keys=(), *, armed=False, deadman=False, shutdown=False):
    return {"keys": list(keys), "armed": armed, "deadman": deadman, "shutdown": shutdown}


def wait_for(receiver, predicate):
    deadline = time.monotonic() + 2
    current = receiver.snapshot()
    while time.monotonic() < deadline:
        current = receiver.snapshot()
        if predicate(current):
            return current
        time.sleep(0.005)
    raise AssertionError(current)


def test_socket_release_multiaxis_and_deadman():
    with KeyboardReceiver(0, 0.5) as receiver:
        peer = Peer(receiver)
        try:
            peer.reply(state())
            peer.reply(state(armed=True))
            peer.reply(state(("w", "q"), armed=True, deadman=True))
            assert velocity_command(receiver.snapshot()["keys"], (0.4, 0.4, 0.8)) == (0.4, 0.0, 0.8)
            peer.reply(state(("q",), armed=True, deadman=True))
            assert velocity_command(receiver.snapshot()["keys"], (0.4, 0.4, 0.8)) == (0.0, 0.0, 0.8)
            peer.reply(state(("w", "s", "q"), armed=True, deadman=True))
            assert velocity_command(receiver.snapshot()["keys"], (0.4, 0.4, 0.8)) == (0.0, 0.0, 0.8)
            peer.reply(state(("w", "q"), armed=True))
            assert receiver.snapshot()["keys"] == ()
        finally:
            peer.close()


def test_disconnect_and_reconnect_do_not_restore_held_motion():
    with KeyboardReceiver(0, 0.5) as receiver:
        peer = Peer(receiver)
        peer.reply(state())
        peer.reply(state(armed=True))
        peer.reply(state(("w",), armed=True, deadman=True))
        assert receiver.snapshot()["keys"] == ("w",)
        peer.close()
        assert not wait_for(receiver, lambda x: not x["fresh"])["armed"]
        peer = Peer(receiver)
        try:
            peer.reply(state(("w",), armed=True, deadman=True))
            assert receiver.snapshot()["keys"] == ()
            peer.reply(state())
            peer.reply(state(armed=True))
            peer.reply(state(("w",), armed=True, deadman=True))
            assert receiver.snapshot()["keys"] == ("w",)
            receiver.reset()
            peer.reply(state(("w",), armed=True, deadman=True))
            assert receiver.snapshot()["keys"] == ()
        finally:
            peer.close()


def test_expired_input_and_replayed_challenge_cancel_motion():
    with KeyboardReceiver(0, 0.15) as receiver:
        peer = Peer(receiver)
        try:
            peer.reply(state())
            peer.reply(state(armed=True))
            old = peer.challenge.copy()
            peer.reply(state(("w",), armed=True, deadman=True))
            assert receiver.snapshot()["keys"] == ("w",)
            packet = {**old, **state(("w",), armed=True, deadman=True)}
            peer.socket.sendall(json.dumps(packet).encode() + b"\n")
            assert not wait_for(receiver, lambda x: not x["fresh"])["keys"]
        finally:
            peer.close()
        peer = Peer(receiver)
        try:
            peer.reply(state())
            peer.reply(state(armed=True))
            peer.reply(state(("w",), armed=True, deadman=True))
            assert receiver.snapshot()["keys"] == ("w",)
            assert not wait_for(receiver, lambda x: not x["fresh"])["keys"]
        finally:
            peer.close()


def test_client_key_state_x_and_repeated_keys_cannot_rearm():
    keys = KeyState()
    keys.press("f8")
    keys.press("f8")
    keys.release("f8")
    assert keys.snapshot()["armed"]
    keys.press("space")
    keys.press("w")
    keys.press("x")
    keys.press("w")
    keys.press("f8")
    assert not keys.snapshot()["armed"]
    keys.release("f8")
    keys.release("w")
    keys.release("space")
    keys.press("f8")
    assert keys.snapshot()["armed"]
    keys.disarm()
    assert not keys.snapshot()["armed"] and not keys.snapshot()["deadman"]


def test_production_client_loop_releases_and_delivers_shutdown():
    with KeyboardReceiver(0, 0.5) as receiver:
        keys = KeyState()
        errors = []

        def client():
            try:
                run_client(keys, port=receiver.port)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=client)
        thread.start()
        try:
            wait_for(receiver, lambda x: x["fresh"])
            keys.press("f8")
            keys.release("f8")
            wait_for(receiver, lambda x: x["armed"])
            keys.press("space")
            keys.press("w")
            assert wait_for(receiver, lambda x: x["keys"] == ("w",))["armed"]
            keys.release("w")
            assert wait_for(receiver, lambda x: not x["keys"])["armed"]
            keys.press("esc")
            thread.join(timeout=2)
            assert not thread.is_alive() and not errors
            assert receiver.snapshot()["shutdown"]
            assert not keys.snapshot()["armed"]
        finally:
            receiver.close()
            thread.join(timeout=2)


def test_echo_cli_blocks_all_robot_package_imports():
    root = Path(__file__).resolve().parents[1]
    guarded = """
import importlib.abc, runpy, sys
class NoRobot(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('robojudo', 'unitree_cpp', 'unitree_sdk2py'):
   raise RuntimeError('Forbidden robot import: '+fullname)
sys.meta_path.insert(0, NoRobot())
sys.argv=['robojudo_keyboard','echo','--port','0']
runpy.run_module('robojudo_keyboard',run_name='__main__')
"""
    process = subprocess.Popen([sys.executable, "-u", "-c", guarded], cwd=root,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout is not None
        ready = process.stdout.readline()
        assert "ECHO ONLY on 127.0.0.1:" in ready
        port = int(ready.split(":", 1)[1].split(";", 1)[0])
        with socket.create_connection(("127.0.0.1", port), timeout=2) as connection:
            with connection.makefile("rb") as stream:
                challenge = json.loads(stream.readline())
                connection.sendall(json.dumps({**challenge, **state(shutdown=True)}).encode() + b"\n")
                stream.readline()
        stdout, stderr = process.communicate(timeout=3)
        assert process.returncode == 0, stderr
        assert '"velocity": [0.0, 0.0, 0.0]' in stdout
        assert "Shutdown request observed" in stdout
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=2)
