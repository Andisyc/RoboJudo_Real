"""Hardware-isolated CLI: neither mode imports RoboJuDo or a robot SDK."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

from . import KeyState, KeyboardReceiver, run_client, velocity_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("echo", "client"))
    parser.add_argument("--port", type=int, default=8765, help="Localhost port; use SSH forwarding between machines")
    parser.add_argument("--timeout-s", type=float, default=0.3, help="Server lease timeout; client network timeout is at least 1 second")
    parser.add_argument("--forward-m-s", type=float, default=0.4)
    parser.add_argument("--lateral-m-s", type=float, default=0.4)
    parser.add_argument("--yaw-rate-rad-s", type=float, default=0.8)
    args = parser.parse_args()
    magnitudes = (args.forward_m_s, args.lateral_m_s, args.yaw_rate_rad_s)
    if not 0 <= args.port <= 65535 or (args.mode == "client" and args.port == 0) or not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("expected a valid port and finite positive timeout")
    try:
        velocity_command((), magnitudes)
    except ValueError as error:
        parser.error(str(error))
    if args.mode == "echo":
        with KeyboardReceiver(args.port, args.timeout_s) as receiver:
            print(f"ECHO ONLY on 127.0.0.1:{receiver.port}; no robot environment or SDK loaded", flush=True)
            previous = None
            while True:
                state = receiver.snapshot()
                report = {**state, "velocity": velocity_command(state["keys"], magnitudes)}
                if report != previous:
                    print(json.dumps(report), flush=True)
                    previous = report
                if state["shutdown"]:
                    print("Shutdown request observed; echo exits without any robot operation", flush=True)
                    return
                time.sleep(0.02)
    else:
        try:
            from pynput import keyboard
        except ImportError:
            parser.error("client requires pynput; run with `uv run --no-project --with pynput python -m robojudo_keyboard client`")
        if not keyboard.Listener.IS_TRUSTED:
            parser.error("grant this terminal macOS Accessibility/Input Monitoring permission, then restart")
        state = KeyState()

        def name(key):
            return key.char.lower() if getattr(key, "char", None) else getattr(key, "name", "")

        print("Global keyboard capture: F8 arm/disarm (release all keys first); hold Space + W/S/A/D/Q/E to move.", flush=True)
        print("Release Space or a motion key to cancel. X disarms. Esc requests pipeline shutdown and exits.", flush=True)
        print("After disconnect, restart and explicitly arm again. Do not use this as the physical emergency stop.", flush=True)
        with keyboard.Listener(on_press=lambda key: state.press(name(key)), on_release=lambda key: state.release(name(key))) as listener:
            run_client(state, port=args.port, timeout_s=max(1.0, args.timeout_s), input_alive=listener.is_alive)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Keyboard input stopped: {error}. Motion state is not retained; restart and re-arm.", file=sys.stderr)
        raise SystemExit(1)
