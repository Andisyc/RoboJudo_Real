import logging
import os
import struct
import time
from queue import Empty, Queue
from threading import Thread

import numpy as np

logger = logging.getLogger(__name__)

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"


# TODO: axis post processing
class JoystickThread(Thread):
    def __init__(self, state_queue: Queue, event_queue: Queue):
        super().__init__(name="JoystickThread", daemon=True)
        self.state_queue = state_queue
        self.event_queue = event_queue

        self.config = self._init_config()

        self.running = True

    # fmt: off
    def _init_config(self):
        config = {
            "button_map": {
                0: "A", 1: "B", 2: "X", 3: "Y",
                4: "LB", 5: "RB", 6: "Back", 7: "Start",
                8: "Xbox", 9: "L", 10: "R",
            },
            "axis_config": {
                "axis_map": {
                    "LeftX": 0,
                    "LeftY": 1,
                    "RightX": 3,
                    "RightY": 4,
                    "LT": 2,
                    "RT": 5
                },
                "axis_range": {
                    "LT": [0, 1],
                    "RT": [0, 1]
                },
                "invert": ["LeftY", "RightY"],
            },
            "dpad_config": {
                "as_button_event": True,  # map dpad to button events
                "dpad_map": {
                    "Up": (1, 1),
                    "Right": (0, 1),
                    "Down": (1, -1),
                    "Left": (0, -1),
                }

            }
        }

        # if windows
        if os.name == 'nt':  # Windows
            config["button_map"].update({
                8: "L", 9: "R",
            })
            del config["button_map"][10]
            config["axis_config"]["axis_map"].update({
                "RightX": 2,
                "RightY": 3,
                "LT": 4,
            })
        return config
    # fmt: on

    @staticmethod
    def normalize_axis(axis_range, name, value):
        target_range = axis_range.get(name)
        if target_range:
            min_val, max_val = -1.0, 1.0  # SDL default range
            min_target, max_target = target_range
            value = (value - min_val) / (max_val - min_val) * (max_target - min_target) + min_target
        return round(value, 3)

    def run(self):
        import pygame

        pygame.init()
        if pygame.joystick.get_count() == 0:
            self.running = False
            logger.error("No joystick connected. Try to fix with: export SDL_JOYSTICK_DEVICE=/dev/input/js0")
            # raise RuntimeError("No joystick connected, try to fix with: export SDL_JOYSTICK_DEVICE=/dev/input/js0")
            return

        joystick = pygame.joystick.Joystick(0)
        joystick.init()

        name = joystick.get_name().lower()
        logger.info(f"[Joystick] Initialized: {name}")
        logger.info(
            f"[Joystick] Buttons: {joystick.get_numbuttons()}, \
                Axes: {joystick.get_numaxes()}, \
                Hats: {joystick.get_numhats()}"
        )

        button_map = self.config.get("button_map", {})
        axis_config = self.config.get("axis_config", {})

        dpad_config = self.config.get("dpad_config", {})
        dpad_as_button = dpad_config.get("as_button_event", True)
        dpad_state = {key: False for key in dpad_config.get("dpad_map", {}).keys()}

        axis_map = axis_config.get("axis_map", {})
        axis_range = axis_config.get("axis_range", {})
        invert = set(axis_config.get("invert", []))

        clock = pygame.time.Clock()
        last_state_time = time.time()
        state_interval = 1.0 / 100  # 100Hz

        while self.running:
            pygame.event.pump()
            now = time.time()

            # Poll events for buttons and DPad
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN or event.type == pygame.JOYBUTTONUP:
                    btn_index = event.button
                    btn_name = button_map.get(btn_index, f"Button_{btn_index}")
                    self.event_queue.put(
                        {
                            "type": "button",
                            "name": btn_name,
                            "pressed": event.type == pygame.JOYBUTTONDOWN,
                            "timestamp": now,
                        }
                    )

                elif event.type == pygame.JOYHATMOTION:
                    if dpad_as_button:
                        dpad_state_new = {
                            name: event.value[axis] == direction
                            for name, (axis, direction) in dpad_config.get("dpad_map", {}).items()
                        }
                        for name, pressed in dpad_state_new.items():
                            if pressed != dpad_state[name]:
                                dpad_state[name] = pressed
                                self.event_queue.put(
                                    {
                                        "type": "button",
                                        "name": name,
                                        "pressed": pressed,
                                        "timestamp": now,
                                    }
                                )
                    else:
                        self.event_queue.put(
                            {
                                "type": "dpad",
                                "value": event.value,
                                "timestamp": now,
                            }
                        )

            # Axes update at fixed rate
            if now - last_state_time >= state_interval:
                axes_state = {}
                for name, index in axis_map.items():
                    val = joystick.get_axis(index)
                    if name in invert:
                        val = -val
                    val = self.normalize_axis(axis_range, name, val)
                    axes_state[name] = val

                while self.state_queue.full():
                    self.state_queue.get()
                self.state_queue.put(
                    {
                        "type": "axes",
                        "axes": axes_state,
                        "timestamp": now,
                    }
                )
                last_state_time = now

            clock.tick(500)  # avoid busy loop


class PyUSBJoystickThread(Thread):
    """Xbox controller reader via pyusb — bypasses the kernel xpad module.

    Supports Xbox 360 (wired) and Xbox One (wired) controllers.
    On Linux, add a udev rule to allow non-root access:
        SUBSYSTEM=="usb", ATTRS{idVendor}=="045e", MODE="0666"
    """

    # (vendor_id, product_id, protocol_type)
    # XInput mode devices all use the same 20-byte Xbox 360 report format.
    SUPPORTED_DEVICES = [
        (0x045e, 0x028e, "xbox360"),  # Xbox 360 wired
        (0x045e, 0x028f, "xbox360"),  # Xbox 360 wired (clone)
        (0x045e, 0x02d1, "xboxone"),  # Xbox One
        (0x045e, 0x02dd, "xboxone"),  # Xbox One (2015 firmware)
        (0x045e, 0x02e3, "xboxone"),  # Xbox One Elite
        (0x045e, 0x02ea, "xboxone"),  # Xbox One S
        (0x045e, 0x0b12, "xboxone"),  # Xbox Series X/S
        # Betop (北通) XInput mode controllers
        (0x20bc, 0x5500, "xbox360"),  # Betop BTP-2163X / BTP-2185X
        (0x20bc, 0x5501, "xbox360"),  # Betop (variant)
        (0x20bc, 0x5506, "xbox360"),  # Betop (variant)
        (0x11c0, 0x5506, "xbox360"),  # Betop (older VID)
        (0x045e, 0x028e, "xbox360"),  # Betop in XInput mode (spoofs Xbox 360 VID/PID)
    ]

    # Xbox 360 — 16-bit button bitmask bit index → name
    # Bytes 2-3 of the 20-byte input report (little-endian uint16)
    XBOX360_BUTTON_BITS = {
        0: "Up", 1: "Down", 2: "Left", 3: "Right",
        4: "Start", 5: "Back", 6: "L", 7: "R",
        8: "LB", 9: "RB", 10: "Xbox",
        12: "A", 13: "B", 14: "X", 15: "Y",
    }

    # Xbox One (GIP) — byte 3 (lo) and byte 4 (hi) of the input report
    XBOXONE_BUTTON_BITS_LO = {
        0: "Up", 1: "Down", 2: "Left", 3: "Right",
        4: "Start", 5: "Back", 6: "L", 7: "R",
    }
    XBOXONE_BUTTON_BITS_HI = {
        0: "LB", 1: "RB",
        4: "A", 5: "B", 6: "X", 7: "Y",
    }

    def __init__(self, state_queue: Queue, event_queue: Queue,
                 custom_vid: int | None = None, custom_pid: int | None = None):
        """
        custom_vid / custom_pid: override auto-detection with a specific USB VID/PID.
        Use when your controller is not in SUPPORTED_DEVICES (e.g. 北通 with unknown PID).
        XInput-mode controllers always use the xbox360 report format regardless of brand.
        Run `lsusb` on Linux to find your device's VID:PID.
        """
        super().__init__(name="PyUSBJoystickThread", daemon=True)
        self.state_queue = state_queue
        self.event_queue = event_queue
        self.running = True

        # Device discovery at init time so callers can catch RuntimeError early
        self._dev, self._dev_type = self._find_device(custom_vid, custom_pid)
        if self._dev is None:
            raise RuntimeError(
                "PyUSBJoystickThread: no supported XInput controller found. "
                "Run `lsusb` to get VID:PID, then set pyusb_vid/pyusb_pid in JoystickCtrlCfg."
            )

        self.config = self._init_config()

    # fmt: off
    def _init_config(self):
        return {
            "axis_config": {
                "axis_map": {
                    "LeftX": 0, "LeftY": 1,
                    "RightX": 3, "RightY": 4,
                    "LT": 2, "RT": 5,
                },
                "invert": ["LeftY", "RightY"],
            }
        }
    # fmt: on

    def _find_device(self, custom_vid: int | None = None, custom_pid: int | None = None):
        try:
            import usb.core
        except ImportError:
            logger.error("PyUSBJoystickThread: pyusb is not installed. Run: pip install pyusb")
            return None, None

        # Custom VID/PID from config takes priority (for unlisted brands like 北通)
        if custom_vid is not None and custom_pid is not None:
            dev = usb.core.find(idVendor=custom_vid, idProduct=custom_pid)
            if dev is not None:
                logger.info(
                    f"[PyUSBJoystick] Found custom XInput controller: "
                    f"VID={custom_vid:#06x} PID={custom_pid:#06x} (xbox360 protocol)"
                )
                return dev, "xbox360"
            logger.warning(
                f"[PyUSBJoystick] Custom device VID={custom_vid:#06x} PID={custom_pid:#06x} not found."
            )
            return None, None

        for vid, pid, dev_type in self.SUPPORTED_DEVICES:
            dev = usb.core.find(idVendor=vid, idProduct=pid)
            if dev is not None:
                logger.info(
                    f"[PyUSBJoystick] Found {dev_type} controller: "
                    f"VID={vid:#06x} PID={pid:#06x}"
                )
                return dev, dev_type
        return None, None

    @staticmethod
    def _norm_axis(raw: int, max_val: int = 32767) -> float:
        return round(max(min(raw / max_val, 1.0), -1.0), 4)

    @staticmethod
    def _norm_trigger_u8(raw: int) -> float:
        """Xbox 360 trigger: 0–255 → 0.0–1.0"""
        return round(raw / 255.0, 4)

    @staticmethod
    def _norm_trigger_u16(raw: int) -> float:
        """Xbox One trigger: 0–1023 → 0.0–1.0"""
        return round(min(raw / 1023.0, 1.0), 4)

    def run(self):
        import usb.core
        import usb.util

        dev = self._dev
        dev_type = self._dev_type

        # Detach any kernel driver that may have partially claimed the device
        try:
            if dev.is_kernel_driver_active(0):
                dev.detach_kernel_driver(0)
                logger.info("[PyUSBJoystick] Detached kernel driver from interface 0.")
        except Exception as e:
            logger.debug(f"[PyUSBJoystick] detach_kernel_driver: {e}")

        try:
            dev.set_configuration()
        except Exception as e:
            logger.debug(f"[PyUSBJoystick] set_configuration: {e}")

        try:
            usb.util.claim_interface(dev, 0)
        except Exception as e:
            logger.error(f"[PyUSBJoystick] Cannot claim interface 0: {e}")
            self.running = False
            return

        ep_in = 0x81  # Interrupt IN endpoint (standard for both Xbox 360 and One)

        # Xbox One (GIP) requires an explicit "open session" packet before sending input
        if dev_type == "xboxone":
            try:
                dev.write(0x02, b'\x05\x20\x00\x01\x00')
                logger.info("[PyUSBJoystick] Sent Xbox One GIP initialization packet.")
            except Exception as e:
                logger.warning(f"[PyUSBJoystick] Xbox One init packet failed: {e}")

        last_button_state: dict = {}
        last_state_time = time.time()
        state_interval = 1.0 / 100  # 100 Hz

        try:
            while self.running:
                try:
                    data = bytes(dev.read(ep_in, 64, timeout=20))
                except usb.core.USBTimeoutError:
                    continue
                except usb.core.USBError as e:
                    if e.errno == 110:  # ETIMEDOUT on some backends
                        continue
                    logger.error(f"[PyUSBJoystick] USB read error: {e}")
                    break

                now = time.time()

                # ── Xbox 360 input report (20 bytes) ──────────────────────────
                # Byte 0: 0x00 (input type), Byte 1: 0x14 (length=20)
                # Bytes 2-3: buttons (uint16 LE), 4: LT, 5: RT
                # Bytes 6-7: LX, 8-9: LY, 10-11: RX, 12-13: RY (int16 LE)
                if dev_type == "xbox360":
                    if len(data) < 14 or data[0] != 0x00:
                        continue
                    buttons_raw = struct.unpack_from("<H", data, 2)[0]
                    lt = self._norm_trigger_u8(data[4])
                    rt = self._norm_trigger_u8(data[5])
                    lx = self._norm_axis(struct.unpack_from("<h", data, 6)[0])
                    ly = -self._norm_axis(struct.unpack_from("<h", data, 8)[0])   # up=positive
                    rx = self._norm_axis(struct.unpack_from("<h", data, 10)[0])
                    ry = -self._norm_axis(struct.unpack_from("<h", data, 12)[0])  # up=positive

                    for bit, name in self.XBOX360_BUTTON_BITS.items():
                        pressed = bool(buttons_raw & (1 << bit))
                        if last_button_state.get(name) != pressed:
                            self.event_queue.put({
                                "type": "button", "name": name,
                                "pressed": pressed, "timestamp": now,
                            })
                            last_button_state[name] = pressed

                # ── Xbox One (GIP) input report (≥17 bytes) ───────────────────
                # Byte 0: 0x20 (input), Byte 1: seq, Byte 2: payload length
                # Byte 3: buttons_lo, Byte 4: buttons_hi
                # Bytes 5-6: LT (uint16), 7-8: RT (uint16)
                # Bytes 9-10: LX, 11-12: LY, 13-14: RX, 15-16: RY (int16 LE)
                elif dev_type == "xboxone":
                    if len(data) < 17 or data[0] != 0x20:
                        continue
                    btn_lo = data[3]
                    btn_hi = data[4]
                    lt = self._norm_trigger_u16(struct.unpack_from("<H", data, 5)[0])
                    rt = self._norm_trigger_u16(struct.unpack_from("<H", data, 7)[0])
                    lx = self._norm_axis(struct.unpack_from("<h", data, 9)[0])
                    ly = -self._norm_axis(struct.unpack_from("<h", data, 11)[0])
                    rx = self._norm_axis(struct.unpack_from("<h", data, 13)[0])
                    ry = -self._norm_axis(struct.unpack_from("<h", data, 15)[0])

                    current: dict = {}
                    for bit, name in self.XBOXONE_BUTTON_BITS_LO.items():
                        current[name] = bool(btn_lo & (1 << bit))
                    for bit, name in self.XBOXONE_BUTTON_BITS_HI.items():
                        current[name] = bool(btn_hi & (1 << bit))
                    for name, pressed in current.items():
                        if last_button_state.get(name) != pressed:
                            self.event_queue.put({
                                "type": "button", "name": name,
                                "pressed": pressed, "timestamp": now,
                            })
                    last_button_state = current

                else:
                    continue

                # Axes at fixed rate
                if now - last_state_time >= state_interval:
                    axes_state = {
                        "LeftX": lx, "LeftY": ly,
                        "RightX": rx, "RightY": ry,
                        "LT": lt, "RT": rt,
                    }
                    while self.state_queue.full():
                        self.state_queue.get()
                    self.state_queue.put({
                        "type": "axes", "axes": axes_state, "timestamp": now,
                    })
                    last_state_time = now

        finally:
            try:
                usb.util.release_interface(dev, 0)
                usb.util.dispose_resources(dev)
            except Exception:
                pass


# Modified From unitree_sdk2_python
class unitreeRemoteController:
    def __init__(self, state_queue, event_queue):
        self.state_queue = state_queue
        self.event_queue = event_queue

        # button
        self.button_map = [
            "R1",
            "L1",
            "Start",
            "Select",
            "R2",
            "L2",
            "F1",
            "F2",
            "A",
            "B",
            "X",
            "Y",
            "Up",
            "Right",
            "Down",
            "Left",
        ]
        self.last_button_state = np.zeros((16), dtype=bool)

    def parse(self, remoteData):
        now = time.time()
        # button
        keys = struct.unpack("H", remoteData[2:4])[0]
        button = [((keys & (1 << i)) >> i) for i in range(16)]
        button_state = np.array(button, dtype=bool)

        # Check for button state changes
        changed = button_state != self.last_button_state
        for i in range(16):
            if changed[i]:
                self.event_queue.put(
                    {
                        "type": "button",
                        "name": self.button_map[i],
                        "pressed": bool(button_state[i]),
                        "timestamp": now,
                    }
                )
        self.last_button_state = button_state.copy()

        # axis
        lx_offset = 4
        LeftX = struct.unpack("<f", remoteData[lx_offset : lx_offset + 4])[0]
        rx_offset = 8
        RightX = struct.unpack("<f", remoteData[rx_offset : rx_offset + 4])[0]
        ry_offset = 12
        RightY = struct.unpack("<f", remoteData[ry_offset : ry_offset + 4])[0]
        # L2_offset = 16
        # L2 = struct.unpack('<f', remoteData[L2_offset:L2_offset + 4])[0] # Placeholder，unused
        ly_offset = 20
        LeftY = struct.unpack("<f", remoteData[ly_offset : ly_offset + 4])[0]

        while self.state_queue.full():
            self.state_queue.get()
        self.state_queue.put(
            {
                "type": "axes",
                "axes": {
                    "LeftX": LeftX,
                    "LeftY": LeftY,
                    "RightX": RightX,
                    "RightY": RightY,
                },
                "timestamp": now,
            }
        )


if __name__ == "__main__":
    state_queue = Queue(maxsize=10)
    event_queue = Queue(maxsize=100)
    js_thread = JoystickThread(state_queue, event_queue)
    js_thread.start()

    print("Press joystick buttons (Ctrl+C to exit)...")
    try:
        while True:
            try:
                state = state_queue.get(timeout=1.0)
                print("State:", state)
            except Empty:
                pass

            while not event_queue.empty():
                try:
                    event = event_queue.get_nowait()
                    print("Event:", event)
                except Empty:
                    break
    except KeyboardInterrupt:
        print("Exiting...")
        js_thread.running = False
        js_thread.join()
