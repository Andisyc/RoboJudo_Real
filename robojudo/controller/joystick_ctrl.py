from __future__ import annotations

import time
from queue import Empty, Queue

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import JoystickCtrlCfg
from robojudo.controller.utils.joystick import JoystickThread, PyUSBJoystickThread

# Axis index → name mapping (must match agent_publisher.py KEY_AXIS_MAP indices)
JOY_AXIS_MAP = {
    0: "LeftX",
    1: "LeftY",
    2: "LT",
    3: "RightX",
    4: "RightY",
    5: "RT",
}

# Button index → name mapping (must match agent_publisher.py KEY_BUTTON_MAP indices)
# Indices 0-7  : standard buttons
# Indices 11-14: D-Pad directions (mirror joystick.py dpad_map event names)
JOY_BUTTON_MAP = {
    0:  "A",
    1:  "B",
    2:  "X",
    3:  "Y",
    4:  "LB",
    5:  "RB",
    6:  "Back",
    7:  "Start",
    11: "Left",
    12: "Right",
    13: "Up",
    14: "Down",
}


def _joy_subscriber_process(data_queue):
    """
    Runs in a separate process to avoid DDS domain conflicts with the main process
    (e.g. Unitree SDK's internal CycloneDDS participant).
    Subscribes to /joy and puts parsed data into data_queue.
    """
    import os
    # Remove environment variables that conflict with rclpy's DDS initialization:
    # - CYCLONEDDS_URI: set by Unitree SDK to its own cyclonedds.xml, conflicts with rclpy
    # - ROS_LOCALHOST_ONLY: when set to 1, rclpy injects a CycloneDDS config that forces
    #   the interface to 'localhost', which CycloneDDS rejects on some systems, and also
    #   blocks cross-machine communication (other developers can't reach this node).
    os.environ.pop('CYCLONEDDS_URI', None)
    os.environ.pop('ROS_LOCALHOST_ONLY', None)

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Joy
    import time as _time

    class _JoyNode(Node):
        def __init__(self):
            super().__init__('joystick_ctrl_subscriber')
            self.last_buttons = None
            self.create_subscription(Joy, '/joy', self._cb, 10)

        def _cb(self, msg):
            now = _time.time()

            axes = {name: 0.0 for name in JOY_AXIS_MAP.values()}
            for i, v in enumerate(msg.axes):
                if i in JOY_AXIS_MAP:
                    axes[JOY_AXIS_MAP[i]] = float(v)

            current_buttons = list(msg.buttons)
            if self.last_buttons is None:
                self.last_buttons = [0] * len(current_buttons)

            button_events = []
            for i, pressed in enumerate(current_buttons):
                if i < len(self.last_buttons) and pressed != self.last_buttons[i]:
                    if i in JOY_BUTTON_MAP:
                        button_events.append({
                            "type": "button",
                            "name": JOY_BUTTON_MAP[i],
                            "pressed": bool(pressed),
                            "timestamp": now,
                        })
            self.last_buttons = current_buttons

            # Keep only the latest frame (drop stale entries)
            while not data_queue.empty():
                try:
                    data_queue.get_nowait()
                except Exception:
                    break
            try:
                data_queue.put_nowait({
                    "axes": axes,
                    "button_event": button_events,
                    "timestamp": now,
                })
            except Exception:
                pass

    rclpy.init()
    node = _JoyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


@ctrl_registry.register
class JoystickCtrl(Controller):
    cfg_ctrl: JoystickCtrlCfg

    def __init__(self, cfg_ctrl: JoystickCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)

        self.state_queue = Queue(maxsize=2)  # for axes
        self.event_queue = Queue(maxsize=100)  # for button/dpad events

        # Select joystick backend based on config.
        # use_pyusb=True: read directly via pyusb (no kernel xpad module needed).
        # use_pyusb=False: use pygame/SDL (requires kernel xpad or similar driver).
        if cfg_ctrl.use_pyusb:
            try:
                self.joystick_thread = PyUSBJoystickThread(
                    self.state_queue, self.event_queue,
                    custom_vid=cfg_ctrl.pyusb_vid,
                    custom_pid=cfg_ctrl.pyusb_pid,
                )
                print("[JoystickCtrl] Using PyUSB backend (xpad kernel module not required).")
            except Exception as e:
                print(f"[JoystickCtrl] PyUSB backend unavailable ({e}), falling back to pygame.")
                self.joystick_thread = JoystickThread(self.state_queue, self.event_queue)
        else:
            # Legacy pygame/SDL path (requires kernel xpad or SDL joystick driver)
            self.joystick_thread = JoystickThread(self.state_queue, self.event_queue)

        self.joystick_thread.start()

        self.axes_names = self.joystick_thread.config["axis_config"]["axis_map"].keys()

        # ROS2 mode switching setup
        self.control_mode = 'local'  # 'local' or 'ros'
        self.last_ros_cmd_time = 0
        self.last_ros_cmd = None
        self.toggle_buttons = {'Back', 'Start'}  # Back+Start to toggle mode
        self.active_toggle_buttons = set()
        self.toggle_debounce = False

        self.init_ros()
        self.reset()

    def init_ros(self):
        """
        Starts the ROS2 Joy subscriber in a separate subprocess to avoid
        DDS domain conflicts with the main process (Unitree SDK uses its
        own CycloneDDS participant which conflicts with rclpy in-process).

        Uses 'spawn' start method (not the Linux default 'fork') so the child
        process starts with a clean slate and does NOT inherit the parent's
        CycloneDDS domain state.
        """
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        self._ros_data_queue = ctx.Queue(maxsize=2)
        self._ros_process = ctx.Process(
            target=_joy_subscriber_process,
            args=(self._ros_data_queue,),
            daemon=True,
            name="JoySubscriberProcess",
        )
        self._ros_process.start()
        print("[JoystickCtrl] ROS2 subscriber started in subprocess (spawn) for /joy (sensor_msgs/Joy).")

    def _update_control_mode(self, events):
        """Toggle between local joystick and ROS control via Back+Start combo."""
        for event in events:
            if event['type'] == 'button' and event['name'] in self.toggle_buttons:
                if event['pressed']:
                    self.active_toggle_buttons.add(event['name'])
                else:
                    self.active_toggle_buttons.discard(event['name'])

        if self.active_toggle_buttons == self.toggle_buttons:
            if not self.toggle_debounce:
                if self.control_mode == 'local':
                    self.control_mode = 'ros'
                    print("\n[JoystickCtrl] Switched to ROS control mode.")
                else:
                    self.control_mode = 'local'
                    print("\n[JoystickCtrl] Switched to Local Joystick control mode.")
                self.toggle_debounce = True
        else:
            self.toggle_debounce = False

    def reset(self):
        self.combination_init_buttons = self.cfg_ctrl.combination_init_buttons
        self.onhold_buttons = set()
        while not self.state_queue.empty():
            try:
                self.state_queue.get_nowait()
            except Empty:
                break

        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except Empty:
                break

        self.last_state = {
            "type": "axes",
            "axes": {name: 0.0 for name in self.axes_names},
            "timestamp": time.time(),
        }

    def get_state(self):
        try:
            state = self.state_queue.get_nowait()
            self.last_state = state.copy()
        except Empty:
            state = self.last_state

        return state

    def get_events(self):
        events = []
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                events.append(event)
            except Empty:
                break
        return events

    def get_data(self):
        # Always drain physical joystick events (needed for mode toggle detection)
        events = self.get_events()
        self._update_control_mode(events)

        if self.control_mode == 'ros':
            # Pull latest data from the subscriber subprocess
            try:
                ros_cmd = self._ros_data_queue.get_nowait()
                self.last_ros_cmd = ros_cmd
                self.last_ros_cmd_time = time.time()
            except Exception:
                pass  # No new data; use last known

            if self.last_ros_cmd and (time.time() - self.last_ros_cmd_time < 0.5):
                ros_cmd = self.last_ros_cmd.copy()
                # Merge physical button events so Back+Start combo can toggle back
                ros_cmd.setdefault('button_event', []).extend(events)
                return ros_cmd
            else:
                # ROS mode but no recent data → safe neutral state
                return {
                    "axes": {name: 0.0 for name in self.axes_names},
                    "button_event": events,
                }
        else:  # 'local' joystick control
            state = self.get_state()
            return {
                "axes": state["axes"],
                "button_event": events,
            }

    def process_triggers(self, ctrl_data):
        commands = []
        if len(self.triggers) == 0:
            return ctrl_data, commands

        for event in ctrl_data["button_event"]:
            if event["type"] == "button":
                if event["name"] in self.combination_init_buttons:
                    if event["pressed"]:
                        self.onhold_buttons.add(event["name"])
                    else:
                        self.onhold_buttons.discard(event["name"])
                else:
                    if event["pressed"]:
                        command = None
                        if len(self.onhold_buttons) == 0:
                            command = self.triggers.get(event["name"], None)
                        else:
                            event_combination = "+".join(sorted(list(self.onhold_buttons)) + [event["name"]])
                            command = self.triggers.get(event_combination, None)
                        if command is not None:
                            commands.append(command)
                            # remove event after triggered
                            ctrl_data["button_event"].remove(event)

        return ctrl_data, commands


if __name__ == "__main__":
    joystick_ctrl = JoystickCtrl(
        cfg_ctrl=JoystickCtrlCfg(
            triggers={
                "A": "[TEST_A]",
                "B": "[TEST_B]",
                "LB+Left": "[TEST_LB_Left]",
                "RB+Right": "[TEST_RB_Right]",
                "LB+RB+A": "[TEST_LB_RB_A]",
            },
        )
    )
    for _ in range(10000):
        ctrl_data = joystick_ctrl.get_data()
        ctrl_data, commands = joystick_ctrl.process_triggers(ctrl_data)
        print(ctrl_data)
        print(commands)
        print("================================")
        time.sleep(0.3)
    exit()
