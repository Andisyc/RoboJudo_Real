import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

# --- Configuration ---
MAX_SPEED = 0.8  # Speed limit for joystick axes, value between 0.0 and 1.0

# Mapping from key to (axis_index, direction)
# Axis indices match JOY_AXIS_MAP in joystick_ctrl.py:
#   0: "LeftX", 1: "LeftY", 2: "LT", 3: "RightX", 4: "RightY", 5: "RT"
KEY_AXIS_MAP = {
    'w': (1,  1.0),   # Forward   -> LeftY
    's': (1, -1.0),   # Backward  -> LeftY
    'a': (0, -1.0),   # Left      -> LeftX
    'd': (0,  1.0),   # Right     -> LeftX
    'i': (4,  1.0),   # Cam Up    -> RightY
    'k': (4, -1.0),   # Cam Down  -> RightY
    'j': (3, -1.0),   # Cam Left  -> RightX
    'l': (3,  1.0),   # Cam Right -> RightX
    'u': (2,  1.0),   # LT
    'o': (5,  1.0),   # RT
}

# Mapping from key to button index
# Indices 0-7  : standard buttons (match joystick_ctrl.py JOY_BUTTON_MAP)
# Indices 11-14: D-Pad directions (match joystick.py dpad_map event names)
KEY_BUTTON_MAP = {
    'h':     0,   # A
    'q':     4,   # LB
    'e':     5,   # RB
    'b':     6,   # Back
    ' ':     7,   # Start
    'left':  11,  # D-Pad Left  (starts stepping in locomotion policy)
    'right': 12,  # D-Pad Right
    'up':    13,  # D-Pad Up
    'down':  14,  # D-Pad Down
}

NUM_AXES = 6
NUM_BUTTONS = 15  # 0-10: standard buttons, 11-14: D-Pad
# --- End Configuration ---

try:
    from pynput import keyboard
except ImportError:
    print("\n[ERROR] pynput library not found.")
    print("Please install it using: pip install pynput\n")
    exit(1)


class AgentPublisher(Node):
    def __init__(self):
        super().__init__('agent_publisher')
        self.publisher_ = self.create_publisher(Joy, '/joy', 10)

        # State management: mirrors the Joy message layout
        self.axes_state = [0.0] * NUM_AXES
        self.buttons_state = [0] * NUM_BUTTONS
        self.active_keys = set()

        self.timer = self.create_timer(0.05, self.publish_command)  # Publish at 20Hz
        self.get_logger().info('Agent Publisher started. Publishing on /joy (sensor_msgs/Joy).')
        self.print_instructions()

    def print_instructions(self):
        print("------------------------------------------")
        print("Keyboard Control for Agent Publisher:")
        print("  - W/A/S/D             : Left Stick (Move)")
        print("  - Arrow Left          : D-Pad Left  (start stepping)")
        print("  - Arrow Up/Down/Right : D-Pad Up/Down/Right")
        print("  - I/J/K/L             : Right Stick (Camera)")
        print("  - U/O                 : Left/Right Triggers")
        print("  - Q/E                 : LB/RB")
        print("  - H                   : A button")
        print("  - Space/B             : Start/Back")
        print("  - Press 'Esc' to exit.")
        print(f"  - Max Speed: {MAX_SPEED}")
        print("------------------------------------------")

    def on_press(self, key):
        try:
            key_char = key.char
        except AttributeError:
            key_char = key.name  # e.g. 'up', 'down', 'left', 'right', 'esc'

        if key_char in self.active_keys:
            return  # Avoid repeat events for held-down keys
        self.active_keys.add(key_char)

        if key_char in KEY_AXIS_MAP:
            axis_index, direction = KEY_AXIS_MAP[key_char]
            self.axes_state[axis_index] = direction * MAX_SPEED

        if key_char in KEY_BUTTON_MAP:
            button_index = KEY_BUTTON_MAP[key_char]
            self.buttons_state[button_index] = 1

    def on_release(self, key):
        try:
            key_char = key.char
        except AttributeError:
            key_char = key.name

        if key_char not in self.active_keys:
            return
        self.active_keys.remove(key_char)

        if key_char in KEY_AXIS_MAP:
            axis_index, _ = KEY_AXIS_MAP[key_char]
            # Only reset if no other key is still pressing the same axis
            is_other_key_active = any(
                active_key in KEY_AXIS_MAP and KEY_AXIS_MAP[active_key][0] == axis_index
                for active_key in self.active_keys
            )
            if not is_other_key_active:
                self.axes_state[axis_index] = 0.0

        if key_char in KEY_BUTTON_MAP:
            button_index = KEY_BUTTON_MAP[key_char]
            self.buttons_state[button_index] = 0

        if key == keyboard.Key.esc:
            print("Escape key pressed. Shutting down...")
            rclpy.shutdown()
            return False  # Stop the listener thread

    def publish_command(self):
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes = [float(v) for v in self.axes_state]
        msg.buttons = list(self.buttons_state)
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    agent_publisher = AgentPublisher()

    listener = keyboard.Listener(
        on_press=agent_publisher.on_press,
        on_release=agent_publisher.on_release)
    listener.start()

    try:
        rclpy.spin(agent_publisher)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        print("Cleaning up and shutting down.")
        listener.stop()
        agent_publisher.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
