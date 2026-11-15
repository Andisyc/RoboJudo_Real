import time
from queue import Full, Queue
from threading import Event, Thread

from pynput import keyboard  # TODO: fix DISPLAY error on Linux without GUI


class KeyboardThread(Thread):
    def __init__(self, event_queue: Queue, *, hardware_motion: bool = False):
        super().__init__(name="KeyboardThread", daemon=True)
        self.event_queue = event_queue
        self._hardware_motion = hardware_motion
        self._closed = Event()
        self._overflowed = Event()
        self._listener = None

    def _emit(self, key, pressed: bool):
        event = {
            "type": "keyboard",
            "name": self.get_key_name(key),
            "pressed": pressed,
            "timestamp": time.time(),
        }
        if not self._hardware_motion:
            self.event_queue.put(event)
            return
        try:
            self.event_queue.put_nowait(event)
        except Full:
            self._overflowed.set()
            return False

    def input_failed(self) -> bool:
        return self._overflowed.is_set() or not self.is_alive()

    def close(self):
        self._closed.set()
        if self._listener is not None:
            self._listener.stop()

    def run(self):
        with keyboard.Listener(
            on_press=lambda key: self._emit(key, True),
            on_release=lambda key: self._emit(key, False),
        ) as listener:
            self._listener = listener
            if self._closed.is_set():
                listener.stop()
            listener.join()

    def get_key_name(self, key):
        try:
            return key.char if key.char is not None else str(key)
        except AttributeError:
            return str(key)


if __name__ == "__main__":
    event_queue = Queue()
    kb_thread = KeyboardThread(event_queue)
    kb_thread.start()

    print("Press keys (ESC to exit)...")
    while True:
        event = event_queue.get()
        print(event)
        if event["name"] == "Key.esc" and event["pressed"]:
            break
