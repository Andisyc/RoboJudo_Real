"""Observation-only, bounded asynchronous motion/control-cycle recording."""
from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import msgpack

logger = logging.getLogger(__name__)


def freeze(value):
    """Detach payloads before the controller can mutate their arrays."""
    if isinstance(value, dict):
        return {str(k): freeze(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [freeze(v) for v in value]
    if hasattr(value, "tolist"):
        return freeze(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported trace value: {type(value).__name__}")


def capture_inference(policy, query_frame, observation_command, outputs, actions, scaled_actions):
    """Called after inference; failures affect recording only."""
    if not getattr(policy, "_motion_trace_enabled", False):
        return
    try:
        policy._motion_trace_snapshot = freeze({
            "query_frame": query_frame,
            "segment_generation": getattr(policy, "_motion_trace_generation", 0),
            "observation_reference": observation_command,
            "output_reference": dict(zip(
                ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w"), outputs[1:5])),
            "raw_actions": outputs[0], "filtered_actions": actions,
            "scaled_actions": scaled_actions,
            "inference_end_monotonic_ns": time.monotonic_ns(),
        })
    except Exception as exc:
        policy._motion_trace_snapshot = None
        logger.error("Motion inference snapshot unavailable: %s", exc)


def capture_target(policy, target):
    if not getattr(policy, "_motion_trace_enabled", False):
        return
    try:
        snapshot = getattr(policy, "_motion_trace_snapshot", None)
        if snapshot is not None:
            snapshot["policy_target"] = freeze(target)
    except Exception as exc:
        policy._motion_trace_snapshot = None
        logger.error("Motion target snapshot unavailable: %s", exc)


class MotionTrace:
    def __init__(self, path, metadata, queue_size=4096):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("xb")
        self._queue = queue.Queue(maxsize=queue_size)
        self._closed = False
        self.error = None
        self.dropped = 0
        self.counts = {}
        self.cycle = 0
        self.segment = 0
        self._last_segment = None
        self._last_phase = None
        self._write(self._event("header", {
            "schema_version": 1, "metadata": metadata,
            "timestamp_semantics": "host receive/call time; DDS tick remains in state log",
            "clock": "time.monotonic_ns on the same host and boot",
            "units": {"q": "rad", "dq": "rad/s", "gyro": "rad/s", "acceleration": "m/s^2"},
            "reference_quaternion_order": "wxyz", "dds_quaternion_order": "wxyz",
            "reference_positions_frame": "exported reference world",
            "query_frame_semantics": "ONNX input index, may be clamped inside model",
        }))
        self._stream.flush()
        self._thread = threading.Thread(target=self._run, name="motion-trace", daemon=True)
        self._thread.start()

    @staticmethod
    def _event(kind, payload):
        return {"kind": kind, "monotonic_time_ns": time.monotonic_ns(),
                "receive_time_ns": time.time_ns(), "payload": freeze(payload)}

    def _write(self, event):
        self._stream.write(msgpack.packb(event, use_bin_type=True))

    def _run(self):
        try:
            last_flush = time.monotonic()
            while True:
                item = self._queue.get()
                if item is None:
                    break
                self._write(item)
                if time.monotonic()-last_flush >= .5:
                    self._stream.flush()
                    last_flush = time.monotonic()
            self._stream.flush()
        except Exception as exc:
            self.error = str(exc)
            logger.error("Motion recording failed; controller output unchanged: %s", exc)
        finally:
            self._stream.close()

    def record_event(self, kind, **payload):
        if self._closed or self.error:
            return
        try:
            self._queue.put_nowait(self._event(kind, payload))
            self.counts[kind] = self.counts.get(kind, 0)+1
        except queue.Full:
            self.dropped += 1
            if self.dropped == 1:
                logger.error("Motion trace queue overflow; recording is incomplete")
        except Exception as exc:
            self.error = str(exc)
            logger.error("Motion recording disabled: %s", exc)

    def record_cycle(self, policy, target, phase, commands=(), pipeline_step=None):
        """Return a cycle ID; this method never owns or changes a control target."""
        try:
            cfg = policy.cfg_policy
            snapshot = getattr(policy, "_motion_trace_snapshot", None)
            key = (id(policy), getattr(policy, "_motion_trace_generation", 0))
            if key != self._last_segment:
                self.segment += 1
                self._last_segment = key
                self.record_event("segment", segment=self.segment,
                                  policy_name=getattr(cfg, "policy_name", type(policy).__name__),
                                  policy_file=str(getattr(cfg, "policy_file", "")),
                                  policy_joint_names=policy.cfg_action_dof.joint_names,
                                  configured_policy_hz=getattr(cfg, "freq", None),
                                  reference_fps=None,
                                  reference_fps_note="not inferred from control frequency; use matching NPZ fps")
            if phase != self._last_phase:
                self.record_event("phase", phase=phase, segment=self.segment)
                self._last_phase = phase
            cycle = self.cycle
            self.cycle += 1
            self.record_event("cycle", cycle=cycle, segment=self.segment, phase=phase,
                              pipeline_step=pipeline_step, commands=commands,
                              inference=snapshot, final_target=target,
                              send_state="attempted")
            return cycle
        except Exception as exc:
            self.error = str(exc)
            logger.error("Motion cycle recording disabled: %s", exc)
            return None

    def close(self):
        if self._closed:
            return
        self.record_event("summary", counts=self.counts.copy(), dropped=self.dropped,
                          error=self.error, complete=not self.error and self.dropped == 0)
        self._closed = True
        # Shutdown is outside the control cycle; never block indefinitely on a failed writer.
        deadline = time.monotonic()+3
        while self._thread.is_alive():
            try:
                self._queue.put(None, timeout=.05)
                break
            except queue.Full:
                if time.monotonic() >= deadline:
                    break
        self._thread.join(timeout=max(0., deadline-time.monotonic()))
        if self._thread.is_alive():
            self.error = self.error or "writer shutdown timed out; incomplete log"
        logger.warning("Motion trace saved: %s counts=%s dropped=%s error=%s",
                       self.path, self.counts, self.dropped, self.error)
