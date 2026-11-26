"""On-site mode switch diagnostic for the official G1 FSM 500 and 801."""

import argparse
from collections.abc import Callable
from datetime import datetime
import hashlib
import hmac
import json
from pathlib import Path
import re
import subprocess
import time


REGULAR = 500
WALKRUN = 801
CYCLES = 5
OBSERVE_SECONDS = 150
ROOT_PORT = Path("/sys/bus/pci/devices/0004:00:00.0")
NVME = Path("/sys/class/nvme/nvme0")
AER_FILES = (
    "aer_rootport_total_err_cor",
    "aer_rootport_total_err_nonfatal",
    "aer_rootport_total_err_fatal",
)


class ProbeError(RuntimeError):
    pass


def emit(event, **fields):
    print(json.dumps({
        "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "monotonic_s": round(time.monotonic(), 3),
        "event": event,
        **fields,
    }), flush=True)


def read(path):
    return Path(path).read_text().strip()


def storage_sample():
    root_options = next(line.split()[3] for line in read("/proc/mounts").splitlines()
                        if line.split()[1] == "/")
    return {
        "nvme": read(NVME / "state"),
        "root_rw": "rw" in root_options.split(","),
        "aer": {name: int(read(ROOT_PORT / name)) for name in AER_FILES},
    }


class StorageWatch:
    """Read in-memory kernel state. The 4090 side records kernel and SMART logs."""

    def __init__(self, sample=storage_sample):
        self.sample = sample
        self.baseline = None
        self.next_sample = 0.0

    def check(self, *, force=False):
        if not force and time.monotonic() < self.next_sample:
            return
        try:
            state = self.sample()
        except (OSError, ValueError, StopIteration) as exc:
            raise ProbeError(f"storage status unavailable: {exc}") from exc
        self.next_sample = time.monotonic() + 0.25
        if state["nvme"] != "live" or not state["root_rw"]:
            raise ProbeError(f"storage lost or root became read-only: {state}")
        if self.baseline is None:
            self.baseline = state
            emit("storage_baseline", **state)
        if self.baseline["aer"] != state["aer"]:
            raise ProbeError(f"PCIe AER counters changed: before={self.baseline}, after={state}")


def confirm_fsm(env, expected):
    actual = env.sync_operator_fsm()
    status = env.get_control_status()
    if (status["authority_state"] != "INTERNAL" or status["publish_enabled"]
            or status["active_publish_count"] != 0
            or actual != expected):
        raise ProbeError(f"expected official FSM {expected}; read back {actual}, status={status}")


def wait_ready(*, expected, entered, dwell, still, timeout, period,
               stationary, check_storage, now, sleep):
    stable_since = None
    reason = "no stationary sample before deadline"
    while now() - entered <= timeout:
        check_storage()
        is_still, reason = stationary()
        sampled = now()
        if not is_still:
            stable_since = None
        elif stable_since is None:
            stable_since = sampled
        if (sampled - entered <= timeout and sampled - entered >= dwell
                and stable_since is not None and sampled - stable_since >= still):
            return
        sleep(period)
    raise ProbeError(f"FSM {expected} did not meet still/dwell gate within {timeout}s: {reason}")


def switch(env, source, target, entered, cycle, check_storage, now, output):
    check_storage(force=True)
    confirm_fsm(env, source)
    check_storage(force=True)
    send = (env.switch_regular_to_walkrun if target == WALKRUN
            else env.switch_walkrun_to_regular)
    started = now()
    output("switch_request", cycle=cycle, source=source, target=target,
           source_dwell_s=round(started - entered, 3))
    result = send()
    trace = (env.last_walkrun_fsm_trace if target == WALKRUN
             else env.last_regular_fsm_trace)
    output("bridge_return", cycle=cycle, target=target, result=result, trace=trace,
           elapsed_s=round(now() - started, 3))
    check_storage(force=True)
    if result != 0:
        raise ProbeError(f"official FSM request {source} → {target} failed: {result}")
    confirm_fsm(env, target)
    confirmed = now()
    output("switch_confirmed", cycle=cycle, fsm=target, publish_count=0,
           elapsed_s=round(confirmed - started, 3))
    return confirmed


def run_phase(env, phase, *, stationary: Callable, check_storage: Callable,
              still_seconds, wait_seconds, period, observation_seconds=OBSERVE_SECONDS,
              now=time.monotonic, sleep=time.sleep, output=emit):
    dwell = 5.0 if phase == "paced" else 0.0
    check_storage(force=True)
    confirm_fsm(env, REGULAR)
    entered = now()
    gates = dict(timeout=wait_seconds, period=period, stationary=stationary,
                 check_storage=check_storage, now=now, sleep=sleep)
    output("phase_started", phase=phase, cycles=CYCLES, dwell_s=dwell,
           still_seconds=still_seconds, still_timeout_s=wait_seconds)
    for cycle in range(1, CYCLES + 1):
        # Require a valid, still sample before entering walk/run.
        wait_ready(expected=REGULAR, entered=entered, dwell=dwell, still=0, **gates)
        entered = switch(env, REGULAR, WALKRUN, entered, cycle, check_storage, now, output)
        wait_ready(expected=WALKRUN, entered=entered, dwell=dwell, still=still_seconds, **gates)
        entered = switch(env, WALKRUN, REGULAR, entered, cycle, check_storage, now, output)
        output("cycle_complete", cycle=cycle)

    output("observation_started", seconds=observation_seconds)
    deadline = now() + observation_seconds
    while now() < deadline:
        check_storage()
        sleep(min(0.25, max(0, deadline - now())))
    check_storage(force=True)
    confirm_fsm(env, REGULAR)


def confirm_host(machine_id_sha256):
    actual = hashlib.sha256(read("/etc/machine-id").encode("ascii")).hexdigest()
    if not hmac.compare_digest(actual, machine_id_sha256):
        raise ProbeError("target machine fingerprint does not match")
    power = subprocess.run(
        ["sudo", "-n", "/usr/sbin/nvpmodel", "-q"], capture_output=True,
        text=True, timeout=5, check=True,
    ).stdout
    if not re.search(r"^NV Power Mode:\s*MAXN\s*$", power, re.MULTILINE):
        raise ProbeError(f"MAXN required; current nvpmodel output: {power!r}")
    latency = read(NVME / "power/pm_qos_latency_tolerance_us")
    if latency != "100000":
        raise ProbeError(f"expected unchanged APST baseline (100000 µs); read {latency!r}")
    emit("preflight", power="MAXN", apst_latency_us=latency,
         boot_id=read("/proc/sys/kernel/random/boot_id"),
         aspm_policy=read("/sys/module/pcie_aspm/parameters/policy"))


def execute(phase, machine_id_sha256):
    confirm_host(machine_id_sha256)
    storage = StorageWatch()
    storage.check(force=True)

    import g1_loco_bridge
    from robojudo.config.g1.g1_baduanjin_suite_cfg import g1_baduanjin_23dof_suite
    from robojudo.environment.g1_loco_env import G1LocoEnv
    from robojudo.pipeline.g1_native_loco_mimic_pipeline import G1NativeLocoMimicPipeline
    from robojudo.tools.mimic_safety import MimicSafetyGuard, SafetyViolation

    if (getattr(g1_loco_bridge, "MIMIC_SAFETY_REVISION", None) != 2026112301
            or not hasattr(g1_loco_bridge, "ControllerAuthority")
            or not hasattr(G1LocoEnv, "attach_control_authority")):
        raise ProbeError("use the installed authority-enabled RoboJudo checkout and bridge (2026112301)")

    cfg = g1_baduanjin_23dof_suite.model_fields["env"].default.model_copy(deep=True)
    if cfg.dof.num_dofs != 23 or cfg.unitree.regular_fsm_id != REGULAR:
        raise ProbeError("installed suite does not have the expected 23DoF/FSM500 environment")
    cfg.act = False
    token = g1_loco_bridge.ControllerAuthority()
    if result := token.acquire():
        raise ProbeError(f"controller lock unavailable: {result}")
    env = G1LocoEnv(cfg, device="cpu")
    try:
        env.attach_control_authority(token)
        if result := env.begin_control_session():
            raise ProbeError(f"control session refused: {result}")
        pipeline = G1NativeLocoMimicPipeline
        guard = MimicSafetyGuard(env.joint_names)
        legs = [i for i, name in enumerate(env.joint_names)
                if any(key in name for key in pipeline.STANCE_JOINT_KEYS)]

        def stationary():
            env.update()
            data = env.get_data()
            try:
                guard.state(data, env.safety_state_age())
            except SafetyViolation as exc:
                return False, str(exc)
            speed = max(abs(data["dof_vel"][i]) for i in legs)
            return speed <= pipeline.REGULAR_STILL_VEL, f"max leg/waist speed {speed:.3f} rad/s"

        run_phase(env, phase, stationary=stationary, check_storage=storage.check,
                  still_seconds=pipeline.REGULAR_STILL_S, wait_seconds=pipeline.REGULAR_WAIT_S,
                  period=cfg.unitree.control_dt)
        if result := env.end_control_session():
            raise ProbeError(f"control session end failed: {result}")
    finally:
        # Native close() only shuts down local resources. It never sets an FSM.
        if result := env.shutdown():
            raise ProbeError(f"controller shutdown failed: {result}; check mode with handset")
    emit("phase_passed", phase=phase, cycles=CYCLES, publish_count=0, final_fsm=REGULAR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fast", "paced"), required=True)
    parser.add_argument("--machine-id-sha256", required=True,
                        help="SHA-256 of the verified target's machine-id without newline")
    parser.add_argument("--execute", action="store_true", help="allow on-site FSM requests")
    args = parser.parse_args()
    if not args.execute:
        parser.error("on-site execution requires --execute")
    if not re.fullmatch(r"[a-f0-9]{64}", args.machine_id_sha256):
        parser.error("--machine-id-sha256 must be a lowercase SHA-256 digest")
    try:
        execute(args.phase, args.machine_id_sha256)
    except (Exception, KeyboardInterrupt) as exc:
        emit("phase_aborted", phase=args.phase, error=repr(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
