# g1_loco_bridge

This is the single RoboJuDo native-locomotion/Mimic bridge for both 23DoF and
29DoF G1. `G1NativeLocoMimicPipeline` owns the Python state machine. This package
owns SDK/DDS, motor mapping, control authority and confirmed FSM transitions.
The 2026110310 implementation is integrated into this original package; no
separately named 29DoF bridge is required.

Build in the Python environment that runs the controller, on the G1 computer:

```bash
python -m pip install --no-deps --no-build-isolation --force-reinstall ./packages/g1_loco_bridge
```

Requires Unitree SDK2 with G1 LocoClient and AudioClient headers, CycloneDDS
C/C++ and C++17. The package exports `MIMIC_SAFETY_REVISION = 2026110310`.
Constructing a controller subscribes to state and queries the FSM; only an
explicit acquire requests user control and starts LowCmd publishing.

`num_dofs=29` uses the full motor array directly. `num_dofs=23` maps joints to
indices 0–11, 12, 15–19 and 22–26; the six unused motors have zero gains.
RoboJuDo passes this value from the selected robot configuration. Do not
construct another low-level DDS owner alongside this controller.

The lab 29DoF command remains:

```bash
env -u CYCLONEDDS_URI python -u scripts/run_pipeline.py -c g1_native_loco_mimic
```

Start in regular locomotion requests WALKRUN before entering Mimic. Motion
completion or A/Select during a dance waits for a stance before handing back.
After WALKRUN settles, the pipeline requests the configured regular FSM (501
in the default lab configuration). Failed handbacks retain the hold target
and wait for the operator. A/Select is not an emergency stop.

Always close through the pipeline: native `close()` only stops local resources
and does not request Damp/PASSIVE. The handback RPC still follows a temporary
stop in LowCmd publishing, as in the supplied implementation.

The former `switch_walkrun_to_normal()`/FSM 500 path has been replaced by
`switch_walkrun_to_regular()` with confirmed, traced FSM switching. Rebuild
after pulling this update; an old binary cannot supply the new API.

See the repository README for configuration and offline checks. No local
offline check establishes real-hardware deployment behavior.
