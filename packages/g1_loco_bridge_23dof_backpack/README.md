# g1_loco_bridge

`g1_loco_bridge` is the RoboJuDo-owned C++ boundary for temporarily handing a
Unitree G1 from its internal locomotion controller to user DDS control and back.
It links against the robot's installed official `unitree_sdk2`; it does not
modify or wrap the third-party `packages/unitree_cpp` module.

The controller starts read-only. Constructing it subscribes to robot state and
queries the locomotion FSM, but does not call `SwitchToUserCtrl()` and does not
publish `rt/user_lowcmd`. An explicit acquire requires native `WALKRUN`,
continuously publishes the first fully framed policy target during the direct
user-control request, and confirms FSM ID `1000` before reporting that user
control is active.

Install on the G1 computer after installing a compatible official SDK2:

```bash
python -m pip install -e packages/g1_loco_bridge
```

The build requires the installed header
`unitree/robot/g1/loco/g1_loco_client.hpp` and `libunitree_sdk2.so`.

Do not construct `G1LocoController` and the legacy `unitree_cpp.UnitreeController`
in the same process. A deployment must choose exactly one low-level DDS owner.

RoboJuDo integration uses the isolated `g1_native_loco_mimic` configuration:

```bash
python scripts/run_pipeline.py -c g1_native_loco_mimic
```

The process starts in Unitree native locomotion. On the Unitree remote, `Start`
performs the direct `WALKRUN -> USER_CTRL` handoff and enters the selected dance,
`Select` returns to native `WALKRUN`, `R1`/`L1` select the next/previous dance
while native locomotion is active, and `A` confirms `PASSIVE` before closing the
bridge.
