# g1_loco_bridge_23dof_backpack

G1 23DoF native-locomotion/Mimic bridge, v3 revision **2026110310**.
The 23 joints map into the SDK's fixed 29-motor array; the six unused motors
keep zero gains. The controller implementation matches `g1_loco_bridge_29dof`.

Build in the actual deployment Python environment after installing Unitree
SDK2 (including `unitree/robot/g1/audio/g1_audio_client.hpp`), CycloneDDS and C++17:

```bash
python -m pip install --no-deps --no-build-isolation --force-reinstall ./packages/g1_loco_bridge_23dof_backpack
python -c 'import g1_loco_bridge_23dof_backpack as m; print(m.MIMIC_SAFETY_REVISION)'
```

Both 23DoF pipeline configs, including the backpack config, select
`g1_loco_bridge_23dof`. Build the separately named backpack package only for
programs that explicitly import it. Use exactly one low-level DDS owner;
do not instantiate this controller alongside `unitree_cpp.UnitreeController`.

The v3 pipeline owns Start/A/Select, stationarity and return timing. Start
in regular locomotion requests WALKRUN before entry; A/Select during a dance
wait for a stance before handback. Native `close()` only stops local resources,
so close through the pipeline to retain the handback/hold behavior.

See the repository [README](../../README.md) for configuration, offline checks,
startup and the retained handback command-stream window. These controls are
not an emergency stop.
