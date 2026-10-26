# g1_loco_bridge_v2

Independent copy of `g1_loco_bridge` for the 29DoF G1 locomotion firmware that
reports WALKRUN as FSM 802. It accepts both 801 and 802 on acquire and on every
WALKRUN release confirmation. USER_CTRL remains exactly 1000 and PASSIVE exactly
1. The original Bridge packages are not replaced or modified.

## Install on the robot

Use the same Python environment as RoboJuDo. The installed Unitree SDK2 must
provide `SwitchToUserCtrl()` and `SwitchToInternalCtrl()`; CycloneDDS C and C++
libraries and development headers are required, as for the original Bridge.

For the private CycloneDDS installation used on this robot:

```bash
conda activate robojudo
(
  export CYCLONEDDS_HOME="$HOME/cyx/cyclonedds/install"
  export CMAKE_PREFIX_PATH="$CYCLONEDDS_HOME${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  python -m pip install -e packages/g1_loco_bridge_v2
  python -c 'import g1_loco_bridge_v2 as b; print(b.__file__); print(b.WALKRUN_FSM_IDS)'
)
```

The reported supported IDs should be `(801, 802)`. This import does not construct
a controller. The package uses its own extension and C++ namespace, so importing
it does not replace `g1_loco_bridge`. Do not instantiate multiple DDS control
owners in the same process.

## Explicit configuration selection

`g1_native_loco_mimic_v2` sets `env.bridge_variant="v2"` and retains the original
29DoF environment, motion policies and remote buttons. The legacy configuration
`g1_native_loco_mimic` still selects `g1_loco_bridge`. Existing 23DoF configurations
retain their original routing; selecting v2 with a 23DoF environment is rejected.
There is no fallback to the old Bridge if the new package is missing.

Run from the repository root, after the normal real-robot safety preparation:

```bash
env -u CYCLONEDDS_URI \
  LD_LIBRARY_PATH="$HOME/cyx/cyclonedds/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  python -u scripts/run_pipeline.py -c g1_native_loco_mimic_v2
```

The process initially observes native locomotion without publishing LowCmd.
`Start` requests Mimic; `Select` returns to native WALKRUN; `R1`/`L1` choose a
policy while native locomotion is active; `A` shuts down to PASSIVE.

Motor ordering, gains, control rate, DDS topics and SDK handoff calls are copied
unchanged from the original 29DoF Bridge. This is a firmware-compatibility variant,
not a 23DoF or backpack-specific motor mapping.

Shutdown behavior is also unchanged: `close()` requests PASSIVE even when control
is already internal. Normal process exit can therefore put a standing robot into
damping; it must not be treated as a return-to-WALKRUN command.
