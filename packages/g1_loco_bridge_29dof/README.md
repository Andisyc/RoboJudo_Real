# g1_loco_bridge_29dof

Same controller source as `g1_loco_bridge_23dof` (byte-identical
`g1_loco_controller.*` and `mimic_safety.hpp`; only the Python module name
differs). With `num_dofs == 29` the joints map 1:1 onto the G1 motor array, so
waist roll/pitch and the wrist pitch/yaw motors are commanded by the policy.

Used by `g1_native_loco_mimic` and `g1_native_loco_mimic_29dof`
(`bridge_variant="v3"`). The existing `-c g1_native_loco_mimic` launch command
now selects this package; `g1_native_loco_mimic_legacy` retains the old bridge.
Build on the machine that runs the controller:

```bash
python -m pip install --no-deps --no-build-isolation --force-reinstall ./packages/g1_loco_bridge_29dof
python -c 'import g1_loco_bridge_29dof as m; print(m.MIMIC_SAFETY_REVISION)'
```
