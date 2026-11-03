# g1_loco_bridge_23dof（历史包）

当前所有 native-loco/Mimic 配置共用 `G1NativeLocoMimicPipeline` 和
`packages/g1_loco_bridge`；不再加载此包。这里保留历史源码，不作为当前部署入口。

请使用原包构建命令：

```bash
python -m pip install --no-deps --no-build-isolation --force-reinstall ./packages/g1_loco_bridge
```

模型和关节配置仍在 `robojudo/config/g1/g1_native_loco_mimic_cfg.py` 选择。
参见仓库 README 的 G1 native Mimic 说明。
