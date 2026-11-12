# G1 真机启动 DoF 检查交接

本次改动防止 23DoF 或 29DoF 的 G1 真机选错启动配置。程序在加载 Mimic checkpoint、接管机器人控制之前，自动比较真机 DoF 和所选配置的 DoF。两种真机共用原来的启动链路，不需要人工登记 DoF 或增加命令行参数。

## 实现

唯一改动的生产文件是 [`packages/g1_loco_bridge/src/g1_loco_controller.cpp`](../packages/g1_loco_bridge/src/g1_loco_controller.cpp)，入口是 `G1LocoController::self_check()`。

1. 复用 `LowStateHandler()` 已有的 DDS 订阅和 CRC 校验，从 `mode_machine_` 读取真机上报的机型编号。没有状态或状态超过现有的 0.1 秒新鲜度阈值时，自检返回 `false`，由 `G1LocoEnv.self_check()` 继续等待，直至原有超时。
2. 根据[宇树官方 G1 机型表](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/README.md#overview)，将 `1 / 4 / 10` 映射为 23DoF，将 `2 / 5 / 11 / 13 / 15 / 18` 映射为 29DoF。其他编号不做推断，直接报 `unsupported mode_machine=<编号>`。
3. 将识别出的 DoF 与启动配置传给 Bridge 的 `num_dofs_` 比较。匹配时继续原有自检；错配时抛出 `RuntimeError`，例如 `G1 startup refused: hardware is 29 DoF (mode_machine=5), configuration expects 23 DoF`。这条检查不发送电机指令，也不请求切换 FSM。

[`G1LocoEnv.__init__()`](../robojudo/environment/g1_loco_env.py) 已经调用 Bridge 自检；[`G1NativeLocoMimicPipeline.__init__()`](../robojudo/pipeline/g1_native_loco_mimic_pipeline.py) 先构造环境，再创建 `PolicyManager`，所以错配会在 checkpoint 加载前被拦下。新检查比较的是**真机和启动配置**，配置和 checkpoint 的关节匹配仍由原有校验负责。23DoF backpack 的硬件 DoF 仍按 23DoF 检查。

## 验证与部署

回归用例位于 [`tests/cpp/test_mimic_controller.cpp`](../tests/cpp/test_mimic_controller.cpp)。已运行 `python3 tests/run_mimic_native_offline.py`，同一份生产 Bridge 源码用假 SDK 分别通过了 23DoF 和 29DoF 测试（各 35 个场景）。新用例覆盖两种匹配、两个错配方向、未知机型、缺失和过期状态，并检查没有发送电机指令或请求切换模式。未连接真机，未验证部署机上的实际二进制。

部署时，在运行 RoboJudo 的 Python 环境中，从仓库根目录重新编译并覆盖安装共享包：

```bash
python -m pip install --no-deps --no-build-isolation --force-reinstall ./packages/g1_loco_bridge
```

23DoF、23DoF backpack、29DoF 和 `v2` 配置均导入 `g1_loco_bridge`，其余同名变体目录不参与这条启动链路。`MIMIC_SAFETY_REVISION` 没有变：只同步源码或仍加载旧二进制时，不会自动提示重编，也不会执行新检查。下一位 Agent 若接手部署，先核对部署环境加载的是从更新后源码构建的 `g1_loco_bridge`，再沿用现有启动命令。
