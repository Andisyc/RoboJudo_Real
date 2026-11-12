请从 https://github.com/Andisyc/RoboJudo_Real 的 **g1loco** 分支拉取最新三段联播版本，部署到此前测试的 **23DoF 背包机器人**。所有模型和代码都在仓库，不需要更新包。

这版已简化启动：**每次开始都自动从 cut1 播放到 cut3，语音 Agent 不需要调用 `/select_policy`。** 单段测试入口保持原样。

请按下面四步完成：

1. **更新并离线检查。** 保存原部署、现场修改、启动命令、Python/bridge 路径和网卡/DDS；在独立目录检出 g1loco，记录完整提交 SHA，保留必要现场定制。只更新本台 23DoF，不改 29DoF；沿用已测通的 bridge 和常规运控请求 500。三个 ONNX 本轮没有再变，先运行：

   ```bash
   python scripts/check_baduanjin_suite.py
   python scripts/check_g1_native_config.py -c g1_baduanjin_23dof_suite --net-if "$UNITREE_NET_IF"
   python -m pytest -q tests/test_baduanjin_http.py tests/test_baduanjin_suite.py tests/test_pose_transition.py tests/test_mimic_safety_offline.py tests/test_g1_native_integration.py tests/test_g1_backpack_contract.py
   ```

   使用 controller 的 Python 环境；HTTP 离线测试需 fastapi/uvicorn/httpx。离线检查不启动真实控制器、不发机器人命令。

2. **保留原语音调用方式。** 沿用现场 HTTP 服务地址、语音唤醒词和服务托管方式，让服务运行本次检出的代码。仅在首次加载或服务重建时调用：`POST /start`，JSON 为 `{"config_name":"g1_baduanjin_23dof_suite"}`。已经加载本配置时不重复 `/start`。

   **每次演出只调用一次 `POST /enter_mimic`，不带请求体。** 首次、整套结束后再次开始、人工退出后重新开始，controller 都自动从 cut1 首帧开始。不要额外 select、逐段 switch 或逐段 stop/start。旧服务若还在舞蹈或交还中，先按原流程完成交还、确认官方运控且 publish_enabled=false，再更换服务；不要同时开 HTTP 服务与 start.sh 两个控制器。

3. **现场测试两轮。** 完成部署与离线检查后，交由现场人员按原流程说原来的语音指令，第一轮确认：常规→走跑→3 秒启动过渡→cut1→3 秒交接→cut2→3 秒交接→cut3→走跑→常规 500。结束后直接再次说同样指令，确认第二轮仍从 cut1 开始，不需要选段或重启。HTTP 200 只表示命令入队，须观察实际入舞与最终常规模式读回。日志应有 `SUITE_START` / `suite_start`（start_index=0）以及两次 segment_transition（0→1、1→2）。

   重点看两处衔接是否突跳、滑脚和转身，以及 cut3 第 2200–2500、3550–3850 帧的前倾/垫步是否改善。已经在跳舞或等待入舞时再次收到开始指令，不应把当前舞蹈强制重置。保留原人工急停；`/return_loco`、A/Select 仍是等待合适站姿退出，不是急停。收尾到默认站姿的旧实验仍未启用。

4. **返回结果和记录。** 沿用现场状态记录器，给 HTTP 服务设置唯一的 ROBOJUDO_RECORD_DIR / ROBOJUDO_RUN_ID，确认 motion.msgpack、state.msgpack、stdout/stderr 和视频均保存。服务不会自动启动 start.sh 的记录器。返回提交 SHA、实际配置/启动命令、模型哈希、完整日志与视频、语音请求时间；分别说明第一轮、第二轮是否从 cut1 开始，两处衔接、cut3 前倾和最终常规 500 是否通过。

仅需单段复测时，继续使用 `g1_baduanjin_23dof_select`：通过原 `/select_policy` 或 R1/L1 选择 0=cut1、1=cut2、2=cut3，再开始，选中的段不会被自动改为 cut1。联播模式不用这些选段操作。

模型与详细部署说明见 `docs/baduanjin_suite_20261112_zh.md`；HTTP 调用与旧日志证据见 `docs/baduanjin_http_compatibility_20261112_zh.md`。本轮未改模型、3 秒过渡、PD、cut2 的 UNITREE 输入或 FSM 设置；软件离线测试通过不等于本版已通过实机测试。
