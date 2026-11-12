请从 https://github.com/Andisyc/RoboJudo_Real 的 **g1loco** 分支拉取最新版本，部署到此前测试的 **23DoF 背包机器人**，测试 cut1→cut2→cut3 三段联播。模型已在仓库，不需要更新包。

这次只更新 Cut3 模型及默认选择、校验和说明。Cut1、Cut2、输入设置、PD、controller 流程、3 秒启动/段间过渡、常规运控请求 500 均沿用之前版本。每次演出自动从 cut1 开始，语音 Agent 无须 `/select_policy`。

请按下面四步完成，不要自行改模型、滤波、补偿或控制流程：

1. **拉取并核对版本。** 保留现有部署、未提交修改、Python/bridge 路径、网卡和 DDS 设置，在独立目录检出 `g1loco`，记录完整提交 SHA。确认默认入口为 `g1_baduanjin_23dof_suite`，索引为 0=cut1、1=cut2、2=cut3，且加载以下三个模型：

   ```text
   0 cut1_scratch_v2_i15000_zero_translation.onnx
     SHA256 2afc0f0cb58605b5abc7eb2c127148f9e5e0993e6dc4b54c8ff0540af155e53d
   1 eight_cut_2_v010_1.2x.onnx
     SHA256 bbca0a34feab8f9f73f150003910190cfe9412ef91b4a00f0b65197a2a0312bc
   2 cut3_jointv1_pitch8_rightless_20261112.onnx
     SHA256 70a509a3082737909a2ca1d02680a68122bba098bd3b7b0cacb42d76d6212e3e
   ```

   新 Cut3 保留旧两段 6°，在《联调-v1》重新定位的查询帧 **2980–3280、4350–4650** 新增最大 8° 减俯仰和少量减右转。不要选成旧 `cut3_pitch_ref6_20261112`，也不要叠加额外髋部补偿或换成仅减8°的对照模型。Cut2 保留原 `UNITREE` 平移输入。

   分支在本次模型更新之前已有同事的 `fecfca2` 启动 DoF 检查提交，涉及共享 bridge 源码。它不是本次模型优化的一部分。本次模型测试沿用现场已测通且兼容的 bridge 二进制，不因换 ONNX 自动重编或替换它；记录实际加载路径、二进制哈希及是否已部署 DoF 检查。拉取源码不等于二进制已更新。该独立改动的说明见 `docs/G1_DOF_STARTUP_CHECK_HANDOFF.md`，不要把它引起的启动拒绝误记成模型问题。

2. **先做离线检查。** 用现场 controller 的 Python 环境执行以下检查；它们不构造真实机器人环境、不发控制指令。先保留/设置现场部署配置中的控制网卡和记录器网卡，使二者一致，再将 `UNITREE_NET_IF` 设为同一网卡；该检查参数不会替你改写控制器配置，不照抄其他机器的名称。

   ```bash
   python scripts/check_baduanjin_suite.py
   python scripts/check_g1_native_config.py -c g1_baduanjin_23dof_suite --net-if "$UNITREE_NET_IF"
   python -m pytest -q tests/test_baduanjin_http.py tests/test_baduanjin_suite.py tests/test_pose_transition.py tests/test_mimic_safety_offline.py tests/test_g1_native_integration.py tests/test_g1_backpack_contract.py
   ```

   HTTP 离线测试需 fastapi/uvicorn/httpx，测试依赖使用现有兼容环境，不全量升级机器人环境。若模型哈希、23DoF 或输入/PD 检查失败，保留错误记录，先排查部署文件和环境。

3. **沿用原语音方式测试两轮。** 让现场 HTTP 服务运行本次检出的代码，保留原服务地址、唤醒词和托管方式。旧服务须先按原流程完成交还，确认官方运控且 `publish_enabled=false` 后再更换；不要同时开 HTTP 服务与 `start.sh` 两个 controller。

   首次加载或服务重建时：`POST /start`，JSON 为 `{"config_name":"g1_baduanjin_23dof_suite"}`。已经加载此配置时不要重复 `/start`。每次演出仅调用一次 **`POST /enter_mimic`，不带请求体**；无需 `/select_policy`，不用逐段启动或重启。由现场人员按原流程说原来的语音指令开始真实动作，部署与离线检查本身不触发跳舞。

   第一轮应为：常规→走跑→3秒启动过渡→cut1→3秒交接→cut2→3秒交接→新cut3→走跑→常规500。结束后再说同样指令，第二轮也应从cut1首帧开始。HTTP 200不是实际模式确认，要看状态读回和日志；保留人工急停，`/return_loco`、A/Select仍是等待站姿退出，不是急停。

   重点看 Cut3 **2980–3280、4350–4650** 帧的前倾、左脚接地和回正垫步，并复查旧 **2200–2500、3550–3850** 帧、两次段间过渡和收尾。新窗口对应旧《联调-v1》约5:55–6:01、6:23–6:29，其中截图约5:57.75；新视频有启动等待差异，**按日志的 policy_name/query_frame 对齐，不机械套用视频秒数**。记录左脚先移动还是躯干先倾、脚跟是否抬起、动作结束是否仍请求并读回500。不要根据仿真结果预先认定真机垫步已解决。

4. **交回结果。** 沿用现场记录器，并为HTTP服务设置唯一的 `ROBOJUDO_RECORD_DIR` / `ROBOJUDO_RUN_ID`，确认 `motion.msgpack`、`state.msgpack`、stdout/stderr和完整视频都保存且能按时间关联。HTTP服务不会自动启动`start.sh`的状态记录器。日志应有 `suite_start` / `SUITE_START` 的 `start_index=0`、0→1和1→2两次段间交接、新Cut3模型名，以及请求500和最终读回记录。

   返回完整提交SHA、实际启动配置/命令、三模型哈希、Python与bridge路径/哈希、日志与原速完整视频、两次语音请求时间。分别说明：两轮是否从cut1开始、cut1/cut2是否保持原效果、两处衔接、新两段前倾/垫步、旧两段及收尾、常规500切换是否通过。若异常，保留现场数据并按原人工退出流程处理，不自行切换501或增加自动阻尼。

仅需单段复测时，使用 `g1_baduanjin_23dof_select`，通过原 `/select_policy` 或 R1/L1 选择2再开始；它共享上述新Cut3模型，但仍只播放所选段，不自动改回cut1。此前其他独立Cut3配置没有被本次更新替换，不要用错入口。

详细改动与离线结果见 `docs/baduanjin_suite_20261112_zh.md`；原HTTP调用合同见 `docs/baduanjin_http_compatibility_20261112_zh.md`。本版仅完成离线验证，尚待本次真机测试。需要回滚动作时，使用保留的上一版部署和Cut3 ref6模型，Cut1、Cut2不换版本。
