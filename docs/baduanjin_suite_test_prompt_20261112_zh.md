请直接从 https://github.com/Andisyc/RoboJudo_Real 的 **g1loco** 分支部署并准备测试“23DoF 八段锦三段联播”。模型、代码、检查脚本和说明均已随仓库发布，不需要更新包。先阅读 `docs/baduanjin_suite_20261112_zh.md` 和 `docs/baduanjin_http_compatibility_20261112_zh.md`。若收到指定发布 SHA，检出该 SHA；否则从 g1loco 拉取并记录完整 SHA，确认包含 `g1_baduanjin_23dof_suite` 和 `cut3_pitch_ref6_20261112.onnx`。

现场已确认原测试的 **23DoF 背包机器人** 请求常规运控 **500** 成功。本次只针对这台机器，不换到其他 Unitree 机器人，不套用 29DoF 的 501。

1. 保存现场原部署 SHA、未提交 diff、启动命令、Python/bridge 路径、网卡和 DDS，保留刚测通的版本用于回滚。在独立目录检出 GitHub 更新、比较差异并合入必要现场定制，不使用 reset --hard 覆盖工作目录。包含 controller、HTTP 服务和配置修改，不能只替换 ONNX。本轮 C++ bridge 未改，沿用刚测通 500 的版本，不擅自重装全套依赖。

2. 核对固定顺序：0=`cut1_scratch_v2_i15000_zero_translation`，1=`eight_cut_2_v010_1.2x`，2=`cut3_pitch_ref6_20261112`。cut1/cut2 未改；cut2 保留 UNITREE 速度输入，不改 DUMMY。新 cut3 SHA-256 必须为 `47b75e8c34d8735d3bd788bfed90935e5e99d93f18f063dadcc7ac0d4966398c`，输入/输出 130/23 维，max_timestep=5299。运行下面的离线检查，不能创建真实机器人环境：

   ```bash
   python scripts/check_baduanjin_suite.py
   python scripts/check_g1_native_config.py -c g1_baduanjin_23dof_suite --net-if "$UNITREE_NET_IF"
   python -m pytest -q tests/test_baduanjin_http.py tests/test_baduanjin_suite.py tests/test_pose_transition.py tests/test_mimic_safety_offline.py tests/test_g1_native_integration.py tests/test_g1_backpack_contract.py
   ```

   使用现场 controller 的 Python 环境。HTTP 路由测试需已安装 fastapi/uvicorn/httpx；缺测试依赖时在离线副本解决并报告，不把启动真实服务当成检查。新配置的默认网卡不一定符合现场，确保 controller 与记录器使用同一实际网卡。

3. 现场使用语音 Agent 调 HTTP 服务，保持原服务地址、启动/托管方式和语音唤醒词。更新该服务实际运行的仓库路径和 Python，不能另启动 `start.sh` 形成第二个控制器。准备以下调用顺序，交由现场人员按原测试流程执行；离线阶段不执行这些运动调用：

   - `GET /status` 检查是否已有 controller。若旧配置在运行，先完成交还、确认官方运控且 publish_enabled=false，再按原管理流程停止并加载新配置，不能叠加 `/start`。
   - `POST /start`，JSON 为 `{"config_name":"g1_baduanjin_23dof_suite"}`。必须显式传新配置名；旧默认配置不会自动变成联播。
   - `GET /policies` 和 `GET /status`，核对模型顺序及 `config_name`。
   - 在官方运控下，`POST /select_policy`，JSON 为 `{"sequence_index":0}`，确认 `selected` / 索引 0。
   - `POST /enter_mimic` 一次，开始完整三段。200 只代表入队，继续确认实际进入 MIMIC_ACTIVE、USER_ACTIVE 和 FSM 1000。
   - 等全部结束，确认最终官方常规模式 500 和 publish_enabled=false。下一次语音演出仍要先 select 0，再 enter；结束索引停在 2，不能直接 enter 只重播 cut3。

   若旧 Agent 使用 policy_name，需改成当前精确模型名或使用索引；若旧流程逐段 stop/start，需改成一次 enter 后等整套完成。旧 `/return_loco`、`/switch_policy?direction=next|last` 路径保留；联播期间不要主动切段。`/select_sequence` 不受支持，不要调用。A/Select、`/return_loco` 为等待站姿退出，不是急停；段间要求退出会先完成当前最多约 3 秒交接，再退出并取消后续段。保留独立人工急停，不新增阻尼。

   仅需遥控器测试时，可停用 HTTP 控制器后用 `bash start.sh -c g1_baduanjin_23dof_suite`，二选一，不能并行。单段复测使用 `g1_baduanjin_23dof_select`，按索引选段再开始，无需反复重启进程。

4. 服务启动前设置本轮唯一 `ROBOJUDO_RECORD_DIR` / `ROBOJUDO_RUN_ID`，使 `/start` 启用 motion.msgpack；沿用现场已有 state 记录器，使用相同 run_id 和实际网卡。确认 motion.msgpack、state.msgpack、stdout/stderr 都在记录。HTTP 服务不会自动执行 start.sh 的记录器，不能只保存服务访问日志。保存语音唤醒/识别文本及上层调用时间，关联到同一轮测试，以验证真正的语音端到端调用。

5. 一次语音开始的目标流程为：常规→走跑→3 秒起始过渡→cut1→3 秒交接→cut2→3 秒交接→cut3→原交还流程→走跑→常规 500。两次 segment_transition begin/complete 应为 0→1、1→2，各约 3 秒，中间控制权不重取，舞蹈参考帧暂停。检查突跳、滑脚、转身、踉跄和双模型推理超时。不要把“过渡完成”直接当作真机姿态稳定。收尾到默认站姿的旧实验没有启用。

6. cut3 两处参考减俯仰从总 4° 改为总 6°。按 cut3 模型帧 2200–2500、3550–3850 对齐，比较前倾、垫步、跳动；联播绝对秒数已改变，不硬套旧视频 47 秒/76 秒。同时观察全部动作、启动和收尾，核对最终请求 500、RPC ACK 与实际读回，不只看 ACK 或 HTTP 200。

7. 返回本次完整 commit/diff、三个模型哈希、配置、启动命令、Python/bridge 路径、完整视频、motion.msgpack、state.msgpack、控制台日志及语音 Agent 请求记录。分别说明语音调用、两处衔接、cut3 两处前俯、整套完成和最终常规运控是否通过。异常按现场原人工流程处置并保留证据，回滚原 ref4/旧入口，不自动改模型或 FSM 参数。

本版已通过离线物理仿真和软件接口检查，尚无这版三段联播的实机通过结论；部署、离线检查与现场人员触发实机测试分开进行。
