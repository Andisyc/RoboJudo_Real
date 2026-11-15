请从 https://github.com/Andisyc/RoboJudo_Real 的 **g1loco** 分支拉取最新代码，在此前测试的 **23DoF 背包机器人**部署并测试三段联播。模型已经在仓库，不需要更新包。

这次只把联播 Cut1 换成指定的 **leftsmall i80**，不是 rhythm i80，也不是 11 月 13 日续训的 guard 候选。Cut2、Cut3、controller、PD、输入设置和切换流程不变。i80 仍需真机验收，不能预设高摩擦左跨已经解决。

1. **备份并同步。** 保留现场原部署及未提交修改，在独立目录检出最新 `g1loco`，记录完整提交 SHA。沿用现场已测通的 Python、bridge 二进制、网卡、DDS 和服务托管配置，不自动重编 bridge 或升级依赖。确认加载的代码来自新目录，并记录实际 bridge 路径与 SHA256。检查三个模型：

   ```text
   0 cut1_leftsmall_i80_20261112.onnx
     SHA256 5558dfe97ecf0e0b7057793fba4cfc08493b3d089690d930054783b992054ebd
   1 eight_cut_2_v010_1.2x.onnx
     SHA256 bbca0a34feab8f9f73f150003910190cfe9412ef91b4a00f0b65197a2a0312bc
   2 cut3_jointv1_pitch8_rightless_20261112.onnx
     SHA256 70a509a3082737909a2ca1d02680a68122bba098bd3b7b0cacb42d76d6212e3e
   ```

   Cut2 保留 UNITREE 平移输入；Cut1/Cut3 的原有零平移输入权重在模型内，不要将环境统一改成 DUMMY。不要添加滤波、髋部补偿或其他后处理。单段入口 `g1_baduanjin_23dof_select` 共享这三个模型，仍按 0/1/2 选段；其他独立单段配置不在这次更新范围。

2. **先做离线检查。** 使用现场 controller 的 Python 环境，保留现场控制网卡和记录器网卡，确认两者一致，将 `UNITREE_NET_IF` 设为该网卡。以下检查不初始化机器人、不发送动作：

   ```bash
   python scripts/check_baduanjin_suite.py
   python scripts/check_g1_native_config.py -c g1_baduanjin_23dof_suite --net-if "$UNITREE_NET_IF"
   python -m pytest -q tests/test_baduanjin_http.py tests/test_baduanjin_suite.py tests/test_pose_transition.py tests/test_mimic_safety_offline.py tests/test_g1_native_integration.py tests/test_g1_backpack_contract.py
   ```

   检查通过后再由现场人员启动真实测试。不要同时运行旧服务、新 HTTP 服务和 `start.sh` 多个 controller。旧服务按原流程交还官方运控、确认 `publish_enabled=false` 后再更换。

3. **沿用原语音调用完成两轮联播。** 入口仍为 `g1_baduanjin_23dof_suite`。首次加载/服务重建：`POST /start`，JSON 为 `{"config_name":"g1_baduanjin_23dof_suite"}`；已加载时不重复 start。每次演出只发送一次 `POST /enter_mimic`，不带请求体，**不需要 `/select_policy`**，每次自动从 Cut1 首帧开始。沿用原唤醒词和 Agent，不逐段启动。

   流程应为：常规→走跑→3 秒启动过渡→新 Cut1→3 秒交接→原 Cut2→3 秒交接→当前 Cut3→走跑→常规 500。第二轮仍从 Cut1 开始。常规模式以请求和实际状态读回为准，不只看 HTTP 200。

   重点比较 Cut1 后半段两次左跨，尤其第一次：是否先移重心、左脚是否离地后横移，还是仍贴地滑动；同时逐次检查三次右跨有没有缩小，以及起身前移、Cut1→Cut2 衔接是否退化。使用当前地面完成后，条件允许再在较高摩擦的平整地面做同样对照，并记录地面。按 `policy_name/query_frame` 对齐，不套用旧视频秒数。Cut2、Cut3只做保持效果检查。

4. **保留完整数据并回传。** HTTP 服务启动前设唯一 `ROBOJUDO_RECORD_DIR` / `ROBOJUDO_RUN_ID`，另启动现场已有状态记录器，确认 `motion.msgpack`、`state.msgpack`、stdout/stderr及完整原速带音频视频均保存；HTTP 服务不会自动运行 `start.sh` 的记录器。记录启动命令、代码 SHA、三个模型哈希、Python/bridge 路径与哈希、语音请求时间、CPU/进程情况和网卡。日志应有 `suite_start` / `SUITE_START` 的 `start_index=0`、两次段间切换、请求500与最终读回。

   11月13日曾有一次原 Cut2 单段运行上肢抖动，原因尚在排查。若本轮再出现，沿用现场人工停止/急停流程，保留发生前后的全部日志和音频，不自行修改 Cut2、PD、通信设置或增加自动阻尼。`/return_loco`、A/Select 是等待站姿退出，不是急停。请额外保留同机同环境的正常 Cut2 对照、进程列表和 controller/记录器标准输出，便于区分控制与记录延迟。

请返回：两轮是否从Cut1开始、两次左跨和三次右跨结果、两处衔接、Cut2/Cut3是否保持、常规500切换、是否再有上肢抖动，以及上述日志和视频。需要回滚时恢复本次更新前部署（Cut1 为 `cut1_scratch_v2_i15000_zero_translation`，Cut2/Cut3仍保留当前版本），不要回退到旧 Cut3 ref6。

本次发布详情见 `docs/cut1_i80_release_20261113_zh.md`，HTTP 合同见 `docs/baduanjin_http_compatibility_20261112_zh.md`。
