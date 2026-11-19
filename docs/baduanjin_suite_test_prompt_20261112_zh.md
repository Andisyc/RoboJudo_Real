请在绿芽现场的 **G1-001（23DoF 背包）** 从 https://github.com/Andisyc/RoboJudo_Real 的 `g1loco` 分支同步本次 Cut3。先测一次单段，再测一次完整联播。

1. 以上次运行目录 `g1_run_261119-152754.MbBvyp`（单段）和 `g1_run_261119-155333.rbev8E`（联播）确认机器和两个实际入口。上次工作目录是 `/home/unitree/RoboJu/RoboJudo_BDJ_Suite_c41d9a7`。保留现场补丁、保护机制和回滚副本；若部署目录无 Git，在独立检出中取得本次差异并同步到实际加载目录。沿用现场 Python、bridge、网卡和记录器。
2. 用实际 controller 的 Python 执行 `python scripts/check_baduanjin_suite.py`。两个入口的索引应为 0 = Cut1 i100、1 = Cut2 v010、2 = `cut3_i5999_advance8_back12_20261119`（5299 帧）。核对实际 ONNX 的 SHA-256 为 `adc55e569be9b2079bba25fc0dee6f556b637287c0b3b229bf548429c3de6fdd`。
3. **先测单段。** 确认原 controller 已交还控制，只保留一路发布。沿用带记录器的 `bash start.sh -c g1_baduanjin_23dof_select`，现场用 R1 选到索引 2，核对控制台选中的完整模型名，再按 Start。观察起步及最后一次转身。结束后确认实际 FSM 500，退出单段 controller。
4. **再测联播。** 使用 `g1_baduanjin_23dof_suite`，按上次带记录器的方式从 Cut1 连续跑过 Cut2 到 Cut3，保留真实的段间交接。确认进入索引 2 的切段日志显示新模型。若使用语音 HTTP 入口，沿用原调用并先打开现场轨迹记录器。任一轮出现明显失稳，按现场人工停止流程处理，保留已录数据。
5. **两轮分别录像、留日志。** 使用同一地面和能看清右脚跟、脚尖与地面的机位，录完整原速全身视频。按 **Cut3 自身约 84–91 秒（帧 4180–4524）** 看最后一次转身，分别记录右脚是贴地、仅抬跟，还是连脚尖也离地；记下时刻、是否向前垫步、是否有人或吊带辅助。同时看 47 秒、74 秒两次回正。分别回传两段视频、两个运行目录（各含 `motion.msgpack`、`state.msgpack`）、对应控制台日志、提交号及实机模型哈希。上一版 Cut3 为 `cut3_field_scratch_i5999`。
