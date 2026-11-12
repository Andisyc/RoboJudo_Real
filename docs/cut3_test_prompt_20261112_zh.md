# 给测试人员 Codex 的部署与测试 prompt

请从 https://github.com/Andisyc/RoboJudo_Real 的 **g1loco** 分支获取本次 Cut3 更新，部署到现场此前测试八段锦的 **23DoF 背包机器人**。模型、代码和说明都在 GitHub，不需要额外模型包。本次包含 controller 改动，不能只替换 ONNX。收到发布提交 SHA 时，检出该提交以固定测试版本，并记录实际 SHA。

先读取 `docs/cut3_pitch_transition_20261112_zh.md`，按以下顺序完成：

1. 检查现场工作目录、当前 commit、未提交修改、原启动命令、Python/bridge 路径、网卡及 DDS 配置，保留原部署用于回滚。在独立目录检出更新并比较差异，合入现场必要定制；不要强制覆盖原部署。不要修改 Cut1、Cut2、三段索引、29DoF 配置、PD、里程计、FK，或增加自动阻尼逻辑。先核实物理机器人是 23DoF，本模型不用于 29DoF。

2. 确认新模型为 `assets/models/g1/beyondmimic/cut3_pitch_ref4_20261112.onnx`，SHA-256 必须是：

   ```text
   7dbc5b3753512fbc227444695c3b076507e5d6c1963c94dcc873adf19a74cf77
   ```

   模型是 130 维输入、23 维动作，`max_timestep=5299`。这是内置参考轨迹修改，没有重新训练演员网络，也没有新增舞蹈中输出滤波。保留旧 `cut3_recovery_r1_i80.onnx`。

3. 在运行 controller 的同一 Python 环境编译共享 bridge，启用每次请求参数和 ACK 日志，再完成离线检查：

   ```bash
   python -m pip install --no-deps --no-build-isolation --force-reinstall ./packages/g1_loco_bridge
   python tests/test_pose_transition.py
   python tests/test_mimic_safety_offline.py
   python -m unittest discover -s tests -p test_g1_native_integration.py
   python scripts/check_g1_native_config.py -c g1_native_loco_mimic_23dof_backpack_cut3_fsm500
   python scripts/check_g1_native_config.py -c g1_native_loco_mimic_23dof_backpack_cut3_pitch
   python scripts/check_g1_native_config.py -c g1_native_loco_mimic_23dof_backpack_cut3_entry_transition
   python scripts/check_mimic_model.py assets/models/g1/beyondmimic/cut3_pitch_ref4_20261112.onnx --max-timestep 5299
   ```

   检查不得创建真实机器人环境或触发模式切换。记录实际导入的 `g1_loco_bridge.__file__`；仅看 revision 不足以证明加载了新二进制。

4. 准备沿用现场记录器及 `start.sh` 的启动命令，保留网卡、Python、DDS 设置。完成部署及离线检查后，把实际命令和回滚方式交给现场人员；在现场人员在位并按其测试安排操作时，依次测试以下 `-c` 入口。不要无人自动触发舞蹈，也不要一次连续跑完三轮而不检查上一轮结果。

   | 顺序 | `-c` 参数 | 本轮变化 |
   |---|---|---|
   | A | `g1_native_loco_mimic_23dof_backpack_cut3_fsm500` | 原 Cut3 模型，只把请求常规运控改为 500 |
   | B | `g1_native_loco_mimic_23dof_backpack_cut3_pitch` | 请求 500 + 两段减俯仰的新 ONNX |
   | C | `g1_native_loco_mimic_23dof_backpack_cut3_entry_transition` | 在 B 基础上增加启动 3 秒参考插值，收尾仍沿用原流程 |

   三个都是单独 Cut3 入口，不是三段切换列表。启动命令模板：沿用现场原命令，只替换 `-c` 后的配置名。例如 `bash start.sh -c g1_native_loco_mimic_23dof_backpack_cut3_pitch`，其余环境变量必须沿用并核对现场配置。

   **禁止本轮实机使用 `g1_native_loco_mimic_23dof_backpack_cut3_transition`。** 该入口同时开启收尾 3 秒恢复，实际站姿跟踪尚未通过离线验收。也不要将 `exit_pose_transition_s` 改成 3 来绕过这个限制。

5. A 轮重点核对退出：请求 `requested_fsm_id=500`、每次 `FSM_REQUEST` 的 attempt/ACK，以及最终 `regular_loco.result` 和 `fsm_id`。ACK=0 不代表完成，须读回常规模式；仍为 801/802 就是没有确认切回。不要自行循环尝试 501；保存日志，沿用现场人工遥控器切换流程。不要将 29DoF 曾接受 501 的经验直接外推为本机结论。

6. B/C 轮重点观察 query_frame **2200–2500、3550–3850** 的前倾、补步、跳动及滑移，同时检查整段、原已优化窗口 1730–1915 / 4270–4580、末姿及交还。C 轮检查 `pose_transition` 的 entry begin/complete：参考首帧保持约 3 秒，随后才开始推进舞蹈；旧视频秒数会后移，必须按模型帧对齐。不能把插值完成直接判为姿态已到位，也不能把仿真通过报告成真机通过。

7. 每轮返回：实际 commit/diff、模型 SHA-256、配置名、启动命令、Python 和 bridge 路径、运行目录、配置快照、完整 `motion.msgpack` / `state.msgpack` / stdout / stderr 及完整视频。分别说明启动抖动、两段前倾跳动、收尾和常规运控切换是否改善，尤其明确“先失败后成功”还是“最终仍失败”。异常时按现场原人工急停流程结束，保留记录并回滚旧部署，不自动修改模型或保护参数。A/Select 是等待站姿交还，不是急停。
