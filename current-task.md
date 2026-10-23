# Current task

Goal: 将 TTT 的旧流式 FADA 和新冻结基座轨迹差分方法接入 RoboJudo_Real 的 G1 真机部署链，保留冻结推理模式，并以离线闭环与单独授权的真机验证区分工程接通和实际效果。
Do not: 本轮仅修改 start.sh 的两处 "$@" 参数透传并移除过时注释，运行一次无硬件 wrapper smoke，同步既有 README/本板。不改 Python 默认配置、键盘后端实现、在线算法、网卡/XML、录制策略、信号/清理逻辑、环境或 native；不提交/推送/部署，不执行远端或机器人控制命令。保留 env -u DISPLAY 原有控制台输入路线和所有原生/时效前置检查。

Current identity:
- 目标: `/Users/chengyuxuan/ArtiIntComVis/RoboJudo_Real`，当前 `release`，基线 `8ee66e918e3d2370bd3c262cbd1388f0872a3ad0`；调研开始时工作区干净。路线以当前 release 为基线，不授权 Git 操作。
- 新方法来源: `/Users/chengyuxuan/ArtiIntComVis/TTT`，`0f8a410b97127a5bff7d3379ac66400e590679f9`，调研开始时工作区干净；主要 owner 为 `src/unilab/algos/torch/distill/fada/{playback,online_data,online_collector,online_controller,online_updater,adaptation}.py`。
- 旧流式 FADA 来源: TTT 历史提交 `c4ff3bf6c3cd6918b5dd5502f083f17d07c33667` 的 online_controller/online_collector/online_updater；不能把中间的单步关节误差 IMP 当作旧方法。当前 HEAD 仍有 causal-window helper，但已没有旧在线 updater 路径。
- 目标正式入口 `scripts/run_pipeline.py`；真机配置 `g1_real_fada_planner_idm`；适配器 `FADAPlannerIDMPolicyAdapter`；环境 `UnitreeCppEnv`。既有 FADA actor obs=98，投影 state=66，action=29，command=3，默认 H=30/K=6、50Hz。
- 目标配置默认 `g1_hang_book_real.onnx` 保留不变；本地 `assets/models/g1/fada/planner_idm_v022/planner_idm.pt` 已经既有严格 loader/新运行时成功加载并实际前向，契约 g1_fada_state_v2、H=30/K=6。兼容性已证明，但未证明它与此前预期生产 ONNX 来自同一权重，不自动改默认选择。

Confirmed decisions:
- 用户已授权第 1–6 步、第 2 步纯计算补验及本轮速度门控实现；本轮只在本地 RoboJudo_Real 修改/离线验证。此前用户允许 tmux 只读核对部署终端，目前计划使用服务器有线连接机器人、网卡由用户自行配置；未授权本轮远端修改、安装依赖、Git 推送或实际机器人控制。整体仍为 5/6，完整原生扩展验收未由本轮关闭。
- 旧方法训练条件是同一历史下的 Y_exec，监督采集时的首动作；新方法训练条件是 Y_nom，固定标签为 U_snapshot + gamma * (First(base(Y_nom)) - First(base(Y_exec)))，省略部分代表同一历史输入。
- Planner 冻结，只更新 IDM LoRA；新方法的 base IDM 独立冻结，标签不随同批在线优化重算。不恢复单步 q 误差或手工除 action_scale 的标签路线；环境既有动作转换保留。
- TTT 已完成的离线 smoke 只证明 donor 链路，不证明目标仓库兼容、目标机器实时性或负重适应效果。TTT 的旧 current-task.md 保持关闭，本板是这次迁移唯一状态入口。
- 新启动约定：prepare 后零速度期间正常控制并记录真实下发历史，不开始学习；首个生效的非零三轴速度一次性开启采集，窗口成熟才更新。速度归零不关闭学习；reset/离开激活控制重新等待，保留已有权重/Adam/已接受任务，并把旧结果的发布延迟到重新触发后。warmup/interval 数值不变，只调整相对起点，真实控制步不重编号。

Progress:
- [x] 总体路线与源代码边界核对完成；工程离线验收进度 5/6 (第 1、3、4、5、6 步完成，第 2 步部分完成)，不是原生或真机部署验收。
  - Evidence: 只读 Git 身份/历史检查和两个独立只读源代码分支，主 Agent 核对模型/投影、真实发送、prepare/dry_run、tick 暴露和 checkpoint 默认值。未运行测试或机器人。
  - Initial evidence (修改前): target `fada_policy.py:_build_runtime` 支持 ONNX/PT，但使用 inference-only playback；`RlPipeline.__init__` 先建硬件 env 再建 policy。`prepare` 通过 dry_run 暖历史并另行下发插值目标，约第 900 步 reset，不能把策略 dry_run 输出当真实动作。
  - Initial evidence (修改前): `UnitreeCppEnv.get_data` 未透传已有 `RobotState.tick`；躯干 IMU 的独立接收新鲜度未在现有 Python 数据中表达。C++ 周期发送器会继续发送最后的 PD 命令；release 启动循环没有异常 finally 关闭路径，不能直接把同步反向传播塞进控制步。

- [x] 1. 冻结接口、模式配置和可训练运行时 — 已实现并通过针对性 SDK-free smoke。
  - Changed/config: `G1FADAPlannerIDMPolicyCfg` 增加 `online_adaptation: FADAOnlineAdaptationCfg` (method=none/fada/expert_difference，共同 LoRA=8/16/0.05) 和显式 checkpoint_path；没有 enabled 第二开关、目录扫描或默认权重替换。
  - Changed/runtime: 新增 `robojudo/policy/fada/{lora,online_runtime,online_data}.py`。活动 Q/V 注入、所有权检查、独立 no_grad 推理/同次前向快照与可微 IDM；`prepare_fada_trainable_runtime` 复用现有严格 checkpoint reader。expert 在 LoRA 注入前独立复制冻结 IDM，fada 不创建 base。没有实现 loss/optimizer/collector/worker；旧 model/checkpoint/onnx_runtime 文件未改。
  - Frozen interfaces: Snapshot(observation_history, action_history, command, nominal_future, action_snapshot)，对应 [1,H,66]/[1,H,29]/[1,3]/[1,K,66]/[1,29]，独立持有无计算图。旧 `FADACausalWindow` 保留 history/command/realized_future/executed_action_chunk ([1,K,29])；新 `FADATrajectoryPair` 和 `FADAOnlineSample` 持配对未来/静态目标。`FADAControlEvidence` 单独持 episode_id/control_step/policy_version/sensor_tick/torso_imu_sequence/policy_control_active/action_dispatched；None 是缺失证据，不是新帧。传感器实际填充和采集仍属第 2 步。
  - Changed/cold path: RlPipeline 对 FADA 先构造并校验 policy，再创建 env；PolicyWrapper 的 keyword-only prepared_policy 复用该对象，其他三个位置参数调用/非 FADA 顺序不变。现有 inference-only adapter 对非 none 明确拒绝，尚未接入的在线控制不能静默按 baseline 运行；无假 controller/空 worker。
  - Changed/evidence owners: 既有 FADA 测试文件新增两个行为用例，覆盖真实网络/梯度/基座隔离和硬件前拒绝；移除已过时的纯配置默认值断言及被正式 checkpoint_path 取代的测试子类。README 已注明第 1/6 步能力及非上机结论；迁入源的 Apache-2.0 全文保留在 fada/LICENSE-TTT，根许可证未改。
  - Evidence A: 首次唯一 smoke 工作流在 import 阶段因共享 uv 环境缺 python-box 中断，尚未执行模型检查。一次依赖清点发现还缺数项已声明包；未安装包、未创建环境，而用临时 PYTHONPATH 引用本机现有缓存的 Box/MuJoCo/joblib/msgpack-numpy 等实际包 (匹配 CPython 3.13/macOS ARM ABI)，没有伪造依赖或接 SDK。
  - Evidence B: 恢复同一工作流后，`TestFADATrainableRuntime` 两个行为用例全部通过 (0.595s，含两模式真实 forward/backward/一次局部 SGD)。验证初始输出一致、只有 LoRA 改变且回归误差下降、Planner/base 不变且 base 存储独立、快照不共享历史；在线执行器不匹配、架构不匹配、缺失 PT 在硬件 env sentinel 前拒绝，在线 ONNX 被模块入口拒绝。没有运行完整套件或已通过用例的额外重复验证。
  - Evidence C: 同次 smoke 用本地真实 planner_idm.pt 构造 expert runtime 并前向，输出 nominal future=(1,6,66)、首动作前三维 [-0.6517170071601868, 0.11584055423736572, 0.23944340646266937]、requires_grad=False；整个恢复调用墙钟 13.26s。这是真实模块/权重的离线输出，不是物理机器人或仿真结果。
  - Scope: 只完成获授权的第 1 步；未修改 TTT 源文件或其共享环境、SDK/C++、step/prepare/reset/shutdown、IMU/PD/关节契约。临时文件由测试上下文回收，没有新增启动脚本、依赖或流程文档。

- [ ] 2. 接入因果轨迹采集与真实帧边界 — 源码、Python 采集及生产年龄纯计算验证通过；完整 SDK 扩展构建/运行验证仍阻塞。Depends on: 1。
  - Changed/native: `unitree_controller.hpp/.cpp` 给有效 LowState 和独立躯干 IMU 记录 steady_clock 接收时刻，躯干 IMU 使用原子递增接收序号；数据/序号/时间在同一缓存快照中复制。get_robot_state 在同一时钟域计算接收年龄，未收到 IMU 时为未知；复制旧躯干 IMU 不更新时间。`py_binding.cpp` 暴露 readonly 年龄/序号和 RobotState.sensor_evidence_version=1。未改 DDS topic/消息、CRC、PD、控制器启动或发送循环。此部分尚未编译/运行证明。
  - Changed/Python readout: `UnitreeCppEnv` 只在初始化缓存公开元信息版本，在 update 调用原生读取前记录 monotonic_ns；get_data 透传 tick、读起点、接收年龄/序号。旧扩展明确给 None，不伪造当前新帧。未改 env.step 返回值或发送确认，不接入控制 lifecycle；该原生读出尚未在真实扩展上验证。
  - Frozen evidence: `FADAControlEvidence` 增加可选 lowstate_age_s/torso_imu_age_s/sensor_read_started_ns/action_submitted_ns；未知为 None。submit 时间仍要求未来第 5 步实际发送 hook 提供，不能把 env.step 的 None 当作发送成功；这些值不进入 66 维观测。
  - Changed/collector: 新增 `fada/online_collector.py`。observe(state, command, evidence) 完成前一控制周期，record_action(snapshot, evidence) 只接受真实策略提交；FADACollectionResult 明确区分接受、成熟样本和拒收原因。要求连续控制步、uint32 tick 前进 (支持回绕)、躯干序号前进、显式 max_sensor_age_s 预算，并以两路接收年龄 < (本次读起点 - 上次提交完成时间) 证明本地后继接收，不跨 C++/Python 绝对时钟域作差。坏帧丢弃未成熟片段，不能把 t+2 补成 t+1；模型版本变动不算 episode 边界。
  - Legacy behavior: H+K-1 闭合 transition 后发射精确 anchor=H-1 窗口，无新增成熟缓存或 T=10 满批门槛。未来 K 帧命令一致即可，允许历史含旧命令。上个动作来自提交账本；坏帧不抹成零，只有真实 control_step=0 的 episode 起点补零；中途开始而上一动作未知时，不伪造 legacy history，等实际账本建立。
  - Expert behavior: 保存实际前向快照，等待 K 个真实后继状态形成 Pair；成熟队列默认 10，get_window 暴露有序快照，acknowledge 按 (episode_id, control_step) 精确消费，晚确认不误清新样本/在途未来。新命令仅使跨界在途未来失效，已成熟 Pair 保留；reset 清本模式成熟与在途数据。未实现 loss、warmup/interval、optimizer 或 worker。
  - Evidence: 一次 `uv run --project TTT --frozen --no-sync` SDK-free 工作流，复用第 1 步已记录的实际缓存依赖路径，不安装包。TestFADACausalCollectors 三个行为用例 0.043s 全部通过：精确历史/未来/动作切片、部分旧窗口发射、tick 回绕、模型版本变化、命令未来边界/允许旧命令历史、reset、缺帧/重复/过期/提交前新 tick、未激活/未发送、晚确认保留新 Pair、坏帧后真实 previous_action 和中途起始不伪造零历史。没有重跑第 1 步或广泛套件。
  - Runtime smoke: 同次工作流直接调用采集器，实际打印 legacy origin=29, future=[30,31,32,33,34,35]；expert origin=0, future=[1,2,3,4,5,6]。墙钟 1.65s；没有实例化 SDK、environment 或 controller，没有仿真/优化/物理发送。
  - Scope/docs: README 已记录模块 API、拒收/确认语义和原生证据限制。没有修改 TTT、依赖、网络输入、PD 或正式 prepare/step/reset/shutdown 接线；inference-only 保护仍保留。只完成第 2 步源代码，不提前实现 updater/worker。
  - Blocked native proof: 按现有 CMake 声明路径只读检查，未找到 Unitree SDK2/CycloneDDS 库/头文件；未安装依赖、未远程连接或用假 SDK 代替编译。必须在有这些依赖和 pybind11 的环境真实构建/导入，并完成不实例化控制器的元信息计算检查后才能勾选本步。Python smoke、源码检查和已有旧二进制都不是新绑定运行证明；DDS 实际收包/时差/硬件生效仍需后续单独授权验证。
  - Follow-up/change: 本轮新增 `packages/unitree_cpp/src/sensor_evidence.hpp` 的 inline `unitree_cpp::compute_receive_ages`，`unitree_controller.cpp:get_robot_state` 调用同一纯计算再填年龄字段；新增 `packages/unitree_cpp/tests/test_sensor_evidence.cpp`。仍保持单次 RobotState 快照复制、同一 steady_clock/秒单位/序号为零时未知，无额外堆分配；未改回调、线程、DDS、pybind、Python 消费或控制协议。
  - Follow-up/evidence: 一次 `uv run --project .../TTT --frozen --no-sync python -u -c ...` 工作流，调用 Apple clang 21 以 `-std=c++17 -O2` 编译并执行真实生产 helper 的独立 C++ 用例，验证秒单位、年龄增长、旧 IMU 不因新 LowState 重置、未知 IMU、有效零年龄。编译产物 stdout 的真实数值再输入旧/新两个 Python collector，均显示 fresh=accepted、stale=rejected、missing=rejected；整条调用墙钟 2.74s。没有重跑既有套件；临时可执行文件由 TemporaryDirectory 回收。
  - Follow-up/build boundary: 同次 find_spec 检查当前共享 uv 环境中 pybind11、scikit_build_core、unitree_cpp 的 origin 均为 None；未安装依赖、未重复运行已知缺 SDK 的完整构建、未构造 SDK controller 或用替身编译。C++ 纯计算编译通过，不代表完整 unitree_controller.cpp/pybind 扩展已构建，真实绑定默认未知值/正数转换及 UnitreeCppEnv.get_data 对新扩展的运行验证仍待具备依赖且匹配目标 ABI 的环境。
  - Follow-up/safety: 已确认 UnitreeController 构造会 Init DDS、ReleaseMode 并创建发送线程，不能用 act=false 或只读 get_robot_state 作为离线安全替代。实际 DDS/接收年龄、序号并发一致性、设备 20ms 和物理停机依旧未验证。只选 skill-router + code-construction，README 同步本轮证据/限制，没有新增流程文档或环境配置。

- [x] 3. 移植新旧两种优化目标 — 已实现并通过一次针对性 SDK-free smoke；不依赖第 2 步尚缺的原生构建证据，不代表正式集成通过。
  - Changed/owner: 新增 `robojudo/policy/fada/online_updater.py` 的 `FADALegacyUpdater`、`build_expert_samples`、`FADAExpertUpdater`；扩展既有 `tests/test_fada_planner_idm_policy.py` 的 `TestFADAOnlineUpdaters` 三个语义用例，README 同步独立 API 与未接线边界。
  - Shared contract: 消费已准备好的匹配方法 `FADATrainableRuntime`，复用既有 LoRA 参数发现/所有权检查；不重复注入、加载 checkpoint 或复制 Planner/base。只含 LoRA 的 Adam，lr=1e-4、weight_decay=0、max_grad_norm=1 (可为 None)，每次 update 一步；超参通过构造器传入，未提前接部署配置/调度。
  - Old: Y_exec 条件、首预测/首真实下发动作的动作维 mean MSE、lambda_forget=0.99 归一化新近权重；保留逐窗口 forward 和 train/dropout，恢复原模式。不加 sigma 或新方法的十条门槛；空缓存不更新、部分成熟批可更新、成功清本批、失败保留缓存，不承诺 optimizer.step 异常回滚。updater 持成熟缓存/Adam，普通 reset 不应重建它。
  - Expert targets: 独立冻结 eval 基座对同一历史下的 Y_nom/Y_exec 各做一次批量 forward，以 gamma=0.05 的首动作 nom-exec 差分叠加真实 U_snapshot，构造自有、脱图的固定 FADAOnlineSample。不使用单步 q 误差、手工 action_scale 转换或虚构未来。
  - Expert loss: 始终用 Y_nom；0.5×动作维平方和再对样本求均值，sigma=1e-4 相对部署初始 LoRA 的参数锚定，eval+autograd 并恢复模式。theta_base 只在 updater 创建时复制，标签不在 compute_loss/update 中重算；不采用 AdamW 向零回拉。
  - Evidence A: 一次实际 `uv run --project /Users/chengyuxuan/ArtiIntComVis/TTT --frozen --no-sync python -c ...` 工作流，复用已有共享环境和缓存依赖，未安装包。仅运行 TestFADAOnlineUpdaters 的 3 个用例，全部通过 (0.562s)。固定随机条件下对照独立 donor 损失表达式/LoRA 梯度，验证 dropout/模式、部分批/成功清批/失败保留、差分符号/快照、固定标签、非零部署锚点、零残差不向零衰减及 Planner/base 不变。
  - Evidence B: 同一工作流另行直接调用两种公共 updater→runtime.infer，使用小型真实 Planner-IDM、H=30/K=6/66 状态/29 动作；旧 loss=0.1681382656，新 loss=0.0041175187，两种更新后实际推理动作 L2 变化分别为 0.0003490912 / 0.0004083767，输出有限。这是模块 API 执行证据，不是机器人/仿真或方法效果证据。
  - Scope: 没有重跑第 1/2 步套件，没有 SDK、仿真、控制器、worker、版本发布、collector acknowledge 或硬件接线；未改模型/LoRA/数据接口、正式控制/传感配置或 TTT。新方法 T=10/warmup=40/interval=10、精确成功确认和 reset 协议仍留第 4 步。

- [x] 4. 后台优化与完整权重版本发布 — 已实现并通过真实 spawn/网络离线 smoke；后续第 5 步 Python 接线已完成，正式部署仍受第 2 步原生证据阻塞。
  - Changed/owners: 新增 `robojudo/policy/fada/online_worker.py` 的 FADALearnerWorker 和 `online_controller.py` 的 FADAOnlineCoordinator；扩展既有 FADAOnlineAdaptationCfg、测试文件的 TestFADAOnlineWorker 和 README。没有修改 model/runtime/LoRA/data/collector/第 3 步 updater 或 adapter/pipeline/SDK/脚本。
  - Worker/identity: 一个 spawn learner，冷启动从完整 CPU 字节快照恢复精确 policy/初始 LoRA/新方法冻结 base，再创建一次 updater；不执行 Planner 前向、不共享可变模型存储、不每批重载/复制。默认 CPU/1 计算线程，初始化/更新/close 预算为 30/5/1s。请求/结果包含 job/generation/episode/control_step/policy_version/sample IDs，结果再带 learner_version、完整 LoRA、stats/耗时；不新增磁盘或哈希协议。
  - Transport: 单个有界 job queue；结果经单向 Pipe，由后台传输线程接收完整消息后写入有界本地 inbox，control poll 不读取可能只到一半的 pipe frame。停止使用父写、子读的单向共享字节标记，不依赖可能被异常子进程遗弃的进程锁。仅冷启动、close 和明确的离线 wait_for_message 可等待，不在控制路径反传或构造专家标签。
  - Coordinator/cadence: observe/record_action 复用真实因果采集；try_submit 默认每 10 个实际控制步评估，旧方法非空部分批即可，新方法 warmup=40/T=10；繁忙时不排训练任务。旧 outbox 默认 40、溢出明确失败；新样本仍走既有 bounded sliding window，已提交批独立持有，成功只精确 acknowledge 其 IDs。一个在途 job/待应用版本，前版应用前不提交下一 job。
  - Publication/reset: 先验证完整 LoRA 键/形状/dtype/有限值/版本，再在单一控制线程步边界准备全部张量、应用全部权重并最后更新 policy_version；学习完成和应用版本分别可见。普通 reset 清未成熟轨迹/未提交新法成熟窗，保留旧法成熟 outbox、在途完整批和 worker 的 LoRA/Adam/base/theta_base；旧 episode 完成只确认原 IDs，可在新 episode 真正激活边界发布，不在准备阶段提交/应用。
  - Stop/errors/config: request_stop 作废待发布版本、停止接收；close 在控制循环外有界 join/terminate/kill，并检查子进程和接收线程退出。异常退出、超时、坏包、非有限值、积压溢出均显式 FAILED，不自动重试或回退 none；状态含 job/积压/完成与应用版本/统计/延迟。原配置新增对应调度/资源/超参约束，max_sensor_age_s 在 coordinator 创建前必须显式给出；none 不创建 learner。
  - Failed attempts/resolution: 首次 MCP shell 30s 传输超时，未假称成功或盲目重复；确认原进程并停止后，用有输出句柄的 uv smoke 定位测试 finally 在强制终止 Event 等待者后再次 set 的死锁，修正测试清理。随后发现冻结全局 monotonic 的超时用例干扰标准库轮询，改为仅推进该 job 的已提交时刻。又依据 stdlib Queue.get 的 framing 行为补齐 Pipe/接收线程和无锁停止标记；补验仅围绕这些具体问题，没有重跑第 1–3 步套件。
  - Evidence A: 最终 `uv run --project .../TTT --frozen --no-sync python -u -c ...` 真实进程 smoke，两模式分别先提交 6/10 个成熟样本；忙时持续实际推理，完成后 actor 在 apply 前逐项不变，apply 后整包一致。跨 reset 的连续两次更新与独立真实 updater 的数值结果一致，证明 Adam/锚点保留；两者完成/应用版本均到 2，晚到确认不删新 episode 样本，最终 child_reaped=True。
  - Evidence B: 同一最终工作流 4 个针对性用例全部 OK (5.211s，含两模式正向流程的整条命令墙钟 10.20s)：坏发布不改 actor/不确认、更新超时与有界积压、真实模型错误保留 job、半个回复仍能推理且 child 突然退出后显式失败/回收接收线程。没有 DDS、机器人、物理仿真或 20ms 阈值测试；共享 uv/缓存依赖未修改。
  - Resource limit: SIGKILL 故障注入仍触发 Python resource_tracker 对命名 semaphore 的退出清理警告。一次仅针对该警告的归属检查确认警告名字不在所记录的本次父侧 queue/Event 名单中，但未完全追溯来源；不能宣称强杀下全部 IPC 资源都已优雅清理。实际 worker/接收线程退出已证明，警告没有被屏蔽；不把其包装成正常路径成功或目标机资源保证。
  - Scope: 第 4 步仅使用 skill-router + code-construction，未增加审查/整套测试、依赖、Git 操作或正式控制入口，没有对外部署；执行器优先安全请求及 worker finally 接线现已由第 5 步完成，见下方证据。

- [x] 5. 接回正式部署和安全生命周期 — 已完成 Python 接线及 SDK-free 集成证明；不代表第 2 步原生/真机验收通过。
  - Changed/owners: 仅修改 `fada_policy.py`、`fada/playback.py`、`pipeline/rl_pipeline.py`、`environment/unitree_cpp_env.py`、`scripts/run_pipeline.py`、既有 FADA 测试、README/本板。没有修改损失/updater/worker/native C++、PD 参数、默认 checkpoint、DoF 数值或其他 pipeline 文件。
  - Startup: FADA adapter 按 method 复用 frozen 或 prepared trainable runtime；online coordinator 创建一次，RlPipeline 在环境/UnitreeController 创建前做 Unitree G1 position/act/torso/evidence v1 冷校验和 learner READY 握手。none 保留 PT/ONNX，不建 learner；多策略不获得在线接线，缺生命周期激活会明确拒绝，不默默按 baseline 运行。
  - Query/receipt: playback 拆成 prepare_query/commit_action，原 act 复用并保持冻结行为。在线推理仅暂存同次真实快照；UnitreeCppEnv.step 在 enabled 的 native body step 返回后给出 monotonic_ns，disabled 为 None，异常不确认。confirm_action_submission 校验 receipt 后才提交 raw U 到历史/last_action/collector 并计实际控制步；学习标签不替换成 scaled/absolute PD target。现有 native step 含立即 writer 调用，但凭据不是 DDS/电机 ack。
  - Phases/reset: self_check/prepare/dry_run 不采集、不提交 job、不发布，也不把试算动作写入在线执行历史。首次真实激活重建历史/从实际步 0 起算，padding 不算观测经验；保留原 1000 步 prepare/900 reset/gait/IMU。online reset 先失效策略采集再重置 env，episode/generation 只推进一次；不重建 worker/Adam/base/anchor，保留学习权重。类级不可变默认兼容跳过 RlPipeline 构造器的既有子类。
  - Safety: RlPipeline should_stop/request_stop/close、adapter stop/close 和 main try/finally 已接通；急停在新动作前处理，跌倒、发送/learner 错误、prepare/初始化失败及 KeyboardInterrupt 都走执行器优先安全请求，再关闭学习资源。UnitreeCppEnv.shutdown 幂等禁写，native handle 已创建但 self_check 失败会局部清理；清理失败显式报告且不掩盖主异常，不改原生 damping 参数或宣称全部 native 线程已停。
  - Evidence A: 唯一针对性 uv 集成工作流实际驱动 none/fada/expert_difference 的真实小网络、adapter/PolicyWrapper/RlPipeline/prepare 和 spawn worker，以临时 SDK/传感/执行替身隔离 DDS。三个模式正向均通过：1000 步准备期零学习样本/job，none 与独立 frozen forward 一致，在线模式 receipt 前历史不推进、后仅提交 raw U；反向 joint 顺序及 execution_scale=1.7 的 PD 映射符合独立逐关节表达式。旧/新首批分别 6/10 条，完成一次真实更新并在下一步应用 version=1，reset 保留权重/同一 worker，初始动作历史重新 padding。
  - Evidence B: 同工作流的 4 个针对性用例全部 OK (4.505s；恢复执行的命令墙钟 7.48s)：发送异常不确认、worker READY 后控制器构造失败仍安全清理、act=false 硬件前拒绝；真实 Python Unitree wrapper 的部分初始化失败/disabled receipt/幂等 shutdown；真实 main 的 prepare 后 KeyboardInterrupt/finally；受影响的非法请求硬件前拒绝契约。额外正向路径覆盖跌倒、learner 进程终止、急停前不再下发。worker/接收线程均已回收。
  - Failed fixture/resume: 首次工作流 none/旧方法已经通过，expert 启动被测试 patch.dict(sys.modules) 卸载 multiprocessing.heap 后的类身份冲突阻塞；只将夹具恢复范围缩到 SDK 两个模块，保留无关模块缓存，然后继续 expert 和未运行用例，没有重跑已通过模式或第 1–4 步套件。过时 inference-only/错误措辞断言已改成行为边界检查，没有新增流程产物。
  - Scope/limits: 只使用 skill-router + code-construction，没有全面审查、依赖安装或 Git 操作。测试仅在本模块替换休眠/传感时钟，未冻结全局 monotonic；临时 paired PT 由上下文回收。SDK 替身不证明 native v1 已构建、真实 DDS 收发/电机动作、目标机 20ms、生产 PT/ONNX 同源或负重适应有效；第 4 步强杀 semaphore 警告边界未被关闭。
  - Velocity-gate follow-up/change: 本轮仅修改 `fada_policy.py` 的构造/reset/set_control_active/get_action/confirm_action_submission/stop 和 `fada/online_controller.py` 的首个 observe 起点/相对调度/status/reset，以及既有测试、README/本板。adapter 的 `_adaptation_started` 与 `_control_active` 分离，另以 `_pending_collection` 固化每次查询的采集资格；等待时仍 infer/snapshot/真实发送/receipt 后提交 raw history 和实际计步、poll worker 健康，但不 observe/record collector、不 try_submit 或 apply。触发不清控制历史，无额外 forward；none 不变。
  - Velocity-gate follow-up/timing: coordinator 增加 collection_start_control_step/collection_step 状态，未触发时均为 None，保持原 worker.state 枚举。使用真实 evidence.control_step 定位与样本身份，首次 active observe 设采集锚点，warmup/interval 仅用相对差值；reset 清锚点但保留既有成熟旧批/在途 job/待发布结果/worker/Adam 规则。没有把中途开始的采集伪造成 control_step=0。
  - Velocity-gate follow-up/evidence: 唯一一次 uv SDK-free 工作流只运行改造后的 `TestFADADeploymentIntegration.test_registered_comparison_profiles_run_through_main`，1 个用例含 none/fada/expert_difference 三模式，4.752s 全部 OK，命令墙钟 6.34s。真实 -c/ConfigManager/main/prepare/control/finally、小型真实网络/LoRA 和 spawn worker；53 个零速实际步 (>warmup且非 interval 整倍数) 内无样本/job/更新，receipt 后动作历史/计步仍推进；w 触发保留原历史，以真实步53为起点，相对40步才提交。首个结果完成但未应用时 reset，随后12个零速步不发布且 actor 不变；s 重新触发才应用 version=1/改变固定探针输出，再 x 归零仍继续采集；同一 worker 保留且最后进程/接收线程均回收。none 全参数不变。
  - Velocity-gate follow-up/tests and delivery: 原 Commands 夹具增加真实格式的一次性 keyboard event；基本 `exercise_mode` 的 online 正向用例改为显式 w，不把零速自动学习旧行为重新固定。只执行上述新入口风险用例，没有重跑其他套件/审查。README 解释一次性门控、相对计时、reset 旧结果发布和新增状态；按用户实际 Conda/Python 部署入口修正命令并保留 DDS XML 选择，不改任何启动脚本或环境。本轮未连接或修改远端，代码尚需用户自行同步到服务器。
  - Wrapper follow-up/change: 用户确认保留 `env -u DISPLAY ./start.sh`；本轮仅把 start.sh 的 Python 调用改成 `python scripts/run_pipeline.py "$@"`、末尾改成 `run_controller "$@"`，删除要求手改 Python 默认参数的过时注释。未改无参数默认行为、DDS、后台录制、信号或退出清理，未新增 launcher。
  - Wrapper follow-up/evidence: 一次 uv 驱动的无硬件 smoke 实际执行 start.sh，临时 python 替身隔离实际 recorder/controller，controller 分支调用真实 run_pipeline.py 的 parse_args/ConfigManager，确认 -c 解析成 g1_real_fada_expert_online、method=expert_difference；原 DISPLAY=:wrapper-probe 被 env -u DISPLAY 清除，controller DDS XML 保留、recorder DDS 环境仍清除，替身录制进程收到终止、触发目录清除。命令墙钟 2.57s；临时文件全部回收，无新永久测试/全面检查，无实际 SDK/机器人启动。
  - Wrapper follow-up/delivery: README 已去掉“start.sh 不转发 -c”的旧说明，给出保留控制台输入的 wrapper 命令和 Linux auto 后端选择原因。仅本地修改，未自动同步服务器；网络、传感器时效和真实绑定仍依原有部署要求，不由参数透传 smoke 代替。

- [x] 6. 离线闭环与启动交付 — 三模式注册配置、正式入口离线验证及启动说明已完成；第 2 步原生依赖仍阻塞，不能记为 6/6。
  - Changed/owners: `robojudo/config/g1/g1_cfg.py` 增加共享 `_fada_comparison_policy` 和注册 `g1_real_fada_baseline` / `g1_real_fada_online` / `g1_real_fada_expert_online`；既有测试文件增加 `TestFADADeploymentIntegration.test_registered_comparison_profiles_run_through_main`；仅同步 README/本板。未修改第 5 步控制代码、优化目标、协议、native 或默认 ONNX。
  - Configuration: 三者继承同一正式 G1/UnitreeCppEnv/keyboard/safety 配置，共用明确的 `planner_idm.pt` 文件名，分别 method=none/fada/expert_difference；共享 helper 的 max_sensor_age_s=None 故意保留，在线部署前必须设置明确、经部署环境确认的正数预算，不能复制测试 0.01s。未添加 CLI/Hydra override、替代启动循环或 checkpoint 搜索/回退。
  - Evidence A: 唯一一次 `uv run --project /Users/chengyuxuan/ArtiIntComVis/TTT --frozen --no-sync python -u -c ...` 工作流，复用原共享环境/实际缓存依赖，无安装。真实 ConfigManager 解析三个注册名，均输出同一现存 `assets/models/g1/fada/planner_idm_v022/planner_idm.pt`；原入口仍输出 `g1_hang_book_real.onnx`。这里只检查实际配置/文件存在，不重跑生产 PT 的旧前向证明。
  - Evidence B: 同次仅运行新增的一个 registered-entry 行为用例 (含三个模式)，整个调用墙钟 5.90s。真实 `-c` parser/ConfigManager/main→prepare→42 个实际策略步→finally；基线全参数不变/无 learner，两种在线方法真实 spawn/更新后完整应用 version=1，且固定探针的实际推理输出随之改变；安全请求及 learner/接收线程退出断言通过。没有重跑第 1–5 步用例或全面审查。
  - Fixture boundary: 复用现有硬件隔离夹具，不创建 SDK controller；ConfigManager 真实加载注册配置后仅临时替换为小型 paired PT、CPU、明确合成接收预算、无 debug/kinematics、加速调度。未覆写 method 或生产优化超参，未伪造网络、梯度或学习结果；临时 PT 随上下文清除。不是生产权重闭环、物理仿真或真实 50Hz 验证。
  - Delivery/docs: README 提供三个方法的明确入口、共享 PT/网卡/年龄预算配置位置、完成/应用版本与 debug 产物、退出与后台时序限制。本轮速度门控追加中已按既有 Conda/Python 使用方式修正启动说明，原第 6 步初版 uv 部署命令不再作为这三模式的要求；本地离线验证仍复用原 TTT uv 环境。
  - Completion limit: SDK2/CycloneDDS/pybind11 环境尚缺，不进行重复已知失败的构建或安装；仅补清原生构建/导入/离线元信息验证要求，不将仅导入/版本属性当作真实接收年龄计算或 DDS/电机证据。第 4 步 SIGKILL semaphore 清理警告未结案。当前目标机 20ms、IMU/DDS 新鲜度、物理停机、PT/ONNX 同源与负重适应效果均未证明。

Blockers:
- 第 1、3、4、5、6 步 Python 工程/启动交付已离线验收；第 2 步尚缺原生构建/运行证据，因此进度为 5/6，非部署验收。强杀 semaphore 清理警告仍保留，不宣称全部 IPC 无遗留。
- 本地原生验证边界：此前默认路径核查未发现 SDK2/CycloneDDS，本地 uv 缺 pybind11/scikit_build_core/unitree_cpp。用户报告远端环境齐全，不能据本地缺包推断远端缺包；但完整新绑定、目标 ABI/Python 读出和实际 DDS/时效仍未由本轮验证，阶段 2 保留未完成。
- 本轮本地离线验证继续使用既有共享 uv/实际缓存依赖，未修改环境。远端网卡和 CycloneDDS XML 由用户手动配置；max_sensor_age_s 仍需部署时显式设置，不用测试预算替代实际预算。目标设备实时余量、停机效果及生产 PT/ONNX 同源性仍未证明。

Next:
- 由用户将本轮 start.sh 参数透传修改同步到服务器，再沿用 env -u DISPLAY ./start.sh 并显式指定 -c；网络及实际部署由用户控制，本轮不远端启动。
