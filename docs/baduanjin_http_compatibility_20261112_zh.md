# 三段联播：旧 HTTP / 语音 Agent 调用方式核对

2026-11-12。用户确认现场语音 Agent 使用 `/start`、`/enter_mimic` 这一组 HTTP 接口。本次保持这些 URL、请求结构和命令语义；切换为三段联播需要更新启动配置参数，并在每次从头播放前明确选择索引 0。

## 旧日志能证明什么

`优化后-数据log.zip` 内的 `run_service.log` 记录的是 **10 月 15 日**，不能按压缩包上传日期写成 11 月 10 日：

- 10:55:03.235：服务加载 `g1_native_loco_mimic_23dof`，模型进度条为 `eight_cut_1`。
- 随后 `POST /start`、`POST /enter_mimic` 均返回 200。
- 10:55:28.645：`G1 user control acquired`，实际状态为 `USER_ACTIVE`、FSM=1000、publish_enabled=true。
- 后续模型帧持续推进至 3532，随后收到 `/shutdown`。

这证明过去确有 **HTTP 启动八段锦并实际进入舞蹈** 的记录，而不只是收到请求。日志没有唤醒词、ASR 文本或语音请求 ID，因此不能仅凭 controller 日志将这一轮确定为语音唤醒来源。用户对现场 HTTP 链路的确认与上述控制器侧证据相互补充，但不是 ASR 端到端测试。

另外，11 月 4 日服务快照中也有 `g1_native_loco_mimic_23dof` / `eight_cut_1` 经 HTTP 入舞记录。但快照取自公司另一台 29DoF 计算系统，配置名不能证明实体机器人自由度或日志是否被复制，不能将它当作本台 23DoF 的成功验收。

## 仓库与现场服务的差异

现场 11 月 5 日源码快照已经有 `/policies`、`/select_policy`、`policy_name` 和 `sequence_index` 状态字段；GitHub 原基线 `136ed28` 的服务缺少这部分。直接覆盖现场服务会令旧 Agent 的精确选段请求返回 404。

本次补齐上述已部署接口及 pipeline 公共选段方法，并增加只读 `config_name` 状态字段。按名称必须使用当前加载的准确模型名；按索引是 0/1/2。跳舞和等待入舞期间拒绝选段，返回 400。没有把曾返回 404 的 `/select_sequence`、`/next_sequence` 等猜测路径新增为别名。

`/start` 默认配置保持原样，不能省略配置名来期望自动选联播。`/start` 的 200 表示服务准备完成；`/enter_mimic` 的 200 表示命令排队，不等于成功跳舞。原 `/switch_policy?direction=next|last`、`/return_loco`、`/shutdown` 保持原语义。

## 给语音 Agent 的调用合同

服务仍由原现场管理方式启动 `scripts/run_service.py`。保持原服务地址、Python 环境、网卡、DDS 和反向代理配置。只运行一个控制器实例，不要同时运行 `start.sh` 和 HTTP 服务。

| 步骤 | 调用 | 需要确认 |
|---|---|---|
| 查询当前状态 | `GET /status` | 是否已加载；不能同时开第二个 controller |
| 尚未加载时准备联播 | `POST /start`，JSON `{"config_name":"g1_baduanjin_23dof_suite"}` | `/status.config_name` 确认是新联播配置 |
| 查询可选模型 | `GET /policies` | 顺序为新 cut1、原 cut2 v010、新 cut3 ref6 |
| 每次从头演出前选 cut1 | `POST /select_policy`，JSON `{"sequence_index":0}` | 返回 `selected`、索引 0，且仍在官方运控 |
| 触发一次完整演出 | `POST /enter_mimic`，不带请求体 | 后续实际进入 `MIMIC_ACTIVE` / USER_ACTIVE / FSM 1000 |
| 等待全部结束 | `GET /status`，读取日志事件 | 最后从走跑进入常规 500、发布关闭；不能只看 state=NATIVE_LOCO |

完整顺序：常规→走跑→3 秒启动过渡→cut1→3 秒交接→cut2→3 秒交接→cut3→原交还流程→走跑→常规 500。

再次语音启动时，当前模型通常停在索引 2，所以必须再次 select 0，再 enter。不要每段调用 `/enter_mimic`，不要中间主动 switch，也不要让旧 Agent 继续按“一段结束就 `/stop` 再启动下一段”的逻辑干预联播。中途要求停止使用原 `/return_loco`，它是等待站姿退出，不能替代人工急停。

若已有旧配置正在运行，不能再次 `/start` 叠加；应先完成原官方运控交还并确认 publish_enabled=false，再由现场原管理流程停止/重启服务并加载新配置。`/stop` 是释放/关闭服务接口，不是舞蹈中急停接口。若 Agent 按旧模型名选择，必须更新模型名映射或改用索引；这里不自动将旧名字映射到新模型。

想继续“一次语音只跳某一段”，使用 `g1_baduanjin_23dof_select` 配置，然后 select 相应索引再 enter；与连续三段入口明确区分。

## 记录与离线验证

服务启动前设置 `ROBOJUDO_RECORD_DIR` 和 `ROBOJUDO_RUN_ID`，`/start` 会按 `run_pipeline.py` 相同机制启用 `motion.msgpack`，包含模型/输出和段间事件。另用现场已有记录器保存 `state.msgpack`，两者保持同一 run_id，网卡与 controller 一致。HTTP 服务没有调用 `start.sh`，所以不能假设 `start.sh` 的记录器会自动启动；部署人员需确认两个文件都在增长。日志目录每轮唯一，旧服务结束且控制已交还后由原管理流程回收记录器。

测试没有启动 HTTP 监听端口、没有构造机器人环境、没有发机器人指令：FastAPI TestClient 在进程内发请求，硬件 SDK 导入被禁止，用假环境/线程运行真实服务路由、命令队列和 pipeline 命令分发。覆盖新配置加载、首次 enter、名称/索引选段、从索引 2 回 0 重播、舞蹈/待入舞时拒绝选段、错误参数、原 switch/return 路由及 motion trace 配置。

本次总计 **80 个测试、48 个 subtests 通过**（8 个 HTTP 测试 + 原 72 个）。HTTP 测试环境为 FastAPI 0.115.12、uvicorn 0.34.2、httpx 0.28.1；离线测试需 pytest/httpx，现场已运行 HTTP 服务的环境继续使用原依赖，不为此全量升级环境。一个测试依赖的弃用提示不影响结果。

这些是软件接口兼容证据，不等于真实语音识别、网络服务、机器人端到端验收。现场仍需说原来的唤醒词/动作指令，确认上层识别、HTTP 请求、实际三段执行和最终 500 读回能关联到同一轮。
