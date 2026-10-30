# G1 真机动作问题：远端采集、离线对比与人工调整交接

更新时间：2026-10-30，Asia/Shanghai。

## 1. 接手后先做什么

**本任务下一步是找到远端实际运行 RoboJudo 控制程序的电脑，确认它具有本次新增代码和正确部署配置，再安排一次带同步日志的执行。不要继续建设自动补偿器。**

用户无法直接登录远端真机并取回数据，希望接手的 Agent 负责衔接远端数据工作。数据应保存在运行控制程序的电脑上，未必在机器人本体内。先区分机器人本体、部署控制电脑和训练服务器，不能假定它们是同一台机器。

当前交接已包含代码和流程；尚未提供可用的远端地址、登录方式、远端仓库路径或真实采集目录。本文不包含账号或密钥，也不代表远程执行、推送代码或启动机器人已获授权。

接手后的第一个反馈应明确：

1. 当前在哪台机器、哪个仓库，能访问哪台实际部署电脑。
2. 远端是否已有下文列出的代码，当前配置实际加载哪个 ONNX。
3. 是否已有成对的 `state.msgpack`、`motion.msgpack`，以及视频。
4. 最小下一步是读取现有数据，还是需要现场操作者配合采集一次。

有真实数据时直接进入离线对比；没有数据时先把采集步骤准备具体，再与用户和现场操作者协调执行。不要要求用户亲自登录去寻找文件。

## 2. 用户已经确定的目标与边界

目标：减少 BeyondMimic 策略在真机上执行动作时出现的不希望发生的补偿迈步。

已经确定的工作流程：

> 记录一次真机执行 → 用户指出具体问题和时间区间 → Agent 离线分析 → Agent 提出针对性调整方案 → 用户确认具体修改与试验 → 用同一个策略复测。

必须保持的约定：

- 异常定位由用户给出，不开发自动异常识别来替代用户判断。
- Agent 的介入发生在两次执行之间，不接管实时控制。
- 当前代码只负责记录和离线展示，不自动根据误差修改控制指令。
- 不预先开发通用补偿层，不启用在线学习、自适应关节偏置或自动平衡修正。
- 不把默认路线重新接回“反复修改 PKL、重新训练、再部署”。优先固定现有动作与已训练策略，针对实测问题讨论部署侧具体调整。
- 不把实测误差简单取反后加到控制指令，也不把真机的补偿迈步复制进参考动作。
- 具体修改的关节、参数、方向、幅度、时间区间和代码位置，必须由本次数据分析支持，再交用户确认。
- 沿用用户已有的部署入口、控制权切换和退出行为，不为采集更换部署方式。
- 用户偏好高速、局部、可解释的工作。只运行与变化直接相关的检查，不追加宽泛审查、长测试、训练或真机动作。

## 3. 当前状态及证据等级

| 项目 | 状态 | 可以说明什么 |
| --- | --- | --- |
| 第一步：同步记录 | 已在本地实现 | 已接入现有控制链路，包含动作帧与 DDS 实测记录 |
| 第二步：离线片段对比 | 已在本地实现 | 可以按用户指定区间生成 HTML 和对齐数据 |
| 第三步：人工分析并提出方案 | 流程已规划，等待真实数据 | 目前没有针对新实测数据确定补偿参数 |
| 离线测试 | 已通过，见第 12 节 | 证明接线、映射和样例处理，不代表真机效果 |
| 远端代码更新 | 未核实 | 不能假定远端已拉取本次改动 |
| 真实采集与分析 | 未完成 | 目前的测试样例不能作为真机故障证据 |
| 自动补偿功能 | 未实现，且不属于当前范围 | 不要把第三步理解为待开发的自动控制模块 |

权威来源顺序：

1. **AUTHORITATIVE：实际远端加载的配置、ONNX 和该次原始日志。** 这些决定真正执行的对象。
2. **AUTHORITATIVE：当前仓库实现。** 文件和字段见下文；远端版本可能不同，先核实。
3. **DERIVED：`report.html` 与 `aligned.npz`。** 它们由指定日志和参数生成，可以重新生成。
4. **HYPOTHESIS：关于多余迈步原因的解释。** 必须与具体实测数据对应。
5. **历史背景：用户此前观察与离线动作修复结果。** 不自动适用于新部署版本。

本文是单文件交接。无需另建 manifest、ledger、审批文件或重复的交接文档。

## 4. 仓库位置与本地工作树

本地仓库：

```text
/Users/chengyuxuan/ArtiIntComVis/RoboJudo_Real
```

仓库实际目录拼写为 `RoboJudo_Real`；用户有时写作 `RoboJuDo_Real`。

关联仓库：

```text
/Users/chengyuxuan/ArtiIntComVis/BeyondMimic
/Users/chengyuxuan/ArtiIntComVis/GMR
```

下文相对路径和终端命令均以实际 RoboJudo 仓库根目录为基准，便于在 Linux 远端使用。不要把上述 Mac 绝对路径原样当作服务器路径。

**交接时实现仍处于未提交的本地工作树中，没有由本会话 commit、push 或在远端 pull。** 传递本文前后，都应确认新增代码也已被传递；只提交这份 Markdown 不会让远端自动具备工具。

本任务相关文件：

```text
# 第一步：同步记录
start.sh
scripts/run_pipeline.py
scripts/record_g1_trajectory.py
robojudo/tools/g1_trajectory_recorder.py
robojudo/tools/motion_trace.py                    # 新增
robojudo/policy/beyondmimic_policy.py
robojudo/pipeline/base_pipeline.py
robojudo/pipeline/rl_pipeline.py
robojudo/pipeline/rl_loco_mimic_pipeline.py
robojudo/pipeline/g1_native_loco_mimic_pipeline.py
tests/test_motion_trace.py                       # 新增

# 第二步：离线片段对比
robojudo/tools/motion_compare.py                  # 新增
scripts/compare_g1_motion.py                     # 新增
tests/test_motion_compare.py                     # 新增
pyproject.toml                                   # 新增 analysis 可选依赖

# 原有、仍可运行的记录器测试
tests/test_g1_trajectory_recorder.py
```

工作树另有用户的 `.DS_Store` 变动及未跟踪 ONNX 文件，保留它们，不自动清理或混入代码提交：

```text
assets/models/g1/beyondmimic/eight_cut_1_v035_1.2x.onnx
assets/models/g1/beyondmimic/eight_cut_2_v010_1.2x.onnx
assets/models/g1/beyondmimic/eight_cut_3_v007.onnx
```

这些文件存在不证明它们已被远端加载，也不证明与某个训练 NPZ 对应。需要从实际配置和该次日志确认。

用户惯用代码同步方式是：本地代码上传 Git 远端，再由服务器拉取。不要未经同意更改为其他分发流程；本文不授权 commit 或 push。

## 5. 已实现的同步记录链路

### 5.1 启动与保存

`start.sh` 当前包含：

```bash
RECORD_TRAJECTORY=True
UNITREE_NET_IF="${UNITREE_NET_IF:-eth0}"
```

开启后：

1. 创建本次执行目录 `logs/g1_run_<时间>.<随机后缀>/`。
2. 导出 `ROBOJUDO_RECORD_DIR` 和 `ROBOJUDO_RUN_ID`。
3. 启动独立 DDS 记录器，立即开始记录，不再等待按 W。
4. 确认记录器订阅就绪，再启动控制程序。
5. 控制程序初始化动作记录，在准备动作之前接入；结束时保存动作日志。
6. 主程序退出后，shell 停止 DDS 记录器并保存实测日志。

每次运行最终得到：

```text
logs/g1_run_<本次编号>/
    state.msgpack
    motion.msgpack
```

设置 `RECORD_TRAJECTORY=False` 会关闭整套自动记录。这个开关当前在脚本内赋值，**不能假定 `RECORD_TRAJECTORY=False ./start.sh` 会覆盖它**；需要修改脚本中的设置。

`start.sh` 仍只调用 `python scripts/run_pipeline.py`，当前没有把额外 shell 参数转交给 Python。不要使用未经实现的 `./start.sh --config ...`。

### 5.2 启动配置不能猜

本地 `scripts/run_pipeline.py` 当前默认配置名为：

```text
g1_beyondmimic_23dof
```

这不是对远端真实配置的确认，也不能仅凭此认定是正确真机入口。启动前读取实际配置的 `env.is_sim`、环境类型、策略类型、模型文件和关节数。

配置还可能沿用旧的 `eight_cut_1_backpack` 名称或旧 `max_timestep`。确认实际 ONNX 与动作长度，不要看到本地有 v035 就假定它已生效。

`UNITREE_NET_IF` 当前只控制独立 DDS 记录器使用的网卡，**不会自动改写控制程序配置里的 `net_if`**。二者应指向同一实际机器人网络；先读远端已在使用的配置，不盲目用 `eth0` 或 `enP8p1s0`。

### 5.3 DDS 实测日志

入口：`scripts/record_g1_trajectory.py`。

实现：`robojudo/tools/g1_trajectory_recorder.py`。

订阅：

| 话题 | 内容 |
| --- | --- |
| `rt/lowstate` | 前 29 路硬件电机的 `q`、`dq`、`tau_est`，机体 IMU，DDS tick |
| `rt/secondary_imu` | 上身 IMU 的 quaternion、gyroscope、accelerometer、rpy |
| `rt/lowcmd` | 前 29 路电机目标、模式、速度、kp、kd、前馈力矩 |

记录器只订阅，不创建电机指令发布器。当前 shell 启动记录器时移除了 `CYCLONEDDS_URI` 环境变量，控制程序仍沿用其现有 DDS 配置，这是已有接线选择，不要随意统一。

### 5.4 动作与发送日志

实现：`robojudo/tools/motion_trace.py`。

接入：

- `BeyondMimicPolicy.reset()` 增加记录用 generation，区分重置。
- `BeyondMimicPolicy.get_action()` 保存本次 ONNX 查询及输出快照。
- `PolicyWrapper.get_pd_target()` 保存映射到环境顺序前的策略关节目标。
- `Pipeline.send_traced()` 在原发送调用前记录目标，之后记录返回或异常。
- 普通 RL、RL locomotion/mimic、native locomotion/mimic 三条入口均接入。
- 准备动作、插值、播放、暂停、结束及原生控制切换有阶段或事件记录。

记录队列有容量上限。数据在入队前复制，后台写盘。活动期间记录失败会报告，不主动改变发送目标、发送返回值或原发送异常。关闭时统计丢失与错误；缺少结束统计的日志应视为不完整。

**注意：`send_result.status=returned` 只说明原发送调用返回，不代表物理动作已经完成。** 仍要结合 DDS 和实测。

## 6. 数据格式与分析约定

两个文件都是连续 MessagePack 对象流，不能当作一个 JSON 或一次 `unpackb()` 读取。

最小只读查看方法：

```python
from pathlib import Path
import msgpack

path = Path("logs/实际执行目录/motion.msgpack")
with path.open("rb") as stream:
    for event in msgpack.Unpacker(stream, raw=False):
        if event["kind"] in ("header", "segment", "phase", "summary"):
            print(event)
```

避免输出整段高频轨迹；先查看头信息、播放段和统计。

### 6.1 公共字段

```text
kind
monotonic_time_ns
receive_time_ns
payload
```

- 两个头信息中的 `payload.metadata.run_id` 必须存在且相同。
- 当前格式版本均为 1。
- 主对齐时间是同一台部署电脑、同一次系统运行中的单调时钟。
- DDS 时间戳是主机接收时间，不是已校正网络延迟的传感器采样时间；原始 tick 仍在 state 日志中。
- 关节位置为 rad，速度为 rad/s，IMU gyro 为 rad/s，加速度为 m/s²。
- 当前记录的 DDS 与 ONNX 身体 quaternion 顺序均标注为 wxyz；不要混用 PKL 常见的 xyzw。

### 6.2 motion 日志关键事件

| kind | 关键内容 |
| --- | --- |
| `header` | 实际配置类、pipeline 类型、环境关节名称、硬件映射、run_id、单位及时间语义 |
| `segment` | 播放段编号、策略名称、模型路径、策略关节名称、配置控制频率 |
| `cycle` | 周期编号、segment、phase、指令、inference、环境顺序的 final_target |
| `send_result` | 对应 cycle 的 returned/raised 及返回值或错误 |
| `phase` / `commands` / `native_state` / `lifecycle` | 阶段与生命周期事件 |
| `summary` | 记录数量、丢失数量、错误和 complete 标记 |

`cycle.payload.inference` 包含：

```text
query_frame
segment_generation
observation_reference
output_reference.joint_pos / joint_vel / body_pos_w / body_quat_w
raw_actions / filtered_actions / scaled_actions
policy_target
inference_end_monotonic_ns
```

`observation_reference` 是本次观测使用的参考命令；`output_reference` 是本次 ONNX 查询返回的参考。二者可能相差一帧，必须分别保留。

`query_frame` 是送入 ONNX 的整数索引，模型内部可能钳制索引。不要把超过模型末尾的查询索引描述为实际存在的新动作帧。

### 6.3 帧率与名称映射

当前 `segment.reference_fps` 故意为 `None`。`configured_policy_hz` 是控制配置频率，不能代替参考动作实际帧率。

第二步命令需要显式指定 `--reference-fps`。应从**该策略对应的训练 NPZ**读取，例如：

```bash
python -c 'import sys,numpy as np; d=np.load(sys.argv[1],allow_pickle=False); print(d["fps"])' /实际路径/对应动作.npz
```

历史常用导出为 50 FPS，但不能仅凭历史写死。

关节数据有三个顺序：

1. 参考及 `policy_target`：策略关节名称顺序。
2. `final_target`：环境关节名称顺序。
3. DDS motor：原始硬件电机索引顺序。

使用头信息中的名称与 `joint_to_hardware_motor` 对齐。23DoF 不是简单取原始 29 路数据的前 23 个。

## 7. 接手 Agent 的远端工作顺序

### 7.1 先进行只读核实

在获得的远端环境中确认仓库、代码和配置。下面是检查命令，不启动机器人：

```bash
pwd
git status --short
git diff --stat
python scripts/compare_g1_motion.py --help
```

检查第 4 节代码清单是否完整。尤其是新文件不能漏传；本地尚未提交的改动不会因服务器执行 `git pull` 自动出现。

先读 `scripts/run_pipeline.py` 的实际默认配置，再顺着注册配置确定环境、模型路径、关节顺序及网卡。不要为验证日志就启动一次真实动作。

### 7.2 确认现场具备执行条件后采集

保持现场原有控制流程，由用户/现场操作者决定何时执行、如何操作机器人。接手 Agent 可以准备命令，但不能因本文就自主启动带电机输出的流程。

确认实际网卡及部署配置之后，仍使用已有入口。例如：

```bash
# 在实际部署仓库根目录；替换占位网卡，确认配置后由现场执行。
UNITREE_NET_IF="实际机器人网卡" ./start.sh
```

预期看到：

```text
Synchronized recording directory: logs/g1_run_...
Trajectory recorder ready and recording ...
```

然后按既有方式进入动作，固定机位拍到全身和双脚。录制前后留出少量余量，并记录视频与动作开始的对应点。用户给出的“23 秒”需注明是动作时间、视频时间还是日志经过时间。

正常退出既有控制程序，让其控制释放和日志保存流程完成。不要为了保存日志改变既有控制权处理或强杀进程。

### 7.3 找到本次目录并检查完整性

记录终端打印的**准确目录**，不要只凭“最新文件”推断是哪次动作。

本次目录应有两份非空文件；查看两份头信息 run_id、结束 summary、各话题计数及 motion segment。

需要确认：

- 两份 run_id 相同。
- state 中 lowstate、lowcmd 是否有数据；secondary IMU 若为空，分析中应标注。
- motion 中有正式播放周期，而不只有 preparing/acquiring。
- 是否有 dropped、写盘错误、发送异常或缺少 summary。
- 实际 policy_file 是否正是本次要分析的模型。

记录不完整时不要覆盖或删除原始文件。对比工具会报告相应信息，应结合目标区间是否受影响判断能否继续分析。

### 7.4 数据交付

优先由能够访问远端控制电脑的 Agent 在原地完成初步分析，再通过已获授权的文件通道返回结果，减少用户操作。

供后续分析的数据应包括：

- 本次完整目录下的 `state.msgpack`、`motion.msgpack`。
- 本次视频，以及视频时间与动作开始的对应说明。
- 用户标注的问题、期望行为、区间和左右侧。
- 对应 NPZ 的 fps，以及实际 ONNX 文件/可访问副本。
- 生成的 `report.html` 与 `aligned.npz`。

完整原始日志仍保留在采集电脑，不仅传截图。转移目的地、SSH 地址或共享目录均由实际已有授权确定；不要猜地址、索取明文密钥或擅自上传第三方服务。

## 8. 离线对比：精确命令与输出

入口：`scripts/compare_g1_motion.py`。

实现：`robojudo/tools/motion_compare.py`，不导入部署 package 的初始化代码，也不连接 DDS、启动策略或写电机指令。

在现有分析 Python 环境补齐依赖即可：

```bash
python -m pip install numpy scipy msgpack matplotlib onnx
```

仓库也声明了 `analysis` 可选依赖组，其中 matplotlib 用于出图，onnx 用于读取身体名称元数据。不要为了离线分析重装远端完整控制环境。

下面使用 shell 变量表达占位路径，先替换为实际值：

```bash
run_dir="/实际部署仓库/logs/实际执行目录"
model_file="/实际部署仓库/assets/models/g1/beyondmimic/实际加载模型.onnx"

# 50 仅为命令示例，必须替换为本次对应 NPZ 的实际 fps。
python scripts/compare_g1_motion.py \
  --run-dir "$run_dir" \
  --reference-fps 50 \
  --start 20 \
  --end 26 \
  --context 2 \
  --max-gap-ms 100 \
  --model-file "$model_file"
```

若一次采集中有多个正式播放段，工具会打印可选编号并要求 `--segment`。先核对日志，再在命令中加入正确段号，不假定永远为 1。

默认输出：

```text
<run-dir>/comparison_s<segment>_<start>_<end>/
    report.html
    aligned.npz
```

可用 `--output-dir /新的输出目录` 指定其他位置。工具拒绝覆盖已经存在的输出目录；重新生成时用新目录，不删原始数据。

### 8.1 对比工具的处理行为

- 只选 playing、paused、done 中具有参考查询快照的周期；准备和 acquiring 不当作正式动作区间。
- 动作区间按 `query_frame / reference_fps` 选取，前后 context 同样使用动作时间。
- 图表横轴为该段首次正式播放后的主机经过时间。暂停时动作时间不变，但横轴时间仍前进。
- 关节实测使用有上限的双侧时间插值，默认超过 100ms 的缺口不插值、不外推。
- DDS 指令使用因果零阶保持，不拿未来指令补前面的周期；过旧指令标为缺失。
- quaternion 在有效短时间括号中做旋转插值。
- 不自动平移曲线来消除跟踪延迟。
- 记录周期本身的长缺口会在绘图时断线。
- 不把各阶段或不同播放段拼成一段连续动作。

### 8.2 模型元数据与姿态限制

参考关节和身体 quaternion 来自**本次日志**。`--model-file` 只用于读取 ONNX 的 `joint_names`、`body_names`，不重新推理，也不替换日志中的轨迹。

只有关节顺序与记录一致时，才按名称提取 `pelvis`、`torso_link`。没有元数据时仍可生成关节与 IMU 报告，不猜身体数组索引。

当前未应用 IMU 安装标定，参考身体姿态和 IMU 姿态分开展示，不计算直接姿态误差。不能把未校准的轴差解释为机器人实际倾斜误差。

仅凭关节和 IMU，不能可靠重建脚相对地面的滑动或世界坐标轨迹。当前报告也不输出“实测世界脚位”或“实测重心”。脚步是否发生由用户和视频确认。

### 8.3 aligned.npz 的主要内容

可用 `np.load(path, allow_pickle=False)` 读取，元信息存于 Unicode 字符串 `metadata_json`，不是 pickle 对象。

```text
monotonic_time_ns / time_s / motion_time_s / query_frame
joint_names / cycle / phase / send_status / core_mask
reference_q / reference_dq / observation_reference_q
policy_q / final_q / dds_q
actual_q / actual_dq / tau_est
base_quaternion / base_gyroscope / base_accelerometer
torso_quaternion / torso_gyroscope / torso_accelerometer
reference_base_quat / reference_torso_quat          # 元数据可用时才有
*_valid / *_age_ms / *_bracket_ms
cycle_gap_before_ms
actual_minus_reference / actual_minus_final_target
actual_minus_dds / velocity_minus_reference
metadata_json
```

图上关节角度显示为 deg，速度为 deg/s；保存数组及统计差值使用 rad、rad/s。报告中的 peak host time、motion time 和 query frame 不是同一种时间坐标。

## 9. 第三步如何提出针对性方案

用户给出问题区间后，Agent 先看发生前的过程，而不只看迈步的一帧。

建议交付以下内容，直接写在会话里即可：

1. **分析对象**：本次 run_id、segment、模型及问题区间。
2. **已确认事实**：参考、策略目标、DDS、关节实测和 IMU 中的变化及先后顺序，引用具体数值和时刻。
3. **解释与剩余不确定性**：哪些原因得到支持，哪些尚未证实；不能只凭相关性确定因果。
4. **一个优先调整方案**：具体文件、函数或配置参数、方向、初始幅度、区间、平滑过渡、依据。
5. **保持内容**：固定 ONNX、动作版本与其他已正常行为，明确本次只改什么。
6. **验证预期**：修改后应观察什么；什么结果会否定当前判断。
7. **回退方式**：恢复原配置或原代码的明确操作。

没有数据支持时，不预填某个髋、膝、踝的补偿角度，不生成虚构“最佳参数”。用户确认某个具体方案后，再按 engineering-plan 规划那一次最小代码改动。

即使后来需要固定部署修正，也必须针对这次问题明确设计，不把用户确认过的方案扩展成未经授权的自动补偿框架。

## 10. 本次问题的历史背景，避免接错版本

用户此前报告：cut1 部署后约 23 秒、52 秒，右脚出现向前补偿迈步。该轮排查基于 v034。随后离线生成了 v035，处理双脚前后错位与重心前偏；**这不是 v035 已通过真机验证的证据，也不意味着现在应继续修改动作或重训。**

本地 cut1 输出：

```text
/Users/chengyuxuan/ArtiIntComVis/GMR/motion_pkl_23dof/cut1/eight_cut_1_v035_1.2x.pkl
/Users/chengyuxuan/ArtiIntComVis/BeyondMimic/motion_csv/eight_cut_1_v035_1.2x.csv
/Users/chengyuxuan/ArtiIntComVis/GMR/videos/cut1/eight_cut_1_v035_1.2x.mp4
```

v035 保留 5177 帧、30 FPS、约 172.57 秒。训练导出的 NPZ 帧率与帧数要从实际文件读取，不能套用 PKL 帧率；动作已做过裁剪与加速，不能直接使用原视频的旧时间戳。

Senior Student 推荐意见的本地来源：

```text
/Users/chengyuxuan/Downloads/adjust.txt
```

其内容包括同步记录状态和指令、用视频定位执行问题的参考做法。文件内的 GR00T/SONIC 路径属于对方工程，不是本仓库运行依赖；其中原 DDS 记录示例仅覆盖少数手臂关节，不能原样当作本项目完整脚步分析工具。

用户随后明确补充：不要反复修动作再重新训练，应由人指出具体问题，Agent 据真机数据提出部署侧针对性方案。以用户这条后续决定为准。

## 11. 常见衔接问题

| 现象 | 优先核对 |
| --- | --- |
| 只有 state，没有 motion | 是否通过新版 start.sh 启动；环境变量是否传入控制进程；新增文件是否漏传 |
| 只有准备动作，没有正式播放段 | 是否真正进入本次 Mimic；记录对应的是哪次执行；不要把 warm-up 当播放 |
| 一直等按 W | 远端还是旧 start.sh，或独立记录器仍使用 --start-trigger-file |
| DDS 没数据 | 实际部署网卡与记录器网卡是否一致；不要先修改动作或策略 |
| run ID 不匹配 | 是否误配了不同执行的两份文件；不要绕过检查强行拼接 |
| ONNX 身体图缺失 | --model-file 路径是否为本机可读文件，onnx 分析依赖是否安装，名称是否匹配 |
| 输出目录已存在 | 用新的 --output-dir，不覆盖原始记录 |
| 参考和实测好像相差一帧 | 区分 observation_reference 与 output_reference；查询帧不是 pipeline 累积步数 |
| 视频时刻和报告时刻不同 | 先明确动作时间、视频时间、host time 三者对应关系 |
| 没有 summary 或出现 dropped | 保留文件，标注不完整，并检查目标片段内的缺口和有效性标记 |

采集失败、数据缺失与机器人实际运动问题应分别说明，不把日志问题解释成动作问题。

## 12. 已执行验证及必要时的复现命令

第一步本地执行并通过：

- `tests/test_g1_trajectory_recorder.py`：2 项，覆盖只订阅记录、触发前忽略和读回。
- `tests/test_motion_trace.py`：4 项，覆盖三条实际 pipeline 发送位置、ONNX 单次查询快照、暂停/重置、数组副本、记录开关输出一致、写盘故障及原发送异常保持。
- 改动 Python 文件语法检查与 `git diff --check`。

第二步本地执行并通过：

- `tests/test_motion_compare.py`：1 项端到端样例，包含非 50 FPS、关节乱序、已知偏差和延迟、暂停、重置、缺口、ONNX 身体名称映射、报告与 NPZ 读回、原始输入未改、错误 run ID、截断记录。
- 对第二步三个 Python 文件进行 Ruff 检查，通过；`git diff --check` 通过。

若远端代码有变化，需要复现时可在无机器人连接的分析环境运行：

```bash
python -m unittest discover -s tests -p test_g1_trajectory_recorder.py -v
python -m unittest discover -s tests -p test_motion_trace.py -v
python -m unittest discover -s tests -p test_motion_compare.py -v
```

测试的模型、发送端和 DDS 为离线样例或模拟对象，不启动机器人。第二步测试需 numpy、scipy、msgpack、matplotlib、onnx。

如果使用临时 uv 环境，可以按最小依赖运行对比测试：

```bash
uv run --no-project --with numpy --with scipy --with msgpack --with matplotlib --with onnx \
  python -m unittest discover -s tests -p test_motion_compare.py -v
```

上述证据证明的是代码处理与记录接线。**尚未测得真实 DDS 采样间隔、现场掉帧率、记录开销，也尚未证明任何调整能消除真机补偿迈步。** 不需要因此先运行整仓测试或重新训练；下一项证据应来自正确配置的一次实际采集。

## 13. 给下一位 Agent 的接手提示

可以将下面内容作为下一会话的起始指令：

> 先阅读 RoboJudo_Real/docs/G1_REAL_MOTION_HANDOFF.md，并核对当前仓库代码。我们已实现同步记录和离线片段对比，尚未取得本轮真实采集数据。用户不能直接登录远端取数据。请先确认你能访问的部署电脑、实际配置和数据目录，报告代码与远端状态差异；优先读取已有日志，没有日志则准备一次由现场操作者配合的采集。异常由用户指定，你负责离线分析后提出一个具体、可撤销的调整方案。不要自动补偿，不要自行启动机器人，不要默认改 PKL 重训，也不要未经授权 commit、push 或转移数据。

远端无访问条件时，明确缺少的是哪台部署电脑的执行/文件访问，并向现场可访问者提供准确命令和所需回传文件；不要以“请用户自己登录下载”结束交接。
