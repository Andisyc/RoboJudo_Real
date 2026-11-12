# Cut3 回正候选测试（23DoF 背包版）

分支：`g1loco`。本次仅新增一个 ONNX、一个独立配置入口和本文档；原入口、Cut1、Cut2、旧 Cut3 文件及 controller 均不修改。这个入口只有一个舞蹈模型，直接选择新的 Cut3，不代表三段切换索引已经更新。

- 模型：`assets/models/g1/beyondmimic/cut3_recovery_r1_i80.onnx`
- SHA-256：`56ca0a2179c7cab9a654db59873f767dd5643d3f1de185f6298d8207799beebe`
- 配置：`g1_native_loco_mimic_23dof_backpack_cut3_recovery`
- 配置文件：`robojudo/config/g1/g1_native_loco_mimic_cfg.py`
- 模型契约：130 维观测、23 维动作、5,299 帧参考，`max_timestep=5299`。减少两段参考前俯（上限 2°、3°）并微调腿部策略；参考动作和权重均包含在 ONNX 中。

保留原来的 PD、控制频率、姿态重对齐、手柄控制和运控交还流程。无新增运行时后处理。六个平移相关输入的首层权重已在 ONNX 固化为零，不需要为此改动现场里程计配置。训练 checkpoint、CSV 和 NPZ 不需要部署。

## 给现场 Codex 的执行 prompt

你负责把 Andisyc/RoboJudo_Real 仓库 g1loco 分支中的 Cut3 回正候选部署到本次八段锦测试使用的 **23DoF 背包机器人**。只更新这一份模型，并使用新的 Cut3 独立配置入口；保留当前已验证的 controller 和 bridge、Cut1/Cut2、原配置及模型备份。

1. 先检查现场运行目录、启动命令、Python 环境、当前 git 提交和未提交修改，记录原配置、模型、网卡和 bridge 版本。使用现场原来测试 23DoF 八段锦的机器人和计算机；不要把本模型用于 29DoF，也不要仅凭 SSH 别名判断机器人型号。
2. 在单独目录获取 https://github.com/Andisyc/RoboJudo_Real.git 的 g1loco 最新提交，记录完整 commit SHA。不要对带现场改动的工作目录执行 reset --hard，也不要用整个仓库覆盖当前部署。读取 docs/cut3_recovery_test_20261111_zh.md，确认本次变更范围。
3. 校验新模型 assets/models/g1/beyondmimic/cut3_recovery_r1_i80.onnx 的 SHA-256 为 56ca0a2179c7cab9a654db59873f767dd5643d3f1de185f6298d8207799beebe，部署时保留这个新文件名。已有配置文件若与仓库不同，仅合入 g1_native_loco_mimic_23dof_backpack_cut3_recovery 这个新增配置类，继承现场原来的 g1_native_loco_mimic_23dof_backpack，保留现场网卡等设置。不要覆盖整个配置文件，不重排 Cut1/Cut2 索引，不更新 controller/bridge，不全局修改里程计。若现场父配置类或模型加载方式不兼容，先说明具体差异，停止依赖该差异的部署步骤。
4. 做离线检查，不实例化机器人环境：确认新配置只选择 cut3_recovery_r1_i80，max_timestep=5299，without_state_estimator=False，模型的观测/动作维度为 130/23，23 个关节名正确，原 PD 参数保持一致，无额外髋部补偿或输出滤波。可运行：

   python scripts/check_g1_native_config.py -c g1_native_loco_mimic_23dof_backpack_cut3_recovery
   python scripts/check_mimic_model.py assets/models/g1/beyondmimic/cut3_recovery_r1_i80.onnx --max-timestep 5299

   第一条检查可能需要现场项目正常安装的依赖，但不应加载硬件 SDK 或创建机器人环境。语法和模型检查通过不等于实机通过。
5. 准备沿用现场记录器的启动命令。仓库 start.sh 会启动记录器并转发 -c 参数，命令模板是：

   UNITREE_NET_IF=<现场控制网卡> ROBOJUDO_PYTHON=<现场Python路径> bash start.sh -c g1_native_loco_mimic_23dof_backpack_cut3_recovery

   确保 UNITREE_NET_IF 与配置内 env.unitree.net_if 一致；保留现场 CYCLONEDDS/DDS 设置。若现场有自定义记录启动脚本，沿用该脚本，只替换 -c 参数，保留完整模型推理轨迹、状态和运行日志。不要因为只更新模型就重装依赖或重新编译 bridge。
6. 完成部署与离线检查后，把实际启动命令和回滚方式交给现场人员，实机按现场原有启动、人工急停和测试流程执行；不要在无人配合时自动启动控制程序或触发舞蹈。先观察接管及启动姿态，再完整运行 Cut3，重点观察旧视频约 47–49 秒和 108–110 秒对应的回正相位，以及约 57 秒的不平衡、末姿和运控交还。新视频时间会受准备过程和控制周期影响，用 query_frame 定位两段：1730–1915、4270–4580，重点回正约 1840–1890、4470–4560。不要把模型帧除以 50 当作实际视频时间。
7. 返回本次提交 SHA、新配置名、模型哈希、现场启动命令、运行目录、完整日志和视频、是否前跳/补步/踮脚以及运控交还结果。若退出、明显失稳或结果不如现版，保留原始日志和视频，按现场流程结束并切回原入口/模型，不自动改奖励、PD、限速或 controller。

离线检查还显示：现有 A/Select 是等待合适站姿后交还，可能达到 20 秒等待上限后进入保持，不是立即急停；不要用它替代现场人工急停。这个逻辑本次未改动。

本模型已通过离线固定 50 Hz 和实测控制周期的完整仿真；较强压力组仍有明显补步，尚未证明实机问题消失。启动踉跄和收尾手部外翻未在这次单独修复。不要把离线通过或模型哈希正确报告成真机验收通过。
