# 001 单段 Cut3 接管测试

请在绿芽现场 G1-001（23DoF 背包）部署并协助现场人员完成一次 Cut3 单段测试。本轮验收手柄 Start 接管后的前半秒，并记录完整动作到交还常规运控。等上午的联播结束、测试日志已保存、001 已交还官方运控后再开始。联播服务和部署目录继续保留原版本。

1. 从 GitHub 的 `Andisyc/RoboJudo_Real` 拉取 `g1loco`，定位提交 `Smooth final targets during standalone Cut3 takeover`，核对并记录 SHA。以 001 **实际使用的部署目录**为基础建独立单测副本。将该提交相对父提交 `0802882` 的差异合入副本，保留现场网卡、解释器、已安装 Bridge、静音、DoF 自检、互斥和记录器。遇到现场改动冲突先核对并合并，避免整文件覆盖。在副本根目录用现场 Python 跑 `python scripts/check_baduanjin_suite.py` 和 `python tests/test_pose_transition.py`，两项通过再启动。此次无需替换 ONNX 或重编 Bridge。
2. 确认 001 上没有其他 controller 正在接管，沿用现场保护和带记录器的入口，在单测副本里运行 `bash start.sh -c g1_baduanjin_23dof_select`。由现场人员确认双脚着地、机器人站稳，用 R1 选中 **索引 2**，核对模型 `cut3_i5999_advance8_back12_20261119` 后按 Start 一次。录像从按键前开始，侧前方构图同时看清膝、双脚和上肢。
3. 重点看接管的前 **0–0.5 秒**有无突然屈膝、甩臂、下沉或脚跟弹起，**1.5 秒附近**有无第二次突动，约 **5 秒**进入动作是否连续。正常则跑完整段并记录最终 FSM；出现明显失稳，现场按既有流程终止并保留已录数据。
4. 从本轮 `motion.msgpack` 核对 `entry_target_blend` 的 `begin`（`duration_s=1.5`）和 `complete`，以及 `pose_transition` 的 `phase=entry`（`duration_s=5`）和完成事件。两段过渡重叠计时。记录首条 `acquiring` 的最终目标，及相对真实关节位置的前半秒膝部和上肢变化。如未触发，回报拒绝原因、实际运行配置和代码路径。
5. 回传原速视频、完整运行目录（含 `motion.msgpack` 和 `state.msgpack`）、控制台日志、提交 SHA、最终 FSM（预期 500）和现场观察。若有扶持或吊绳承重，注明时刻。结束后退出单测 controller。
