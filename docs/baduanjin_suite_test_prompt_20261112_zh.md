请在 **003（23DoF 背包）** 同步 GitHub 仓库 https://github.com/Andisyc/RoboJudo_Real 的 `g1loco` 分支。本轮只实测一次 Cut3 单段，联播同步更新默认模型。

1. 找到上次 003 手动单段与语音联播各自实际使用的部署目录、Python 和启动入口。保留现场改动（含运控保护）和可回滚版本，从 GitHub 同步发布的 ONNX、共享配置中索引 2 的模型选择和离线校验。如两入口使用不同代码副本，分别核对。如果工作树有现场修改，保留修改并合入本次差异。沿用现有 bridge、网卡、DDS 和服务启动方式。记录来源提交与实际部署路径。
2. 在两个入口实际使用的环境中运行 `python scripts/check_baduanjin_suite.py`。应得到 0 = Cut1 i100，1 = Cut2 v010，2 = `cut3_field_scratch_i5999`（5299 帧）。核对实际加载的 `assets/models/g1/beyondmimic/cut3_field_scratch_i5999.onnx` 的 SHA-256 为 `042833fcfa9b55fc5b45902f34865f491cf60354c0b5e438386053ea9e8784ab`。单段与联播都要生效，包括服务下次启动时使用的路径。
3. 由现场人员按原交还和停服流程释放语音联播 controller，确认只留一路控制。沿用 003 上次带记录器的单段入口（仓库入口为 `bash start.sh -c g1_baduanjin_23dof_select`）。现场用 R1 选到索引 2，看到当前选中 i5999 后按 Start，只跑一次。录制原速全身视频，双脚、双手均入镜；保留本轮控制台输出及同一运行目录下的 `motion.msgpack`、`state.msgpack`。
4. 从 Cut3 正式播放起，观察约 47 秒和 74 秒回正时的前倾、踮脚或垫步，记录哪只脚、对应时间；同时看开头 0–5 秒、15–20 秒及收尾 1:35–1:40 的肩臂动作。记录结束后常规运控 FSM 500 的实际读回。异常时按现场既有人工停止流程处理，保留已录数据。
5. 单段交还控制并退出后，按原托管方式恢复更新后的联播服务；核对 `g1_baduanjin_23dof_suite` 的模型清单，联播仍从 Cut1 开始，本轮无需实跑三段。请返回提交、实际 Cut3 哈希、运行目录、完整视频及两处垫步和上肢观察。需要回滚时恢复此次留存的部署，原 Cut3 为 `cut3_jointv1_pitch8_rightless_20261112`。
