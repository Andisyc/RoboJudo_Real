# 003 穿鞋八段锦测试

请在绿芽 **G1-003，23DoF 背包** 上测试 Cut1、Cut2、Cut3。全程用同一双鞋、同一块平整地面和一致的起始站位。现场人员负责上机操作、录像和必要时停止。

## 同步与核对

1. 从 GitHub `Andisyc/RoboJudo_Real` 的 `g1loco` 分支同步提交 **Add opt-in G1 shoe trial for Cut1 and Cut3**，记录实际部署的 SHA。将该提交的变更合入 003 实际运行目录的独立测试副本，沿用现场 Python 环境、网卡、Bridge、互斥保护和轨迹记录器。现场目录有本地改动时先核对差异。此次只新增模型、配置和离线校验，无需重编 Bridge。
2. 在单测副本中用实际运行的 Python 执行 `python scripts/check_baduanjin_suite.py --shoes`。确认输出 `Suite contract passed`，并核对模型和手柄索引。

| 手柄索引 | 动作 | 模型 |
|---:|---|---|
| 0 | Cut1 | `cut1_i100_shoe_finish_20261121` |
| 1 | Cut2 | `eight_cut_2_v010_1.2x` |
| 2 | Cut3 | `cut3_i5999_shoe_recover_20261121` |

## 现场测试

- 按 Cut1、Cut2、Cut3 的顺序各做一次手柄单段测试。每段分别启动 `bash start.sh -c g1_baduanjin_23dof_shoe_select`，用 R1 选择模型并核对日志，让现场人员确认站稳后按 Start。每段保留独立运行目录。单段 Cut3 启动仍有 5 秒参考过渡，前 1.5 秒同时平滑最终关节目标。
- Cut1 重点拍结束前约 5 秒的收脚和前倾。Cut2 完整记录，做同一双鞋下的对照。Cut3 重点拍动作开始后约 **8–10 秒右脚外跨**、**22–25 秒前俯回正**，再记录末段转身和最后举臂。这里时间从正式播放第 0 帧算起。录像从按 Start 前开始，使用能同时看清双脚和全身的侧前方角度。
- Cut3 的离线压力回放仍出现失稳。若某段明显前扑、脚部异常或需要人工扶持，按现场流程停止并保留已录材料。如三段单测都稳定且现场具备继续测试条件，可用 `bash start.sh -c g1_baduanjin_23dof_shoe_suite` 再跑一次联播，重点记录 Cut2→Cut3 过渡和 Cut3 收尾。
- 切换测试入口前，先确认上一轮已交还官方控制并退出相应 controller。正常完成后记录最终 FSM，预期为 500。

## 回收材料

回传每段原速视频、`start.sh` 打印的完整运行目录、其中的 `motion.msgpack` 和 `state.msgpack`、控制台日志、部署 SHA、模型名和最终 FSM。请按 **003_穿鞋_Cut1 / Cut2 / Cut3 / 联播** 标注，并记录鞋、地面、手动停止或扶持的时间。有异常也请回传已有日志。只要尚未传输，也可以先回报 003 上日志的绝对路径和运行编号。
