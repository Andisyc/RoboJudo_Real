# cut1 v034 固定髋关节补偿：修改报告与 Agent 交接

日期：2026-11-06。

## 1. 当前交付与下一步

已实现按模型查询帧生效的双侧髋俯仰固定偏置，接入现有 `G1NativeLocoMimicPipeline`。补偿配置默认关闭；保留现有启动入口、策略选择、ONNX、动作文件和底层通信。

本次交付是部署侧补偿代码，不是重新训练或重做动作序列。下一步是由现场同事将录像中第一次举臂对应到日志查询帧，填写确认后的帧段，在用户决定进行试验后启用。当前候选帧段不能当成已完成现场对齐的参数。

本机仓库根目录为 `/Users/chengyuxuan/ArtiIntComVis/RoboJudo_Real`。下文源码路径均相对仓库根目录，服务器使用自己的实际 checkout 路径。

## 2. 用户目标与已选方案

用户观察到 cut1 v034 真机播放时，第一次和后续一次举臂阶段出现计划外的向前踏步。用户希望沿用现有已训练策略，以人工决定的小幅部署补偿进行调整，避免反复修改动作、重新训练。

这次只覆盖第一次举臂。双侧 `left_hip_pitch_joint`、`right_hip_pitch_joint` 同时加入正向小偏置，初始峰值为 +0.5°。目的在于稍微向后收身体，减少前移需求。0.5°是初始试调值，不是根据实测重心拟合的最优解。

补偿参数由人事先指定。运行时仅根据策略名称和查询帧执行固定曲线，不读取 IMU、关节误差或足底接触状态计算新的补偿，也不进行参数学习、优化或自动调参。

先前讨论的误差滤波、残差优化方案不在本次实现中。

## 3. 已修改的文件与函数

| 文件 | 修改内容 |
| --- | --- |
| `robojudo/pipeline/pipeline_cfgs.py` | 新增 `FixedHipBiasCfg` 和参数校验；为 `G1NativeLocoMimicPipelineCfg` 增加 `fixed_hip_bias` 配置。 |
| `robojudo/config/g1/g1_native_loco_mimic_cfg.py` | 在 `g1_native_loco_mimic_23dof_backpack` 显式填写候选配置，保持 `enabled=False`。 |
| `robojudo/pipeline/g1_native_loco_mimic_pipeline.py` | 初始化时按环境关节名称缓存两侧髋俯仰索引；新增 `_apply_fixed_hip_bias()`；在 `_step_mimic()` 发送前接入；在成功进入 Mimic 后记录一次配置事件。 |
| `tests/test_g1_native_integration.py` | 新增 `test_fixed_hip_bias_at_send_boundary`，用真实配置和生产方法、模拟策略及发送函数检查补偿。 |
| 本文档 | 记录实现、启用、回退、验证和接手边界。 |

开始任务时，`g1_native_loco_mimic_cfg.py` 已有两处用户修改：`g1_native_loco_mimic_v2.return_to_regular_loco` 与 `walkrun_on_start` 均由 False 改为 True。本次保留，不将它们视为本次补偿修改。

## 4. 每周期实际执行顺序

1. `G1NativeLocoMimicPipeline._step_mimic()` 调用现有策略构造观测。
2. 读取 `int(self.policy.timestep)`，与这一轮 ONNX `time_step` 的整数查询规则一致。
3. 调用原来的 `PolicyWrapper.get_pd_target()`。这一步完成推理、动作缩放、默认关节位置相加及模型顺序到环境顺序的映射；原始策略目标仍按原方式记录。
4. 调用 `_apply_fixed_hip_bias(target, query_frame)`。
5. 将返回目标交给现有 `send_traced()`，记录最终目标并按原路径发送。
6. 外层现有 `post_step_callback()` 再推进策略帧。

因此，补偿单位是最终 PD 关节目标的弧度，不是网络动作的归一化单位，也不是动作参考轨迹本身。不能把 `hip_pitch_deg` 直接加到 ONNX 原始动作里，那里还存在缩放和默认位置语义。

## 5. 补偿曲线与参数

```python
fixed_hip_bias: FixedHipBiasCfg = FixedHipBiasCfg(
    enabled=False,
    policy_name="eight_cut_1_v034_1.2x",
    hip_pitch_deg=0.5,
    ramp_in_start=1100,
    ramp_in_end=1140,
    ramp_out_start=1310,
    ramp_out_end=1350,
)
```

| 查询帧 | 偏置 |
| --- | --- |
| ≤1100 | 0° |
| 1100–1140 | 从 0°平滑增加至 +0.5° |
| 1140–1310 | 保持 +0.5° |
| 1310–1350 | 从 +0.5°平滑减至 0° |
| ≥1350 | 0° |

过渡使用 `s(x) = 10x³ − 15x⁴ + 6x⁵`，其中 x 从 0 变化到 1。曲线在各过渡端点的一阶、二阶导数为零。两侧髋关节加入同一个 `radians(hip_pitch_deg) * s(x)`，其他关节不改。

实现对原目标执行 copy-on-write：有偏置时复制后修改；关闭、模型不匹配、幅度为零或窗口外直接返回原目标。多次调用不会累加角度。关节索引从 `env.joint_names` 查找，不能换成模型关节顺序中的固定下标。

配置实例化时要求：

- `policy_name` 非空。
- `hip_pitch_deg` 是有限数，范围为 `[0, 0.5]` 度。本次不开放负偏置或更大幅度。
- `0 <= ramp_in_start < ramp_in_end < ramp_out_start < ramp_out_end`。

参数应修改配置源文件后重新启动进程，不提供运行中的参数热更新。

## 6. 查询帧与现场时间的对应

候选帧来自先前对 `g1_run_261104-112131.POFsKX` 的检查。该记录使用的策略名称为 `eight_cut_1_v034_1.2x`，第一处明显腿部反向目标脉冲位于查询帧 1169–1176。本次候选窗口从 1100 帧提前介入，覆盖后续举臂，再于 1350 帧退出。

参考 NPZ 为 50 FPS；按参考时间换算，1100、1140、1310、1350 帧分别为 22.0、22.8、26.2、27.0 秒。这不是录像播放时间或日志实际经过时间。先前记录的策略平均执行频率约为 40.29 Hz，而配置为 50 Hz，因此不能直接用录像秒数乘 50 作为现场触发帧。

本机返回的数据目录中没有用于独立同步的现场录像，录像第22秒与日志查询帧的对应尚未确认。现场 Agent 应使用同次 run 的视频和 `motion.msgpack` 中查询帧对齐。文件名中的 `1.2x` 不意味着这里还要再次将帧号乘以 1.2。

## 7. 当前策略名称与启用步骤

当前本地 `g1_native_loco_mimic_23dof_backpack.mimic_policies` 仍然选择 `eight_cut_1_backpack`，与补偿指定的 `eight_cut_1_v034_1.2x` 不匹配。这是保留已有模型选择的结果，本次没有偷偷替换 ONNX 或播放长度。

现场启用时按以下顺序处理：

1. 使用现场已经部署过、待调整的 v034 原策略配置，确认其实际 `policy_name` 和 `policy_file`。沿用该策略原本的 `max_timestep`，不要直接套用本地另一个模型的播放长度。
2. 在现有背包配置的 `fixed_hip_bias.policy_name` 填入该策略的准确名称。名称采用精确匹配，不按文件名前缀或相近版本自动匹配。名称本身不验证权重内容，现场仍需确认没有用错模型文件。
3. 对齐录像和查询帧后填写四个过渡帧参数。
4. 首次试验保持 +0.5°候选幅度，将 `enabled` 改为 True，重启现有进程。现场试验由用户决定和操作，不由 Agent 自行启动。
5. 使用现场原来的启动命令、网络接口和配置名，无需增加新启动脚本或重新编译 native bridge。不要假设 `start.sh` 会转发新参数；本次没有修改其参数处理。
6. 观察第一次举臂时额外踏步是否减少，以及原有动作是否仍然连贯。第二处保持原样，供后续单独判断。

用户原本的协作方式继续有效：用户指出具体问题，Agent 离线提出固定调整，用户决定是否试验。不要将本次代码扩展为基于传感器反馈的自动补偿器。

## 8. 生命周期与日志

- 预热 `prepare()`、首次控制权接管 `_enter_mimic()` 的目标生成不加入偏置；仅后续 `_step_mimic()` 调用补偿。首次试验应从窗口之前开始播放，不把 `start_timestep` 直接设在补偿平台中间。
- 暂停或健康策略冻结参考时，查询帧不变，补偿也保持同值，不因为 `play_speed=0` 突然撤销。
- 重播由现有策略 reset 恢复查询帧；补偿本身没有累积状态。
- HANDBACK、原生步态、停止和错误处理沿用现有逻辑；bridge 保持最后目标的行为不改。
- 没有修改 ONNX 输入构造、`last_action`、参考命令或动作滤波。实际运动变化仍会通过正常传感器反馈进入后续策略观测。
- 没有修改 PD 增益、动作限幅、电机索引映射、控制频率或原生 FSM 切换。

成功进入 Mimic 时调用一次 `trace_event("fixed_hip_bias", ...)`，包含本次参数、两侧关节名称及 `policy_matches`。只有现有 motion trace 开启时才会写入；没有新增逐帧控制台输出。它用于记录当次配置，不能仅凭此事件认定目标已经变化，还要看 `enabled`、名称匹配和帧段。

原有 `policy_target` 仍是模型关节顺序的原始策略目标；`final_target` 是环境关节顺序的实际发送目标。对比时必须先按关节名称重排，再计算 `final_target - mapped_policy_target`。启用后，差值应只在两侧髋俯仰出现预定的小角度曲线。

## 9. 回退

把 `fixed_hip_bias.enabled=False`，重启现有进程，即恢复原发送目标；也可以把 `hip_pitch_deg` 设为 0。不要在正在控制机器人时突然热切换参数。

回退补偿不需要回退模型、动作文件或 native bridge，也不要覆盖任务开始前已有的两处 v2 配置修改。

## 10. 验证方法与结果

新增的单项离线检查使用生产 `_step_mimic()` 和 `_apply_fixed_hip_bias()`，以 mock 替代策略推理和发送函数，不构造环境、SDK 或机器人控制器。检查覆盖：

- 默认关闭，原模型选择保持不变。
- 环境关节顺序刻意反转后仍只改变两侧髋俯仰。
- 接入中点为 0.25°，平台为 0.5°，退出中点为 0.25°，窗口外为零。
- 使用与 ONNX 一致的整数查询帧；1120.9 按 1120 处理。
- 暂停帧保持、重复调用不累加、重播回到零帧后恢复原目标。
- 策略名称不匹配、关闭、角度为零时直通。
- 原始目标和 `last_action` 未被修改，手部目标及遥控命令继续原样传递。
- 非法幅度、非有限数、错误帧序和空名称在配置构建时被拒绝。

在已有项目依赖的环境中，仓库根目录执行：

```bash
python tests/test_g1_native_integration.py TestNativeIntegration.test_fixed_hip_bias_at_send_boundary
```

本机 `.venv/bin/python` 缺少 torch，首次调用在导入阶段退出，没有运行到测试。为避免修改项目依赖，改用 uv 临时环境运行同一项测试：

```bash
uv run --no-project --with-requirements requirements.txt python tests/test_g1_native_integration.py TestNativeIntegration.test_fixed_hip_bias_at_send_boundary
```

最终结果：上述 uv 命令运行成功，`Ran 1 test in 0.007s`、`OK`。`git diff --check` 通过。没有运行全仓测试、仿真、训练、真机试验、提交或推送。

离线检查的结论范围是补偿配置与发送链路正确，不包含真机减少踏步的效果结论。

## 11. 接手 Agent 阅读顺序

先读本文，再看 `FixedHipBiasCfg`、`_apply_fixed_hip_bias()`、`_step_mimic()` 及新增测试。更完整的采集与人类决策边界见同目录 `G1_REAL_MOTION_HANDOFF.md`。

本机已有离线分析材料位于 `/Users/chengyuxuan/Downloads/postprocess/inspection_20261106/`，原始返回记录位于 `/Users/chengyuxuan/Downloads/postprocess/logs/g1_run_261104-112131.POFsKX/`。这些是本机路径，不假设服务器存在；需要时由用户或同事提供同次 run 数据。

下一步只需处理策略身份、录像帧同步和人工决定的首次对照试验，不要重新启动 PKL 修复、重新训练、残差优化或全链路改造。
