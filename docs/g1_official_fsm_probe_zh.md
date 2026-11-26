# G1 官方运控快慢切换测试

这份工具用于确认模式切换能否复现 PCIe AER、NVMe 掉盘或 SSH 失联。各组复用现场 Bridge 的状态读回、重试和跨进程控制锁，均不加载动作模型。

| 组别 | 测试节奏 | 轮数 |
| --- | --- | ---: |
| `fast` | 常规 500 → 走跑 801 → 常规 500。读到有效静止状态就请求 801。确认 801 后，连续站稳 1 秒才请求 500，最多等待 8 秒。 | 5 |
| `paced` | 常规 500 停留至少 5 秒 → 走跑 801 停留至少 5 秒 → 常规 500。站稳检查同时进行，仍沿用 8 秒上限。 | 5 |
| `debug` | 常规 500 → 走跑 801 → 调试 1000 → 走跑 801 → 常规 500。调试接管确认后约 2 秒将腿和腰过渡到配置默认站姿，双臂保持接管时的实测姿态。保持该关节目标 5 秒，再验证实测站姿并交还。 | 1 |

每组结束后在常规 500 观察 150 秒。先执行 `fast`。确认首组通过、监控链路正常，再在同一开机周期执行 `paced`。快组和慢组的关节指令发布计数必须为零。

## 现场准备

1. 用现场清单核对目标机器人与机型，准备对应主机 `/etc/machine-id` 的 SHA-256。计算时使用去掉末尾换行的值。用事先核实的指纹作为预期值。
2. 现场操作员确认机器人双脚落地、在常规运控站稳、遥控器可用于接管。暂停 Agent 表演、导航和手柄动作。
3. 按该台机器的功耗配置恢复 MAXN，并读回确认。检查 NVMe 为 `live`，根分区可写，APST 开启且延迟容忍为 `100000` 微秒，根端口与 SSD 端实际读回 ASPM 关闭。本工具采样 `nvme0` 与 PCIe 根端口 `0004:00:00.0`。监控机继续保存内核、SMART 和 SSH 连通性记录。
4. 脚本复用现场正在使用的 23DoF 部署及其 Python 环境。需要有跨进程互斥的 Bridge，修订号 `2026112301`。GitHub 提交用于取诊断脚本，既有部署和解释器用现场版本。

## 执行

将 `scripts/g1_official_fsm_probe.py` 复制到目标机的 `/dev/shm/`。下面是在目标机上执行的命令。事先将 `DEPLOY_ROOT`、`DEPLOY_PYTHON` 和 `EXPECTED_MACHINE_SHA256` 设成现场确认的值。

```bash
sudo -v &&
  mkdir -p /dev/shm/g1-official-fsm-probe &&
  cd /dev/shm/g1-official-fsm-probe &&
  PYTHONPATH="$DEPLOY_ROOT" "$DEPLOY_PYTHON" -Bu /dev/shm/g1_official_fsm_probe.py \
    --machine-id-sha256 "$EXPECTED_MACHINE_SHA256" --phase fast --execute
```

通过 SSH 把输出保存到跳板机或采集机。RoboJudo 初始化时如创建日志，也会落在上述内存目录。看到 `phase_passed`、`final_fsm` 为 500、`publish_count` 为 0，且外部监控无新增异常后，把同一命令的 `--phase fast` 改为 `--phase paced`。

出现切换失败、状态不符、新 AER、NVMe 非在线、SSH 中断或异常姿态，停止后续测试，保留外部日志。脚本在状态确认和请求前后检查存储状态。正在执行的 SDK 调用按照原 Bridge 逻辑完成或超时。现场操作员确认当前模式并处理恢复。

## 第三组现场操作

确认快组和慢组的结果、存储基线和监控连接。由现场操作员确认地面平整、双脚落地、常规模式下站稳、防摔措施就绪，并持手柄待命。调试期间使用固定站姿目标，机器人会收到关节位置指令，站姿没有舞蹈策略的动态平衡能力。

保持与前两组一致的部署和监控，用上面的命令将 `--phase fast` 改为 `--phase debug`，手动运行一次。计划观察 `debug_confirmed`（1000）、`debug_stand_hold_complete`（至少 5 秒）、`debug_handback_confirmed`（801、发布关闭），以及最后的 `phase_passed`（500）。交还 801 后，累计发布计数应保持不变。

进入 1000 前若腿或腰与默认站姿相差过大，或任一关节目标超限，脚本在走跑停止。`debug_stance_target` 记录站姿偏差和与配置默认值相差最大的关节。若已在 1000，出现 `debug_manual_takeover_required`、姿态异常或 SSH 失联时，现场立即用手柄接管或急停。进程仍在且由 Bridge 持有控制权时，Bridge 维持最后一个目标。等手柄切出 1000 再结束进程。

## 回收结果

保留各组原始输出和同期外部监控。记录每组完成轮数、开机 ID、首条内核报错、最后一次读回的 FSM 和发生时间。快慢任一组复现可说明问题在不发关节指令时也能发生。仅调试组复现时，对照首个错误与接管、站姿发布、交还各自的时间。各组都通过时，结论限于本次流程未复现。

离线用例 `python3 -B -m unittest discover -s tests -p 'test_g1_official_fsm_probe.py'`。
