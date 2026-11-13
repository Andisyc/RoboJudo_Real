# Cut1 leftsmall i80 联播测试版（2026-11-13）

按用户指定，将 g1loco 联播配置的 Cut1 从 scratch v2 i15000 换成既有 leftsmall i80。该文件是 11月12日离线实验的原始 i80，SHA256 `5558dfe97ecf0e0b7057793fba4cfc08493b3d089690d930054783b992054ebd`，未重新导出、修改或叠加后处理。不是 11月13日两轮 guard 续训候选；那些候选未采用。

## 变更范围

- 新增 `assets/models/g1/beyondmimic/cut1_leftsmall_i80_20261112.onnx`，保留旧文件便于回滚。
- 修改 `g1_baduanjin_23dof_suite` 的索引0及模型校验清单。继承它的 `g1_baduanjin_23dof_select` 也使用该新Cut1，但单段选择语义不变；其他独立入口不改。
- 更新测试prompt和当前版本说明。
- Cut2 v010、Cut3 jointv1 pitch8/rightless 文件和配置均不变。controller、bridge、PD、输入设置、3秒启动/段间过渡、自动从Cut1开始以及500交还不变。

Cut1 i80 相对于原发布 Cut1 包含此前局部参考动作修改和小幅续训，变化已经封装在ONNX内；不应描述为“只改参考表”或“与原发布模型参考动作完全相同”。它与本次上传前离线i80字节完全一致。保持原130维输入、23维输出、8627帧，模型内6个平移相关输入列仍为零权重。Cut2仍使用原UNITREE平移信息。

## 离线结果与限制

已核对模型SHA256、130/23接口、参考长度、元数据/PD/关节顺序、入口与末帧参考、三段插值兼容性、首次与再次HTTP启动、单段选择和交还回归。86个Python回归及59个subtests通过（HTTP、联播、插值、交还和23DoF契约）；没有执行机器人。

i80 的完整离线仿真已完成两档摩擦对比。μ=0.65两次左跨实际离地峰值约2.10/3.57毫米；μ=1.5约0.45/1.42毫米。第一次左跨及高摩擦抬脚仍是未解决项，本次按用户选择交现场比较，不能写成已通过全部左跨优化验收。仿真摩擦系数不等于具体真实地面。

## 测试与回滚

使用仓库中的 `docs/baduanjin_suite_test_prompt_20261112_zh.md`（内容已更新为11月13日）。无需模型包。保留现场服务设置和已测通的bridge，不因更换模型重编。每次POST /enter_mimic自动从Cut1开始，不需要/select_policy。

回滚只恢复本次前的部署/配置，Cut1为 `cut1_scratch_v2_i15000_zero_translation`（SHA256 `2afc0f0cb58605b5abc7eb2c127148f9e5e0993e6dc4b54c8ff0540af155e53d`）；当前Cut2和Cut3不变。旧Cut3文档中的ref6回滚说明属于11月12日发布历史，不适用于本次。
