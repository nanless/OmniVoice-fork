# 多轮声纹/CER 漏斗补量设计

## 目标与权威口径

每个 canonical speaker 的达标时长固定为 1,800 秒，且只累计两类音频：合并数据集
`audio/<dataset>/<speaker>/` 中的原音频，以及已经通过完整质量漏斗并复制到 accepted root 的
复刻音频。raw round 不直接计入目标，避免失败样本、重复副本或中断产物被误计。每轮质量契约为：
全量 raw cosine SIM，严格选择 `SIM > 0.8`，只对该选择域运行确定性 CER，再严格选择
`CER < 0.1`。CER canonical key 集必须与 SIM-pass 清单完全相等。

round 完成后先原子复制最终通过项到 accepted root，再删除本轮未通过项的 WAV 和指标 sidecar。
主 clone JSON 不删除，而是改写为 `status=rejected` 并记录原 WAV 签名、拒绝阶段、阈值和选择清单
指纹，使删除可审计且后续 inventory 不会把它当作有效 generated 音频。已通过项在 raw round 中保留，
但后续规划只读取 accepted root，因此不会重复计时。复制与删除必须可重复执行；目标已存在时只有内容一致
才允许跳过，任何冲突都失败关闭。

## 轮次数据流与效率

控制器按 `plan → clone → SIM → SIM筛选 → scoped CER → 联合筛选 → publish/prune → target check`
串行执行。每一阶段完成后都检查带签名的 manifest；失败时停止当前轮，重复启动则复用有效 sidecar。
下一轮 planner 重新读取原音频和 accepted root 的真实 WAV header，因此实际生成时长、筛选损耗和轻微超调
都会反映到新缺口中。

为避免按约 2% 通过率机械运行大量小轮次，planner 支持有上限的 generation multiplier：接受目标仍为
1,800 秒，但单轮 raw 生成预算可取当前缺口的若干倍。默认采用 4 倍，兼顾轮数和单轮风险；每轮完成后
仍按真实 accepted 时长校正，绝不把预估通过率当成已接受时长。上一轮 SIM aggregate 仅用于对原始参考
音频排序：优先使用产生过 `SIM > 0.8` 结果的 reference，并让文本遍历随 round ID 改变，减少重复的
reference/text 组合。历史 SIM 只影响效率，不影响质量门槛或时长正确性。

## 安全、失败恢复与验收

删除前必须验证最终清单是当前 schema-v3 inventory 的子集、路径全部位于精确 raw root、无重复、manifest
为 complete，且 scoped CER 与 SIM-pass 指纹一致。执行顺序固定为“复制全部通过项并核验 → 发布操作清单 →
删除 rejects”；复制失败时不允许删除。删除目标在 manifest 中逐条记录，不能通过 glob 或未解析变量确定。

每轮验收包括：accepted copy 数等于最终清单数；raw 中 rejects WAV 为零；拒绝 JSON 均为 rejected；
每个 accepted 文件能映射到唯一 canonical speaker；target checker 不重复计时；最终 1,646 个 speaker 的
`original + accepted >= 1800s`。控制器设置最大轮数与“本轮 accepted 增量必须大于零”的停机条件，防止
质量门槛下无法补齐时无限消耗。所有长任务运行在私有 tmux socket 的独立窗口中，日志按 round 隔离。
