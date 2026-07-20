# 确定性 CER v4

生产路径固定为：`完整克隆音频 → Qwen3-ASR → ref/hyp 各自规则 TN → 字级 CER`。这里不调用 LLM，不读取 LLM endpoint/cache，也不会同时查看 ref 和 hyp 后挑更低的分数。

## 指标与缓存契约

- `cer_metric`: `deterministic_char_cer`
- `eval_schema_version`: `4`
- `cer_score_version`: `4`
- `normalization_profile`: `safe`
- `normalization_version`: `4`
- 每条 `.eval.json` 绑定 TN 源码指纹、参考文本及其语言上下文、ASR hypothesis、WAV 签名、ASR 模型文件指纹和 decode 配置指纹。
- canonical 字段只有 `cer`、`ref_normalized`、`hypo_normalized`、编辑次数和 `reference_chars`；旧 `manual_cer`、`llm_cer` 不参与报告和筛选。

任一 WAV、clone metadata、参考语言、TN 规则、ASR 模型或 decode 配置变化，旧 CER 都会失效。TN-only 变化时仍可复用来源与当前模型/decode/WAV 完全一致的原始 ASR hypothesis；强制中文解码得到的旧 hypothesis 不能冒充当前自动语言解码结果。

## TN v4 的顺序与边界

`cer_normalization.py` 提供 `strict` 和 `safe`。生产筛选使用 `safe`，顺序如下：

1. 只解码完整且带分号的 HTML entity；未知、非法或控制字符 entity 保留为 `HTML` 证据。
2. 统一 NFC、全半角、大小写、空白和不承载语义的标点。
3. 仅 reference 展开白名单 speech tag：显式 `language` 优先，其次 `lang_type`；旧 sidecar 两者都缺失时，仅按 CJK/拉丁字符做确定性回退。判定值和来源都写入 normalization context。hypothesis 从不读取 reference 或其 metadata，未知方括号 tag 保留为 `TAG` 证据。
4. 按明确上下文依次处理日期、时间、百分比、金额、数量/单位、分数、序数、号码和普通数字。
5. 用类型 token 保留语义：`NUM`、`DIGITS`、`ORDINAL`、`FRACTION`、`PERCENT`、`DATE`、`TIME`、`MONEY`、`QTY`、`TAG`、`HTML`。

已覆盖的确定性形式包括中文万/亿大数、合法千分位、英文 cardinal/decimal、显式百分比和单位、合法日期、带明确格式/时段的时间、中英序数、阿拉伯/中文/Unicode/英文分数，以及带“号码/电话/区号/phone number/area code”标签的逐位数字。

这些差异不会被抹掉：

- `80` vs `80%`，`12` vs `12th`，`1/2` vs `50%`；
- `0012` vs `12`，`one two` vs `twelve`；
- filler、语气词、儿化、重复/自修正、同音词、人名、contraction；
- `03/04`、`三点一四`、`有一点希望`、`给我两点建议`、比分/ratio/version、非法时间/序数等歧义或畸形写法。

## 自检

```bash
python eval_cer/check_cer_normalization.py
python eval_cer/check_cer_contract.py
```

第一个是规则与反例规范；第二个验证语言 metadata 变化会让 CER 失效，且旧 ASR decode 不能被复用。

对一个 clone 根目录做只读全量 TN 审计（检查 metadata 完整性、空输出、幂等性和 token 分布）：

```bash
python eval_cer/audit_cer_normalization_corpus.py --root /path/to/clones
```

## 全量评测

```bash
python eval_cer/eval_cloned.py \
  --out-dir /path/to/raw_clone_round \
  --batch-size 16 --skip-existing
```

常用参数：

- `--skip-asr`：只使用模型、decode 和 WAV 签名均有效的 ASR cache/sidecar hypothesis。
- `--refresh-asr-cache`：不复用缓存，全量重跑 ASR。
- `--refresh-cer`：复用有效 ASR hypothesis，只按当前 TN 重算 CER。
- `--skip-existing`：跳过当前有效的 v4 sidecar，但仍全量重建 canonical 报告。
- `--allow-partial`：仅在明确接受部分 clone inventory 时使用；生产筛选不要开启。

固定样本：

```bash
python eval_cer/eval_batch_200.py --sample-size 200
```

输出：

- `text_*.eval.json`：权威逐条 sidecar；
- `eval_cer_details.jsonl`：filter/prune 的 canonical 输入；
- `eval_summary.json`、`eval_details.txt`：全量汇总和人工检查报告；
- `eval_asr_cache.json` + `.meta.json`：绑定音频、模型和 decode 配置的 ASR cache。

## 从 v2/v3 安全迁移

迁移只复用模型、decode、WAV 和原始文本身份都能证明仍然有效的 `asr_hypo`；旧 normalized/LLM 字段全部忽略。缺少 decode 指纹的旧结果会拒绝迁移，需要重新跑 ASR。

```bash
# 默认只预览
python eval_cer/migrate_cer_to_v4.py --out-dir /path/to/clones

# 确认后原子写入
python eval_cer/migrate_cer_to_v4.py --out-dir /path/to/clones --write
```

旧入口 `migrate_cer_v2_to_v3.py` 只保留为兼容 wrapper，实际同样生成 v4。

遗留 `compare_itn_batch_sizes.py` 和 `analyze_1000_samples.py` 已 fail-fast 禁用，不再发起 LLM ITN 或读取其评分作为生产依据。
