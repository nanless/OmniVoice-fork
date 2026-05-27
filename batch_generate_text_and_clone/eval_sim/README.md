# 克隆语音说话人相似度评测（eval_sim）

用 **samresnet100**（voxblink2 说话人验证模型）计算 `voice_clone` 克隆音与 sidecar 中 `ref_audio` 原音的余弦相似度，归一化到 **[0, 1]**，衡量音色保留程度。

> 文本正确性请用 **[eval_cer](../eval_cer/README.md)**，自然度请用 **[eval_mos](../eval_mos/README.md)**；本目录只评说话人相似度。

**自包含**：模型结构、权重、`eval_clone_similarity.py` 均在 `eval_sim/` 内，**运行时不需要安装 wespeaker**。

## 目录

```
eval_sim/
├── README.md                    # 本文档
├── run_eval.sh                  # omnivoice 环境一键运行
├── eval_clone_similarity.py     # 主脚本：扫描克隆库、成对算相似度
├── speaker_encoder.py           # 模型加载 + embedding 提取
├── speaker_similarity.py          # embedding 磁盘缓存 + 相似度
├── verify_parity.py             # 与 wespeaker 对齐校验（开发用）
├── models/
│   └── samresnet.py             # SimAM_ResNet100 + ASP（内嵌实现）
└── model/
    ├── config.yaml              # 模型配置
    └── avg_model.pt             # samresnet100 权重（需存在）
```

## 环境

```bash
conda activate omnivoice
cd batch_generate_text_and_clone/eval_sim
```

| 依赖 | 说明 |
|------|------|
| torch / torchaudio | 推理、fbank |
| numpy / pyyaml / tqdm | 数据处理 |

## 模型权重

首次使用前确认 `model/avg_model.pt` 存在：

```bash
# 从已有 checkpoint 复制或软链
ln -sf /path/to/voxblink2_samresnet100/avg_model.pt model/avg_model.pt
```

默认 checkpoint 来源：`voxblink2_samresnet100`（与 `wespeaker/examples/extract_and_conclude_similarities/v2_organized` 相同）。

## 快速开始

```bash
# 全库
bash run_eval.sh

# 随机 200 条（快速试跑）
GPU=0 SAMPLE_SIZE=200 bash run_eval.sh

# 直接 python
python eval_clone_similarity.py --gpu 0 --sample-size 200 --seed 42
```

环境变量（可选）：

| 变量 | 默认 | 说明 |
|------|------|------|
| `GPU` | 0 | CUDA 设备 |
| `CLONED_VOICES_ROOT` | `batch_cloned_voices/` | 克隆输出根目录 |
| `SAMPLE_SIZE` | 全量 | `run_eval.sh` 抽样大小 |

## 评测逻辑

对每条 `text_*.json`（`status=generated`）：

1. 读 `ref_audio`（原参考 wav）
2. 读 `text_*.wav`（克隆 wav，16 kHz）
3. 分别提 256 维 speaker embedding
4. 算余弦相似度 → [0, 1]

同一 `ref_audio` 对应 10 条克隆时，embedding 缓存会复用 ref 向量，加速全库评测。

## 打分方式（与 wespeaker v2_organized 一致）

预处理必须与 wespeaker CLI 完全一致（已通过 `verify_parity.py` 验证）：

```
torchaudio.load(..., normalize=False)   # int16，勿用默认 float 归一化
  → pcm.to(float)
  → Resample(16 kHz)
  → Kaldi fbank × 2^15
  → CMN（逐 utterance 减均值）
  → SimAM_ResNet100_ASP
cosine = dot(e1, e2) / (||e1|| * ||e2||)
similarity = (cosine + 1) / 2
```

### 对齐校验

开发机若已安装 wespeaker，可用 3dspeaker 环境跑：

```bash
/root/miniforge3/envs/3dspeaker/bin/python verify_parity.py
# 8 对样本，embedding / similarity 差异应为 0
```

## CLI（`eval_clone_similarity.py`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--out-dir` | `batch_cloned_voices/` | voice_clone 输出根目录 |
| `--model-dir` | `eval_sim/model/` | checkpoint 目录 |
| `--gpu` | 0 | CUDA 设备 |
| `--sample-size` | 全量 | 随机子集大小 |
| `--seed` | 42 | 抽样 seed |
| `--cache-dir` | `<out-dir>/eval_sim_embedding_cache` | embedding 缓存 |
| `--skip-cache` | off | 清空并重建缓存 |
| `--no-sidecar` | off | 不写 `text_*.sim.json` |

## 输出

| 文件 | 说明 |
|------|------|
| `batch_cloned_voices/eval_sim_summary.json` | overall + 分数据集 mean/p50/p10/p90 |
| `batch_cloned_voices/eval_sim_details.jsonl` | 逐条 JSONL |
| `{说话人}/text_XXX.sim.json` | 每条 sidecar 结果 |
| `eval_sim_embedding_cache/` | ref/clone embedding 磁盘缓存 |

抽样时文件名带后缀，如 `eval_sim_summary_200.json`。

### `.sim.json` 示例

```json
{
  "cloned_audio": ".../text_004.wav",
  "ref_audio": ".../original.wav",
  "similarity": 0.823,
  "dataset": "BAAI-ChildMandarin41.25H_...",
  "gen_text": "...",
  "ref_text": "...",
  "speed": 0.94,
  "model_dir": ".../eval_sim/model",
  "evaluated_at": "2026-05-27T15:25:03"
}
```

## 输入约定

sidecar（`voice_clone` 写入的 `text_*.json`）需含：

- `status`: `"generated"`
- `ref_audio`: 原 wav 绝对路径
- `cloned_audio` 或同目录 `text_*.wav`

原音任意采样率均可；克隆 wav 已为 16 kHz。

## 与 eval_cer / eval_mos 的区别

| | eval_sim | eval_cer | eval_mos |
|---|----------|----------|----------|
| 对比对象 | 克隆 wav vs **原 ref wav** | ASR vs **gen_text** | 克隆 wav |
| 指标 | similarity ∈ [0,1] | CER | UTMOS ∈ [1,5] |
| 含义 | 音色像不像 | 内容对不对 | 听感自然度 |

## 常见问题

**相似度偏低但 CER 很好？**  
克隆读了正确内容，但音色偏离参考说话人——需调 OmniVoice 克隆参数或 checkpoint。

**换权重后旧缓存不准？**  
`python eval_clone_similarity.py --skip-cache`

**能否评不同 checkpoint 的克隆？**  
可以；sidecar 里 `model` 字段会写入报告，按 `--out-dir` 扫描即可。

**为何不用 wespeaker 包？**  
为在 omnivoice 环境闭环运行；模型结构与 v2_organized 对齐，校验脚本可证一致。
