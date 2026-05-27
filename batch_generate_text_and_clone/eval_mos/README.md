# 克隆语音 UTMOS 评测（eval_mos）

用 **UTMOS22Strong**（OmniVoice 官方 TTS 评测 MOS 预测器）对 `voice_clone` 生成的克隆音打分，衡量自然度/质量（1–5 分制，越高越好）。

评分器与模型实现来自 **`omnivoice/eval/mos/utmos.py`**，与 ZipVoice / OmniVoice 论文评测一致。

> 文本正确性 → [eval_cer](../eval_cer/README.md)  
> 说话人相似度 → [eval_sim](../eval_sim/README.md)

## 目录

```
eval_mos/
├── README.md              # 本文档
├── run_eval.sh            # omnivoice 一键运行
├── eval_clone_mos.py      # 主脚本：扫描克隆库、逐条 UTMOS
└── utmos_scorer.py        # 封装 omnivoice UTMOS22Strong + load_eval_waveform
```

## 环境

```bash
conda activate omnivoice
cd batch_generate_text_and_clone/eval_mos
```

| 依赖 | 来源 |
|------|------|
| `UTMOS22Strong` | `omnivoice/eval/models/utmos.py`（通过 importlib 加载，避免触发 omnivoice 包初始化） |
| `load_eval_waveform` | `omnivoice/eval/utils.py` |
| torch / soundfile / torchaudio | omnivoice 环境 |

## 模型权重

默认读取（按优先级）：

1. `$TTS_EVAL_MODEL_DIR/mos/utmos22_strong_step7459_v1.pt`
2. 仓库根 `TTS_eval_models/mos/utmos22_strong_step7459_v1.pt`
3. 回退 `download/tts_eval_models/mos/...`

从 Hugging Face 下载（与 `examples/run_eval.sh` Stage 1 相同）：

```bash
cd /path/to/OmniVoice-fork
huggingface-cli download \
  --local-dir TTS_eval_models \
  k2-fsa/TTS_eval_models \
  mos/utmos22_strong_step7459_v1.pt
```

或下载完整评测模型包：

```bash
huggingface-cli download --local-dir download/tts_eval_models k2-fsa/TTS_eval_models
export TTS_EVAL_MODEL_DIR=/path/to/download/tts_eval_models
```

## 快速开始

```bash
# 全库
bash run_eval.sh

# 随机 200 条
GPU=0 SAMPLE_SIZE=200 bash run_eval.sh

# 直接 python
python eval_clone_mos.py --gpu 0 --sample-size 200
```

环境变量：

| 变量 | 说明 |
|------|------|
| `GPU` | CUDA 设备（默认 0） |
| `CLONED_VOICES_ROOT` | 克隆输出根目录 |
| `TTS_EVAL_MODEL_DIR` / `UTMOS_MODEL_DIR` | UTMOS 权重根目录 |

## 评测逻辑

对每条 `text_*.json`（`status=generated`）：

1. 读取对应 `text_*.wav`（16 kHz 克隆音）
2. `load_eval_waveform` → mono、重采样 16 kHz
3. `UTMOS22Strong` 前向 → utterance 级 MOS 预测

打分公式与 `omnivoice/eval/mos/utmos.py` 一致（`score_series.mean * 2 + 3`）。

## CLI（`eval_clone_mos.py`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--out-dir` | `batch_cloned_voices/` | voice_clone 输出根目录 |
| `--model-dir` | `TTS_eval_models/` | 评测模型根目录 |
| `--gpu` | 0 | CUDA 设备 |
| `--sample-size` | 全量 | 随机子集 |
| `--seed` | 42 | 抽样 seed |
| `--no-sidecar` | off | 不写 `text_*.mos.json` |

## 输出

| 文件 | 说明 |
|------|------|
| `batch_cloned_voices/eval_mos_summary.json` | overall + 分数据集 / 分 language 统计 |
| `batch_cloned_voices/eval_mos_details.jsonl` | 逐条明细 |
| `{说话人}/text_XXX.mos.json` | 每条 sidecar |

抽样时带后缀，如 `eval_mos_summary_200.json`。

### `.mos.json` 示例

```json
{
  "cloned_audio": ".../text_004.wav",
  "utmos": 3.82,
  "dataset": "BAAI-ChildMandarin41.25H_...",
  "language": "zh",
  "gen_text": "...",
  "speed": 0.94,
  "model_dir": ".../TTS_eval_models",
  "evaluated_at": "2026-05-27T16:00:00"
}
```

## 输入约定

- sidecar 需 `status=generated`
- wav 路径来自 `cloned_audio` 或同目录 `text_*.wav`
- 任意采样率均可（内部 resample 到 16 kHz）

## 三类评测对比

| | eval_mos | eval_cer | eval_sim |
|---|----------|----------|----------|
| 评什么 | 听感自然度 | 内容对不对 | 音色像不像 |
| 输入 | 克隆 wav | 克隆 wav + gen_text | 克隆 wav + ref_audio |
| 指标 | UTMOS ↑ | CER ↓ | similarity ↑ |
| 模型 | UTMOS22Strong | Qwen3-ASR + ITN | samresnet100 |

三类评测相互独立，可并行运行。

## 与 omnivoice 官方脚本的关系

| 本目录 | omnivoice |
|--------|-----------|
| `utmos_scorer.py` | `eval/models/utmos.py` + `eval/utils.load_eval_waveform` |
| `eval_clone_mos.py` | 适配 `batch_cloned_voices` 目录结构 |
| 官方批量入口 | `python -m omnivoice.eval.mos.utmos --wav-path ... --test-list ...` |

官方脚本需 JSONL test list 且 wav 按 `{id}.wav` 命名；本目录直接扫描 `text_*.wav` + sidecar，更适合克隆流水线。

## 常见问题

**报错 checkpoint not found？**  
按上文从 Hugging Face 下载 UTMOS 权重。

**UTMOS 多少算好？**  
非绝对阈值，建议在固定 test set 上与 baseline 对比；一般 >3.5 较可听，>4.0 较好（视域而定）。

**OOM？**  
单条克隆音通常较短；若极长音频可在 `load_eval_waveform` 加 `max_seconds`（需改 `utmos_scorer.py`）。

## 上游文档

- 总览：[../README.md](../README.md)
- 克隆：[../voice_clone/README.md](../voice_clone/README.md)
