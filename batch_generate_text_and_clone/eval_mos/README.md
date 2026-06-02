# 克隆语音质量评测（eval_mos）

用 **4 种指标** 对 `voice_clone` 生成的克隆音打分，衡量自然度 / 质量 / 可懂度。

| 指标 | 包 | 类型 | 说明 |
|------|----|----|------|
| **UTMOS22Strong** | `omnivoice/eval/models/utmos.py` | MOS 预测 (1–5) | OmniVoice 内置 UTMOS，与论文评测一致 |
| **SCOREQ** | `scoreq` (pip) | NR 质量分 | Microsoft 非侵入式语音质量评估 (ONNX) |
| **TTSDS2** | `ttsds` (pip) | 基准套件 | TTS 评测基准（智能度 / 韵律 / 说话人 / 环境） |
| **UTMOSv2** | `utmosv2` (git) | MOS 预测 (1–5) | UT-Sarulab MOS Prediction System v2 |

> 文本正确性 → [eval_cer](../eval_cer/README.md)  
> 说话人相似度 → [eval_sim](../eval_sim/README.md)

## 目录

```
eval_mos/
├── README.md              # 本文档
├── run_eval.sh            # 一键运行
├── eval_clone_mos.py      # 主脚本：扫描克隆库、多指标评测
├── scorers.py             # 统一 scorer 模块（4 种指标）
├── utmos_scorer.py        # 向后兼容封装（→ scorers.py）
└── eval_common.py         # 共享工具（上层目录）
```

## 环境

```bash
conda activate omnivoice
cd batch_generate_text_and_clone/eval_mos
```

| 依赖 | 安装方式 | 说明 |
|------|----------|------|
| UTMOS22Strong | 内置 | `omnivoice/eval/models/utmos.py` + `omnivoice/eval/utils.py` |
| SCOREQ | `pip install scoreq` | ONNX 推理，自动下载模型 |
| TTSDS2 | `pip install ttsds voicerestore pyworld allosaurus` | 依赖较多，详见下方 |
| UTMOSv2 | `pip install git+https://github.com/sarulab-speech/UTMOSv2.git` | 需从 GitHub 安装 |

### 安装依赖

```bash
# SCOREQ（推荐）
pip install scoreq

# TTSDS2（可选，依赖较多）
pip install ttsds voicerestore pyworld allosaurus

# UTMOSv2（可选，从 GitHub 安装）
pip install git+https://github.com/sarulab-speech/UTMOSv2.git
```

缺失的包会自动跳过，不会阻塞其他指标的运行。

## 模型权重

### UTMOS22Strong

默认读取（按优先级）：

1. `$TTS_EVAL_MODEL_DIR/mos/utmos22_strong_step7459_v1.pt`
2. 仓库根 `TTS_eval_models/mos/utmos22_strong_step7459_v1.pt`
3. 回退 `download/tts_eval_models/mos/...`

```bash
cd /path/to/OmniVoice-fork
huggingface-cli download \
  --local-dir TTS_eval_models \
  k2-fsa/TTS_eval_models \
  mos/utmos22_strong_step7459_v1.pt
```

### SCOREQ / TTSDS2 / UTMOSv2

首次运行时自动从 HuggingFace / PyPI 下载模型权重。

## 快速开始

```bash
# 运行所有可用指标
bash run_eval.sh

# 只运行指定指标
METRICS=UTMOS22Strong,SCOREQ bash run_eval.sh

# 随机 200 条
SAMPLE_SIZE=200 bash run_eval.sh

# 直接 python
python eval_clone_mos.py --metrics UTMOS22Strong,SCOREQ --gpus 0 --sample-size 200
```

环境变量：

| 变量 | 说明 |
|------|------|
| `GPU` | CUDA 设备（默认 0） |
| `CLONED_VOICES_ROOT` | 克隆输出根目录 |
| `TTS_EVAL_MODEL_DIR` | 评测模型根目录 |
| `METRICS` | 逗号分隔的指标列表（默认全部可用指标） |
| `SAMPLE_SIZE` | 随机子集大小 |

## 评测逻辑

对每条 `text_*.json`（`status=generated`）：

1. 读取对应 `text_*.wav`（16 kHz 克隆音）
2. 对每个已安装的指标运行评分
3. 结果写入 sidecar `.eval.json` + 汇总 JSONL

指标缺失时自动跳过，不影响其他指标。

## CLI（`eval_clone_mos.py`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--out-dir` | `batch_cloned_voices/` | voice_clone 输出根目录 |
| `--model-dir` | `TTS_eval_models/` | 评测模型根目录 |
| `--gpus` | 0 | CUDA 设备（逗号分隔） |
| `--metrics` | 全部可用 | 逗号分隔指标名 |
| `--sample-size` | 全量 | 随机子集 |
| `--seed` | 42 | 抽样 seed |
| `--no-sidecar` | off | 不写 `text_*.eval.json` |
| `--skip-existing` | off | 跳过已评测文件 |
| `--workers` | 4 | 进程数 |

## 输出

| 文件 | 说明 |
|------|------|
| `batch_cloned_voices/eval_summary.json` | overall + 分数据集 / 分 language 统计 |
| `batch_cloned_voices/eval_details.jsonl` | 逐条明细 |
| `{说话人}/text_XXX.mos.json` | 每条 sidecar |

抽样时带后缀，如 `eval_summary_200.json`。

### `.mos.json` 示例

```json
{
  "cloned_audio": ".../text_004.wav",
  "utmos22strong": 3.82,
  "scoreq": 3.45,
  "ttsds2": 0.78,
  "utmosv2": null,
  "dataset": "BAAI-ChildMandarin41.25H_...",
  "language": "zh",
  "gen_text": "...",
  "speed": 0.94,
  "evaluated_at": "2026-05-30T12:00:00"
}
```

### `eval_summary.json` 结构

```json
{
  "UTMOS22Strong": {
    "overall": {"count": 1000, "mean": 3.82, "p50": 3.91, ...},
    "by_dataset": {...},
    "by_language": {...},
    "failed_count": 0
  },
  "SCOREQ": {
    "overall": {"count": 1000, "mean": 3.45, ...},
    ...
  },
  "metrics": ["UTMOS22Strong", "SCOREQ", "TTSDS2", "UTMOSv2"],
  "items_done": 1000,
  "items_total": 1000
}
```

## 输入约定

- sidecar 需 `status=generated`
- wav 路径来自 `cloned_audio` 或同目录 `text_*.wav`
- 任意采样率均可（内部 resample 到 16 kHz）

## 四类评测对比

| | eval_mos | eval_cer | eval_sim |
|---|----------|----------|----------|
| 评什么 | 听感质量 (4 指标) | 内容对不对 | 音色像不像 |
| 输入 | 克隆 wav | 克隆 wav + gen_text | 克隆 wav + ref_audio |
| 指标 | UTMOS↑ SCOREQ↑ TTSDS2↑ UTMOSv2↑ | CER ↓ | similarity ↑ |

三类评测相互独立，可并行运行。

## 指标详情

### UTMOS22Strong
- 来源：omnivoice 内置（`omnivoice/eval/models/utmos.py`）
- 输出：MOS 预测分 (1–5)，越高越好
- 模型：Wav2Vec2 + Transformer，自定义 checkpoint
- 公式：`score_series.mean * 2 + 3`

### SCOREQ
- 来源：`pip install scoreq`（Microsoft）
- 输出：NR 质量分，越高越好
- 模型：ONNX 推理，自动下载
- 模式：Non-Reference（无需参考音频）

### TTSDS2
- 来源：`pip install ttsds`
- 输出：0–1 综合分，越高越好
- 子基准：智能度 / 韵律 / 说话人 / 环境
- 回退：包不可用时使用简化代理评分

### UTMOSv2
- 来源：`pip install git+https://github.com/sarulab-speech/UTMOSv2.git`
- 输出：MOS 预测分 (1–5)，越高越好
- 模型：UT-Sarulab MOS Prediction System v2（SSL + 频谱图双特征融合）

## 常见问题

**报错 checkpoint not found？**  
按上文从 Hugging Face 下载 UTMOS 权重。

**某个指标显示 Skipping？**  
说明对应 pip 包未安装。按上文安装后重运行。缺失指标不影响其他指标。

**UTMOS 多少算好？**  
非绝对阈值，建议在固定 test set 上与 baseline 对比；一般 >3.5 较可听，>4.0 较好（视域而定）。

**OOM？**  
单条克隆音通常较短；若极长音频可在 `load_eval_waveform` 加 `max_seconds`（需改 `scorers.py`）。

## 上游文档

- 总览：[../README.md](../README.md)
- 克隆：[../voice_clone/README.md](../voice_clone/README.md)
