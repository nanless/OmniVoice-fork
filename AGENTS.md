# AGENTS.md — OmniVoice 开发指南

## 项目概述

OmniVoice：基于离散扩散的大规模多语言零样本 TTS 系统。底层是 Qwen3-0.6B（双向注意力）+ 音频 Embedding/Head。支持 600+ 语言，voice cloning 和 voice design。

## 环境与依赖管理

- **Python**: >= 3.10
- **包管理**: `uv sync`（推荐）或 `pip install -e .`
- **PyTorch**: 2.8.0（`pyproject.toml` 中通过 `uv.lock` 锁定），Linux/Win 使用 CUDA 12.8 版本
- **国内镜像**: `uv sync --default-index "https://mirrors.aliyun.com/pypi/simple"`
- 如果 HuggingFace 下载慢: `export HF_ENDPOINT="https://hf-mirror.com"`

## 常用命令

### 推理

```bash
# 单条推理
omnivoice-infer --model k2-fsa/OmniVoice --text "你好" --output out.wav

# 声音克隆
omnivoice-infer --model k2-fsa/OmniVoice --text "你好" \
    --ref_audio ref.wav --ref_text "参考文本" --output out.wav

# Web UI
omnivoice-demo --ip 0.0.0.0 --port 8001
```

### Python API

```python
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
import torch
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
audio = model.generate(text="你好", ref_audio="ref.wav", ref_text="参考文本", language="zh")
# 返回 list of np.ndarray, 采样率 24000
```

### 训练

```bash
# 多卡训练
accelerate launch --gpu_ids "0,1" --num_processes 2 \
    -m omnivoice.cli.train \
    --train_config train_config.json \
    --data_config data_config.json \
    --output_dir exp/my_finetune
```

参考完整流程：`examples/run_finetune.sh`、`finetune_children.sh`

## 架构要点

### 目录结构

| 目录 | 职责 |
|------|------|
| `omnivoice/models/omnivoice.py` | **核心文件 (~1600 行)**，模型定义 + generate() + 推理全流程 |
| `omnivoice/cli/` | CLI 入口：train.py, infer.py, infer_batch.py, demo.py |
| `omnivoice/data/` | 数据流水线：dataset.py(WebDataset读取), processor.py(样本mask/tokenize), batching.py+pillator.py |
| `omnivoice/training/` | 训练框架：config.py(TrainingConfig), builder.py(构建model/dataloader), trainer.py(OmniTrainer), checkpoint.py |
| `omnivoice/eval/` | 评估：WER, 说话人相似度(ECAPA+WavLM), UTMOS自然度 |
| `omnivoice/scripts/` | 预处理脚本：extract_audio_tokens.py(音频→RVQ token), jsonl_to_webdataset.py, denoise_audio.py |
| `omnivoice/utils/` | 工具函数：audio.py, text.py, duration.py, lang_map.py, voice_design.py |

### 模型架构

- **基座**: Qwen3-0.6B（28层, hidden=1536, 12头，**双向注意力**）
- **新增模块**:
  - `audio_embeddings`: nn.Embedding(8×1025, 1536) — 8层 codebook 共 享一个表，通过 `codebook_layer_offsets` 区分
  - `audio_heads`: nn.Linear(1536, 8×1025) — 从 hidden state 预测 8 层 audio token
- **音频表示**: Higgs Audio V2 Tokenizer, 8 层 RVQ, 24kHz, 75Hz 帧率
- **参数量**: ~0.625B（基座 0.6B + 新增 ~25M）

### 训练机制（自监督，无 speaker_id）

- 训练时从同一段音频截取前缀 (0%~30%) 作为 prompt，其余作为预测目标
- 目标区域随机 mask，模型预测被 mask 的 token
- `drop_cond_ratio=0.1`: 10% 样本丢弃所有条件（用于 CFG 训练）
- Loss: 8 层分别计算交叉熵，加权求和 `[8,8,6,6,4,4,2,2]`
- 数据格式：WebDataset tar shards（含 .npy audio token）+ 对应 jsonl label

### 推理机制

- 迭代去掩码解码（默认 32 步），每步 reveal 置信度最高的 K 个位置
- CFG：同时跑 conditional + unconditional，`final = cond + guidance_scale * (cond - uncond)`
- Layer Penalty（默认 5.0）：让低层（Layer 0-1）先 reveal
- 长文本自动分块（> 30 秒触发）

### Batching 策略

| attn_implementation | 策略 | 要求 |
|---------------------|------|------|
| `flex_attention`（默认） | Sequence Packing: 多样本拼接成长序列 | PyTorch >= 2.5, Ampere+ GPU |
| `sdpa` | Length-grouped Padding: 相近长度样本 padding 到 batch | 无特殊要求 |

## 关键注意事项

### `from_pretrained` 的 `train` 参数
- `train=True`: **不加载** audio_tokenizer/text_tokenizer（训练不需要）
- `train=False`（默认）: 加载所有推理组件，包括 audio_tokenizer（从模型目录的 `audio_tokenizer/` 子目录或自动下载 `eustlb/higgs-audio-v2-tokenizer`）

### Checkpoint 结构
模型 checkpoint 目录必须包含（或通过 symlink 指向）`audio_tokenizer/` 子目录。`finetune_children.sh` 中会自动创建 symlink。

### 训练数据准备流程
1. 原始音频 → JSONL: `{"id": "...", "audio_path": "...", "text": "...", "language_id": "zh"}`
2. JSONL → WebDataset: `python -m omnivoice.scripts.extract_audio_tokens --input_jsonl ...`
3. data.lst 格式: `<tar路径> <jsonl路径> <样本数> <总时长(秒)>`
4. 训练: `accelerate launch -m omnivoice.cli.train`

### Special Tokens
训练时自动注入的 special tokens（`builder.py:73-84`）:
`<|denoise|>`, `<|lang_start|>`, `<|lang_end|>`, `<|instruct_start|>`, `<|instruct_end|>`, `<|text_start|>`, `<|text_end|>`

### 显存调优
- `batch_tokens=8192`（默认），显存不足时减半（4096、2048）
- `flex_attention` 比 `sdpa` 更省显存
- 老 GPU（V100、RTX 2080Ti）需 `"attn_implementation": "sdpa"`，同时设置 `max_sample_tokens`, `min_sample_tokens`, `max_batch_size`

### 推理时文本拼接细节
`ref_text` 和 `text` 会被拼接成一个整体送入模型：`"<|text_start|>ref_text text<|text_end|>"`。这有助于模型理解参考音频的语速和语调。`ref_text` 可选（不提供时 Whisper 自动转写），但推荐提供。

### 非代码文件
- `docs/` 有中英文双语版本（`docs/zh/` 为中文版）
- `README_技术详解.md` 和 `HOW_IT_WORKS.md` 是中文技术文档
- `OmniVoice.ipynb` 是 Google Colab 示例 notebook

## 不要做的事

- **不要**在代码中假定 `speaker_id` 字段 — 训练是自监督的，不需要
- **不要**假定音频采样率 — 始终使用 `model.sampling_rate` (24000)
- **不要**忽略 `train` 参数 — 训练和推理模式的 `from_pretrained` 加载完全不同
- **不要**在 CUDA 上使用 MPS device_map — audio tokenizer 不支持 MPS
