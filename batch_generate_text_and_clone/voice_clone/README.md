# 儿童语音批量克隆（voice_clone）

用 OmniVoice 对儿童说话人验证（SV）数据集的参考音进行批量 voice cloning：每条参考 wav 随机抽 **10 句** LLM 生成文本，以随机语速合成克隆音，输出 16 kHz wav 及 sidecar JSON。

## 目录

```
voice_clone/
├── README.md              # 本文档
├── clone_dataset.py       # 主脚本：加载模型、分 worker、逐条克隆
└── run_clone_8workers.sh  # 双 GPU × 4 worker 生产启动脚本
```

## 环境

```bash
conda activate omnivoice
```

| 依赖 | 说明 |
|------|------|
| OmniVoice | `from omnivoice import OmniVoice` |
| PyTorch + CUDA | fp16 推理 |
| soundfile / torchaudio | 写 wav、24k→16k 重采样 |

## 输入

### 1. 文本池（JSONL）

默认读取：

```
batch_generated_text/llm_children_100k_asr_complete.jsonl
```

由 `text_generation/run_100k_asr_complete.py` 产出。每行至少含 `text`、`language` 字段。

### 2. 参考音频（4 个儿童 SV 数据集）

脚本内 `DATASETS` 常量（可按环境修改）：

| 数据集 | 说明 |
|--------|------|
| BAAI-ChildMandarin41.25H | 中文儿童 |
| Chinese_English_Scripted_Speech_Corpus_Children | 中英儿童 |
| King-ASR-EN-Kid | 英文儿童 |
| speechocean762 | 中文儿童（发音评测） |

每个数据集优先读 `kaldi_files/wav.scp` + `text`；若无 kaldi 目录则递归扫描 `.wav/.mp3/.flac`。

### 3. OmniVoice 模型

默认 checkpoint（脚本内 `MODEL_PATH`）：

```
exp/children_finetune_20260519_1418/checkpoints/checkpoint-62000
```

## 输出

根目录（默认 `batch_cloned_voices/`）：

```
batch_cloned_voices/
├── logs/                          # run_clone_8workers.sh 的 worker 日志
└── {数据集名}/{相对路径}/{utt_id}/
    ├── text_001.wav               # 克隆音频（16 kHz）
    ├── text_001.json              # sidecar
    ├── text_002.wav
    └── ...
```

### Sidecar 字段（`text_*.json`）

```json
{
  "status": "generated",
  "ref_audio": "/path/to/original.wav",
  "ref_text": "参考句 transcript",
  "gen_text": "LLM 生成、用于克隆的文本",
  "cloned_audio": "/path/to/text_001.wav",
  "speed": 1.12,
  "language": "zh",
  "model": "/path/to/checkpoint",
  "model_sr": 24000,
  "generated_at": "2026-05-27T15:20:03.933281"
}
```

- `eval_cer` 读 `gen_text` 作 CER 参考  
- `eval_sim` 读 `ref_audio` 与原音比相似度  
- `eval_mos` 对克隆 wav 打 UTMOS 自然度分  

## 克隆策略

| 参数 | 值 | 说明 |
|------|-----|------|
| `TEXTS_PER_AUDIO` | 10 | 每条参考音随机抽 10 句 |
| `SPEED_MIN/MAX` | 0.85–1.15 | 每句独立随机语速 |
| `SEED` | 42 | 数据集 shuffle、文本抽样可复现 |
| `OUTPUT_SR` | 16000 | 模型 24k 输出经 sinc 重采样 |
| `GEN_CONFIG` | num_step=32, guidance_scale=2.0, … | 见 `clone_dataset.py` |

文本抽样：`Random(f"{SEED}:{utt_id}").sample(all_texts, 10)`，同一参考音每次运行抽到相同 10 句。

语种映射：`language` 字段 → OmniVoice `zh`/`en`（`en_mostly`→`en`，混读→`zh`）。

## 快速开始

### 单 worker 调试

```bash
conda activate omnivoice
cd /root/code/github_repos/OmniVoice-fork

# 只看分配结果、不跑模型
python batch_generate_text_and_clone/voice_clone/clone_dataset.py \
  --gpu 0 --worker-id 0 --num-workers 1 --dry-run

# 实际克隆 2 条参考音
python batch_generate_text_and_clone/voice_clone/clone_dataset.py \
  --gpu 0 --worker-id 0 --num-workers 1 --limit 2
```

### 生产：8 worker（双卡）

```bash
bash batch_generate_text_and_clone/voice_clone/run_clone_8workers.sh
```

- GPU 0：worker 0–3  
- GPU 1：worker 4–7  
- 日志：`batch_cloned_voices/logs/gpu{0,1}_worker{N}.log`

监控：

```bash
tail -f batch_cloned_voices/logs/gpu0_worker0.log
nvidia-smi -l 1
```

## CLI 参数（`clone_dataset.py`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--gpu` | 0 | 逻辑 GPU（配合 `CUDA_VISIBLE_DEVICES`） |
| `--worker-id` | 0 | 当前 worker 编号 [0, num_workers) |
| `--num-workers` | 1 | 总 worker 数；参考音按 `i % num_workers` 分配 |
| `--limit` | 无 | 每个数据集最多处理 N 条（调试） |
| `--dry-run` | off | 只打印分配样本，不加载模型 |

## Worker 分配逻辑

1. 每个数据集内所有参考音 shuffle（seed=`{SEED}:shuffle:{ds_name}`）  
2. 按索引 `% num_workers` 分给各 worker，保证负载大致均衡  
3. 已存在 `text_*.wav` 的条目 **skip**（断点续跑）

## 生成配置修改

需改路径或超参时，编辑 `clone_dataset.py` 顶部常量：

```python
DATASETS = [...]           # 参考数据集根目录
TEXTS_PATH = "..."         # JSONL 路径
MODEL_PATH = "..."         # OmniVoice checkpoint
OUT_ROOT = Path("...")     # 克隆输出根目录
TEXTS_PER_AUDIO = 10
SPEED_MIN, SPEED_MAX = 0.85, 1.15
GEN_CONFIG = OmniVoiceGenerationConfig(...)
```

## 与上下游关系

```
text_generation  →  llm_children_100k_asr_complete.jsonl
       ↓
voice_clone      →  batch_cloned_voices/**/text_*.wav + json
       ↓
eval_cer         →  CER（内容）
eval_sim         →  说话人相似度（音色）
eval_mos         →  UTMOS（自然度）
```

## 常见问题

**OOM / 某 worker 报错？**  
查看对应 `logs/gpu*_worker*.log`；单 worker 重跑即可，skip 已有 wav。

**想换 checkpoint？**  
改 `MODEL_PATH`，sidecar 会记录实际使用的模型路径。

**输出采样率？**  
固定 16 kHz；与 eval_sim 的 samresnet100 输入一致（模型内部会 resample）。

**10 句不够 / 太多？**  
改 `TEXTS_PER_AUDIO`；注意已有输出不会自动重生成，需删目录或换 `OUT_ROOT`。
