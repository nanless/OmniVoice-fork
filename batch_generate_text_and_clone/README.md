# 批量文本生成与儿童语音克隆流水线

**LLM 生成儿童口语文本 → 参考音批量克隆 → CER / 相似度 / UTMOS 三类评测**

本目录是 OmniVoice 儿童语音克隆实验的端到端工具链：从文本池生成、多 GPU 批量克隆，到三类独立评测（内容 / 音色 / 自然度）。

## 目录结构

```
batch_generate_text_and_clone/
├── README.md                    # 本总览
├── text_generation/             # ① LLM 生成 10 万条儿童口语文本
├── voice_clone/                 # ② 多 GPU 批量克隆
├── eval_cer/                    # ③ 字级 CER 评测（ASR + ITN）
├── eval_sim/                    # ④ 说话人相似度评测（samresnet100）
└── eval_mos/                    # ⑤ UTMOS 自然度评测（omnivoice/eval）
```

各子目录均有独立 README，详见文末链接。

## 数据流

```mermaid
flowchart LR
  TG[text_generation] --> JSONL[(batch_generated_text/<br/>100k JSONL)]
  REF[(儿童 SV 数据集<br/>参考 wav)] --> VC[voice_clone]
  JSONL --> VC
  VC --> OUT[(batch_cloned_voices/<br/>text_*.wav + json)]
  OUT --> CER[eval_cer]
  OUT --> SIM[eval_sim]
  OUT --> MOS[eval_mos]
  CER --> RPT1[CER 报告]
  SIM --> RPT2[相似度报告]
  MOS --> RPT3[UTMOS 报告]
```

### 关键路径（默认）

| 阶段 | 输入 | 输出 |
|------|------|------|
| 文本生成 | LLM API | `batch_generated_text/llm_children_100k_asr_complete.jsonl` |
| 语音克隆 | JSONL + 4 个儿童 SV 数据集参考音 | `batch_cloned_voices/{数据集}/{说话人}/text_*.wav` |
| CER 评测 | 克隆 wav + sidecar `gen_text` | `batch_cloned_voices/eval_*` + `text_*.eval.json` |
| 相似度评测 | 克隆 wav + sidecar `ref_audio` | `batch_cloned_voices/eval_sim_*` + `text_*.sim.json` |
| UTMOS 评测 | 克隆 wav | `batch_cloned_voices/eval_mos_*` + `text_*.mos.json` |

## 环境一览

| 阶段 | Conda 环境 | 外部依赖 |
|------|-----------|----------|
| text_generation | 任意 Python 3.10+ | LLM API（`.env`） |
| voice_clone | **omnivoice** | OmniVoice checkpoint、GPU |
| eval_cer | **omnivoice** | Qwen3-ASR 本地模型、LLM ITN API（`.env`） |
| eval_sim | **omnivoice** | GPU；模型权重在 `eval_sim/model/` |
| eval_mos | **omnivoice** | GPU；UTMOS 权重在 `TTS_eval_models/mos/` |

**勿将 `.env` 中的 API Key 提交到 Git。**

## 快速开始（全流程）

### ① 生成文本

```bash
cd batch_generate_text_and_clone/text_generation
# 配置 .env：LLM_API_KEY、LLM_BASE_URL、LLM_MODEL
python run_100k_asr_complete.py
```

→ 详见 [text_generation/README.md](text_generation/README.md)

### ② 批量克隆

```bash
conda activate omnivoice
cd /root/code/github_repos/OmniVoice-fork

# 调试：单 worker、2 条参考音
python batch_generate_text_and_clone/voice_clone/clone_dataset.py \
  --gpu 0 --worker-id 0 --num-workers 1 --limit 2

# 生产：双 GPU × 4 worker = 8 并发
bash batch_generate_text_and_clone/voice_clone/run_clone_8workers.sh
```

→ 详见 [voice_clone/README.md](voice_clone/README.md)

### ③ CER 评测（内容正确性）

```bash
conda activate omnivoice
cd batch_generate_text_and_clone/eval_cer
# 配置 .env：ITN_LLM_API_KEY、ITN_LLM_BASE_URL、ITN_LLM_MODEL

python eval_batch_200.py      # 固定 200 条，日常迭代
python eval_cloned.py         # 全库 + 逐条 .eval.json
```

→ 详见 [eval_cer/README.md](eval_cer/README.md)

### ④ 说话人相似度（克隆 vs 原音）

```bash
conda activate omnivoice
cd batch_generate_text_and_clone/eval_sim
bash run_eval.sh
# 或: GPU=0 SAMPLE_SIZE=200 bash run_eval.sh
```

→ 详见 [eval_sim/README.md](eval_sim/README.md)

### ⑤ UTMOS 自然度（听感质量）

```bash
conda activate omnivoice
cd batch_generate_text_and_clone/eval_mos
bash run_eval.sh
# 或: GPU=0 SAMPLE_SIZE=200 bash run_eval.sh
```

→ 详见 [eval_mos/README.md](eval_mos/README.md)

### 全库三类评测（CER → SIM → MOS，tmux 后台）

```bash
bash batch_generate_text_and_clone/run_eval_all.sh   # 自动创建 tmux session eval_all
tmux attach -t eval_all                              # 查看进度
tail -f batch_cloned_voices/logs/eval_all/cer.log     # 或单独看某阶段日志
```

顺序：**CER → SIM → MOS**；`--skip-existing` 断点续跑；日志实时写入 `logs/eval_all/`。

## 选用哪个脚本？

### 克隆

| 场景 | 命令 |
|------|------|
| 本地调试 | `clone_dataset.py --limit N --num-workers 1` |
| 全量生产 | `run_clone_8workers.sh`（8 worker，断点续跑：已存在 wav 自动 skip） |

### 评测

| 需求 | 脚本 | 指标 |
|------|------|------|
| 固定 200/500 条、迭代 ITN | `eval_cer/eval_batch_200.py` | 字级 CER |
| 全库 CER、写 sidecar | `eval_cer/eval_cloned.py` | 字级 CER |
| 克隆音 vs 原音音色 | `eval_sim/eval_clone_similarity.py` | 余弦相似度 [0,1] |
| 克隆音自然度 | `eval_mos/eval_clone_mos.py` | UTMOS [1,5] |

CER、相似度、UTMOS 评测**相互独立**，可并行跑；共用同一批 `batch_cloned_voices/` 输出。

## Sidecar 约定（`text_*.json`）

克隆脚本为每条 wav 写 sidecar，供两类评测读取：

```json
{
  "status": "generated",
  "ref_audio": "/path/to/original.wav",
  "ref_text": "原参考句",
  "gen_text": "克隆用 LLM 文本（CER 参考）",
  "cloned_audio": "/path/to/text_001.wav",
  "speed": 1.12,
  "language": "zh",
  "model": "/path/to/checkpoint",
  "generated_at": "2026-05-27T15:20:03"
}
```

| 字段 | eval_cer | eval_sim | eval_mos |
|------|----------|----------|----------|
| `gen_text` | ASR 参考文本 | — | — |
| `ref_audio` | — | 原说话人 wav | — |
| `cloned_audio` / `text_*.wav` | ASR 输入 | 克隆 wav | 克隆 wav |
| `status=generated` | 仅评测成功克隆 | 同左 | 同左 |

## 常见问题

**克隆中断了怎么办？**  
重新启动 worker 即可；已生成的 `text_*.wav` 会被 skip，只补缺失项。

**CER、相似度、UTMOS 该看哪个？**  
- CER 低 → 克隆内容读对了  
- 相似度高 → 音色接近原说话人  
- UTMOS 高 → 听感更自然  
三者需结合业务需求判断。

**换机器 / 换路径？**  
各脚本顶部有路径常量（`DATASETS`、`MODEL_PATH`、`OUT_ROOT` 等），按环境修改；或通过环境变量（如 `CLONED_VOICES_ROOT`）覆盖部分默认值。

## 子模块文档

- **[text_generation/README.md](text_generation/README.md)** — LLM 生成、去重质检、JSONL 格式
- **[voice_clone/README.md](voice_clone/README.md)** — 多 GPU 克隆、worker 分配、输出结构
- **[eval_cer/README.md](eval_cer/README.md)** — ASR/ITN 流程、CER 缓存与报告
- **[eval_sim/README.md](eval_sim/README.md)** — samresnet100 相似度、与 wespeaker 对齐说明
- **[eval_mos/README.md](eval_mos/README.md)** — UTMOS 自然度（omnivoice/eval）
