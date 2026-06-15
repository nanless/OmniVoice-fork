# 成人口语文本生成（adult_text_generation）

参考 [higgs_audio_v3_text_generator](https://github.com/boson-ai/higgs-audio/tree/main/higgs_audio_v3_text_generator) 的模块化结构，为 OmniVoice 语音克隆生成**成人口语** JSONL（非儿童语料）。

## 目录

```
adult_text_generation/
├── README.md
├── generate_single.py          # 单 batch 试跑 prompt
├── run_batch_generation.py     # 通用批量入口（可断点续跑）
├── run_100k_adult.py           # 生产入口：10 万条成人 ASR 友好完整句
├── qa_generated_texts.py       # 生成后 QA 统计
└── omnivoice_adult_text_gen/
    ├── config.py               # GenConfig + 环境变量
    ├── scenarios.py            # 10 类成人场景（日常/商务/情感/旁白等）
    ├── prompt_builder.py       # 紧凑 prompt + 多样性轴
    ├── task_generator.py       # 分层采样任务列表
    ├── worker.py               # LLM 调用 + text_tn
    ├── quality_filter.py       # 成人质检（禁儿童叠词/网络梗）
    ├── tags.py                 # 复用 OmniVoice [laughter] 等标签定义
    ├── llm_client.py           # 复用 text_generation/llm_generate_texts.call_llm
    └── checkpoint.py           # 复用断点/task 状态/去重
```

## 与儿童版 `text_generation/` 的区别

| 维度 | 儿童版 | 成人版 |
|------|--------|--------|
| 说话人 | 3-12 岁 | 18-60 岁成人 |
| 场景 | 幼儿园/小学/课文朗读 | 日常/商务/客服/vlog/旁白等 |
| 句长 | 偏短（最多 ~40 字） | 更长（ultra_short ~ very_long，最长 ~500 字） |
| 语言特征 | 叠词、童言童语 | 职场/生活口语、自我修正 |
| 标签 | OmniVoice `[laughter]` 等 | 同一套 OmniVoice 标签 |
| 输出 | `llm_children_100k_asr_complete.jsonl` | `llm_adult_100k_asr_complete.jsonl` |

## 环境

与儿童版共用 `text_generation/.env` 或本目录 `.env`：

```bash
LLM_MODEL=qwen3.6-27b
LLM_API_KEY=EMPTY
LLM_BASE_URL=http://localhost:8000/v1
```

可选环境变量：`GEN_TOTAL_TARGET`、`GEN_BATCH_SIZE`、`GEN_MAX_WORKERS`、`GEN_OUTPUT_DIR`、`GEN_OVERSAMPLE_RATIO`

## 快速开始

```bash
cd batch_generate_text_and_clone/adult_text_generation

# 试跑 8 条
python generate_single.py --scenario daily_chat --emotion happy --length short --lang pure_cn --count 8

# 小规模批量
GEN_TOTAL_TARGET=500 python run_batch_generation.py --total 500 --resume

# 生产 10 万
python run_100k_adult.py
```

## 输出格式

```json
{
  "text": "嗯，那个，今天的会议改到下午三点了[sigh]",
  "length_type": "short",
  "lang_type": "pure_cn",
  "scenario": "business",
  "subscene": "临时会议",
  "emotion": "frustrated",
  "age_tier": "adult",
  "language": "zh",
  "text_tn": "嗯那个今天的会议改到下午三点了",
  "id": "adult_...",
  "tag_count": 1,
  "tags_used": ["sigh"]
}
```

## 与克隆流水线对接

生成完成后，将 `clone_dataset.py` 的 `TEXTS_PATH` 指向：

```
batch_generated_text/llm_adult_100k_asr_complete.jsonl
```

或增加 CLI 参数 `--texts-path`（推荐后续改造）。sidecar 会写入 `age_tier=adult` 及完整 metadata。

## 场景列表

- `daily_chat` 日常对话
- `business` 商务职场
- `education` 教育讲解
- `emotional` 情感表达
- `entertainment` 娱乐创意
- `narration` 叙述旁白
- `social_media` 社交媒体/vlog
- `service` 客服服务
- `creative_writing` 文学朗读
- `asr_stress` ASR 压力测试
