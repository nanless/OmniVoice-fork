# 儿童语音批量克隆（voice_clone）

用 OmniVoice 对儿童参考音进行批量 voice cloning。推荐的 plan 模式按 speaker 的已接受总时长补齐到目标值，并输出可恢复、可审计的 16 kHz WAV 和 schema v3 sidecar；旧 dataset-scan 模式仍可用于小规模调试。

## 目录

```
voice_clone/
├── README.md              # 本文档
├── speaker_topup_common.py # 精确 WAV 时长、speaker/数据集公共契约
├── plan_speaker_topup.py   # 生成不可变的 speaker 补量计划
├── check_speaker_target.py # 检查 accepted 总时长是否全部达标
├── clone_dataset.py        # 加载模型、按 speaker 分 worker、断点续跑
└── run_clone_8workers.sh   # 默认 8 GPU × 2 worker 生产启动脚本
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

### 生产：16 worker（默认 8 卡 × 每卡 2 worker）

```bash
bash batch_generate_text_and_clone/voice_clone/run_clone_8workers.sh
```

- GPU 0–7：默认每卡两个 worker，以提高 A800 利用率；可用 `GPUS`、`WORKERS_PER_GPU` 调整。
- 日志：`$CLONED_VOICES_ROOT/logs/clone_$RUN_ID/gpu{GPU}_worker{N}.log`

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
| `--plan-jsonl` | 无 | 不可变补量计划；启用按 speaker 分片和 schema v3 |
| `--out-dir` | 环境变量或内置路径 | 本轮 raw clone 根目录 |
| `--max-attempts` | 3 | 同一任务允许的最大生成尝试次数 |

## Worker 分配逻辑

plan 模式对 `speaker_key` 做稳定 SHA-256 分片，同一个 speaker 只会归一个 worker；每个 worker 会流式校验整份计划，但仅在内存中保留自己的任务。完整且 plan/model/reference/output 签名都匹配的结果才会 skip。

旧 dataset-scan 模式继续按数据集 shuffle 后的索引 `% num_workers` 分片。

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

## 每个 speaker 补齐到 30 分钟

30 分钟按“合并数据集 `audio/<dataset>/<speaker>` 中的原音频 + 一个或多个已经通过质量筛选的 clone root”计算。未筛选的 raw clone 不计入达标时长。

```bash
BASE=/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130

python batch_generate_text_and_clone/voice_clone/plan_speaker_topup.py \
  --original-root "$BASE/audio" \
  --accepted-root "$BASE/audio_omnivoice_clone_sim0.8_filtered" \
  --target-seconds 1800 \
  --round-id 20260720_r01 \
  --plan-jsonl "$BASE/omnivoice_topup/round_001.plan.jsonl"

CLONE_PLAN_JSONL="$BASE/omnivoice_topup/round_001.plan.jsonl" \
CLONED_VOICES_ROOT="$BASE/omnivoice_topup/round_001_raw" \
bash batch_generate_text_and_clone/voice_clone/run_clone_8workers.sh
```

随后对 `round_001_raw` 先跑 raw-cosine SIM，按 `SIM > 0.8` 限定 CER 域，再按
`CER < 0.1` 复制到 accepted root，最后检查：

```bash
python batch_generate_text_and_clone/voice_clone/check_speaker_target.py \
  --original-root "$BASE/audio" \
  --accepted-root "$BASE/audio_omnivoice_clone_sim0.8_filtered" \
  --strict-accepted-root "$BASE/audio_omnivoice_topup_sim0.8_cer0.1_accepted" \
  --target-seconds 1800
```

checker 全部达标返回 0；仍有 speaker 缺口返回 1。此时用更新后的 accepted root 生成下一轮 plan。plan/task ID、speaker 分片、文件锁和 schema v3 sidecar 保证断点续跑及多 worker 幂等。

### 自动多轮漏斗

生产闭环使用独立的 top-up accepted root；旧 accepted root 保持不变。累计达标口径始终是
`原音频 + legacy accepted + 已提交 top-up accepted`，raw clone 即使生成成功也不参与 30 分钟判断。

```bash
python batch_generate_text_and_clone/voice_clone/run_speaker_topup_loop.py \
  --start-round 2 \
  --generation-multiplier 4 \
  --history-reference-limit 8 \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 2 \
  --wait-for-gpus
```

每轮严格执行：

```text
plan → clone → raw cosine SIM → SIM > 0.8
     → scoped CER → CER < 0.1
     → publish accepted → delete rejected WAV → accepted-only target check
```

`generation-multiplier` 只放大本轮 raw 生成预算，不改变 1,800 秒接受目标。下一轮会重新读取实际
accepted WAV 时长。planner 读取所有 prior plan，拒绝复用历史 `(ref_audio, text_id)`；历史 SIM
aggregate 只用于优先选择曾产生高 raw-cosine 分数的原始 reference，不改变筛选阈值。
有通过历史时每位 speaker 默认只保留表现最好的 8 条 reference；没有通过历史时回退到历史 raw
cosine 最高的 8 条。这个上限只改变生成候选的效率，不改变 `SIM > 0.8` 与 `CER < 0.1`。
`--wait-for-gpus` 默认要求所选 GPU 连续 3 次（间隔 30 秒）满足利用率不高于 10%、显存不高于
1,024 MiB 才启动 clone，避免和服务器上的其他任务抢卡；阈值和连续次数均可显式调整。

发布/清理可以单独预演。默认只生成不可变操作计划，不复制或删除：

```bash
python batch_generate_text_and_clone/voice_clone/promote_and_prune_round.py \
  --round-root "$BASE/omnivoice_topup/round_001_raw" \
  --accepted-root "$BASE/audio_omnivoice_topup_sim0.8_cer0.1_accepted" \
  --sim-pass-list "$BASE/omnivoice_topup/round_001_raw/filtered/sim_gt0.8.txt" \
  --accepted-list "$BASE/omnivoice_topup/round_001_raw/filtered/cer_lt0.1_sim_gt0.8.txt"
```

复核操作计划后加 `--execute`。工具先幂等复制全部通过项，再同时写 raw round commit 和
`accepted_root/commits/<round_id>.accepted.json`；只有带有效整轮 commit 的 top-up WAV 才计入
30 分钟。未提交的中断副本会被忽略，重跑发布即可恢复。commit 落盘后才删除未通过 WAV 和指标
sidecar；主 clone JSON 会保留并改为 `status=rejected`，用于审计和跨轮恢复。
