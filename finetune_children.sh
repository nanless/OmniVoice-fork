#!/bin/bash
# ============================================================
#  OmniVoice Fine-tune — 儿童 TTS 全流程脚本（从头开始）
#
#  数据集（kaldi 格式，自动合并）：
#    - BAAI-ChildMandarin41.25H                (zh)
#    - Chinese_English_Scripted_Speech_Children (en)
#    - King-ASR-EN-Kid                          (en)
#    - speechocean762                           (en)
#
#  用法：
#    bash finetune_children.sh              # 全流程 stage 0-2（tmux 后台）
#    bash finetune_children.sh --stage 1    # 仅从 tokenize 开始
#    bash finetune_children.sh --stage 2    # 仅训练
#    bash finetune_children.sh --stop  1    # 只跑到 tokenize
#    bash finetune_children.sh --fg         # 前台运行（不用 tmux）
#
#  断点续训（从已有 checkpoint 继续）：
#    RESUME_FROM=/path/to/checkpoint bash finetune_children.sh --stage 2
# ============================================================

set -euo pipefail

# ── 参数解析 ──────────────────────────────────────────────────
stage=0
stop_stage=2
use_tmux=1   # 默认 tmux 后台
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) stage="$2";      shift 2 ;;
        --stop)  stop_stage="$2"; shift 2 ;;
        --fg)    use_tmux=0;      shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── tmux 自动后台逻辑 ─────────────────────────────────────────
# 如果没在 tmux 里，且 use_tmux=1，则自动创建 session 后台运行
if [ "${use_tmux}" -eq 1 ] && [ -z "${TMUX:-}" ]; then
    SESSION="omnivoice_ft_$(date +%Y%m%d_%H%M%S)"
    # 把当前所有环境变量和参数一起传进去
    ARGS="$*"
    ENV_VARS=""
    for v in EXP_NAME GPU_IDS NUM_GPUS TOKENIZER_PATH PRETRAINED_MODEL \
              RESUME_FROM LR STEPS BATCH_TOKENS; do
        val="${!v:-}"
        [ -n "$val" ] && ENV_VARS="${ENV_VARS} ${v}='${val}'"
    done

    SCRIPT_PATH="$(realpath "$0")"
    CONDA_BASE="$(conda info --base 2>/dev/null || echo '/root/miniforge3')"
    CONDA_ENV="${OMNIVOICE_CONDA_ENV:-omnivoice}"
    CMD="source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${CONDA_ENV}' && cd '$(pwd)' && ${ENV_VARS} bash '${SCRIPT_PATH}' --fg --stage ${stage} --stop ${stop_stage} ${ARGS}"

    echo "启动 tmux session: ${SESSION}"
    echo "conda 环境: ${CONDA_ENV} (${CONDA_BASE})"
    echo "命令: ${CMD}"
    tmux new-session -d -s "${SESSION}" "bash -c \"${CMD}\""
    echo ""
    echo "已在后台启动，查看日志："
    echo "  tmux attach -t ${SESSION}          # 进入 session"
    echo "  tail -f exp/\${EXP_NAME}/train.log  # 查看训练日志"
    echo ""
    echo "注意：EXP_NAME 默认含时间戳，实际目录请运行后用 ls exp/ 确认"
    exit 0
fi

# ── 路径配置（按需修改这里）────────────────────────────────────
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="${EXP_NAME:-children_finetune_$(date +%Y%m%d_%H%M)}"
EXP_DIR="${REPO_DIR}/exp/${EXP_NAME}"

# GPU
GPU_IDS="${GPU_IDS:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"

# 本地 audio tokenizer（避免网络下载）
TOKENIZER_PATH="${TOKENIZER_PATH:-/root/.cache/modelscope/k2-fsa/OmniVoice/audio_tokenizer}"

# 预训练模型（本地缓存，避免网络下载）
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/root/.cache/modelscope/k2-fsa/OmniVoice}"

# 断点续训（空则从预训练模型开始）
RESUME_FROM="${RESUME_FROM:-}"

# 训练超参
LR="${LR:-3e-6}"
STEPS="${STEPS:-100000}"
BATCH_TOKENS="${BATCH_TOKENS:-16384}"

# kaldi 数据集列表：格式为 "语言:kaldi_dir"，空格分隔
KALDI_DATASETS=(
    "zh:/root/group-shared/voiceprint/data/speech/speaker_verification/BAAI-ChildMandarin41.25H_integrated_by_groundtruth/kaldi_files"
    "en:/root/group-shared/voiceprint/data/speech/speaker_verification/Chinese_English_Scripted_Speech_Corpus_Children_integrated_by_groundtruth/kaldi_files"
    "en:/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_integrated_by_groundtruth/kaldi_files"
    "en:/root/group-shared/voiceprint/data/speech/speaker_verification/speechocean762_integrated_by_groundtruth/kaldi_files"
)

# ── 内部路径（不需要修改）─────────────────────────────────────
DATA_DIR="${EXP_DIR}/data"
TOKEN_DIR="${EXP_DIR}/tokens"
CONFIG_DIR="${EXP_DIR}/config"
OUTPUT_DIR="${EXP_DIR}/checkpoints"
TRAIN_JSONL="${DATA_DIR}/train.jsonl"
DEV_JSONL="${DATA_DIR}/dev.jsonl"
TRAIN_CONFIG="${CONFIG_DIR}/train_config.json"
DATA_CONFIG="${CONFIG_DIR}/data_config.json"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "实验目录: ${EXP_DIR}"
log "GPU: ${GPU_IDS} (${NUM_GPUS} 个)"
log "stage ${stage} → ${stop_stage}"
mkdir -p "${DATA_DIR}" "${CONFIG_DIR}" "${OUTPUT_DIR}"

# ══════════════════════════════════════════════════════════════
# Stage 0: kaldi → JSONL
# ══════════════════════════════════════════════════════════════
if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    log "Stage 0: 从 kaldi 格式生成 JSONL"

    python3 - <<PYEOF
import json, random
from pathlib import Path

datasets = [$(for ds in "${KALDI_DATASETS[@]}"; do
    lang="${ds%%:*}"; kdir="${ds#*:}"
    echo "    ('${lang}', '${kdir}'),"
done)]

def load_scp(path):
    d = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    d[parts[0]] = parts[1]
    return d

all_samples = []
for lang, kdir in datasets:
    kdir = Path(kdir)
    name = kdir.parent.name
    wav  = load_scp(kdir / 'wav.scp')
    text = load_scp(kdir / 'text')
    n = 0
    for utt, path in wav.items():
        if utt in text and text[utt].strip():
            all_samples.append({
                'id': f'{name}_{utt}'.replace('.', '_'),
                'audio_path': path,
                'text': text[utt].strip(),
                'language_id': lang,
            })
            n += 1
    print(f'  [{name}] lang={lang}, loaded={n}')

print(f'Total: {len(all_samples)} samples')
random.seed(42)
random.shuffle(all_samples)
n_dev = max(1, int(len(all_samples) * 0.02))
dev, train = all_samples[:n_dev], all_samples[n_dev:]

for path, split in [('${TRAIN_JSONL}', train), ('${DEV_JSONL}', dev)]:
    with open(path, 'w', encoding='utf-8') as f:
        for s in split:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'Wrote {len(split)} samples → {path}')
PYEOF

    log "Stage 0 完成。train=$(wc -l < "${TRAIN_JSONL}"), dev=$(wc -l < "${DEV_JSONL}")"
fi

# ══════════════════════════════════════════════════════════════
# Stage 1: 音频 tokenization
# ══════════════════════════════════════════════════════════════
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: 音频 tokenization"

    # 检查 tokenizer 是否存在
    if [ ! -d "${TOKENIZER_PATH}" ]; then
        log "ERROR: tokenizer 不存在: ${TOKENIZER_PATH}"
        log "请设置 TOKENIZER_PATH 环境变量指向本地 audio tokenizer 目录"
        exit 1
    fi

    for split in train dev; do
        split_jsonl="${DATA_DIR}/${split}.jsonl"
        manifest="${TOKEN_DIR}/${split}/data.lst"

        if [ -f "${manifest}" ]; then
            log "  ${split} 已存在 ${manifest}，跳过"
            continue
        fi

        log "  Tokenizing ${split} ..."
        CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
            python -m omnivoice.scripts.extract_audio_tokens \
            --input_jsonl  "${split_jsonl}" \
            --tar_output_pattern  "${TOKEN_DIR}/${split}/audios/shard-%06d.tar" \
            --jsonl_output_pattern "${TOKEN_DIR}/${split}/txts/shard-%06d.jsonl" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --nj_per_gpu 4 \
            --shuffle True \
            --skip_errors

        log "  Done → ${manifest}"
        log "  Errors: $(wc -l < "${TOKEN_DIR}/${split}/errors.jsonl" 2>/dev/null || echo 0)"
    done

    log "Stage 1 完成"
fi

# ══════════════════════════════════════════════════════════════
# Stage 2: 微调训练
# ══════════════════════════════════════════════════════════════
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 2: 微调训练"

    # 确认 tokenizer 的 symlink 在 checkpoint 里
    if [ -n "${RESUME_FROM}" ] && [ ! -e "${RESUME_FROM}/audio_tokenizer" ]; then
        ln -s "${TOKENIZER_PATH}" "${RESUME_FROM}/audio_tokenizer"
        log "  Created audio_tokenizer symlink in ${RESUME_FROM}"
    fi

    # 生成 data config（绝对路径）
    cat > "${DATA_CONFIG}" <<EOF
{
    "train": [{"manifest_path": ["${TOKEN_DIR}/train/data.lst"]}],
    "dev":   [{"manifest_path": ["${TOKEN_DIR}/dev/data.lst"]}]
}
EOF
    log "  Data config: ${DATA_CONFIG}"

    # 生成 train config
    INIT_CKPT="${PRETRAINED_MODEL}"
    RESUME_FIELD="null"
    if [ -n "${RESUME_FROM}" ]; then
        INIT_CKPT="${RESUME_FROM}"
        # resume_from_checkpoint 用于恢复 optimizer/scheduler 状态
        # init_from_checkpoint 仅加载模型权重，这里用 init 方式（新 schedule）
    fi

    cat > "${TRAIN_CONFIG}" <<EOF
{
    "llm_name_or_path": "Qwen/Qwen3-0.6B",
    "audio_vocab_size": 1025,
    "audio_mask_id": 1024,
    "num_audio_codebook": 8,

    "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
    "drop_cond_ratio": 0.1,
    "prompt_ratio_range": [0.0, 0.3],
    "mask_ratio_range": [0.0, 1.0],
    "language_ratio": 0.8,
    "use_pinyin_ratio": 0.0,
    "instruct_ratio": 0.0,
    "only_instruct_ratio": 0.0,

    "resume_from_checkpoint": ${RESUME_FIELD},
    "init_from_checkpoint": "${INIT_CKPT}",

    "learning_rate": ${LR},
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "steps": ${STEPS},
    "seed": 42,
    "warmup_type": "ratio",
    "warmup_ratio": 0.01,
    "warmup_steps": 0,

    "batch_tokens": ${BATCH_TOKENS},
    "gradient_accumulation_steps": 1,
    "num_workers": 4,

    "mixed_precision": "bf16",
    "allow_tf32": true,

    "logging_steps": 50,
    "eval_steps": 500,
    "save_steps": 1000,
    "keep_last_n_checkpoints": -1
}
EOF
    log "  Train config: ${TRAIN_CONFIG}"
    log "  init_from: ${INIT_CKPT}, lr=${LR}, steps=${STEPS}, batch_tokens=${BATCH_TOKENS}"

    accelerate launch \
        --gpu_ids "${GPU_IDS}" \
        --num_processes "${NUM_GPUS}" \
        -m omnivoice.cli.train \
        --train_config "${TRAIN_CONFIG}" \
        --data_config  "${DATA_CONFIG}" \
        --output_dir   "${OUTPUT_DIR}" \
        2>&1 | tee "${EXP_DIR}/train.log"

    log "Stage 2 完成。Checkpoints: ${OUTPUT_DIR}"
fi

log "全部完成！实验目录: ${EXP_DIR}"
log ""
log "推理示例:"
log "  python -c \""
log "  from omnivoice import OmniVoice; import soundfile as sf, torch"
log "  m = OmniVoice.from_pretrained('${OUTPUT_DIR}/checkpoint-${STEPS}', device_map='cuda:0', dtype=torch.float16)"
log "  sf.write('out.wav', m.generate(text='你好', ref_audio='ref.wav', ref_text='ref text', language='zh')[0], 24000)"
log "  \""
