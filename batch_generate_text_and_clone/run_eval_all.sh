#!/bin/bash
# 全库评测：CER → SIM → MOS(多指标)，sidecar 写在音频旁，tmux 后台运行
#
# Sidecar（与 text_001.wav 同目录）：
#   text_001.eval.json  CER
#   text_001.sim.json   说话人相似度
#   text_001.mos.json   质量评测（UTMOS22Strong / SCOREQ / TTSDS2 / UTMOSv2）
#
# Usage:
#   bash batch_generate_text_and_clone/run_eval_all.sh          # 自动 tmux 后台
#   tmux attach -t eval_all                                       # 查看进度
#
#   IN_TMUX=1 bash batch_generate_text_and_clone/run_eval_all.sh  # 前台/已在 tmux 内
#   PARALLEL_SIM=1  CER 与 SIM 并行（默认 1）；MOS 仍等两者结束后跑
#   SKIP_EXISTING=0  全量重算（默认 1 跳过已有 sidecar）

set -euo pipefail

SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
SESSION="${EVAL_TMUX_SESSION:-eval_all}"

if [[ "${IN_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already running. Attach: tmux attach -t $SESSION"
    exit 1
  fi
  tmux new-session -d -s "$SESSION" \
    "IN_TMUX=1 GPU='${GPU:-0}' GPUS='${GPUS:-${GPU:-0}}' SKIP_CER='${SKIP_CER:-0}' SKIP_SIM='${SKIP_SIM:-0}' SKIP_MOS='${SKIP_MOS:-0}' SKIP_EXISTING='${SKIP_EXISTING:-1}' EVAL_WORKERS='${EVAL_WORKERS:-4}' PARALLEL_SIM='${PARALLEL_SIM:-1}' exec bash '$SCRIPT'"
  echo "Started tmux session: $SESSION"
  echo "  attach: tmux attach -t $SESSION"
  echo "  logs:   tail -f ${CLONED_VOICES_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/batch_cloned_voices}/logs/eval_all/main.log"
  exit 0
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${CLONED_VOICES_ROOT:-$ROOT/batch_cloned_voices}"
GPU="${GPU:-0}"
LOG_DIR="$OUT_DIR/logs/eval_all"
mkdir -p "$LOG_DIR"

export PYTHONUNBUFFERED=1
# One thread per worker process; parallelism comes from --workers, not OpenMP.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
# CER uses GPU from env; SIM/MOS spawn workers with --gpus
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

PYTHON="${PYTHON:-/root/miniforge3/envs/omnivoice/bin/python}"
PYTHON_CER="${PYTHON_CER:-/root/miniforge3/envs/qwen3-asr/bin/python}"
MODEL_SIM="$ROOT/batch_generate_text_and_clone/eval_sim/model"
MODEL_MOS="${TTS_EVAL_MODEL_DIR:-$ROOT/TTS_eval_models}"

MAIN_LOG="$LOG_DIR/main.log"
CER_LOG="$LOG_DIR/cer.log"
SIM_LOG="$LOG_DIR/sim.log"
MOS_LOG="$LOG_DIR/mos.log"

: > "$MAIN_LOG"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"
}

run_py() {
  local logfile=$1
  shift
  stdbuf -oL -eL "$PYTHON" -u "$@" 2>&1 | stdbuf -oL tee -a "$logfile"
}

run_py_cer() {
  local logfile=$1
  shift
  stdbuf -oL -eL "$PYTHON_CER" -u "$@" 2>&1 | stdbuf -oL tee -a "$logfile"
}

COMMON=(--out-dir "$OUT_DIR")
[[ "${SKIP_EXISTING:-1}" == "1" ]] && COMMON+=(--skip-existing)

ASR_BATCH_SIZE="${ASR_BATCH_SIZE:-16}"
LLM_CONCURRENCY="${LLM_CONCURRENCY:-5}"
GPUS="${GPUS:-${GPU:-0}}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
PARALLEL_SIM="${PARALLEL_SIM:-1}"

log "======== Full eval: CER ${PARALLEL_SIM:+(+ SIM parallel) }→ MOS ========"
log "OUT=$OUT_DIR GPUS=$GPUS ASR_BATCH=$ASR_BATCH_SIZE ITN_WORKERS=$LLM_CONCURRENCY WORKERS=$EVAL_WORKERS PARALLEL_SIM=$PARALLEL_SIM SKIP_EXISTING=${SKIP_EXISTING:-1}"
log "PYTHON=$PYTHON PYTHON_CER=$PYTHON_CER"
log "Logs: $LOG_DIR"

CER_ARGS=(--out-dir "$OUT_DIR" --batch-size "$ASR_BATCH_SIZE" --llm-concurrency "$LLM_CONCURRENCY")
[[ "${SKIP_EXISTING:-1}" == "1" ]] && CER_ARGS+=(--skip-existing)
[[ "${SKIP_LLM:-0}" == "1" ]] && CER_ARGS+=(--skip-llm)
[[ "${SKIP_ASR:-0}" == "1" ]] && CER_ARGS+=(--skip-asr)
[[ "${REFRESH_LLM:-0}" == "1" ]] && CER_ARGS+=(--refresh-llm-cache)

run_cer() {
  log "[CER] START eval_cloned.py ($PYTHON_CER)"
  : > "$CER_LOG"
  (cd "$ROOT/batch_generate_text_and_clone/eval_cer" && \
    run_py_cer "$CER_LOG" eval_cloned.py "${CER_ARGS[@]}")
  log "[CER] END"
}

run_sim() {
  log "[SIM] START eval_clone_similarity.py"
  : > "$SIM_LOG"
  run_py "$SIM_LOG" "$ROOT/batch_generate_text_and_clone/eval_sim/eval_clone_similarity.py" \
    "${COMMON[@]}" --model-dir "$MODEL_SIM" --gpus "$GPUS" --workers "$EVAL_WORKERS"
  log "[SIM] END"
}

if [[ "${SKIP_CER:-0}" != "1" ]] && [[ "${SKIP_SIM:-0}" != "1" ]] && [[ "$PARALLEL_SIM" == "1" ]]; then
  log "[1/3] START CER + SIM (parallel, shared GPU $GPUS)"
  run_cer &
  CER_PID=$!
  run_sim &
  SIM_PID=$!
  CER_OK=0
  SIM_OK=0
  wait "$CER_PID" && CER_OK=1 || true
  wait "$SIM_PID" && SIM_OK=1 || true
  if [[ "$CER_OK" != "1" ]]; then
    log "ERROR: CER failed (exit != 0)"
    exit 1
  fi
  if [[ "$SIM_OK" != "1" ]]; then
    log "ERROR: SIM failed (exit != 0)"
    exit 1
  fi
  log "[1/3] END CER + SIM"
elif [[ "${SKIP_CER:-0}" != "1" ]]; then
  log "[1/3] START CER (eval_cer)"
  run_cer
  log "[1/3] END CER"
fi

if [[ "${SKIP_SIM:-0}" != "1" ]] && [[ "$PARALLEL_SIM" != "1" || "${SKIP_CER:-0}" == "1" ]]; then
  log "[2/3] START SIM (eval_sim)"
  run_sim
  log "[2/3] END SIM"
fi

if [[ "${SKIP_MOS:-0}" != "1" ]]; then
  log "[3/3] START MOS (eval_mos)"
  : > "$MOS_LOG"
  run_py "$MOS_LOG" "$ROOT/batch_generate_text_and_clone/eval_mos/eval_clone_mos.py" \
    "${COMMON[@]}" --model-dir "$MODEL_MOS" --gpus "$GPUS" --workers "$EVAL_WORKERS"
  log "[3/3] END MOS"
fi

log "======== ALL DONE ========"
log "Sidecars: text_*.eval.json (CER) / text_*.sim.json (SIM) / text_*.mos.json (质量多指标)"
