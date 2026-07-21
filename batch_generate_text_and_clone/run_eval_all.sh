#!/bin/bash
# 漏斗评测：SIM → SIM筛选 → CER(SIM通过项) → CER筛选 → 可选MOS
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
#   SKIP_EXISTING=0  全量重算（默认 1 跳过已有 sidecar）

set -euo pipefail

SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
SESSION="${EVAL_TMUX_SESSION:-eval_all}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${CLONED_VOICES_ROOT:-/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned}"
CER_GPU="${CER_GPU:-${GPU:-0}}"
SIM_GPUS="${SIM_GPUS:-${GPUS:-1,2,3,4}}"
MOS_GPUS="${MOS_GPUS:-$SIM_GPUS}"
SIM_THRESHOLD="${SIM_THRESHOLD:-0.8}"
CER_THRESHOLD="${CER_THRESHOLD:-0.1}"
FILTER_DIR="${FILTER_DIR:-$OUT_DIR/filtered}"
SIM_PASS_LIST="${SIM_PASS_LIST:-$FILTER_DIR/sim_gt${SIM_THRESHOLD}.txt}"
CER_SCOPE_DIR="${CER_SCOPE_DIR:-$OUT_DIR/eval_scopes/sim_gt${SIM_THRESHOLD}}"
FINAL_PASS_LIST="${FINAL_PASS_LIST:-$FILTER_DIR/cer_lt${CER_THRESHOLD}_sim_gt${SIM_THRESHOLD}.txt}"
EVAL_RUN_ID="${EVAL_RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"

if [[ "${IN_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already running. Attach: tmux attach -t $SESSION"
    exit 1
  fi
  tmux new-session -d -s "$SESSION" \
    "IN_TMUX=1 CLONED_VOICES_ROOT='$OUT_DIR' CER_GPU='$CER_GPU' SIM_GPUS='$SIM_GPUS' MOS_GPUS='$MOS_GPUS' SKIP_CER='${SKIP_CER:-0}' SKIP_SIM='${SKIP_SIM:-0}' SKIP_MOS='${SKIP_MOS:-0}' SKIP_EXISTING='${SKIP_EXISTING:-1}' SKIP_ASR='${SKIP_ASR:-0}' REFRESH_CER='${REFRESH_CER:-0}' REFRESH_ASR='${REFRESH_ASR:-0}' ALLOW_PARTIAL='${ALLOW_PARTIAL:-0}' ASR_BATCH_SIZE='${ASR_BATCH_SIZE:-16}' EVAL_WORKERS='${EVAL_WORKERS:-}' PYTHON='${PYTHON:-}' PYTHON_CER='${PYTHON_CER:-}' TTS_EVAL_MODEL_DIR='${TTS_EVAL_MODEL_DIR:-}' SIM_THRESHOLD='$SIM_THRESHOLD' CER_THRESHOLD='$CER_THRESHOLD' FILTER_DIR='$FILTER_DIR' SIM_PASS_LIST='$SIM_PASS_LIST' CER_SCOPE_DIR='$CER_SCOPE_DIR' FINAL_PASS_LIST='$FINAL_PASS_LIST' EVAL_RUN_ID='$EVAL_RUN_ID' exec bash '$SCRIPT'"
  echo "Started tmux session: $SESSION"
  echo "  attach: tmux attach -t $SESSION"
  echo "  logs:   tail -f $OUT_DIR/logs/eval_all/$EVAL_RUN_ID/main.log"
  exit 0
fi

if [[ ! -d "$OUT_DIR" ]]; then
  echo "ERROR: clone output directory does not exist: $OUT_DIR" >&2
  exit 1
fi

LOG_DIR="$OUT_DIR/logs/eval_all/$EVAL_RUN_ID"
mkdir -p "$LOG_DIR"

export PYTHONUNBUFFERED=1
# One thread per worker process; parallelism comes from --workers, not OpenMP.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
# CER receives one isolated GPU; SIM/MOS spawn workers from explicit GPU lists.

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
SIM_COMMON=("${COMMON[@]}")
[[ "${ALLOW_PARTIAL:-0}" == "1" ]] && SIM_COMMON+=(--allow-partial)
MOS_COMMON=("${COMMON[@]}")
[[ "${ALLOW_PARTIAL:-0}" == "1" ]] && MOS_COMMON+=(--allow-partial)

ASR_BATCH_SIZE="${ASR_BATCH_SIZE:-16}"
IFS=',' read -r -a SIM_GPU_ARRAY <<< "$SIM_GPUS"
EVAL_WORKERS="${EVAL_WORKERS:-${#SIM_GPU_ARRAY[@]}}"

log "======== Funnel eval: SIM → SIM filter → CER → CER filter → optional MOS ========"
log "OUT=$OUT_DIR CER_GPU=$CER_GPU SIM_GPUS=$SIM_GPUS MOS_GPUS=$MOS_GPUS ASR_BATCH=$ASR_BATCH_SIZE WORKERS=$EVAL_WORKERS SKIP_EXISTING=${SKIP_EXISTING:-1}"
log "STRICT THRESHOLDS: raw cosine > $SIM_THRESHOLD; deterministic CER < $CER_THRESHOLD"
log "SIM_PASS_LIST=$SIM_PASS_LIST CER_SCOPE_DIR=$CER_SCOPE_DIR FINAL_PASS_LIST=$FINAL_PASS_LIST"
log "PYTHON=$PYTHON PYTHON_CER=$PYTHON_CER"
log "Logs: $LOG_DIR"

CER_ARGS=(--out-dir "$OUT_DIR" --batch-size "$ASR_BATCH_SIZE" --wav-list "$SIM_PASS_LIST" --report-dir "$CER_SCOPE_DIR")
if [[ "${SKIP_EXISTING:-1}" == "1" ]] \
  && [[ "${REFRESH_CER:-0}" != "1" ]] \
  && [[ "${REFRESH_ASR:-0}" != "1" ]]; then
  CER_ARGS+=(--skip-existing)
fi
[[ "${SKIP_ASR:-0}" == "1" ]] && CER_ARGS+=(--skip-asr)
[[ "${REFRESH_CER:-0}" == "1" ]] && CER_ARGS+=(--refresh-cer)
[[ "${REFRESH_ASR:-0}" == "1" ]] && CER_ARGS+=(--refresh-asr-cache)
[[ "${ALLOW_PARTIAL:-0}" == "1" ]] && CER_ARGS+=(--allow-partial)

run_cer() {
  log "[CER] START eval_cloned.py ($PYTHON_CER)"
  : > "$CER_LOG"
  (export CUDA_VISIBLE_DEVICES="$CER_GPU"
    cd "$ROOT/batch_generate_text_and_clone/eval_cer" && \
    run_py_cer "$CER_LOG" eval_cloned.py "${CER_ARGS[@]}")
  log "[CER] END"
}

run_sim() {
  log "[SIM] START eval_clone_similarity.py"
  : > "$SIM_LOG"
  run_py "$SIM_LOG" "$ROOT/batch_generate_text_and_clone/eval_sim/eval_clone_similarity.py" \
    "${SIM_COMMON[@]}" --model-dir "$MODEL_SIM" --gpus "$SIM_GPUS" --workers "$EVAL_WORKERS"
  log "[SIM] END"
}

if [[ "${SKIP_SIM:-0}" != "1" ]]; then
  log "[1/5] START SIM (full current clone inventory)"
  run_sim
  log "[1/5] END SIM"
fi

log "[2/5] START strict SIM filter (> $SIM_THRESHOLD)"
FILTER_SIM_ARGS=(--out-dir "$OUT_DIR" --min-sim "$SIM_THRESHOLD" --output "$SIM_PASS_LIST")
[[ "${ALLOW_PARTIAL:-0}" == "1" ]] && FILTER_SIM_ARGS+=(--allow-partial)
run_py "$SIM_LOG" "$ROOT/batch_generate_text_and_clone/filter_cloned.py" "${FILTER_SIM_ARGS[@]}"
log "[2/5] END strict SIM filter: $(wc -l < "$SIM_PASS_LIST") candidates"

if [[ "${SKIP_CER:-0}" != "1" ]]; then
  log "[3/5] START CER on SIM-pass candidates only"
  run_cer
  log "[3/5] END CER"
fi

log "[4/5] START strict joint filter (SIM > $SIM_THRESHOLD AND CER < $CER_THRESHOLD)"
FILTER_FINAL_ARGS=(
  --out-dir "$OUT_DIR"
  --candidate-list "$SIM_PASS_LIST"
  --cer-details "$CER_SCOPE_DIR/eval_cer_details.jsonl"
  --max-cer "$CER_THRESHOLD"
  --min-sim "$SIM_THRESHOLD"
  --output "$FINAL_PASS_LIST"
)
[[ "${ALLOW_PARTIAL:-0}" == "1" ]] && FILTER_FINAL_ARGS+=(--allow-partial)
run_py "$CER_LOG" "$ROOT/batch_generate_text_and_clone/filter_cloned.py" "${FILTER_FINAL_ARGS[@]}"
log "[4/5] END strict joint filter: $(wc -l < "$FINAL_PASS_LIST") accepted"

if [[ "${SKIP_MOS:-0}" != "1" ]]; then
  log "[5/5] START MOS (independent full-inventory stage)"
  : > "$MOS_LOG"
  run_py "$MOS_LOG" "$ROOT/batch_generate_text_and_clone/eval_mos/eval_clone_mos.py" \
    "${MOS_COMMON[@]}" --model-dir "$MODEL_MOS" --gpus "$MOS_GPUS" --workers "$EVAL_WORKERS"
  log "[5/5] END MOS"
fi

log "======== ALL DONE ========"
log "Accepted list: $FINAL_PASS_LIST"
log "Sidecars: text_*.sim.json (SIM) / text_*.eval.json (CER on SIM-pass only) / text_*.mos.json (optional MOS)"
