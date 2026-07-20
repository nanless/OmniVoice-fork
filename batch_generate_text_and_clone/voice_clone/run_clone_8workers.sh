#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  DEFAULT_PYTHON="/root/miniforge3/envs/omnivoice/bin/python"
fi

PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
# A single OmniVoice stream uses only a small fraction of an A800.  Two
# independent streams overlap generation with CPU resampling and durable I/O
# while keeping ample memory headroom.  Override for smaller GPUs if needed.
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
OUT_DIR="${CLONED_VOICES_ROOT:-/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned}"
MODEL_PATH="${OMNIVOICE_MODEL_PATH:-$REPO_ROOT/exp/children_finetune_20260519_1418/checkpoints/checkpoint-62000}"
TEXTS_PATH="${CLONE_TEXTS_PATH:-$REPO_ROOT/batch_generated_text/llm_children_100k_asr_complete.jsonl}"
PLAN_JSONL="${CLONE_PLAN_JSONL:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LAUNCH_DRY_RUN="${LAUNCH_DRY_RUN:-0}"

for arg in "$@"; do
  case "$arg" in
    --gpu|--gpu=*|--worker-id|--worker-id=*|--num-workers|--num-workers=*|\
    --out-dir|--out-dir=*|--model-path|--model-path=*|--texts-path|--texts-path=*|\
    --plan-jsonl|--plan-jsonl=*)
      echo "Reserved launcher argument must be configured through environment: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ ! "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS_PER_GPU must be a positive integer: $WORKERS_PER_GPU" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  exit 2
fi
if [[ -z "$PLAN_JSONL" && ! -f "$TEXTS_PATH" ]]; then
  echo "Clone texts not found: $TEXTS_PATH" >&2
  exit 2
fi
if [[ -n "$PLAN_JSONL" && ! -f "$PLAN_JSONL" ]]; then
  echo "Clone plan not found: $PLAN_JSONL" >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "OmniVoice model not found: $MODEL_PATH" >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if (( ${#GPU_LIST[@]} == 0 )); then
  echo "GPUS must contain at least one GPU id" >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for i in "${!GPU_LIST[@]}"; do
  GPU_LIST[$i]="${GPU_LIST[$i]//[[:space:]]/}"
  if [[ ! "${GPU_LIST[$i]}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id: ${GPU_LIST[$i]}" >&2
    exit 2
  fi
  if [[ -n "${SEEN_GPUS[${GPU_LIST[$i]}]+present}" ]]; then
    echo "Duplicate GPU id: ${GPU_LIST[$i]}" >&2
    exit 2
  fi
  SEEN_GPUS[${GPU_LIST[$i]}]=1
done

NUM_WORKERS=$(( ${#GPU_LIST[@]} * WORKERS_PER_GPU ))
COMMON_ARGS=(
  --num-workers "$NUM_WORKERS"
  --out-dir "$OUT_DIR"
  --model-path "$MODEL_PATH"
)
if [[ -n "$PLAN_JSONL" ]]; then
  COMMON_ARGS+=(--plan-jsonl "$PLAN_JSONL")
else
  COMMON_ARGS+=(--texts-path "$TEXTS_PATH")
fi

echo "Clone configuration:"
echo "  PYTHON=$PYTHON"
echo "  GPUS=$GPUS"
echo "  WORKERS_PER_GPU=$WORKERS_PER_GPU"
echo "  NUM_WORKERS=$NUM_WORKERS"
echo "  OUT_DIR=$OUT_DIR"
echo "  MODEL_PATH=$MODEL_PATH"
if [[ -n "$PLAN_JSONL" ]]; then
  echo "  PLAN_JSONL=$PLAN_JSONL"
else
  echo "  TEXTS_PATH=$TEXTS_PATH"
fi

worker_id=0
if [[ "$LAUNCH_DRY_RUN" == "1" ]]; then
  for gpu in "${GPU_LIST[@]}"; do
    for ((local_worker = 0; local_worker < WORKERS_PER_GPU; local_worker++)); do
      printf 'CUDA_VISIBLE_DEVICES=%q %q -u %q --gpu %q --worker-id %q' \
        "$gpu" "$PYTHON" \
        "batch_generate_text_and_clone/voice_clone/clone_dataset.py" \
        "$gpu" "$worker_id"
      printf ' %q' "${COMMON_ARGS[@]}" "$@"
      printf '\n'
      worker_id=$((worker_id + 1))
    done
  done
  exit 0
fi

mkdir -p "$OUT_DIR"
exec 9>"$OUT_DIR/.clone_dataset.lock"
if ! flock -n 9; then
  echo "Another clone launcher holds $OUT_DIR/.clone_dataset.lock" >&2
  exit 1
fi

LOG_DIR="$OUT_DIR/logs/clone_$RUN_ID"
if ! mkdir "$LOG_DIR"; then
  echo "Log directory already exists; choose a unique RUN_ID: $LOG_DIR" >&2
  exit 1
fi
pids=()
labels=()

cleanup() {
  local status=$?
  trap - INT TERM EXIT
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  exit "$status"
}
trap cleanup INT TERM EXIT

for gpu in "${GPU_LIST[@]}"; do
  for ((local_worker = 0; local_worker < WORKERS_PER_GPU; local_worker++)); do
    log="$LOG_DIR/gpu${gpu}_worker${worker_id}.log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON" -u \
      batch_generate_text_and_clone/voice_clone/clone_dataset.py \
      --gpu "$gpu" --worker-id "$worker_id" \
      "${COMMON_ARGS[@]}" "$@" >"$log" 2>&1 &
    pids+=("$!")
    label="gpu=$gpu worker=$worker_id log=$log"
    labels+=("$label")
    echo "Started $label pid=$!"
    worker_id=$((worker_id + 1))
  done
done

echo "All $NUM_WORKERS workers started. Logs: $LOG_DIR"

failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "FAILED: ${labels[$i]}" >&2
    failed=1
  fi
done

trap - INT TERM EXIT
if [[ "$failed" != "0" ]]; then
  echo "Some clone workers failed or exhausted retries." >&2
  exit 1
fi

echo "All $NUM_WORKERS workers completed successfully."
