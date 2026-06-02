#!/bin/bash
set -euo pipefail

# 4卡 × 4 worker = 16 并发
cd /root/code/github_repos/OmniVoice-fork

PYTHON="${PYTHON:-/root/miniforge3/envs/omnivoice/bin/python}"

LOG_DIR="/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned/logs"
mkdir -p "$LOG_DIR"

NUM_WORKERS=16
pids=()

# GPU 0: workers 0-3
for w in 0 1 2 3; do
  log="$LOG_DIR/gpu0_worker${w}.log"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PYTHON" -u batch_generate_text_and_clone/voice_clone/clone_dataset.py \
    --gpu 0 --worker-id "$w" --num-workers "$NUM_WORKERS" \
    > "$log" 2>&1 &
  pids+=("$!")
  echo "Started GPU 0 worker $w (pid=$!) log=$log"
done

# GPU 1: workers 4-7
for w in 4 5 6 7; do
  log="$LOG_DIR/gpu1_worker${w}.log"
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "$PYTHON" -u batch_generate_text_and_clone/voice_clone/clone_dataset.py \
    --gpu 1 --worker-id "$w" --num-workers "$NUM_WORKERS" \
    > "$log" 2>&1 &
  pids+=("$!")
  echo "Started GPU 1 worker $w (pid=$!) log=$log"
done

# GPU 2: workers 8-11
for w in 8 9 10 11; do
  log="$LOG_DIR/gpu2_worker${w}.log"
  CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PYTHON" -u batch_generate_text_and_clone/voice_clone/clone_dataset.py \
    --gpu 2 --worker-id "$w" --num-workers "$NUM_WORKERS" \
    > "$log" 2>&1 &
  pids+=("$!")
  echo "Started GPU 2 worker $w (pid=$!) log=$log"
done

# GPU 3: workers 12-15
for w in 12 13 14 15; do
  log="$LOG_DIR/gpu3_worker${w}.log"
  CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PYTHON" -u batch_generate_text_and_clone/voice_clone/clone_dataset.py \
    --gpu 3 --worker-id "$w" --num-workers "$NUM_WORKERS" \
    > "$log" 2>&1 &
  pids+=("$!")
  echo "Started GPU 3 worker $w (pid=$!) log=$log"
done

echo ""
echo "All $NUM_WORKERS workers started."
echo "Logs: $LOG_DIR/"
echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/gpu0_worker0.log"
echo "  nvidia-smi -l 1"
echo ""

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" != "0" ]]; then
  echo "Some workers failed." >&2
  exit 1
fi

echo "All workers completed."
