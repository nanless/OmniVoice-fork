#!/bin/bash
set -euo pipefail

# 双卡 × 4 worker = 8 并发
cd /root/code/github_repos/OmniVoice-fork

LOG_DIR="batch_cloned_voices/logs"
mkdir -p "$LOG_DIR"

NUM_WORKERS=8
pids=()

# GPU 0: workers 0-3
for w in 0 1 2 3; do
  log="$LOG_DIR/gpu0_worker${w}.log"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u batch_generate_text_and_clone/voice_clone/clone_dataset.py \
    --gpu 0 --worker-id "$w" --num-workers "$NUM_WORKERS" \
    > "$log" 2>&1 &
  pids+=("$!")
  echo "Started GPU 0 worker $w (pid=$!) log=$log"
done

# GPU 1: workers 4-7
for w in 4 5 6 7; do
  log="$LOG_DIR/gpu1_worker${w}.log"
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python -u batch_generate_text_and_clone/voice_clone/clone_dataset.py \
    --gpu 1 --worker-id "$w" --num-workers "$NUM_WORKERS" \
    > "$log" 2>&1 &
  pids+=("$!")
  echo "Started GPU 1 worker $w (pid=$!) log=$log"
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
