#!/bin/bash
set -e

OUT_DIR="/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned"
LOG_DIR="$OUT_DIR/logs"
LOG="$LOG_DIR/eval_mos_full.log"
mkdir -p "$LOG_DIR"
: > "$LOG"

echo "=== Full Quality Eval (4 metrics, all files) ===" >> "$LOG"
echo "OUT: $OUT_DIR" >> "$LOG"
echo "GPUs: 0,1 | Workers: 4 (2/GPU)" >> "$LOG"
echo "Metrics: UTMOS22Strong, SCOREQ, TTSDS2, UTMOSv2" >> "$LOG"
echo "Start: $(date)" >> "$LOG"
echo "============================================" >> "$LOG"

cd /root/code/github_repos/OmniVoice-fork/batch_generate_text_and_clone/eval_mos

conda run -n omnivoice python eval_clone_mos.py \
  --out-dir "$OUT_DIR" \
  --gpus 0,1 \
  --workers 4 \
  --metrics UTMOS22Strong,SCOREQ,TTSDS2,UTMOSv2 \
  --model-dir /root/code/github_repos/OmniVoice-fork/TTS_eval_models \
  >> "$LOG" 2>&1

echo "=== DONE $(date) ===" >> "$LOG"
