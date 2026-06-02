#!/bin/bash
set -e
cd "$(dirname "$0")"

GPU="${GPU:-0}"
OUT_DIR="${CLONED_VOICES_ROOT:-/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned}"
MODEL_DIR="${TTS_EVAL_MODEL_DIR:-$(cd ../.. && pwd)/TTS_eval_models}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"
METRICS="${METRICS:-}"

ARGS=(--out-dir "$OUT_DIR" --gpus "$GPU")
[ -n "$MODEL_DIR" ] && ARGS+=(--model-dir "$MODEL_DIR")
[ -n "$SAMPLE_SIZE" ] && ARGS+=(--sample-size "$SAMPLE_SIZE")
[ -n "$METRICS" ] && ARGS+=(--metrics "$METRICS")

echo "=== Clone Audio Quality Evaluation ==="
echo "Metrics: ${METRICS:-UTMOS22Strong,SCOREQ,TTSDS2,UTMOSv2 (all available)}"
echo "Out: $OUT_DIR | GPU: $GPU"
[ -n "$MODEL_DIR" ] && echo "Models: $MODEL_DIR"
[ -n "$SAMPLE_SIZE" ] && echo "Sample: $SAMPLE_SIZE"
echo "========================================"

conda run -n omnivoice python eval_clone_mos.py "${ARGS[@]}" "$@"
