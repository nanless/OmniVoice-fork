#!/bin/bash
set -e
cd "$(dirname "$0")"

GPU="${GPU:-0}"
OUT_DIR="${CLONED_VOICES_ROOT:-/root/code/github_repos/OmniVoice-fork/batch_cloned_voices}"
MODEL_DIR="${TTS_EVAL_MODEL_DIR:-$(cd ../.. && pwd)/TTS_eval_models}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"

ARGS=(--out-dir "$OUT_DIR" --gpu "$GPU" --model-dir "$MODEL_DIR")
[ -n "$SAMPLE_SIZE" ] && ARGS+=(--sample-size "$SAMPLE_SIZE")

echo "=== Clone Audio UTMOS (MOS) ==="
echo "Env: omnivoice | Model: $MODEL_DIR/mos/utmos22_strong_step7459_v1.pt"
echo "Out: $OUT_DIR | GPU: $GPU"
[ -n "$SAMPLE_SIZE" ] && echo "Sample: $SAMPLE_SIZE"
echo "================================"

conda run -n omnivoice python eval_clone_mos.py "${ARGS[@]}" "$@"
