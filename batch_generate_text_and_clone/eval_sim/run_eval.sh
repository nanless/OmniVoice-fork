#!/bin/bash
set -e
cd "$(dirname "$0")"

GPU="${GPU:-0}"
OUT_DIR="${CLONED_VOICES_ROOT:-/root/code/github_repos/OmniVoice-fork/batch_cloned_voices}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"

ARGS=(--out-dir "$OUT_DIR" --gpu "$GPU")
[ -n "$SAMPLE_SIZE" ] && ARGS+=(--sample-size "$SAMPLE_SIZE")

echo "=== Clone vs Ref Speaker Similarity ==="
echo "Env: omnivoice | Model: $(pwd)/model"
echo "Out: $OUT_DIR | GPU: $GPU"
[ -n "$SAMPLE_SIZE" ] && echo "Sample: $SAMPLE_SIZE"
echo "========================================"

conda run -n omnivoice python eval_clone_similarity.py "${ARGS[@]}" "$@"
