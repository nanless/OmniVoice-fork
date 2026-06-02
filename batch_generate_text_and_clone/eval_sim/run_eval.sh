#!/bin/bash
set -e
cd "$(dirname "$0")"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

GPU="${GPU:-0}"
OUT_DIR="${CLONED_VOICES_ROOT:-/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"

ARGS=(--out-dir "$OUT_DIR" --gpu "$GPU")
[ -n "$SAMPLE_SIZE" ] && ARGS+=(--sample-size "$SAMPLE_SIZE")

echo "=== Clone vs Ref Speaker Similarity ==="
echo "Env: omnivoice | Model: $(pwd)/model"
echo "Out: $OUT_DIR | GPU: $GPU"
[ -n "$SAMPLE_SIZE" ] && echo "Sample: $SAMPLE_SIZE"
echo "========================================"

LOG_FILE="$OUT_DIR/logs/eval_sim_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUT_DIR/logs"

eval "$(conda shell.bash hook)"
conda activate omnivoice
export PYTHONUNBUFFERED=1
echo "Scanning clone pairs (this may take a while)..."
python -u eval_clone_similarity.py "${ARGS[@]}" "$@" 2>&1 | tee "$LOG_FILE"
