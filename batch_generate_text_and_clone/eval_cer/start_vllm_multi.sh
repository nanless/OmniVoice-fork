#!/bin/bash
# Start 3 vllm instances on GPUs 1-3 for LLM ITN (GPU 0 reserved for ASR)
# Usage: bash start_vllm_multi.sh

MODEL_PATH="/root/.cache/huggingface/hub/Qwen/Qwen3___6-27B-FP8"
CONDA_ENV="qwen3-asr"

# Register qwen3_5 config first
conda run -n $CONDA_ENV python3 -c "
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
try:
    CONFIG_MAPPING.register('qwen3_5', Qwen3_5Config)
except ValueError:
    pass
print('qwen3_5 registered')
"

# Start 3 vllm instances on GPUs 1-3 (GPU 0 reserved for ASR)
for GPU_ID in 1 2 3; do
    PORT=$((8000 + GPU_ID))
    echo "Starting vllm on GPU $GPU_ID, port $PORT..."
    tmux kill-session -t "vllm${GPU_ID}" 2>/dev/null
    tmux new-session -d -s "vllm${GPU_ID}" \
        "CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n $CONDA_ENV vllm serve $MODEL_PATH \
        --language-model-only \
        --enable-prefix-caching \
        --port $PORT \
        --gpu-memory-utilization 0.95 \
        --max-model-len 8192 \
        --max-num-seqs 16 \
        --served-model-name qwen3.6-27b \
        --default-chat-template-kwargs '{\"enable_thinking\": false}' \
        2>&1 | tee /tmp/vllm_qwen3.6_gpu${GPU_ID}.log"
done

echo "Waiting for all vllm instances to start..."
sleep 300

# Verify all instances
for GPU_ID in 1 2 3; do
    PORT=$((8000 + GPU_ID))
    STATUS=$(curl -s http://localhost:$PORT/v1/models 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
    if [ -n "$STATUS" ]; then
        echo "GPU $GPU_ID (port $PORT): OK ($STATUS)"
    else
        echo "GPU $GPU_ID (port $PORT): FAILED"
    fi
done
