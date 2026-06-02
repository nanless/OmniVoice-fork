#!/bin/bash
# Run eval_cloned.py with multi-GPU support
# Usage: bash run_eval_cloned_multi.sh [args...]

CONDA_ENV="qwen3-asr"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

# Register qwen3_5 config first
conda run -n $CONDA_ENV python3 -c "
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
try:
    CONFIG_MAPPING.register('qwen3_5', Qwen3_5Config)
except ValueError:
    pass
"

# Run eval_cloned.py
conda run -n $CONDA_ENV python eval_cloned.py "$@"
