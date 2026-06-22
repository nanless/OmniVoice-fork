#!/bin/bash
# ============================================================
#  500K 文本生成 — 4 GPU vLLM 并行
#
#  vLLM 服务: qwen3-asr conda 环境 (vllm 0.19.1 + torch 2.10)
#  文本生成:  omnivoice conda 环境
#  模型: Qwen3.6-27B-FP8（每卡 ~29GB / 80GB A800）
#  输出: batch_generated_text_500k/
#
#  用法:
#    bash run_500k_4gpu.sh              # 全流程（启动服务+生成）
#    bash run_500k_4gpu.sh --stage 3    # 跳过服务启动，只跑生成（服务已运行）
#    bash run_500k_4gpu.sh --stop       # 停止所有 vLLM 服务
# ============================================================
set -euo pipefail

# ── 配置 ──────────────────────────────────────────────
MODEL="/root/.cache/huggingface/hub/Qwen/Qwen3.6-27B-FP8"
SERVED_NAME="qwen3.6-27b"
PORTS=(8000 8001 8002 8003)
GPUS=(0 1 2 3)

VLLM_PY="/root/miniforge3/envs/qwen3-asr/bin/python"
GEN_PY="/root/miniforge3/envs/omnivoice/bin/python"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/batch_generated_text_500k"
GEN_DIR="$(cd "$(dirname "$0")" && pwd)"

TARGET=500000
BATCH_SIZE=10
MAX_WORKERS=80
SEED=12345

# ── 参数解析 ──────────────────────────────────────────
stage=2
if [[ "${1:-}" == "--stop" ]]; then
    echo "Stopping all vLLM servers..."
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null && echo "  Done" || echo "  No running servers"
    exit 0
fi
[[ "${1:-}" == "--stage" ]] && stage="${2:-2}"

# ── 清理残留 ──────────────────────────────────────────
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

# ── Stage 2: 启动 4 个 vLLM 实例 ──────────────────────
if [[ "$stage" -le 2 ]]; then
    echo "=== Stage 2: 启动 vLLM 服务 ==="
    echo "  Python: ${VLLM_PY}"
    echo "  模型: ${MODEL}"
    mkdir -p "${OUTPUT_DIR}/logs"

    VLLM_PIDS=()
    for i in "${!PORTS[@]}"; do
        port=${PORTS[$i]}
        gpu=${GPUS[$i]}

        echo "  启动 GPU${gpu} → port ${port}..."
        CUDA_VISIBLE_DEVICES=${gpu} ${VLLM_PY} -m vllm.entrypoints.openai.api_server \
            --model "${MODEL}" \
            --port "${port}" \
            --served-model-name "${SERVED_NAME}" \
            --max-model-len 32768 \
            --gpu-memory-utilization 0.90 \
            > "${OUTPUT_DIR}/logs/vllm_gpu${gpu}.log" 2>&1 &

        VLLM_PIDS+=($!)
        echo "    PID=${VLLM_PIDS[-1]}"
    done

    # 退出时自动杀 vLLM
    cleanup() {
        echo ""
        echo "清理 vLLM 进程..."
        for pid in "${VLLM_PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        wait 2>/dev/null
    }
    trap cleanup EXIT

    # 等待服务就绪
    echo ""
    echo "等待 vLLM 服务就绪..."
    max_wait=600
    for port in "${PORTS[@]}"; do
        elapsed=0
        while ! curl -s "http://localhost:${port}/health" &>/dev/null; do
            sleep 5
            elapsed=$((elapsed + 5))
            if [[ $elapsed -ge $max_wait ]]; then
                echo "  ✗ 端口 ${port} 超时 (${max_wait}s)"
                echo "  查看日志: tail -50 ${OUTPUT_DIR}/logs/vllm_gpu*.log"
                exit 1
            fi
            printf "  port ${port}: 等待中 (%ds)...\r" $elapsed
        done
        echo "  ✓ port ${port} 就绪 (${elapsed}s)              "
    done
    echo "全部 ${#PORTS[@]} 个 vLLM 实例就绪"
    echo ""
fi

# ── Stage 3: 生成文本 ──────────────────────────────────
echo "=== Stage 3: 生成 ${TARGET} 条文本 ==="
echo "  输出目录: ${OUTPUT_DIR}"
echo "  模型: ${SERVED_NAME}"
echo "  端口: ${PORTS[*]}"
echo "  MAX_WORKERS: ${MAX_WORKERS}"

URLS=""
for port in "${PORTS[@]}"; do
    [[ -n "$URLS" ]] && URLS="${URLS},"
    URLS="${URLS}http://localhost:${port}/v1"
done

cd "${GEN_DIR}"

LLM_MODEL="${SERVED_NAME}" \
LLM_API_KEY=EMPTY \
LLM_BASE_URLS="${URLS}" \
GEN_TOTAL_TARGET="${TARGET}" \
GEN_BATCH_SIZE="${BATCH_SIZE}" \
GEN_MAX_WORKERS="${MAX_WORKERS}" \
GEN_SEED="${SEED}" \
GEN_OUTPUT_DIR="${OUTPUT_DIR}" \
GEN_OVERSAMPLE_RATIO=1.20 \
${GEN_PY} run_100k_asr_complete.py 2>&1 | tee "${OUTPUT_DIR}/generation.log"

echo ""
echo "=== 完成 ==="
echo "输出: ${OUTPUT_DIR}/llm_children_100k_asr_complete.jsonl"
wc -l "${OUTPUT_DIR}/llm_children_100k_asr_complete.jsonl" 2>/dev/null || true
