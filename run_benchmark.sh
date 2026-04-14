#!/bin/bash
# Simple benchmark script for dual-size KV cache

set -e

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
NUM_REQUESTS="${NUM_REQUESTS:-30}"
HANG_TIMEOUT="${HANG_TIMEOUT:-0}"
DEBUG_BENCHMARK="${DEBUG_BENCHMARK:-0}"

echo "=========================================="
echo "Dual-Size KV Cache Benchmark"
echo "=========================================="
echo "Model: $MODEL"
echo "Requests: $NUM_REQUESTS"
if [ "$HANG_TIMEOUT" != "0" ]; then
    echo "Hang timeout: ${HANG_TIMEOUT}s"
fi
if [ "$DEBUG_BENCHMARK" != "0" ]; then
    echo "Debug logging: enabled"
fi
echo ""

# Check if we have the custom vLLM built
echo "Checking vLLM installation..."
python -c "from vllm import LLM; print('vLLM OK')" || {
    echo "Error: vLLM not installed or not built"
    echo "Run: uv pip install -e . --torch-backend=auto"
    exit 1
}

# Run baseline (fixed 16)
echo ""
echo ">>> Running Fixed-16 baseline..."
python benchmark_dual_kv.py --baseline --model "$MODEL" --num-requests $NUM_REQUESTS \
    --hang-timeout "$HANG_TIMEOUT" $( [ "$DEBUG_BENCHMARK" != "0" ] && echo "--debug" )

# Run dual-size
echo ""
echo ">>> Running Dual-Size (16/32)..."
python benchmark_dual_kv.py --dual-kv --model "$MODEL" --num-requests $NUM_REQUESTS \
    --hang-timeout "$HANG_TIMEOUT" $( [ "$DEBUG_BENCHMARK" != "0" ] && echo "--debug" )

# Show final comparison
echo ""
echo "=========================================="
echo "Final Results"
echo "=========================================="
python -c "
import json
with open('benchmark_results.json') as f:
    results = json.load(f)
    
for r in results:
    print(f\"{r['config']:12} {r['throughput_tokens_per_sec']:6.2f} tok/s ({r['elapsed_seconds']:.1f}s)\")

if len(results) == 2:
    fixed = results[0]['throughput_tokens_per_sec']
    dual = results[1]['throughput_tokens_per_sec']
    improvement = (dual / fixed - 1) * 100
    print(f'')
    print(f'Improvement: {improvement:+.1f}%')
"

echo ""
echo "Full results in: benchmark_results.json"
