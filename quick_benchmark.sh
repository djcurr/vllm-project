#!/bin/bash
# Quick benchmark: Fixed 16 vs Dual-Size KV blocks
# Usage: ./quick_benchmark.sh [model_name]

MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
NUM_REQUESTS="${NUM_REQUESTS:-30}"

export MODEL
export NUM_REQUESTS

echo "=========================================="
echo "Quick Benchmark: Fixed 16 vs Dual-Size KV"
echo "=========================================="
echo "Model: $MODEL"
echo "Requests: $NUM_REQUESTS"
echo ""

./run_benchmark.sh
