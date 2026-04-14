#!/usr/bin/env python3
"""
Simple benchmark for dual-size KV cache vs fixed 16-token blocks.

Usage:
    # Fixed 16 baseline
    python benchmark_dual_kv.py --baseline
    
    # Dual-size (16/32)
    python benchmark_dual_kv.py --dual-kv
    
Environment variables:
    MODEL: Model name (default: meta-llama/Llama-2-7b-chat-hf)
    NUM_REQUESTS: Number of requests (default: 100)
"""

import time
import random
import json
import os
import sys
import argparse
import gc
import threading
import faulthandler
from dataclasses import dataclass
from typing import List

import torch

# vLLM imports
from vllm import LLM, SamplingParams


@dataclass
class Request:
    prompt: str
    input_len: int
    output_len: int


def generate_workload(num_requests: int = 100, short_ratio: float = 0.7) -> List[Request]:
    """Generate a mixed workload: 70% short, 30% long requests."""
    workload = []
    for i in range(num_requests):
        is_short = random.random() < short_ratio
        
        if is_short:
            input_len = random.randint(50, 150)
            output_len = random.randint(50, 100)
        else:
            input_len = random.randint(500, 1000)
            output_len = random.randint(300, 500)
        
        # Simple repeatable prompt
        prompt = " ".join(["word"] * input_len)
        workload.append(Request(prompt, input_len, output_len))
    
    return workload


def _dump_stacks(reason: str) -> None:
    print(f"\n[hang-dump] {reason}", file=sys.stderr, flush=True)
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


def _run_with_timeout(fn, timeout_s: float, label: str):
    if timeout_s <= 0:
        return fn()

    result = {"value": None, "error": None}

    def _target():
        try:
            result["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=_target, name=f"benchmark-{label}")
    thread.start()
    thread.join(timeout_s)

    if thread.is_alive():
        _dump_stacks(f"timeout after {timeout_s:.0f}s during {label}")
        raise TimeoutError(f"Timed out after {timeout_s:.0f}s during {label}")

    if result["error"] is not None:
        raise result["error"]
    return result["value"]


def benchmark(name: str, workload: List[Request], model: str, engine_args: dict,
             hang_timeout_s: float) -> dict:
    """Run benchmark with given engine args."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    # Create LLM
    llm = LLM(model=model, **engine_args)
    
    # Prepare prompts and sampling params
    prompts = [r.prompt for r in workload]
    max_output_len = max(r.output_len for r in workload)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_output_len,
    )
    
    # Warmup
    print("Warming up...")
    _run_with_timeout(lambda: llm.generate(prompts[:5], sampling_params),
                      hang_timeout_s, f"{name} warmup")
    
    # Actual benchmark
    print(f"Running {len(workload)} requests...")
    torch.cuda.synchronize() if hasattr(__import__('torch'), 'cuda') else None
    start_time = time.time()
    outputs = _run_with_timeout(lambda: llm.generate(prompts, sampling_params),
                                hang_timeout_s, f"{name} generate")
    end_time = time.time()
    
    # Calculate metrics
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    elapsed = end_time - start_time
    throughput = total_output_tokens / elapsed
    
    results = {
        "config": name,
        "num_requests": len(workload),
        "total_output_tokens": total_output_tokens,
        "elapsed_seconds": round(elapsed, 2),
        "throughput_tokens_per_sec": round(throughput, 2),
    }
    
    print(f"Results:")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Output tokens: {total_output_tokens}")
    print(f"  Throughput: {throughput:.2f} tok/s")
    
    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark dual-size KV cache")
    parser.add_argument("--baseline", action="store_true", help="Run fixed-16 baseline only")
    parser.add_argument("--dual-kv", action="store_true", help="Run dual-size only")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf", help="Model name")
    parser.add_argument("--num-requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--hang-timeout", type=float, default=0,
                        help="Seconds before dumping stacks and failing (0 disables)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose iteration logs")
    args = parser.parse_args()
    
    model = args.model
    num_requests = args.num_requests
    
    print("=" * 60)
    print("Dual-Size KV Cache Benchmark")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Requests: {num_requests}")
    
    # Generate workload once
    print("\nGenerating mixed workload (70% short, 30% long)...")
    workload = generate_workload(num_requests=num_requests, short_ratio=0.7)
    
    short_reqs = [r for r in workload if r.output_len <= 100]
    long_reqs = [r for r in workload if r.output_len > 100]
    print(f"  Short: {len(short_reqs)} ({len(short_reqs)/num_requests*100:.0f}%)")
    print(f"  Long:  {len(long_reqs)} ({len(long_reqs)/num_requests*100:.0f}%)")
    
    results = []
    hang_timeout_s = args.hang_timeout
    if hang_timeout_s > 0:
        faulthandler.enable()
    
    # Run benchmarks
    run_baseline = args.baseline or not args.dual_kv
    run_dual = args.dual_kv or not args.baseline
    
    if run_baseline:
        # Fixed 16 - standard config
        engine_args = {
            "block_size": 16,
            "max_num_seqs": 128,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": True,
            "enable_prefix_caching": False,
        }
        if args.debug:
            engine_args["enable_logging_iteration_details"] = True

        results.append(benchmark(
            name="Fixed-16",
            workload=workload,
            model=model,
            engine_args=engine_args,
            hang_timeout_s=hang_timeout_s,
        ))
    
    if run_dual:
        # Dual-size - need to use env var and the engine args from your implementation
        os.environ["VLLM_EXPERIMENTAL_DUAL_KV_MIXED_KERNEL"] = "1"
        
        engine_args = {
            "block_size": 16,
            "max_num_seqs": 128,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "experimental_dual_kv_blocks": True,
            "experimental_small_kv_block_size": 16,
            "experimental_large_kv_block_size": 32,
            "experimental_dual_kv_threshold_tokens": 256,
            "experimental_dual_kv_mixed_kernel": True,
            "experimental_small_kv_pool_fraction": 0.5,
        }
        if args.debug:
            engine_args["enable_logging_iteration_details"] = True

        results.append(benchmark(
            name="Dual-Size",
            workload=workload,
            model=model,
            engine_args=engine_args,
            hang_timeout_s=hang_timeout_s,
        ))
    
    # Summary
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        fixed_16 = results[0]
        dual_size = results[1]
        
        print(f"Fixed-16:     {fixed_16['throughput_tokens_per_sec']:.2f} tok/s")
        print(f"Dual-Size:    {dual_size['throughput_tokens_per_sec']:.2f} tok/s")
        
        improvement = (dual_size['throughput_tokens_per_sec'] / fixed_16['throughput_tokens_per_sec'] - 1) * 100
        print(f"Improvement:  {improvement:+.1f}%")
    
    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()
