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
from statistics import mean

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


def generate_fixed_workload(
    num_requests: int,
    input_len: int,
    output_len: int,
) -> List[Request]:
    """Generate a fixed-length workload for sequence-length sweeps."""
    prompt = " ".join(["word"] * input_len)
    return [
        Request(prompt=prompt, input_len=input_len, output_len=output_len)
        for _ in range(num_requests)
    ]


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


def _log_kv_memory_stats(llm, name: str) -> None:
    """Extract and print KV-cache tail-waste stats from the model runner."""
    stats = None
    exc_info = None
    try:
        engine = llm.llm_engine
        if hasattr(engine, "collective_rpc"):
            results = engine.collective_rpc("get_kv_memory_stats")
            if results:
                stats = results[0]
    except Exception as exc:
        exc_info = exc

    if stats is not None:
        total_allocated = stats["total_allocated"]
        total_actual = stats["total_actual"]
        total_waste = stats["total_waste"]
        by_class = stats["by_class"]
        print(f"  KV Memory Stats ({name}):")
        print(f"    Total allocated tokens: {total_allocated}")
        print(f"    Total actual tokens:    {total_actual}")
        print(f"    Total tail waste:       {total_waste}")
        if total_allocated > 0:
            print(f"    Waste %%:                 {total_waste / total_allocated * 100:.2f}%")
        for cls, st in by_class.items():
            print(f"    -- class={cls}, count={st['count']}, "
                  f"allocated={st['allocated']}, actual={st['actual']}, "
                  f"waste={st['waste']} ({st['waste'] / st['allocated'] * 100:.2f}%)")
        return

    runner = None
    try:
        engine = llm.llm_engine
        if hasattr(engine, "model_executor") and engine.model_executor is not None:
            me = engine.model_executor
            if hasattr(me, "driver_worker") and hasattr(me.driver_worker, "worker") and hasattr(me.driver_worker.worker, "model_runner"):
                runner = me.driver_worker.worker.model_runner
    except Exception as exc:
        if exc_info is None:
            exc_info = exc

    if runner is None:
        msg = f"  [KV stats] model_runner not found"
        if exc_info:
            msg += f" ({exc_info})"
        print(msg)
        return

    kernel_block_size = (
        runner._kernel_block_sizes[0]
        if hasattr(runner, "_kernel_block_sizes") and runner._kernel_block_sizes
        else 16
    )

    total_allocated = 0
    total_actual = 0
    total_waste = 0
    by_class: dict[str, dict] = {}

    for req in runner.requests.values():
        seq_len = req.num_computed_tokens
        logical_block_size = req.kv_block_size or kernel_block_size
        num_kernel_blocks = len(req.block_ids[0]) if req.block_ids else 0
        blocks_per_logical = max(1, logical_block_size // kernel_block_size)
        num_logical_blocks = num_kernel_blocks // blocks_per_logical
        allocated = num_logical_blocks * logical_block_size
        waste = max(0, allocated - seq_len)

        total_allocated += allocated
        total_actual += seq_len
        total_waste += waste

        size_class = req.kv_size_class or "default"
        if size_class not in by_class:
            by_class[size_class] = {
                "count": 0,
                "allocated": 0,
                "actual": 0,
                "waste": 0,
            }
        by_class[size_class]["count"] += 1
        by_class[size_class]["allocated"] += allocated
        by_class[size_class]["actual"] += seq_len
        by_class[size_class]["waste"] += waste

    print(f"  KV Memory Stats ({name}):")
    print(f"    Total allocated tokens: {total_allocated}")
    print(f"    Total actual tokens:    {total_actual}")
    print(f"    Total tail waste:       {total_waste}")
    if total_allocated > 0:
        print(f"    Waste %%:                 {total_waste / total_allocated * 100:.2f}%")
    for cls, st in by_class.items():
        print(f"    -- class={cls}, count={st['count']}, "
              f"allocated={st['allocated']}, actual={st['actual']}, "
              f"waste={st['waste']} ({st['waste'] / st['allocated'] * 100:.2f}%)")


def _get_kv_memory_stats(llm):
    try:
        engine = llm.llm_engine
        if hasattr(engine, "collective_rpc"):
            results = engine.collective_rpc("get_kv_memory_stats")
            if results:
                return results[0]
    except Exception:
        return None
    return None


def _reset_execution_timing_stats(llm) -> None:
    try:
        engine = llm.llm_engine
        if hasattr(engine, "collective_rpc"):
            engine.collective_rpc("reset_execution_timing_stats")
    except Exception:
        pass


def _get_execution_timing_stats(llm):
    try:
        engine = llm.llm_engine
        if hasattr(engine, "collective_rpc"):
            results = engine.collective_rpc("get_execution_timing_stats")
            if results:
                return results[0]
    except Exception:
        return None
    return None


def _get_promotion_stats(llm):
    try:
        engine = llm.llm_engine
        if hasattr(engine, "collective_rpc"):
            results = engine.collective_rpc("get_promotion_stats")
            if results:
                return results[0]
    except Exception:
        return None
    return None


def _print_kv_memory_stats(name: str, stats, suffix: str = "") -> None:
    if stats is None:
        print("  [KV stats] unavailable")
        return

    total_allocated = stats["total_allocated"]
    total_actual = stats["total_actual"]
    total_waste = stats["total_waste"]
    by_class = stats["by_class"]
    label = f"{name}{suffix}"
    print(f"  KV Memory Stats ({label}):")
    print(f"    Total allocated tokens: {total_allocated}")
    print(f"    Total actual tokens:    {total_actual}")
    print(f"    Total tail waste:       {total_waste}")
    if total_allocated > 0:
        print(f"    Waste %%:                 {total_waste / total_allocated * 100:.2f}%")
    for cls, st in by_class.items():
        print(f"    -- class={cls}, count={st['count']}, "
              f"allocated={st['allocated']}, actual={st['actual']}, "
              f"waste={st['waste']} ({st['waste'] / st['allocated'] * 100:.2f}%)")


def _print_execution_timing_stats(name: str, stats) -> None:
    if not stats:
        print("  [Execution timing] unavailable")
        return
    print(f"  Execution Timing ({name}):")
    for metric_name, metric_stats in stats.items():
        print(
            f"    -- {metric_name}: count={metric_stats['count']}, "
            f"total={metric_stats['total_ms']:.2f} ms, "
            f"avg={metric_stats['avg_ms']:.2f} ms, "
            f"max={metric_stats['max_ms']:.2f} ms"
        )


def _classify_request(expected_total_tokens: int, engine_args: dict) -> tuple[str, int]:
    if not engine_args.get("experimental_dual_kv_blocks", False):
        block_size = engine_args.get("block_size", 16)
        return "default", block_size

    small_block = engine_args["experimental_small_kv_block_size"]
    large_block = engine_args["experimental_large_kv_block_size"]
    configured_threshold = engine_args["experimental_dual_kv_threshold_tokens"]
    effective_large_threshold = max(configured_threshold, large_block * 32)
    if expected_total_tokens <= effective_large_threshold:
        return "small", small_block
    return "large", large_block


def _summarize_admission_policy(workload: List[Request], engine_args: dict) -> dict:
    by_class: dict[str, dict[str, float | int]] = {}
    for req in workload:
        expected_total = req.input_len + req.output_len
        size_class, block_size = _classify_request(expected_total, engine_args)
        stats = by_class.setdefault(
            size_class,
            {
                "count": 0,
                "expected_total_sum": 0,
                "min_expected_total": expected_total,
                "max_expected_total": expected_total,
                "block_size": block_size,
            },
        )
        stats["count"] += 1
        stats["expected_total_sum"] += expected_total
        stats["min_expected_total"] = min(stats["min_expected_total"], expected_total)
        stats["max_expected_total"] = max(stats["max_expected_total"], expected_total)

    return by_class


def _print_initial_admission_summary(name: str, summary: dict) -> None:
    print(f"  Initial Admission Summary ({name}):")
    for cls, stats in summary.items():
        count = stats["count"]
        avg_expected_total = stats["expected_total_sum"] / count
        print(
            f"    -- class={cls}, count={count}, block_size={stats['block_size']}, "
            f"expected_total avg={avg_expected_total:.1f} "
            f"min={stats['min_expected_total']} max={stats['max_expected_total']}"
        )


def _print_promotion_stats(name: str, stats) -> None:
    if not stats:
        print(f"  Runtime Promotion Stats ({name}): unavailable")
        return
    print(f"  Runtime Promotion Stats ({name}):")
    print(f"    Promotions:            {stats['count']}")
    print(f"    Kernel blocks copied:  {stats['kernel_blocks_copied']}")
    print(f"    Tokens copied:         {stats['logical_tokens_copied']}")


def _print_step_metrics(name: str, metrics: dict) -> None:
    step_ms = metrics["step_ms"]
    if not step_ms:
        print("  [Step metrics] unavailable")
        return
    sorted_step_ms = sorted(step_ms)
    p50 = sorted_step_ms[len(sorted_step_ms) // 2]
    p95 = sorted_step_ms[min(len(sorted_step_ms) - 1, int(len(sorted_step_ms) * 0.95))]
    print(f"  Step Metrics ({name}):")
    print(f"    Total steps:           {metrics['total_steps']}")
    print(f"    Output-producing steps:{metrics['output_steps']}")
    print(f"    Avg step time:         {mean(step_ms):.2f} ms")
    print(f"    P50 step time:         {p50:.2f} ms")
    print(f"    P95 step time:         {p95:.2f} ms")
    print(f"    Max step time:         {max(step_ms):.2f} ms")
    if metrics["first_output_s"] is not None:
        print(f"    Time to first output:  {metrics['first_output_s']:.3f} s")


def _step_generate(llm, prompts, sampling_params_list, collect_peak_stats: bool = False):
    engine = llm.llm_engine
    latest_outputs = {}
    peak_stats = None
    step_ms = []
    output_steps = 0
    first_output_s = None

    for i, prompt in enumerate(prompts):
        engine.add_request(str(i), prompt, sampling_params_list[i])

    loop_start = time.time()
    while engine.has_unfinished_requests():
        step_start = time.time()
        step_outputs = engine.step()
        step_elapsed_ms = (time.time() - step_start) * 1000
        step_ms.append(step_elapsed_ms)
        for out in step_outputs:
            latest_outputs[out.request_id] = out
        if step_outputs:
            output_steps += 1
            if first_output_s is None:
                first_output_s = time.time() - loop_start
        if collect_peak_stats:
            stats = _get_kv_memory_stats(llm)
            if stats is not None and (
                peak_stats is None
                or stats["total_allocated"] > peak_stats["total_allocated"]
            ):
                peak_stats = stats

    outputs = [latest_outputs[str(i)] for i in range(len(prompts))]
    step_metrics = {
        "total_steps": len(step_ms),
        "output_steps": output_steps,
        "step_ms": step_ms,
        "first_output_s": first_output_s,
    }
    return outputs, peak_stats, step_metrics


def benchmark(name: str, workload: List[Request], model: str, engine_args: dict,
             hang_timeout_s: float) -> dict:
    """Run benchmark with given engine args."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    # Create LLM
    llm = LLM(model=model, **engine_args)
    
    # Prepare prompts and per-request sampling params so the dual-KV
    # admission policy sees realistic output lengths.
    prompts = [r.prompt for r in workload]
    sampling_params_list = [
        SamplingParams(temperature=0.0, max_tokens=r.output_len)
        for r in workload
    ]
    admission_summary = _summarize_admission_policy(workload, engine_args)
    _print_initial_admission_summary(name, admission_summary)

    # Warmup
    print("Warming up...")
    _run_with_timeout(
        lambda: _step_generate(llm, prompts[:5], sampling_params_list[:5]),
        hang_timeout_s,
        f"{name} warmup",
    )
    _reset_execution_timing_stats(llm)

    # Actual benchmark
    print(f"Running {len(workload)} requests...")
    torch.cuda.synchronize() if hasattr(__import__('torch'), 'cuda') else None
    start_time = time.time()
    outputs, peak_kv_stats, step_metrics = _run_with_timeout(
        lambda: _step_generate(
            llm, prompts, sampling_params_list, collect_peak_stats=True
        ),
        hang_timeout_s,
        f"{name} generate",
    )
    end_time = time.time()
    _print_kv_memory_stats(name, peak_kv_stats, suffix=" peak live")
    _print_step_metrics(name, step_metrics)
    execution_timing_stats = _get_execution_timing_stats(llm)
    _print_execution_timing_stats(name, execution_timing_stats)
    promotion_stats = _get_promotion_stats(llm)
    _print_promotion_stats(name, promotion_stats)
    
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
        "admission_summary": admission_summary,
        "peak_kv_stats": peak_kv_stats,
        "execution_timing_stats": execution_timing_stats,
        "promotion_stats": promotion_stats,
        "step_metrics": {
            "total_steps": step_metrics["total_steps"],
            "output_steps": step_metrics["output_steps"],
            "avg_step_ms": round(mean(step_metrics["step_ms"]), 2)
            if step_metrics["step_ms"]
            else 0.0,
            "p50_step_ms": round(
                sorted(step_metrics["step_ms"])[len(step_metrics["step_ms"]) // 2], 2
            )
            if step_metrics["step_ms"]
            else 0.0,
            "p95_step_ms": round(
                sorted(step_metrics["step_ms"])[
                    min(
                        len(step_metrics["step_ms"]) - 1,
                        int(len(step_metrics["step_ms"]) * 0.95),
                    )
                ],
                2,
            )
            if step_metrics["step_ms"]
            else 0.0,
            "max_step_ms": round(max(step_metrics["step_ms"]), 2)
            if step_metrics["step_ms"]
            else 0.0,
            "first_output_s": round(step_metrics["first_output_s"], 4)
            if step_metrics["first_output_s"] is not None
            else None,
        },
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
    parser.add_argument("--short-ratio", type=float, default=0.7,
                        help="Fraction of requests that are short (default: 0.7)")
    parser.add_argument("--input-len", type=int, default=None,
                        help="Use a fixed prompt length for every request")
    parser.add_argument("--output-len", type=int, default=None,
                        help="Use a fixed max output length for every request")
    parser.add_argument("--results-file", default="benchmark_results.json",
                        help="Path to write benchmark results JSON")
    parser.add_argument("--append-results", action="store_true",
                        help="Append results to --results-file instead of overwriting")
    parser.add_argument("--hang-timeout", type=float, default=0,
                        help="Seconds before dumping stacks and failing (0 disables)")
    parser.add_argument("--debug", action="store_true", help="(deprecated) verbose iteration logs are now controlled by VLLM_LOGGING_LEVEL")
    args = parser.parse_args()
    
    model = args.model
    num_requests = args.num_requests
    short_ratio = args.short_ratio

    print("=" * 60)
    print("Dual-Size KV Cache Benchmark")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Requests: {num_requests}")

    # Generate workload once
    if (args.input_len is None) != (args.output_len is None):
        parser.error("--input-len and --output-len must be provided together")

    if args.input_len is not None:
        if args.input_len <= 0 or args.output_len <= 0:
            parser.error("--input-len and --output-len must be positive")
        print(f"Fixed input length:  {args.input_len}")
        print(f"Fixed output length: {args.output_len}")
        print("\nGenerating fixed-length workload...")
        workload = generate_fixed_workload(
            num_requests=num_requests,
            input_len=args.input_len,
            output_len=args.output_len,
        )
        print(f"  Requests: {len(workload)}")
        print(f"  Expected total tokens/request: {args.input_len + args.output_len}")
    else:
        print(f"Short ratio: {short_ratio:.0%}")
        print(f"\nGenerating workload ({short_ratio:.0%} short)...")
        workload = generate_workload(num_requests=num_requests, short_ratio=short_ratio)

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
        for block_size in (16, 32):
            results.append(benchmark(
                name=f"Fixed-{block_size}",
                workload=workload,
                model=model,
                engine_args={
                    "block_size": block_size,
                    "max_num_seqs": 128,
                    "gpu_memory_utilization": 0.85,
                    "enforce_eager": True,
                    "enable_prefix_caching": False,
                },
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
    if len(results) >= 2:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        for r in results:
            print(f"{r['config']:<16} {r['throughput_tokens_per_sec']:.2f} tok/s")

        # Show improvement of dual vs fixed-16 if both are present
        fixed_16 = next((r for r in results if r['config'] == 'Fixed-16'), None)
        dual_size = next((r for r in results if 'Dual' in r['config']), None)
        if fixed_16 and dual_size:
            improvement = (dual_size['throughput_tokens_per_sec'] / fixed_16['throughput_tokens_per_sec'] - 1) * 100
            print(f"Dual vs Fixed-16:  {improvement:+.1f}%")
    
    for result in results:
        result["workload"] = {
            "num_requests": num_requests,
            "short_ratio": None if args.input_len is not None else short_ratio,
            "input_len": args.input_len,
            "output_len": args.output_len,
        }

    # Save results
    if args.append_results and os.path.exists(args.results_file):
        with open(args.results_file) as f:
            existing_results = json.load(f)
        if not isinstance(existing_results, list):
            raise ValueError(
                f"Cannot append to {args.results_file}: expected a JSON list"
            )
        results = existing_results + results

    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.results_file}")


if __name__ == "__main__":
    main()
