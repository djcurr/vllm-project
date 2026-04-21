#!/usr/bin/env python3
"""Plot benchmark_dual_kv.py results for presentation slides.

Usage:
    python plot_benchmark.py benchmark_results.json
    python plot_benchmark.py benchmark_results.json --out figures/
"""

import argparse
import json
import os
import matplotlib.pyplot as plt


COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="benchmark_results.json")
    parser.add_argument("--out", default="figures", help="output directory")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    os.makedirs(args.out, exist_ok=True)
    names  = [r["config"] for r in results]
    colors = COLORS[: len(names)]

    # Throughput
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [r["throughput_tokens_per_sec"] for r in results]
    bars = ax.bar(names, vals, color=colors, width=0.5, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f", padding=4)
    ax.set_ylabel("Tokens / second")
    ax.set_title("Throughput")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "throughput.png"), dpi=150)
    plt.close(fig)

    # Tail Waste
    waste_vals = []
    for r in results:
        kv = r.get("peak_kv_stats") or {}
        alloc = kv.get("total_allocated", 0)
        waste_vals.append(kv.get("total_waste", 0) / alloc * 100 if alloc else 0.0)

    if any(v > 0 for v in waste_vals):
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(names, waste_vals, color=colors, width=0.5, edgecolor="white")
        ax.bar_label(bars, fmt="%.1f%%", padding=4)
        ax.set_ylabel("Tail waste (% of allocated KV tokens)")
        ax.set_title("KV cache tail waste")
        ax.set_ylim(0, max(waste_vals) * 1.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "tail_waste.png"), dpi=150)
        plt.close(fig)

    # P95 step latency 
    fig, ax = plt.subplots(figsize=(6, 4))
    p95 = [r["step_metrics"]["p95_step_ms"] for r in results]
    bars = ax.bar(names, p95, color=colors, width=0.5, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f ms", padding=4)
    ax.set_ylabel("P95 step latency (ms)")
    ax.set_title("P95 step latency")
    ax.set_ylim(0, max(p95) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "latency_p95.png"), dpi=150)
    plt.close(fig)

    print(f"Saved figures to {args.out}/")


if __name__ == "__main__":
    main()
