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


def _plot_sweep_lines(results, out_dir):
    """Generate input-length sweep plots when workload metadata is available."""
    sweep_rows = [
        r for r in results
        if (r.get("workload") or {}).get("input_len") is not None
    ]
    if not sweep_rows:
        return

    grouped = {}
    for row in sweep_rows:
        grouped.setdefault(row["config"], []).append(row)

    ordered_names = [r["config"] for r in results if r["config"] in grouped]
    seen = set()
    ordered_names = [n for n in ordered_names if not (n in seen or seen.add(n))]
    colors = {name: COLORS[i % len(COLORS)] for i, name in enumerate(ordered_names)}

    for rows in grouped.values():
        rows.sort(key=lambda r: r["workload"]["input_len"])

    # Throughput vs input length.
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ordered_names:
        rows = grouped[name]
        xs = [r["workload"]["input_len"] for r in rows]
        ys = [r["throughput_tokens_per_sec"] for r in rows]
        ax.plot(xs, ys, marker="o", linewidth=2, label=name, color=colors[name])
    ax.set_xlabel("Input length")
    ax.set_ylabel("Tokens / second")
    ax.set_title("Throughput vs input length")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "throughput_vs_input_len.png"), dpi=150)
    plt.close(fig)

    # Tail waste vs input length.
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ordered_names:
        rows = grouped[name]
        xs = [r["workload"]["input_len"] for r in rows]
        ys = []
        for r in rows:
            kv = r.get("peak_kv_stats") or {}
            alloc = kv.get("total_allocated", 0)
            waste = kv.get("total_waste", 0)
            ys.append(waste / alloc * 100 if alloc else 0.0)
        ax.plot(xs, ys, marker="o", linewidth=2, label=name, color=colors[name])
    ax.set_xlabel("Input length")
    ax.set_ylabel("Tail waste (%)")
    ax.set_title("KV tail waste vs input length")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tail_waste_vs_input_len.png"), dpi=150)
    plt.close(fig)

    # Average _model_forward vs input length.
    if any(
        (r.get("execution_timing_stats") or {}).get("_model_forward")
        for r in sweep_rows
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in ordered_names:
            rows = grouped[name]
            xs = [r["workload"]["input_len"] for r in rows]
            ys = [
                (r.get("execution_timing_stats") or {})
                .get("_model_forward", {})
                .get("avg_ms", 0.0)
                for r in rows
            ]
            ax.plot(xs, ys, marker="o", linewidth=2, label=name,
                    color=colors[name])
        ax.set_xlabel("Input length")
        ax.set_ylabel("Average _model_forward (ms)")
        ax.set_title("Average _model_forward vs input length")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "model_forward_vs_input_len.png"),
                    dpi=150)
        plt.close(fig)


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

    _plot_sweep_lines(results, args.out)

    print(f"Saved figures to {args.out}/")


if __name__ == "__main__":
    main()
