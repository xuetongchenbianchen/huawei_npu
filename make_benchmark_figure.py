#!/usr/bin/env python3
"""从 benchmark CSV 生成可视化图。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def make_label(row):
    op = row["op"]
    if row.get("case_shape"):
        return f"{op}\nshape={row['case_shape']}"
    parts = []
    for key in ["case_batch", "case_in_c", "case_h", "case_w", "case_out_c", "case_k", "case_heads", "case_seq_len", "case_head_dim", "case_features", "case_hidden"]:
        value = row.get(key)
        if value not in ("", None):
            parts.append(f"{key.replace('case_', '')}={value}")
    extra = ", ".join(parts[:3])
    if len(parts) > 3:
        extra += ", ..."
    return f"{op}\n{extra}" if extra else op


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark figure from CSV")
    parser.add_argument("input_csv")
    parser.add_argument("output_png")
    args = parser.parse_args()

    rows = []
    with open(args.input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["speedup_cpu_div_npu"] = float(row["speedup_cpu_div_npu"])
            rows.append(row)

    labels = [make_label(row) for row in rows]
    values = [row["speedup_cpu_div_npu"] for row in rows]

    fig_h = max(8, len(rows) * 0.55)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    colors = ["#1f77b4" if row["precision_ok"] == "True" else "#d62728" for row in rows]
    bars = ax.barh(range(len(rows)), values, color=colors)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Speedup (CPU / NPU)")
    ax.set_title("Full Benchmark Rerun: Speedup by Operator Case")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()

    for bar, value in zip(bars, values):
        ax.text(value + max(values) * 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.1f}x", va="center", fontsize=8)

    fig.tight_layout()
    output = Path(args.output_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
