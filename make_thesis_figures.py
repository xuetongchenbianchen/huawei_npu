#!/usr/bin/env python3
"""生成可直接放入论文的图表。"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _display_endpoint(meta, fallback):
    if not isinstance(meta, dict):
        return fallback
    return str(meta.get("device") or meta.get("label") or meta.get("module") or fallback).upper()


def _schema_info(payload):
    if isinstance(payload.get("reference"), dict) or isinstance(payload.get("target"), dict):
        return {
            "reference_key": "reference",
            "target_key": "target",
            "reference_label": _display_endpoint(payload.get("reference"), "Reference"),
            "target_label": _display_endpoint(payload.get("target"), "Target"),
        }
    return {
        "reference_key": "cpu",
        "target_key": "npu",
        "reference_label": "CPU",
        "target_label": "NPU",
    }


def _stats(item, key):
    stats = item.get(key)
    if stats is None:
        raise KeyError(f"result item for op={item.get('op')} missing stats field {key!r}")
    return stats


def _comparison_label(rows):
    if not rows:
        return ("Reference", "Target")
    reference_labels = sorted({row["reference_label"] for row in rows})
    target_labels = sorted({row["target_label"] for row in rows})
    reference = reference_labels[0] if len(reference_labels) == 1 else "Reference"
    target = target_labels[0] if len(target_labels) == 1 else "Target"
    return reference, target


def _case_compact_label(op, case):
    case = case or {}
    if op == "matmul" and "shape" in case:
        return f"{op}-{case['shape'][0]}"
    if op == "bmm":
        return f"{op}-b{case.get('batch', '?')}-m{case.get('m', '?')}-n{case.get('n', '?')}"
    if "shape" in case:
        shape = case["shape"]
        if isinstance(shape, list):
            shape = "x".join(str(v) for v in shape)
        return f"{op}-{shape}"
    if op == "conv2d":
        return f"{op}-b{case.get('batch', '?')}-h{case.get('h', '?')}"
    if op in {"maxpool2d", "avgpool2d"}:
        return f"{op}-b{case.get('batch', '?')}-h{case.get('h', '?')}"
    if op == "embedding":
        return f"{op}-v{case.get('vocab_size', '?')}-d{case.get('embedding_dim', '?')}"
    if op == "sdpa":
        return f"{op}-s{case.get('seq_len', '?')}"
    if op == "softmax":
        return f"{op}-f{case.get('features', '?')}"
    if op in {"layernorm", "rmsnorm"}:
        return f"{op}-h{case.get('hidden', '?')}"
    return op


def _save(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(pad=1.1)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_multi(fig, output_paths):
    output_paths = [Path(path) for path in output_paths]
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(pad=1.1)
    for output_path in output_paths:
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_benchmark_rows(paths):
    rows = []
    for path in paths:
        payload = _load_json(path)
        schema = _schema_info(payload)
        for item in payload.get("results", []):
            reference = _stats(item, schema["reference_key"])
            target = _stats(item, schema["target_key"])
            speedup = reference["mean"] / target["mean"] if target["mean"] else None
            rows.append(
                {
                    "source": str(path),
                    "op": item["op"],
                    "case": item.get("case") or {},
                    "label": _case_compact_label(item["op"], item.get("case")),
                    "reference_label": schema["reference_label"],
                    "target_label": schema["target_label"],
                    "reference_mean_ms": reference["mean"] * 1000,
                    "target_mean_ms": target["mean"] * 1000,
                    "cpu_mean_ms": reference["mean"] * 1000,
                    "npu_mean_ms": target["mean"] * 1000,
                    "speedup": speedup,
                    "precision_ok": (
                        item.get("precision", {}).get("ok")
                        if item.get("precision") is not None
                        else None
                    ),
                    "max_abs_err": (
                        item.get("precision", {}).get("max_abs_err")
                        if item.get("precision") is not None
                        else None
                    ),
                }
            )
    return rows


def plot_speedup_summary(rows, output_dir):
    reference_label, target_label = _comparison_label(rows)
    grouped = {}
    for row in rows:
        if row["speedup"] is not None:
            grouped.setdefault(row["op"], []).append(row["speedup"])

    ordered = sorted(
        ((op, sum(values) / len(values), len(values)) for op, values in grouped.items()),
        key=lambda item: item[1],
    )
    ops = [item[0] for item in ordered]
    values = [item[1] for item in ordered]
    counts = [item[2] for item in ordered]

    fig_h = max(6.5, len(ops) * 0.36)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    bars = ax.barh(ops, values, color="#d55e00")
    ax.set_xlabel(f"Average Speedup ({reference_label} / {target_label})")
    ax.set_title(f"{reference_label}/{target_label} Mean Speedup by Operator")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    max_value = max(values) if values else 1.0
    for bar, value, count in zip(bars, values, counts):
        ax.text(
            value + max_value * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}x  n={count}",
            ha="left",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(right=max_value * 1.16)
    _save_multi(
        fig,
        [
            output_dir / "speedup_summary.png",
            output_dir / "operator_average_speedup.png",
        ],
    )


def plot_case_speedup(rows, output_dir):
    reference_label, target_label = _comparison_label(rows)
    ordered = sorted([row for row in rows if row["speedup"] is not None], key=lambda item: item["speedup"], reverse=True)
    labels = [row["label"] for row in ordered]
    values = [row["speedup"] for row in ordered]
    colors = ["#009e73" if row["precision_ok"] is not False else "#cc3311" for row in ordered]

    fig_h = max(6, len(ordered) * 0.42)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    bars = ax.barh(range(len(ordered)), values, color=colors)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"Speedup ({reference_label} / {target_label})")
    ax.set_title(f"{reference_label}/{target_label} Speedup by Benchmark Case")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    max_value = max(values) if values else 1.0
    for bar, value in zip(bars, values):
        ax.text(
            value + max_value * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}x",
            va="center",
            fontsize=8,
        )
    _save(fig, output_dir / "speedup_by_case.png")


def plot_precision_scatter(rows, output_dir):
    reference_label, target_label = _comparison_label(rows)
    filtered = [row for row in rows if row["max_abs_err"] is not None and row["speedup"] is not None]
    if not filtered:
        return

    positive_errors = [row["max_abs_err"] for row in filtered if row["max_abs_err"] and row["max_abs_err"] > 0]
    error_floor = min(positive_errors) * 0.1 if positive_errors else 1e-12

    palette = {
        "matmul": "#0072b2",
        "bmm": "#332288",
        "add": "#56b4e9",
        "sub": "#88ccee",
        "mul": "#44aa99",
        "div": "#117733",
        "relu": "#999933",
        "gelu": "#ddcc77",
        "silu": "#cc6677",
        "sum": "#882255",
        "mean": "#aa4499",
        "max": "#6699cc",
        "transpose_contiguous": "#999999",
        "conv2d": "#e69f00",
        "maxpool2d": "#ee7733",
        "avgpool2d": "#cc3311",
        "embedding": "#000000",
        "where_mask": "#bbbbbb",
        "sdpa": "#009e73",
        "softmax": "#cc79a7",
        "layernorm": "#d55e00",
        "rmsnorm": "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for row in filtered:
        ax.scatter(
            row["speedup"],
            max(row["max_abs_err"], error_floor),
            s=72,
            color=palette.get(row["op"], "#444444"),
            alpha=0.85,
        )
        ax.text(row["speedup"], max(row["max_abs_err"], error_floor), row["label"], fontsize=7, alpha=0.75)

    ax.set_yscale("log")
    ax.set_ylim(bottom=error_floor * 0.5)
    ax.set_xlabel(f"Speedup ({reference_label} / {target_label})")
    ax.set_ylabel("Max Absolute Error")
    ax.set_title("Performance and Precision Trade-off")
    ax.grid(True, linestyle="--", alpha=0.25)
    _save(fig, output_dir / "precision_vs_speedup.png")


def plot_conv2d_precision(conv2d_data, output_dir):
    labels = [_case_compact_label("conv2d", item["case"]) for item in conv2d_data]
    max_err = [item["max_abs_err"] for item in conv2d_data]
    p99_err = [item["p99_abs_err"] for item in conv2d_data]
    mean_err = [item["mean_abs_err"] for item in conv2d_data]
    x = list(range(len(labels)))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width for i in x], max_err, width=width, label="max_abs_err", color="#d55e00")
    ax.bar(x, p99_err, width=width, label="p99_abs_err", color="#e69f00")
    ax.bar([i + width for i in x], mean_err, width=width, label="mean_abs_err", color="#0072b2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Absolute Error")
    ax.set_title("Conv2d Precision Breakdown")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    _save(fig, output_dir / "conv2d_precision_breakdown.png")


def load_case_study_rows(paths):
    rows = []
    for path in paths:
        payload = _load_json(path)
        rows.extend(payload.get("results", []))
    return rows


def plot_warmup_curves(rows, output_dir):
    warmup_rows = [row for row in rows if row.get("study") == "warmup"]
    if not warmup_rows:
        return

    fig, axes = plt.subplots(len(warmup_rows), 1, figsize=(9.5, max(4.2, len(warmup_rows) * 3.0)))
    if len(warmup_rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, warmup_rows):
        iterations = list(range(1, row["iterations"] + 1))
        ax.plot(iterations, row["cpu"]["durations_ms"], marker="o", label="CPU", color="#0072b2")
        ax.plot(iterations, row["npu"]["durations_ms"], marker="o", label="NPU", color="#d55e00")
        ax.set_title(f"Warmup Curve - {_case_compact_label(row['op'], row.get('case'))}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Latency (ms)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
    _save(fig, output_dir / "warmup_curves.png")


def plot_dtype_tradeoff(rows, output_dir):
    dtype_rows = [row for row in rows if row.get("study") == "dtype"]
    if not dtype_rows:
        return

    grouped = {}
    for row in dtype_rows:
        grouped.setdefault(row["op"], {})[row["dtype"]] = row

    ops = list(grouped.keys())
    fp32_values = [grouped[op].get("float32", {}).get("npu", {}).get("mean", 0.0) * 1000 for op in ops]
    fp16_values = [grouped[op].get("float16", {}).get("npu", {}).get("mean", 0.0) * 1000 for op in ops]
    x = list(range(len(ops)))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar([i - width / 2 for i in x], fp32_values, width=width, label="float32", color="#0072b2")
    bars = ax.bar([i + width / 2 for i in x], fp16_values, width=width, label="float16", color="#d55e00")
    ax.set_xticks(x)
    ax.set_xticklabels(ops)
    ax.set_ylabel("NPU Mean Latency (ms)")
    ax.set_title("NPU Latency by Data Type")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    for idx, op in enumerate(ops):
        fp16_row = grouped[op].get("float16")
        speedup = fp16_row.get("speedup_vs_npu_fp32") if fp16_row else None
        if speedup:
            ax.text(
                idx + width / 2,
                bars[idx].get_height(),
                f"{speedup:.2f}x",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    _save(fig, output_dir / "dtype_latency_tradeoff.png")


def plot_scaling_curves(rows, output_dir):
    scaling_rows = [row for row in rows if row.get("study") == "scaling"]
    if not scaling_rows:
        return

    ops = sorted({row["op"] for row in scaling_rows})
    fig, axes = plt.subplots(len(ops), 1, figsize=(9.5, max(4.0, len(ops) * 3.0)))
    if len(ops) == 1:
        axes = [axes]

    for ax, op in zip(axes, ops):
        subset = sorted(
            [row for row in scaling_rows if row["op"] == op],
            key=lambda item: item["scale_value"],
        )
        x = [row["scale_value"] for row in subset]
        cpu_ms = [row["cpu"]["mean"] * 1000 for row in subset]
        npu_ms = [row["npu"]["mean"] * 1000 for row in subset]
        speedup = [row["speedup_cpu_div_npu"] for row in subset]

        ax.plot(x, cpu_ms, marker="o", label="CPU latency", color="#0072b2")
        ax.plot(x, npu_ms, marker="o", label="NPU latency", color="#d55e00")
        ax2 = ax.twinx()
        ax2.plot(x, speedup, marker="s", linestyle="--", label="Speedup", color="#009e73")

        ax.set_xlabel(subset[0]["scale_label"])
        ax.set_ylabel("Latency (ms)")
        ax2.set_ylabel("Speedup (CPU / NPU)")
        ax.set_title(f"Scaling Study - {op}")
        ax.grid(True, linestyle="--", alpha=0.25)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")
    _save(fig, output_dir / "scaling_curves.png")


def main():
    parser = argparse.ArgumentParser(description="生成论文图表")
    parser.add_argument("--benchmark-json", action="append", default=[])
    parser.add_argument("--conv2d-precision-json", default="")
    parser.add_argument("--case-study-json", action="append", default=[])
    parser.add_argument("--output-dir", default="workspace/thesis_materials/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    benchmark_rows = load_benchmark_rows(args.benchmark_json)
    if benchmark_rows:
        plot_speedup_summary(benchmark_rows, output_dir)
        plot_case_speedup(benchmark_rows, output_dir)
        plot_precision_scatter(benchmark_rows, output_dir)

    if args.conv2d_precision_json:
        plot_conv2d_precision(_load_json(args.conv2d_precision_json), output_dir)

    case_study_rows = load_case_study_rows(args.case_study_json)
    if case_study_rows:
        plot_warmup_curves(case_study_rows, output_dir)
        plot_dtype_tradeoff(case_study_rows, output_dir)
        plot_scaling_curves(case_study_rows, output_dir)

    print(f"图表已输出到: {output_dir}")


if __name__ == "__main__":
    main()
