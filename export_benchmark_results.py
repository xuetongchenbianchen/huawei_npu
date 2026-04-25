#!/usr/bin/env python3
"""将 benchmark JSON 结果导出为 CSV。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _display_endpoint(meta, fallback):
    if not isinstance(meta, dict):
        return fallback
    return str(meta.get("device") or meta.get("label") or meta.get("module") or fallback)


def resolve_schema(payload):
    reference_meta = payload.get("reference")
    target_meta = payload.get("target")
    if isinstance(reference_meta, dict) or isinstance(target_meta, dict):
        return {
            "reference_key": "reference",
            "target_key": "target",
            "reference_label": _display_endpoint(reference_meta, "reference"),
            "target_label": _display_endpoint(target_meta, "target"),
            "schema": "reference_target",
        }
    return {
        "reference_key": "cpu",
        "target_key": "npu",
        "reference_label": "cpu",
        "target_label": "npu",
        "schema": "cpu_npu",
    }


def _stats(item, key):
    stats = item.get(key)
    if stats is None:
        raise KeyError(f"result item for op={item.get('op')} missing stats field {key!r}")
    return stats


def _throughput_gflops(stats):
    mean = stats.get("mean")
    work_units = stats.get("work_units")
    if not mean or work_units is None:
        return None
    return work_units / mean / 1e9


def flatten_case(case):
    case = case or {}
    return {f"case_{k}": v for k, v in case.items()}


def main():
    parser = argparse.ArgumentParser(description="Export benchmark_results.json to CSV")
    parser.add_argument("input_path", nargs="?", help="benchmark_results.json path")
    parser.add_argument("output_path", nargs="?", help="output csv path")
    parser.add_argument("--input", dest="input_option", help="benchmark_results.json path")
    parser.add_argument("--output", dest="output_option", help="output csv path")
    args = parser.parse_args()

    input_path = args.input_option or args.input_path
    output_path = args.output_option or args.output_path
    if not input_path or not output_path:
        parser.error("input and output are required, either as positional args or --input/--output")

    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    schema = resolve_schema(data)
    rows = []
    for item in data["results"]:
        reference = _stats(item, schema["reference_key"])
        target = _stats(item, schema["target_key"])
        reference_mean = reference.get("mean")
        target_mean = target.get("mean")
        speedup = reference_mean / target_mean if reference_mean and target_mean else None
        precision = item.get("precision")

        row = {
            "op": item["op"],
            "case_index": item["case_index"],
            "schema": schema["schema"],
            "reference_label": schema["reference_label"],
            "target_label": schema["target_label"],
            "reference_mean_ms": reference_mean * 1000 if reference_mean is not None else None,
            "target_mean_ms": target_mean * 1000 if target_mean is not None else None,
            "reference_gflops": _throughput_gflops(reference),
            "target_gflops": _throughput_gflops(target),
            "speedup_reference_div_target": speedup,
            "precision_ok": precision["ok"] if precision else None,
            "max_abs_err": precision.get("max_abs_err") if precision else None,
            "mean_abs_err": precision.get("mean_abs_err") if precision else None,
            "cosine_similarity": precision.get("cosine_similarity") if precision else None,
        }
        if schema["schema"] == "cpu_npu":
            row.update(
                {
                    "cpu_mean_ms": row["reference_mean_ms"],
                    "npu_mean_ms": row["target_mean_ms"],
                    "cpu_gflops": row["reference_gflops"],
                    "npu_gflops": row["target_gflops"],
                    "speedup_cpu_div_npu": row["speedup_reference_div_target"],
                }
            )
        row.update(flatten_case(item.get("case")))
        rows.append(row)

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
