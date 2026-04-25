#!/usr/bin/env python3
"""分析 conv2d 在 CPU/NPU 之间的数值误差。"""
from __future__ import annotations

import argparse
import json

import numpy as np

import src.cpu as cpu
import src.npu as npu


DEFAULT_CASES = [
    {"batch": 1, "in_c": 3, "h": 32, "w": 32, "out_c": 8, "k": 3},
    {"batch": 4, "in_c": 16, "h": 64, "w": 64, "out_c": 32, "k": 3},
]


def analyze_case(case):
    shared = cpu.generate_input_conv2d(**case)
    cpu_out = cpu.compute_conv2d(shared).astype(np.float32)
    npu_out = npu.compute_conv2d(npu.materialize_input(shared))
    if hasattr(npu_out, "detach"):
        npu_out = npu_out.detach().cpu().numpy()
    npu_out = npu_out.astype(np.float32)

    diff = np.abs(cpu_out - npu_out)
    rel = diff / (np.abs(cpu_out) + 1e-8)
    cosine = float(
        np.dot(cpu_out.reshape(-1), npu_out.reshape(-1))
        / (np.linalg.norm(cpu_out.reshape(-1)) * np.linalg.norm(npu_out.reshape(-1)))
    )
    return {
        "case": case,
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "p99_abs_err": float(np.quantile(diff, 0.99)),
        "max_rel_err": float(rel.max()),
        "mean_rel_err": float(rel.mean()),
        "cosine_similarity": cosine,
        "allclose_atol_1e_2_rtol_1e_2": bool(np.allclose(cpu_out, npu_out, atol=1e-2, rtol=1e-2)),
        "allclose_atol_5e_3_rtol_5e_3": bool(np.allclose(cpu_out, npu_out, atol=5e-3, rtol=5e-3)),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze conv2d precision differences")
    parser.add_argument("--output", default="", help="optional json output path")
    args = parser.parse_args()

    results = [analyze_case(case) for case in DEFAULT_CASES]
    print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
