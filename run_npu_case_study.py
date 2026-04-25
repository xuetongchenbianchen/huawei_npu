#!/usr/bin/env python3
"""面向论文展示的 NPU 专项实验脚本。

支持三类实验：
- warmup: 观察前若干次迭代的时延收敛情况
- dtype: 对比 float32/float16 在 NPU 上的性能与精度
- scaling: 观察输入规模增长时 CPU/NPU 的时延与加速比变化
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from bench_utils import validate_outputs
from run_benchmark import (
    benchmark,
    call_init,
    compute_output,
    generate_input,
    load_module,
    maybe_sync,
    precision_thresholds_for,
    prepare_backend_input,
    sanitize_for_json,
)


DEFAULT_WARMUP_CASES = {
    "matmul": {"shape": (1024, 1024)},
    "bmm": {"batch": 16, "m": 128, "k": 256, "n": 128},
    "gelu": {"shape": (2048, 2048)},
    "sum": {"shape": (2048, 2048), "axis": 1},
    "conv2d": {"batch": 4, "in_c": 16, "h": 64, "w": 64, "out_c": 32, "k": 3},
    "maxpool2d": {"batch": 8, "channels": 32, "h": 128, "w": 128, "k": 2},
    "embedding": {"vocab_size": 16384, "embedding_dim": 512, "batch": 32, "seq_len": 256},
    "sdpa": {"batch": 4, "heads": 8, "seq_len": 256, "head_dim": 64},
}

DEFAULT_DTYPE_CASES = {
    "matmul": {"shape": (1024, 1024)},
    "bmm": {"batch": 16, "m": 128, "k": 256, "n": 128},
    "mul": {"shape": (2048, 2048)},
    "gelu": {"shape": (2048, 2048)},
    "sum": {"shape": (2048, 2048), "axis": 1},
    "conv2d": {"batch": 4, "in_c": 16, "h": 64, "w": 64, "out_c": 32, "k": 3},
    "maxpool2d": {"batch": 8, "channels": 32, "h": 128, "w": 128, "k": 2},
    "embedding": {"vocab_size": 16384, "embedding_dim": 512, "batch": 32, "seq_len": 256},
    "sdpa": {"batch": 4, "heads": 8, "seq_len": 256, "head_dim": 64},
    "softmax": {"batch": 64, "features": 8192},
    "layernorm": {"batch": 32, "seq_len": 256, "hidden": 1024},
    "rmsnorm": {"batch": 32, "seq_len": 256, "hidden": 1024},
}

DEFAULT_SCALING_CASES = {
    "matmul": [
        {"shape": (256, 256)},
        {"shape": (512, 512)},
        {"shape": (1024, 1024)},
        {"shape": (1536, 1536)},
    ],
    "bmm": [
        {"batch": 4, "m": 64, "k": 128, "n": 64},
        {"batch": 8, "m": 128, "k": 128, "n": 128},
        {"batch": 16, "m": 128, "k": 256, "n": 128},
    ],
    "gelu": [
        {"shape": (512, 512)},
        {"shape": (1024, 1024)},
        {"shape": (2048, 2048)},
    ],
    "sum": [
        {"shape": (512, 512), "axis": 1},
        {"shape": (1024, 1024), "axis": 1},
        {"shape": (2048, 2048), "axis": 1},
    ],
    "conv2d": [
        {"batch": 1, "in_c": 3, "h": 32, "w": 32, "out_c": 8, "k": 3},
        {"batch": 2, "in_c": 8, "h": 64, "w": 64, "out_c": 16, "k": 3},
        {"batch": 4, "in_c": 16, "h": 96, "w": 96, "out_c": 32, "k": 3},
        {"batch": 4, "in_c": 16, "h": 128, "w": 128, "out_c": 32, "k": 3},
    ],
    "sdpa": [
        {"batch": 2, "heads": 4, "seq_len": 64, "head_dim": 64},
        {"batch": 2, "heads": 4, "seq_len": 128, "head_dim": 64},
        {"batch": 4, "heads": 8, "seq_len": 256, "head_dim": 64},
        {"batch": 4, "heads": 8, "seq_len": 512, "head_dim": 64},
    ],
}


def _parse_ops(raw_ops):
    return [item.strip() for item in raw_ops.split(",") if item.strip()]


def _np_dtype(dtype_name):
    if dtype_name == "float16":
        return np.float16
    if dtype_name == "float64":
        return np.float64
    return np.float32


def _cast_shared_input(value, dtype_name):
    if isinstance(value, tuple):
        return tuple(_cast_shared_input(item, dtype_name) for item in value)
    if isinstance(value, list):
        return [_cast_shared_input(item, dtype_name) for item in value]
    if isinstance(value, np.ndarray):
        return value.astype(_np_dtype(dtype_name), copy=False)
    if hasattr(value, "to") and hasattr(value, "dtype"):
        target_dtype = getattr(value, dtype_name, None)
        if target_dtype is not None:
            return value.to(dtype=target_dtype)
    return value


def _measure_iteration_series(mod, op, inp, iterations):
    durations = []
    for _ in range(iterations):
        maybe_sync(mod)
        t0 = time.perf_counter()
        compute_output(mod, op, inp)
        maybe_sync(mod)
        t1 = time.perf_counter()
        durations.append(t1 - t0)
    return durations


def _warmup_metrics(durations):
    first = durations[0]
    stable_window = durations[-5:] if len(durations) >= 5 else durations
    stable_mean = statistics.mean(stable_window)
    return {
        "first_iter_ms": first * 1000,
        "stable_mean_ms": stable_mean * 1000,
        "best_ms": min(durations) * 1000,
        "warmup_gain": first / stable_mean if stable_mean else None,
        "durations_ms": [value * 1000 for value in durations],
    }


def run_warmup_study(cpu_mod, npu_mod, ops, iterations):
    results = []
    for op in ops:
        case = DEFAULT_WARMUP_CASES[op]
        shared_input = generate_input(cpu_mod, op, case)
        cpu_inp = prepare_backend_input(cpu_mod, shared_input)
        npu_inp = prepare_backend_input(npu_mod, shared_input)

        cpu_durations = _measure_iteration_series(cpu_mod, op, cpu_inp, iterations)
        npu_durations = _measure_iteration_series(npu_mod, op, npu_inp, iterations)

        results.append(
            {
                "study": "warmup",
                "op": op,
                "case": sanitize_for_json(case),
                "iterations": iterations,
                "cpu": sanitize_for_json(_warmup_metrics(cpu_durations)),
                "npu": sanitize_for_json(_warmup_metrics(npu_durations)),
            }
        )
    return results


def run_dtype_study(cpu_mod, npu_mod, ops, dtypes, repeat, warmup, atol, rtol):
    results = []
    for op in ops:
        case = DEFAULT_DTYPE_CASES[op]
        shared_input = generate_input(cpu_mod, op, case)
        cpu_ref = prepare_backend_input(cpu_mod, shared_input)
        cpu_output = compute_output(cpu_mod, op, cpu_ref)

        dtype_rows = []
        for dtype_name in dtypes:
            candidate_input = _cast_shared_input(shared_input, dtype_name)
            npu_inp = prepare_backend_input(npu_mod, candidate_input)
            stats = benchmark(npu_mod, op, repeat=repeat, warmup=warmup, inp=npu_inp)

            report = None
            try:
                npu_output = compute_output(npu_mod, op, npu_inp)
                thresholds = precision_thresholds_for(op, atol, rtol)
                report = validate_outputs(
                    cpu_output,
                    npu_output,
                    atol=thresholds["atol"],
                    rtol=thresholds["rtol"],
                )
            except Exception as exc:
                report = {"ok": False, "reason": f"precision_failed: {type(exc).__name__}: {exc}"}

            dtype_rows.append(
                {
                    "study": "dtype",
                    "op": op,
                    "case": sanitize_for_json(case),
                    "dtype": dtype_name,
                    "npu": sanitize_for_json({k: v for k, v in stats.items() if k != "last_output"}),
                    "precision": sanitize_for_json(report),
                }
            )

        fp32_mean = next(
            (row["npu"]["mean"] for row in dtype_rows if row["dtype"] == "float32"),
            None,
        )
        for row in dtype_rows:
            mean_value = row["npu"]["mean"]
            row["speedup_vs_npu_fp32"] = fp32_mean / mean_value if fp32_mean and mean_value else None
            results.append(row)
    return results


def _case_scale_value(op, case):
    if op == "matmul":
        return int(case["shape"][0])
    if op == "bmm":
        return int(case["batch"] * case["m"] * case["n"])
    if op in {"gelu", "sum"}:
        return int(case["shape"][0])
    if op == "conv2d":
        return int(case["h"])
    if op == "sdpa":
        return int(case["seq_len"])
    return 0


def _case_scale_label(op):
    if op == "matmul":
        return "matrix_size"
    if op == "bmm":
        return "batch_m_n_product"
    if op in {"gelu", "sum"}:
        return "tensor_size"
    if op == "conv2d":
        return "spatial_size"
    if op == "sdpa":
        return "seq_len"
    return "scale"


def run_scaling_study(cpu_mod, npu_mod, ops, repeat, warmup, check_precision, atol, rtol):
    results = []
    for op in ops:
        for idx, case in enumerate(DEFAULT_SCALING_CASES[op]):
            shared_input = generate_input(cpu_mod, op, case)
            cpu_inp = prepare_backend_input(cpu_mod, shared_input)
            npu_inp = prepare_backend_input(npu_mod, shared_input)

            cpu_stats = benchmark(cpu_mod, op, repeat=repeat, warmup=warmup, inp=cpu_inp)
            npu_stats = benchmark(npu_mod, op, repeat=repeat, warmup=warmup, inp=npu_inp)

            precision = None
            if check_precision:
                cpu_output = compute_output(cpu_mod, op, cpu_inp)
                npu_output = compute_output(npu_mod, op, npu_inp)
                thresholds = precision_thresholds_for(op, atol, rtol)
                precision = validate_outputs(
                    cpu_output,
                    npu_output,
                    atol=thresholds["atol"],
                    rtol=thresholds["rtol"],
                )

            results.append(
                {
                    "study": "scaling",
                    "op": op,
                    "case_index": idx,
                    "case": sanitize_for_json(case),
                    "scale_label": _case_scale_label(op),
                    "scale_value": _case_scale_value(op, case),
                    "cpu": sanitize_for_json({k: v for k, v in cpu_stats.items() if k != "last_output"}),
                    "npu": sanitize_for_json({k: v for k, v in npu_stats.items() if k != "last_output"}),
                    "speedup_cpu_div_npu": cpu_stats["mean"] / npu_stats["mean"] if npu_stats["mean"] else None,
                    "precision": sanitize_for_json(precision),
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description="运行论文用 NPU 专项实验")
    parser.add_argument("--study", default="all", choices=["all", "warmup", "dtype", "scaling"])
    parser.add_argument("--cpu-module", default="src.cpu")
    parser.add_argument("--npu-module", default="src.npu")
    parser.add_argument("--warmup-ops", default="matmul,bmm,gelu,sum,conv2d,maxpool2d,embedding,sdpa")
    parser.add_argument("--dtype-ops", default="matmul,bmm,mul,gelu,sum,conv2d,maxpool2d,embedding,sdpa,softmax,layernorm,rmsnorm")
    parser.add_argument("--scaling-ops", default="matmul,bmm,gelu,sum,conv2d,sdpa")
    parser.add_argument("--dtype-list", default="float32,float16")
    parser.add_argument("--iterations", type=int, default=12, help="warmup 实验记录的迭代次数")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--check-precision", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--output", default="workspace/thesis_materials/results/npu_case_study.json")
    args = parser.parse_args()

    cpu_mod = load_module(args.cpu_module)
    npu_mod = load_module(args.npu_module)
    call_init(cpu_mod)
    call_init(npu_mod)

    studies = ["warmup", "dtype", "scaling"] if args.study == "all" else [args.study]
    payload = {
        "cpu_module": args.cpu_module,
        "npu_module": args.npu_module,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "results": [],
    }

    if "warmup" in studies:
        payload["results"].extend(
            run_warmup_study(cpu_mod, npu_mod, _parse_ops(args.warmup_ops), args.iterations)
        )
    if "dtype" in studies:
        payload["results"].extend(
            run_dtype_study(
                cpu_mod,
                npu_mod,
                _parse_ops(args.dtype_ops),
                _parse_ops(args.dtype_list),
                repeat=args.repeat,
                warmup=args.warmup,
                atol=args.atol,
                rtol=args.rtol,
            )
        )
    if "scaling" in studies:
        payload["results"].extend(
            run_scaling_study(
                cpu_mod,
                npu_mod,
                _parse_ops(args.scaling_ops),
                repeat=args.repeat,
                warmup=args.warmup,
                check_precision=args.check_precision,
                atol=args.atol,
                rtol=args.rtol,
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"实验结果已保存到: {output}")


if __name__ == "__main__":
    main()
