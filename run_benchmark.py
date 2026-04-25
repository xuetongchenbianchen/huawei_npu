#!/usr/bin/env python3
"""统一多 device benchmark 入口。

支持：
- 基础算子与融合算子测试
- 参数扫描（如 matmul/conv2d/sdpa 不同规模）
- reference/target device 精度校验
- CPU/CUDA/NPU 后端设备状态检查
- 结果 JSON 落盘
"""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from copy import deepcopy
from pathlib import Path

from bench_utils import (
    format_backend_info,
    format_precision_report,
    format_stats,
    validate_outputs,
)
from src.device_registry import resolve_backend


DEFAULT_OPS = [
    "matmul",
    "bmm",
    "add",
    "sub",
    "mul",
    "div",
    "relu",
    "gelu",
    "silu",
    "sum",
    "mean",
    "max",
    "transpose_contiguous",
    "conv2d",
    "maxpool2d",
    "avgpool2d",
    "embedding",
    "where_mask",
    "sdpa",
    "softmax",
    "layernorm",
    "rmsnorm",
]

DEFAULT_PRECISION_THRESHOLDS = {
    "conv2d": {"atol": 1e-2, "rtol": 1e-2},
}

DEFAULT_SCAN_CASES = {
    "matmul": [
        {"shape": (256, 256)},
        {"shape": (512, 512)},
        {"shape": (1024, 1024)},
    ],
    "bmm": [
        {"batch": 8, "m": 128, "k": 128, "n": 128},
        {"batch": 16, "m": 128, "k": 256, "n": 128},
    ],
    "add": [
        {"shape": (1024, 1024)},
        {"shape": (2048, 2048)},
    ],
    "sub": [
        {"shape": (1024, 1024)},
        {"shape": (2048, 2048)},
    ],
    "mul": [
        {"shape": (1024, 1024)},
        {"shape": (2048, 2048)},
    ],
    "div": [
        {"shape": (1024, 1024)},
        {"shape": (2048, 2048)},
    ],
    "relu": [
        {"shape": (1024, 1024)},
        {"shape": (2048, 2048)},
    ],
    "gelu": [
        {"shape": (1024, 1024)},
        {"shape": (2048, 2048)},
    ],
    "silu": [
        {"shape": (1024, 1024)},
        {"shape": (2048, 2048)},
    ],
    "sum": [
        {"shape": (1024, 1024), "axis": 1},
        {"shape": (2048, 2048), "axis": 1},
    ],
    "mean": [
        {"shape": (1024, 1024), "axis": 1},
        {"shape": (2048, 2048), "axis": 1},
    ],
    "max": [
        {"shape": (1024, 1024), "axis": 1},
        {"shape": (2048, 2048), "axis": 1},
    ],
    "transpose_contiguous": [
        {"shape": (1024, 1024)},
        {"shape": (2048, 1024)},
    ],
    "conv2d": [
        {"batch": 1, "in_c": 3, "h": 32, "w": 32, "out_c": 8, "k": 3},
        {"batch": 4, "in_c": 16, "h": 64, "w": 64, "out_c": 32, "k": 3},
    ],
    "maxpool2d": [
        {"batch": 4, "channels": 16, "h": 64, "w": 64, "k": 2},
        {"batch": 8, "channels": 32, "h": 128, "w": 128, "k": 2},
    ],
    "avgpool2d": [
        {"batch": 4, "channels": 16, "h": 64, "w": 64, "k": 2},
        {"batch": 8, "channels": 32, "h": 128, "w": 128, "k": 2},
    ],
    "embedding": [
        {"vocab_size": 8192, "embedding_dim": 256, "batch": 32, "seq_len": 128},
        {"vocab_size": 16384, "embedding_dim": 512, "batch": 32, "seq_len": 256},
    ],
    "where_mask": [
        {"shape": (1024, 1024), "true_ratio": 0.5},
        {"shape": (2048, 2048), "true_ratio": 0.25},
    ],
    "sdpa": [
        {"batch": 2, "heads": 4, "seq_len": 128, "head_dim": 64},
        {"batch": 4, "heads": 8, "seq_len": 256, "head_dim": 64},
    ],
    "softmax": [
        {"batch": 32, "features": 4096},
        {"batch": 64, "features": 8192},
    ],
    "layernorm": [
        {"batch": 16, "seq_len": 128, "hidden": 768},
        {"batch": 32, "seq_len": 256, "hidden": 1024},
    ],
    "rmsnorm": [
        {"batch": 16, "seq_len": 128, "hidden": 768},
        {"batch": 32, "seq_len": 256, "hidden": 1024},
    ],
}


def load_module(path):
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError as exc:
        if exc.name == path or path.startswith(f"{exc.name}."):
            raise ModuleNotFoundError(
                f"无法导入后端模块 {path!r}。如果使用 --target-device cuda，"
                "请先确认 src.cuda 后端已实现；也可用 --target-module 指定自定义模块。"
            ) from exc
        raise


def call_init(mod):
    if hasattr(mod, "init"):
        mod.init()


def maybe_sync(mod):
    if hasattr(mod, "synchronize"):
        mod.synchronize()


def resolve_callable(mod, prefix, op):
    name = f"{prefix}_{op}"
    if hasattr(mod, name):
        return getattr(mod, name)
    return None


def generate_input(mod, op, case=None):
    generator = resolve_callable(mod, "generate_input", op)
    if generator is None:
        generator = getattr(mod, "generate_input", None)
    if generator is None:
        raise RuntimeError(f"模块 {mod.__name__} 未提供 generate_input_{op} 或 generate_input")
    case = case or {}
    return generator(**case)


def compute_output(mod, op, inp):
    compute_fn = resolve_callable(mod, "compute", op)
    if compute_fn is not None:
        return compute_fn(inp)

    run_fn = resolve_callable(mod, "run_once", op)
    if run_fn is not None:
        return run_fn(inp)

    if hasattr(mod, "run_once"):
        return mod.run_once(inp)
    if hasattr(mod, "run"):
        return mod.run(inp)
    raise RuntimeError(f"模块 {mod.__name__} 未实现 compute_{op} / run_once_{op} / run_once")


def workload_size(mod, op, inp):
    fn = resolve_callable(mod, "workload_size", op)
    if fn is not None:
        return fn(inp)
    fn = getattr(mod, "workload_size", None)
    if fn is not None:
        return fn(inp)
    return 0


def benchmark(mod, op, repeat=10, warmup=2, inp=None):
    run_target = lambda x: compute_output(mod, op, x)

    for _ in range(warmup):
        run_target(inp)
        maybe_sync(mod)

    durations = []
    last_output = None
    for _ in range(repeat):
        maybe_sync(mod)
        t0 = time.perf_counter()
        last_output = run_target(inp)
        maybe_sync(mod)
        t1 = time.perf_counter()
        durations.append(t1 - t0)

    return {
        "count": len(durations),
        "mean": statistics.mean(durations),
        "median": statistics.median(durations),
        "min": min(durations),
        "max": max(durations),
        "stdev": statistics.stdev(durations) if len(durations) > 1 else 0.0,
        "durations": durations,
        "work_units": workload_size(mod, op, inp),
        "last_output": last_output,
    }


def collect_backend_info(mod, label):
    if hasattr(mod, "backend_info"):
        return {"label": label, **mod.backend_info()}
    return {"label": label, "device": "unknown"}


def backend_label(device, role):
    device = (device or "").upper()
    return device if device else role


def compare_and_print_dynamic(reference_stats, target_stats, reference_label, target_label):
    reference_mean = reference_stats["mean"]
    target_mean = target_stats["mean"]
    if target_mean <= 0:
        print(f"无法计算对比（{target_label} 平均耗时为 0）")
        return
    speedup = reference_mean / target_mean
    diff_ms = (reference_mean - target_mean) * 1000
    diff_pct = (reference_mean - target_mean) / reference_mean * 100 if reference_mean != 0 else float("nan")

    print(
        f"{reference_label} 平均: {reference_mean * 1000:.3f} ms | "
        f"{target_label} 平均: {target_mean * 1000:.3f} ms"
    )
    print(f"Speedup ({reference_label} / {target_label}): {speedup:.3f}x")
    print(f"绝对差异: {diff_ms:.3f} ms, 相对差异: {diff_pct:.2f}%")

    reference_units = reference_stats.get("work_units", 0)
    target_units = target_stats.get("work_units", 0)
    if reference_units and target_units:
        reference_gflops = reference_units / reference_mean / 1e9
        target_gflops = target_units / target_mean / 1e9
        print(
            f"{reference_label} 吞吐量: {reference_gflops:.3f} GFLOPS | "
            f"{target_label} 吞吐量: {target_gflops:.3f} GFLOPS"
        )
        print(f"吞吐比 ({target_label}/{reference_label}): {target_gflops / reference_gflops:.3f}x")


def summarize_results_dynamic(results, reference_label, target_label):
    if not results:
        return "无结果。"

    lines = []
    for item in results:
        op = item["op"]
        case = item.get("case")
        reference_mean = item["reference"]["mean"] * 1000
        target_mean = item["target"]["mean"] * 1000
        speedup = (
            item["reference"]["mean"] / item["target"]["mean"]
            if item["target"]["mean"] > 0
            else float("nan")
        )
        prefix = f"{op}"
        if case:
            prefix += f" {case}"
        line = (
            f"{prefix}: {reference_label}={reference_mean:.3f}ms, "
            f"{target_label}={target_mean:.3f}ms, speedup={speedup:.3f}x"
        )
        if item.get("precision"):
            line += f", precision={'ok' if item['precision'].get('ok') else 'fail'}"
        lines.append(line)
    return "\n".join(lines)


def parse_ops(raw_ops):
    if raw_ops == "all":
        return list(DEFAULT_OPS)
    return [item.strip() for item in raw_ops.split(",") if item.strip()]


def scan_cases_for(op, enable_scan):
    if not enable_scan:
        return [None]
    return deepcopy(DEFAULT_SCAN_CASES.get(op, [None]))


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "shape"):
        try:
            return {"shape": [int(v) for v in value.shape], "type": type(value).__name__}
        except Exception:
            return str(value)
    return str(value)


def prepare_backend_input(mod, shared_input):
    if hasattr(mod, "materialize_input"):
        return mod.materialize_input(shared_input)
    return shared_input


def precision_thresholds_for(op, default_atol, default_rtol):
    thresholds = DEFAULT_PRECISION_THRESHOLDS.get(op, {})
    return {
        "atol": thresholds.get("atol", default_atol),
        "rtol": thresholds.get("rtol", default_rtol),
    }


def resolve_cli_backends(args):
    reference_module_override = args.reference_module or args.cpu_module
    target_module_override = args.target_module or args.npu_module
    reference_device, reference_module = resolve_backend(args.reference_device, reference_module_override)
    target_device, target_module = resolve_backend(args.target_device, target_module_override)
    return {
        "reference": {"device": reference_device, "module": reference_module},
        "target": {"device": target_device, "module": target_module},
    }


def main():
    parser = argparse.ArgumentParser(description="对比 reference 与 target device 实现的 benchmark 工具")
    parser.add_argument("--reference-device", default="cpu", help="参考设备类型，如 cpu/cuda/npu")
    parser.add_argument("--target-device", default="npu", help="目标设备类型，如 cpu/cuda/npu")
    parser.add_argument("--reference-module", default="", help="覆盖 reference device 默认后端模块")
    parser.add_argument("--target-module", default="", help="覆盖 target device 默认后端模块")
    parser.add_argument("--cpu-module", default="", help="兼容旧用法：等价于 --reference-module")
    parser.add_argument("--npu-module", default="", help="兼容旧用法：等价于 --target-module")
    parser.add_argument("--ops", default="all", help="算子集合，逗号分隔或 all")
    parser.add_argument("--repeat", type=int, default=10, help="测量重复次数")
    parser.add_argument("--warmup", type=int, default=2, help="预热次数")
    parser.add_argument("--scan", action="store_true", help="开启内置参数扫描")
    parser.add_argument("--check-precision", action="store_true", help="比较 reference/target 输出一致性")
    parser.add_argument("--rtol", type=float, default=1e-3, help="精度校验相对误差阈值")
    parser.add_argument("--atol", type=float, default=1e-4, help="精度校验绝对误差阈值")
    parser.add_argument("--save-json", default="", help="将结果保存到 JSON 文件")
    args = parser.parse_args()

    backend_meta = resolve_cli_backends(args)
    reference_meta = backend_meta["reference"]
    target_meta = backend_meta["target"]
    reference_label = backend_label(reference_meta["device"], "Reference")
    target_label = backend_label(target_meta["device"], "Target")

    reference_mod = load_module(reference_meta["module"])
    target_mod = load_module(target_meta["module"])
    call_init(reference_mod)
    call_init(target_mod)

    print("加载模块：", reference_meta["module"], target_meta["module"])
    print(format_backend_info(collect_backend_info(reference_mod, reference_label)))
    print(format_backend_info(collect_backend_info(target_mod, target_label)))

    ops = parse_ops(args.ops)
    all_results = []

    for op in ops:
        cases = scan_cases_for(op, args.scan)
        for idx, case in enumerate(cases):
            case_label = f"{op}#{idx + 1}" if case else op
            print(f"\n===== 算子: {case_label} =====")
            if case:
                print("参数:", sanitize_for_json(case))

            shared_input = generate_input(reference_mod, op, case)
            reference_inp = prepare_backend_input(reference_mod, shared_input)
            target_inp = prepare_backend_input(target_mod, shared_input)

            reference_stats = benchmark(
                reference_mod,
                op,
                repeat=args.repeat,
                warmup=args.warmup,
                inp=reference_inp,
            )
            print(
                format_stats(
                    reference_stats,
                    label=reference_label,
                    work_units=reference_stats.get("work_units", 0),
                )
            )

            target_stats = benchmark(
                target_mod,
                op,
                repeat=args.repeat,
                warmup=args.warmup,
                inp=target_inp,
            )
            print(format_stats(target_stats, label=target_label, work_units=target_stats.get("work_units", 0)))

            print("对比结果：")
            compare_and_print_dynamic(reference_stats, target_stats, reference_label, target_label)

            precision_report = None
            if args.check_precision:
                reference_output = compute_output(reference_mod, op, reference_inp)
                target_output = compute_output(target_mod, op, target_inp)
                thresholds = precision_thresholds_for(op, args.atol, args.rtol)
                precision_report = validate_outputs(
                    reference_output,
                    target_output,
                    atol=thresholds["atol"],
                    rtol=thresholds["rtol"],
                )
                print(format_precision_report(precision_report))

            reference_json = sanitize_for_json({k: v for k, v in reference_stats.items() if k != "last_output"})
            target_json = sanitize_for_json({k: v for k, v in target_stats.items() if k != "last_output"})
            all_results.append(
                {
                    "op": op,
                    "case_index": idx,
                    "case": sanitize_for_json(case),
                    "reference": reference_json,
                    "target": target_json,
                    "cpu": reference_json,
                    "npu": target_json,
                    "precision": sanitize_for_json(precision_report),
                }
            )

    print("\n===== 汇总 =====")
    print(summarize_results_dynamic(all_results, reference_label, target_label))

    if args.save_json:
        payload = {
            "reference": reference_meta,
            "target": target_meta,
            "cpu_module": reference_meta["module"],
            "npu_module": target_meta["module"],
            "legacy_alias_note": "cpu/npu result fields are compatibility aliases for reference/target.",
            "ops": ops,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "scan": args.scan,
            "check_precision": args.check_precision,
            "results": all_results,
        }
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
