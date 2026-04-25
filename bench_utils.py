"""Benchmark 辅助函数。"""
from __future__ import annotations

import math

import numpy as np


def _to_numpy(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def format_stats(stats, label="", work_units: int = 0):
    lines = [f"--- {label} 基准统计 ---"]
    lines.append(f"次数: {stats['count']}")
    lines.append(f"平均耗时: {stats['mean'] * 1000:.3f} ms")
    lines.append(f"中位耗时: {stats['median'] * 1000:.3f} ms")
    lines.append(f"最短耗时: {stats['min'] * 1000:.3f} ms")
    lines.append(f"最长耗时: {stats['max'] * 1000:.3f} ms")
    lines.append(f"标准差: {stats['stdev'] * 1000:.3f} ms")
    if work_units and stats["mean"] > 0:
        throughput = work_units / stats["mean"]
        lines.append(f"吞吐量: {throughput / 1e9:.3f} GFLOPS ({throughput:.0f} FLOP/s)")
    return "\n".join(lines)


def compare_and_print(cpu_stats, npu_stats):
    cpu_mean = cpu_stats["mean"]
    npu_mean = npu_stats["mean"]
    if npu_mean <= 0:
        print("无法计算对比（NPU 平均耗时为 0）")
        return
    speedup = cpu_mean / npu_mean
    diff_ms = (cpu_mean - npu_mean) * 1000
    diff_pct = (cpu_mean - npu_mean) / cpu_mean * 100 if cpu_mean != 0 else math.nan

    print(f"CPU 平均: {cpu_mean * 1000:.3f} ms | NPU 平均: {npu_mean * 1000:.3f} ms")
    print(f"Speedup (CPU / NPU): {speedup:.3f}x")
    print(f"绝对差异: {diff_ms:.3f} ms, 相对差异: {diff_pct:.2f}%")

    cpu_units = cpu_stats.get("work_units", 0)
    npu_units = npu_stats.get("work_units", 0)
    if cpu_units and npu_units:
        cpu_gflops = cpu_units / cpu_mean / 1e9
        npu_gflops = npu_units / npu_mean / 1e9
        print(f"CPU 吞吐量: {cpu_gflops:.3f} GFLOPS | NPU 吞吐量: {npu_gflops:.3f} GFLOPS")
        print(f"吞吐比 (NPU/CPU): {npu_gflops / cpu_gflops:.3f}x")


def validate_outputs(reference, candidate, atol=1e-4, rtol=1e-3):
    ref = _to_numpy(reference).astype(np.float32, copy=False)
    cand = _to_numpy(candidate).astype(np.float32, copy=False)
    if ref.shape != cand.shape:
        return {
            "ok": False,
            "reason": "shape_mismatch",
            "reference_shape": tuple(int(v) for v in ref.shape),
            "candidate_shape": tuple(int(v) for v in cand.shape),
        }

    diff = np.abs(ref - cand)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    mean_abs = float(np.mean(diff)) if diff.size else 0.0

    ref_norm = float(np.linalg.norm(ref.reshape(-1))) if ref.size else 0.0
    cand_norm = float(np.linalg.norm(cand.reshape(-1))) if cand.size else 0.0
    if ref_norm == 0.0 or cand_norm == 0.0:
        cosine = 1.0 if ref_norm == cand_norm else 0.0
    else:
        cosine = float(np.dot(ref.reshape(-1), cand.reshape(-1)) / (ref_norm * cand_norm))

    return {
        "ok": bool(np.allclose(ref, cand, atol=atol, rtol=rtol)),
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "cosine_similarity": cosine,
        "reference_shape": tuple(int(v) for v in ref.shape),
        "candidate_shape": tuple(int(v) for v in cand.shape),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def format_precision_report(report):
    if report is None:
        return "未执行精度校验。"
    if not report.get("ok") and report.get("reason") == "shape_mismatch":
        return (
            "精度校验失败: 输出 shape 不一致 "
            f"{report['reference_shape']} vs {report['candidate_shape']}"
        )
    return (
        "精度校验: "
        f"{'通过' if report.get('ok') else '失败'} | "
        f"max_abs_err={report.get('max_abs_err', float('nan')):.6e} | "
        f"mean_abs_err={report.get('mean_abs_err', float('nan')):.6e} | "
        f"cosine={report.get('cosine_similarity', float('nan')):.6f} | "
        f"atol={report.get('atol', float('nan')):.6e} | "
        f"rtol={report.get('rtol', float('nan')):.6e}"
    )


def format_backend_info(info):
    parts = [f"[{info.get('label', 'backend')}]"]
    for key in [
        "device",
        "torch_available",
        "torch_npu_available",
        "npu_available",
        "npu_device_count",
        "torch_version",
    ]:
        if key in info:
            parts.append(f"{key}={info[key]}")
    return " ".join(parts)


def summarize_results(results):
    if not results:
        return "无结果。"

    lines = []
    for item in results:
        op = item["op"]
        case = item.get("case")
        cpu_mean = item["cpu"]["mean"] * 1000
        npu_mean = item["npu"]["mean"] * 1000
        speedup = item["cpu"]["mean"] / item["npu"]["mean"] if item["npu"]["mean"] > 0 else math.nan
        prefix = f"{op}"
        if case:
            prefix += f" {case}"
        line = f"{prefix}: CPU={cpu_mean:.3f}ms, NPU={npu_mean:.3f}ms, speedup={speedup:.3f}x"
        if item.get("precision"):
            line += f", precision={'ok' if item['precision'].get('ok') else 'fail'}"
        lines.append(line)
    return "\n".join(lines)
