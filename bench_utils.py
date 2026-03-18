"""
bench_utils.py
辅助函数：格式化统计并打印对比结果。
"""
import math


def format_stats(stats, label="", work_units: int = 0):
    s = []
    s.append(f"--- {label} 基准统计 ---")
    s.append(f"次数: {stats['count']}")
    s.append(f"平均耗时: {stats['mean']*1000:.3f} ms")
    s.append(f"中位耗时: {stats['median']*1000:.3f} ms")
    s.append(f"最短耗时: {stats['min']*1000:.3f} ms")
    s.append(f"最长耗时: {stats['max']*1000:.3f} ms")
    s.append(f"标准差: {stats['stdev']*1000:.3f} ms")
    if work_units and stats['mean'] > 0:
        # throughput in FLOPs per second
        t = work_units / stats['mean']
        s.append(f"吞吐量: {t/1e9:.3f} GFLOPS ({t:.0f} FLOP/s)")
    return "\n".join(s)


def compare_and_print(cpu_stats, npu_stats):
    cpu_mean = cpu_stats['mean']
    npu_mean = npu_stats['mean']
    if npu_mean <= 0:
        print("无法计算对比（npu 平均耗时为 0）")
        return
    speedup = cpu_mean / npu_mean
    diff_ms = (cpu_mean - npu_mean) * 1000
    diff_pct = (cpu_mean - npu_mean) / cpu_mean * 100 if cpu_mean != 0 else math.nan

    print(f"CPU 平均: {cpu_mean*1000:.3f} ms | NPU 平均: {npu_mean*1000:.3f} ms")
    print(f"Speedup (CPU / NPU): {speedup:.3f}x")
    print(f"绝对差异: {diff_ms:.3f} ms, 相对差异: {diff_pct:.2f}%")

    # 如果两边都有 work_units，可以比较 GFLOPS
    cpu_units = cpu_stats.get('work_units', 0)
    npu_units = npu_stats.get('work_units', 0)
    if cpu_units and npu_units:
        cpu_gflops = cpu_units / cpu_mean / 1e9
        npu_gflops = npu_units / npu_mean / 1e9
        print(f"CPU 吞吐量: {cpu_gflops:.3f} GFLOPS | NPU 吞吐量: {npu_gflops:.3f} GFLOPS")
        print(f"吞吐比 (NPU/CPU): {npu_gflops/cpu_gflops:.3f}x")
