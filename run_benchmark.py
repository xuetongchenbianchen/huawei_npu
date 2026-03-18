#!/usr/bin/env python3
"""
简单基准框架：对比 `src.cpu` 与 `src.npu` 实现的性能。

约定：被测模块应实现以下之一的函数：
 - `init()` (可选)：做一次性初始化
 - `generate_input()` (可选)：返回要传递给 `run_once` 的输入
 - `run_once(input)`：执行一次被测工作（必须）

示例用法：
  python3 run_benchmark.py --repeat 20 --warmup 3
"""
import time
import statistics
import importlib
import argparse
from bench_utils import format_stats, compare_and_print


def load_module(path):
    return importlib.import_module(path)


def call_init(mod):
    if hasattr(mod, "init"):
        try:
            mod.init()
        except Exception:
            pass


def get_input(mod):
    if hasattr(mod, "generate_input"):
        return mod.generate_input()
    return None


def run_once(mod, inp):
    # 支持 run_once 或 run
    if hasattr(mod, "run_once"):
        return mod.run_once(inp)
    if hasattr(mod, "run"):
        return mod.run(inp)
    raise RuntimeError("模块未实现 run_once 或 run")


def benchmark(mod, repeat=10, warmup=2, inp=None):
    call_init(mod)
    if inp is None:
        inp = get_input(mod)

    # warmup
    for _ in range(warmup):
        run_once(mod, inp)

    durations = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        run_once(mod, inp)
        t1 = time.perf_counter()
        durations.append(t1 - t0)

    stats = {
        "count": len(durations),
        "mean": statistics.mean(durations),
        "median": statistics.median(durations),
        "min": min(durations),
        "max": max(durations),
        "stdev": statistics.stdev(durations) if len(durations) > 1 else 0.0,
        "durations": durations,
    }
    # attach work units if module provides workload_size
    if hasattr(mod, 'workload_size'):
        try:
            stats['work_units'] = mod.workload_size(inp)
        except Exception:
            stats['work_units'] = 0
    else:
        stats['work_units'] = 0
    return stats


def main():
    parser = argparse.ArgumentParser(description="对比 src.cpu 与 src.npu 实现的基准工具")
    parser.add_argument("--cpu-module", default="src.cpu", help="CPU 实现的模块路径（默认: src.cpu）")
    parser.add_argument("--npu-module", default="src.npu", help="NPU 实现的模块路径（默认: src.npu）")
    parser.add_argument("--ops", default="all", help="要跑的算子，逗号分隔（默认: all，支持 matmul,add,conv2d,sdpa）")
    parser.add_argument("--repeat", type=int, default=10, help="测量重复次数（默认: 10）")
    parser.add_argument("--warmup", type=int, default=2, help="预热次数（默认: 2）")
    args = parser.parse_args()

    print("加载模块：", args.cpu_module, args.npu_module)
    cpu_mod = load_module(args.cpu_module)
    npu_mod = load_module(args.npu_module)

    ops = [o.strip() for o in args.ops.split(',')] if args.ops != 'all' else ['matmul', 'add', 'conv2d', 'sdpa']

    results = {}
    for op in ops:
        print(f"\n===== 算子: {op} =====")
        # CPU
        cpu_inp = None
        gen_name = f'generate_input_{op}'
        run_name = f'run_once_{op}'
        wname = f'workload_size_{op}'
        if hasattr(cpu_mod, gen_name):
            cpu_inp = getattr(cpu_mod, gen_name)()
        else:
            cpu_inp = get_input(cpu_mod)

        if hasattr(cpu_mod, run_name):
            tmp = type('M', (), {})()
            setattr(tmp, 'run_once', getattr(cpu_mod, run_name))
            cpu_stats = benchmark(tmp, repeat=args.repeat, warmup=args.warmup, inp=cpu_inp)
        else:
            cpu_stats = benchmark(cpu_mod, repeat=args.repeat, warmup=args.warmup, inp=cpu_inp)
        if hasattr(cpu_mod, wname):
            cpu_stats['work_units'] = getattr(cpu_mod, wname)(cpu_inp)
        print(format_stats(cpu_stats, label="CPU", work_units=cpu_stats.get('work_units', 0)))

        # NPU
        npu_inp = None
        if hasattr(npu_mod, gen_name):
            npu_inp = getattr(npu_mod, gen_name)()
        else:
            npu_inp = get_input(npu_mod)

        if hasattr(npu_mod, run_name):
            tmp = type('M', (), {})()
            setattr(tmp, 'run_once', getattr(npu_mod, run_name))
            npu_stats = benchmark(tmp, repeat=args.repeat, warmup=args.warmup, inp=npu_inp)
        else:
            npu_stats = benchmark(npu_mod, repeat=args.repeat, warmup=args.warmup, inp=npu_inp)
        if hasattr(npu_mod, wname):
            npu_stats['work_units'] = getattr(npu_mod, wname)(npu_inp)
        print(format_stats(npu_stats, label="NPU", work_units=npu_stats.get('work_units', 0)))

        print("对比结果：")
        compare_and_print(cpu_stats, npu_stats)
        results[op] = (cpu_stats, npu_stats)

    return


if __name__ == "__main__":
    main()
