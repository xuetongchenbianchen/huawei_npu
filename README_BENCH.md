# 基准对比说明

目的：比较 `src/cpu` 与 `src/npu` 两种实现的性能，并打印分析结果。

使用方法：

1. 确保 Python 能导入项目根目录（直接在项目根运行）

2. 运行基准脚本：

```bash
python3 run_benchmark.py --repeat 20 --warmup 3

# 仅运行 SDPA 融合算子
python3 run_benchmark.py --ops sdpa --repeat 20 --warmup 3
```

3. 约定接口：
- 被测模块（例如 `src/cpu`、`src/npu`）应提供 `run_once(input)` 函数；可选 `init()` 与 `generate_input()`。

4. 输出：脚本会打印每个实现的耗时统计（平均/中位/最短/最长/标准差），并给出速度对比（倍数与百分比）。

当前示例已支持算子：`matmul`、`add`、`conv2d`、`sdpa`。
