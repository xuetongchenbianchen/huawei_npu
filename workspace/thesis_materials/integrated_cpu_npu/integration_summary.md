# CPU/NPU 并行工作整合总结

## 数据来源

本整合版已更新为真实 Ascend NPU 扩展算子 benchmark，使用以下实验数据：

- `workspace/experiments/2026-04-25-real-npu-extended/results/benchmark_results.json`
- `workspace/experiments/2026-04-25-real-npu-extended/results/benchmark_results.csv`

环境来源：

- `workspace/experiments/2026-04-25-real-npu-extended/env/python_env.txt`
- `workspace/experiments/2026-04-25-real-npu-extended/env/npu_smi_before.txt`
- `workspace/experiments/2026-04-25-real-npu-extended/env/npu_smi_after.txt`

运行命令：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
.venv/bin/python run_benchmark.py \
  --ops all \
  --scan \
  --check-precision \
  --repeat 20 \
  --warmup 10 \
  --save-json workspace/experiments/2026-04-25-real-npu-extended/results/benchmark_results.json
```

## 可直接使用的表格

- `tables/operator_overview.csv`：真实 NPU 结果按算子聚合的 CPU/NPU 延迟、加速比、吞吐和精度通过数。建议用于第 6.4 节主表。
- `tables/precision_overview.csv`：真实 NPU 结果按算子聚合的 allclose、最大绝对误差、平均绝对误差和最低余弦相似度。建议用于第 6.3.2 节。
- `tables/representative_cases.csv`：真实 NPU Top 10 代表性 case。建议用于第 6.4 节正文表。
- `tables/all_cases.csv`：45 条逐 case 明细。建议放附录或作为论文数据来源备查。
- `tables/run_comparison_by_case.csv`：本次真实 NPU 数据源索引。

## 可直接使用的图表

- `figures/operator_average_speedup.png`：按算子聚合的平均加速比。
- `figures/operator_latency_logscale.png`：CPU/NPU 平均延迟对比，对数坐标。
- `figures/representative_case_speedup.png`：Top 10 代表性 case 的加速比。
- `figures/speedup_summary.png`：平均加速比汇总图。
- `figures/speedup_by_case.png`：全 case 加速比分布图。
- `figures/precision_vs_speedup.png`：性能与精度关系图。

## 已整合的核心结论

本次真实 Ascend NPU benchmark 共统计 45 条 case 运行记录，覆盖 `matmul`、`bmm`、`add`、`sub`、`mul`、`div`、`relu`、`gelu`、`silu`、`sum`、`mean`、`max`、`transpose_contiguous`、`conv2d`、`maxpool2d`、`avgpool2d`、`embedding`、`where_mask`、`sdpa`、`softmax`、`layernorm` 和 `rmsnorm`。所有 case 的 CPU/NPU 输出精度校验均通过。

按算子聚合后，平均加速比较高的算子包括 `gelu`（829.04x）、`silu`（211.13x）、`conv2d`（203.09x）、`layernorm`（77.80x）、`sdpa`（73.61x）、`transpose_contiguous`（68.65x）和 `rmsnorm`（39.61x）。其中 Top 10 case 中，`gelu#1` 达到 1333.09x，`conv2d#1` 达到 323.71x，`sdpa#1` 达到 128.69x。

精度方面，45 条 case 均通过 allclose 校验。`conv2d` 在当前脚本使用 `atol=1e-2, rtol=1e-2` 的算子级阈值，两个 case 均通过；其它算子使用默认 `atol=1e-4, rtol=1e-3`，均通过。论文中应说明阈值来源，并避免把旧实验中 conv2d 的严格阈值失败结论继续写入新版结果。

图融合方面，当前仓库根目录的 `fusion_result.json` 内容为 `null`，不能作为有效融合统计证据。论文中只能写已设计融合统计方法，不能写具体 pass 生效次数、`effect_ratio` 或性能提升比例。

## 待补事项

1. 若论文必须保留 6.3.1 图融合效果，需要重新导出有效的 `fusion_result.json` 或 CANN profiler 结果。
2. Word 中需统一替换 `表 6-x`、`图 6-x` 占位。
3. 参考文献需要按学校格式补入 Huawei CANN、Ascend Extension for PyTorch、PyTorch 等官方资料。
4. 若 GPU/CUDA 部分也要合并，需避免把 CPU/NPU 数据写成 GPU/NPU 结论。
