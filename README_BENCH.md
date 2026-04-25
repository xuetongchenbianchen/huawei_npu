# 多 Device 基准对比说明

本项目用于支撑论文《GPU 与 NPU 软件开发方法对比研究》中的异构设备 benchmark 实验。当前仓库已经具备 CPU/NPU 算子对比能力，并正在扩展为更通用的多 device benchmark 框架：以 CPU 作为参考设备，以 CUDA GPU、华为昇腾 NPU 或后续其他设备作为目标设备，在同一套算子、计时、精度校验和结果导出流程下进行对比。

当前代码状态说明：

- 已可运行：旧版 CPU/NPU 模块式命令，即 `--cpu-module src.cpu --npu-module src.npu`。
- 迁移目标：新版 reference/target device 命令，即 `--reference-device cpu --target-device cuda|npu`。
- CUDA 后端和新版 device CLI 需要等待对应工程分支合并后使用。未合并前，本文档中的 CUDA 命令作为迁移目标和待运行命令，不代表当前仓库已经完成 CUDA 实测。
- 没有真实 CUDA GPU 或昇腾 NPU 时，只能做导入检查、CPU fallback 或 smoke test，不能把 fallback 结果写成 GPU/NPU 性能数据。

## 支持的算子范围

当前 benchmark 算子集合包括：

- 矩阵计算类：`matmul`、`bmm`、`sdpa`
- 逐元素类：`add`、`sub`、`mul`、`div`、`relu`、`gelu`、`silu`
- 归约与归一化类：`sum`、`mean`、`max`、`softmax`、`layernorm`、`rmsnorm`
- CV 类：`conv2d`、`maxpool2d`、`avgpool2d`
- 访存/layout 类：`transpose_contiguous`、`embedding`、`where_mask`

结果解释建议按算子族分组，不宜只给出一个总平均值。矩阵和卷积类通常更偏计算密集；逐元素、归约、Embedding 和 layout 类更容易受数据访问、同步和任务调度开销影响。

## 环境安装

### CPU-only 环境

CPU-only 环境用于功能开发、导入检查和 CPU 参考结果生成。建议使用 Python 3.10：

```bash
uv sync
```

如果不用 `uv`，可按 `pyproject.toml` 或 `environment.yml` 安装 NumPy、SciPy、Matplotlib、PyTorch 等依赖。CPU-only 环境可以运行 CPU vs CPU smoke test，但不能产生 CUDA GPU 或 Ascend NPU 性能结论。

### NVIDIA CUDA 环境

CUDA GPU 环境需要安装与本机 NVIDIA 驱动和 CUDA 运行时匹配的 PyTorch CUDA 版本。安装方式应以 PyTorch 官方选择器或实验室统一环境为准，例如：

```bash
# 示例，具体 cuXXX 版本需按机器驱动/CUDA 版本调整
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

注意事项：

- PyTorch CUDA 包与 `torch-npu` 不是同一套后端依赖，不能混用为同一个设备环境结论。
- CUDA 可用性应通过 `torch.cuda.is_available()`、`torch.cuda.device_count()` 和设备名称确认。
- 如果 `src.cuda` fallback 到 CPU，只能说明 CUDA 后端接口可导入，不能说明 GPU 性能。

### 华为 Ascend NPU 环境

Ascend NPU 环境需要 CANN/Ascend Toolkit、PyTorch 和 torch_npu 版本匹配。当前仓库提供初始化脚本：

```bash
bash setup_ascend_env.sh
```

如果 CANN 不在默认路径 `/usr/local/Ascend/ascend-toolkit/latest`，先设置：

```bash
export ASCEND_HOME=/your/ascend/path
bash setup_ascend_env.sh
```

手动同步依赖时可执行：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
uv sync --extra ascend
```

当前 `pyproject.toml` 中 Ascend 额外依赖为 PyTorch 2.1.0 和 torch-npu 2.1.0.post13。实际使用时必须以实验室 CANN、驱动和硬件型号匹配结果为准。

## 运行 benchmark

### 当前可用：CPU vs NPU 旧命令

在当前未合并 device CLI 前，使用旧参数运行 CPU/NPU：

```bash
python3 run_benchmark.py \
  --cpu-module src.cpu \
  --npu-module src.npu \
  --ops all \
  --scan \
  --check-precision \
  --repeat 20 \
  --warmup 10 \
  --save-json workspace/experiments/cpu_npu/results/benchmark_results.json
```

只跑部分算子：

```bash
python3 run_benchmark.py \
  --ops matmul,conv2d,sdpa,softmax,layernorm \
  --repeat 20 \
  --warmup 3 \
  --check-precision
```

### 迁移目标：CPU vs CPU smoke test

合并 device CLI 后，可用 CPU vs CPU 进行 smoke test：

```bash
python3 run_benchmark.py \
  --reference-device cpu \
  --target-device cpu \
  --ops matmul,add \
  --repeat 2 \
  --warmup 1 \
  --check-precision
```

若仍使用模块覆盖形式：

```bash
python3 run_benchmark.py \
  --reference-module src.cpu \
  --target-module src.cpu \
  --ops add \
  --repeat 2 \
  --warmup 1 \
  --check-precision
```

### 迁移目标：CPU vs CUDA

合并 `src.cuda` 和 device CLI 后，可用以下命令运行 CPU/CUDA benchmark：

```bash
python3 run_benchmark.py \
  --reference-device cpu \
  --target-device cuda \
  --ops matmul,bmm,add,relu,softmax,layernorm,conv2d \
  --scan \
  --check-precision \
  --repeat 20 \
  --warmup 10 \
  --save-json workspace/experiments/cpu_cuda/results/benchmark_results.json
```

该命令只有在 `torch.cuda.is_available()` 为 True 且目标后端实际使用 `cuda:0` 时，才能作为 CUDA GPU 性能实验。若 CUDA 不可用并 fallback 到 CPU，结果只能标记为“待运行”。

### 迁移目标：CPU vs NPU

合并 device CLI 后，CPU/NPU 可改用统一 reference/target 写法：

```bash
python3 run_benchmark.py \
  --reference-device cpu \
  --target-device npu \
  --ops all \
  --scan \
  --check-precision \
  --repeat 20 \
  --warmup 10 \
  --save-json workspace/experiments/cpu_npu/results/benchmark_results.json
```

## 结果 JSON 与加速比解释

旧版 JSON 顶层使用：

```json
{
  "cpu_module": "src.cpu",
  "npu_module": "src.npu"
}
```

每个 case 使用 `cpu` 和 `npu` 字段保存统计结果。

新版多 device JSON 建议使用：

```json
{
  "reference": {"device": "cpu", "module": "src.cpu"},
  "target": {"device": "cuda", "module": "src.cuda"}
}
```

每个 case 建议使用 `reference` 和 `target` 字段保存统计结果。为了兼容旧脚本，迁移期可同时保留 `cpu`/`npu` 或由导出脚本兼容两种 schema。

加速比统一定义为：

```text
Speedup = reference mean latency / target mean latency
```

因此：

- reference=CPU、target=CUDA 时，Speedup 表示 CPU/CUDA 加速比。
- reference=CPU、target=NPU 时，Speedup 表示 CPU/NPU 加速比。
- target fallback 到 CPU 时，不能写成 GPU 或 NPU 加速比。

## 导出 CSV 与生成图表

旧 schema 当前可用：

```bash
python3 export_benchmark_results.py \
  workspace/experiments/cpu_npu/results/benchmark_results.json \
  workspace/experiments/cpu_npu/results/benchmark_results.csv
```

生成论文图表：

```bash
python3 make_thesis_figures.py \
  --benchmark-json workspace/experiments/cpu_npu/results/benchmark_results.json \
  --conv2d-precision-json workspace/experiments/2026-04-24-full-benchmark/results/conv2d_precision_analysis.json \
  --output-dir workspace/experiments/cpu_npu/figures
```

迁移到 reference/target schema 后，`export_benchmark_results.py` 和 `make_thesis_figures.py` 需要兼容新字段，并将图表标题从固定 CPU/NPU 改为 CPU/CUDA、CPU/NPU 或 Reference/Target。

## NPU 专项实验

NPU 专项实验仍使用：

```bash
python3 run_npu_case_study.py \
  --study all \
  --check-precision \
  --repeat 20 \
  --warmup 5 \
  --output workspace/thesis_materials/results/npu_case_study.json
```

该脚本用于预热、dtype 和 scaling 分析，适合支撑论文第 6.3.2 和第 6.4 节。若新增 transfer、dynamic_shape、layout、utilization_sample 等专项实验，应在独立实验目录保存命令、结果和环境快照。

## 与论文实验章节的对应关系

- 第 2 章：CUDA 编程模型、NPU/CANN 编程模型、PyTorch 设备调用和同步机制。
- 第 3 章：多 device benchmark 的功能需求、非功能需求和实验环境需求。
- 第 4 章：reference/target 设备抽象、后端模块接口和结果 schema 设计。
- 第 5 章：CPU、CUDA、NPU 后端实现，以及 benchmark 主流程。
- 第 6 章：CPU/CUDA、CPU/NPU 的延迟、吞吐量、加速比和精度对比。
- 第 7 章：扩展更多设备、模型级 workload、profiler 和能效评估。

## 迁移风险

- 旧脚本和图表工具固定读取 `cpu`/`npu` 字段，迁移到 reference/target schema 后需要兼容处理。
- `src.cuda` 未合并或 CUDA 不可用时，CPU/CUDA 命令只能标记为“待运行”。
- PyTorch CUDA 与 torch_npu 依赖分别对应 NVIDIA GPU 和昇腾 NPU，不应混淆环境或把一个后端的结果写成另一个后端。
- benchmark 默认测的是 Python/PyTorch 层端到端算子调用，不等于硬件理论峰值。

