# CPU/CUDA Benchmark 使用说明

本文档面向 CPU/CUDA GPU 方向的使用者，说明多 device benchmark 框架迁移后的推荐用法。当前仓库若尚未合并 `src.cuda` 和 `--reference-device/--target-device` CLI，则本文中的 CPU/CUDA 命令为待运行命令，不能作为已完成 CUDA 实测记录。

## 目标

CPU/CUDA benchmark 的目标是以 CPU 作为 reference device，以 NVIDIA CUDA GPU 作为 target device，在相同输入、相同算子集合、相同预热和重复计时规则下比较延迟、吞吐量、加速比和精度误差。

统一加速比定义：

```text
Speedup = reference mean latency / target mean latency
```

当 reference=CPU、target=CUDA 时，Speedup 即 CPU/CUDA 加速比。

## 环境准备

CPU-only smoke test：

```bash
uv sync
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
PY
```

CUDA 环境需安装匹配 NVIDIA 驱动的 PyTorch CUDA 包。示例：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

实际 `cu121` 等版本必须按机器驱动、CUDA runtime 和实验室规范调整。验证命令：

```bash
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
PY
```

## 推荐命令

CPU vs CPU smoke test：

```bash
python3 run_benchmark.py \
  --reference-device cpu \
  --target-device cpu \
  --ops matmul,add,softmax \
  --repeat 2 \
  --warmup 1 \
  --check-precision
```

CPU vs CUDA smoke test：

```bash
python3 run_benchmark.py \
  --reference-device cpu \
  --target-device cuda \
  --ops matmul,add,relu,softmax,layernorm \
  --repeat 5 \
  --warmup 2 \
  --check-precision \
  --save-json workspace/experiments/cpu_cuda_smoke/results.json
```

CPU vs CUDA 全量扫描：

```bash
python3 run_benchmark.py \
  --reference-device cpu \
  --target-device cuda \
  --ops all \
  --scan \
  --check-precision \
  --repeat 20 \
  --warmup 10 \
  --save-json workspace/experiments/cpu_cuda/results/benchmark_results.json
```

自定义模块：

```bash
python3 run_benchmark.py \
  --reference-module src.cpu \
  --target-module src.cuda \
  --ops matmul,conv2d,sdpa \
  --scan \
  --check-precision
```

## 当前旧 CLI 的临时替代

在 device CLI 未合并前，`run_benchmark.py` 仍使用 `--cpu-module` 和 `--npu-module`。如果 CUDA 后端已经存在但 CLI 尚未迁移，可临时把 CUDA 后端传给旧的 `--npu-module` 参数进行功能验证：

```bash
python3 run_benchmark.py \
  --cpu-module src.cpu \
  --npu-module src.cuda \
  --ops matmul,add,relu \
  --repeat 2 \
  --warmup 1 \
  --check-precision
```

这种命令的终端文案和 JSON 字段仍可能显示为 CPU/NPU 或 `npu`，只能作为迁移期临时检查，不建议作为正式论文结果。

## CSV 与图表

导出 CSV：

```bash
python3 export_benchmark_results.py \
  workspace/experiments/cpu_cuda/results/benchmark_results.json \
  workspace/experiments/cpu_cuda/results/benchmark_results.csv
```

生成图表：

```bash
python3 make_thesis_figures.py \
  --benchmark-json workspace/experiments/cpu_cuda/results/benchmark_results.json \
  --output-dir workspace/experiments/cpu_cuda/figures
```

要求：导出和图表脚本需要兼容 reference/target schema；如果还只支持旧 `cpu`/`npu` 字段，应等待工程师 3 的 schema 兼容改造。

## 结果记录规范

建议每轮 CPU/CUDA 实验保存：

- `notes/command.txt`：完整命令。
- `env/python_env.txt`：Python、PyTorch、CUDA、GPU 名称和设备数量。
- `logs/benchmark.log`：终端原始输出。
- `results/benchmark_results.json`：结构化结果。
- `results/benchmark_results.csv`：导出表格。
- `figures/`：图表输出。

若 `torch.cuda.is_available()` 为 False，结果目录应标注为 dry-run 或 fallback，不应进入正式 CPU/CUDA 性能结论。

## 论文写作边界

可以写：

- 本项目设计了 CPU/CUDA benchmark 命令和结果 schema。
- CUDA 后端计划基于 PyTorch CUDA，不手写 CUDA C++ kernel。
- CUDA 计时需要 warmup 和 `torch.cuda.synchronize()`。
- 精度校验沿用 allclose、最大绝对误差、平均绝对误差和余弦相似度。

不能写：

- 未运行真实 GPU 时的 CUDA 延迟、吞吐量或加速比。
- fallback 到 CPU 的结果是 GPU 性能。
- 将 torch_npu/CANN 结果解释为 CUDA 结果。

