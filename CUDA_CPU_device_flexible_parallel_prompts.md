# CUDA GPU/CPU 灵活 Device 框架四路并行代码任务提示词

本文档用于把当前项目改造成更灵活的多 device benchmark 框架。当前机器是 NPU 环境，没有 CUDA GPU，因此本轮四个工程师**只负责编写代码和说明，不要求运行 CUDA GPU 测试，不要求验证真实性能，不要求生成实验数据**。

目标是让项目后续在有 CUDA GPU 的机器上，可以通过 `device` 类型选择 CPU、CUDA GPU、NPU 等不同后端，而不是把主流程固定写成 CPU/NPU。

## 共享项目背景

当前已有核心文件：

- `run_benchmark.py`：统一 benchmark 主入口，目前偏 CPU/NPU 对比。
- `bench_utils.py`：统计、加速比、精度校验公共工具。
- `src/cpu/__init__.py`：CPU 后端。
- `src/npu/__init__.py`：昇腾 NPU 后端。
- `export_benchmark_results.py`：结果导出脚本。
- `make_thesis_figures.py`：图表生成脚本。
- `README_BENCH.md`：当前 benchmark 说明。

当前已有后端接口风格：

```text
init()
backend_info()
synchronize()
materialize_input(inp)
generate_input_<op>(...)
compute_<op>(inp)
run_once_<op>(inp)
workload_size_<op>(inp)
```

当前算子集合：

```text
matmul, bmm, add, sub, mul, div, relu, gelu, silu,
sum, mean, max, transpose_contiguous,
conv2d, maxpool2d, avgpool2d,
embedding, where_mask,
sdpa, softmax, layernorm, rmsnorm
```

## 统一要求

- 不运行 CUDA GPU 测试，因为当前环境没有 CUDA GPU。
- 不编造 CUDA 性能数据。
- 不改动 `workspace/` 下已有实验结果、论文材料和图片。
- 四个工程师写入范围必须尽量独立，避免互相覆盖。
- 保留现有 CPU/NPU 用法，不破坏旧命令。
- 所有新增接口要尽量兼容现有 `src.cpu` 和 `src.npu` 的命名风格。
- 若发现需要跨任务协作的接口，只写清楚约定，不等待其他工程师完成。

## 最终期望形态

后续有 GPU 的机器上，希望可以使用类似命令：

```bash
python run_benchmark.py \
  --reference-device cpu \
  --target-device cuda \
  --ops all \
  --scan \
  --check-precision \
  --repeat 20 \
  --warmup 10 \
  --save-json workspace/experiments/cpu_cuda/results/benchmark_results.json
```

也希望保留模块覆盖能力：

```bash
python run_benchmark.py \
  --reference-module src.cpu \
  --target-module src.cuda \
  --ops matmul,conv2d,sdpa
```

建议 device 映射：

```text
cpu  -> src.cpu
cuda -> src.cuda
npu  -> src.npu
```

建议新的结果 JSON 元信息：

```json
{
  "reference": {"device": "cpu", "module": "src.cpu"},
  "target": {"device": "cuda", "module": "src.cuda"}
}
```

为了兼容旧脚本，可以短期保留旧字段：

```json
{
  "cpu_module": "src.cpu",
  "npu_module": "src.npu"
}
```

---

## 工程师 1：Device Registry 与主流程 CLI 改造

```text
你负责把 run_benchmark.py 从固定 CPU/NPU 对比改造成更通用的 reference/target device 对比框架。

你主要负责的文件：
- run_benchmark.py
- 可新增 src/device_registry.py
- 可新增 workspace/parallel_work_cuda/01_device_cli/device_cli_notes.md

不要修改：
- src/cuda/__init__.py，CUDA 后端由工程师 2 负责。
- export_benchmark_results.py、make_thesis_figures.py，结果兼容由工程师 3 负责。
- README_BENCH.md，文档由工程师 4 负责。

任务目标：
让主流程支持按 device 类型选择后端模块，并保留当前 CPU/NPU 旧参数兼容。

具体任务：
1. 新增 device registry 逻辑。
   - `cpu` 映射到 `src.cpu`
   - `cuda` 映射到 `src.cuda`
   - `npu` 映射到 `src.npu`
   - 未知 device 给出明确错误。

2. 改造 CLI 参数。
   - 新增 `--reference-device`，默认 `cpu`。
   - 新增 `--target-device`，默认 `npu`。
   - 新增 `--reference-module`，用于覆盖 reference device 默认模块。
   - 新增 `--target-module`，用于覆盖 target device 默认模块。
   - 保留旧参数 `--cpu-module`、`--npu-module`，作为兼容旧命令的别名或 fallback。

3. 改造变量命名和输出文案。
   - 主流程内部尽量使用 `reference_mod`、`target_mod`。
   - 输出标签从固定 `CPU`、`NPU` 改成动态标签，例如 `Reference(cpu)`、`Target(cuda)`。
   - `compare_and_print()` 如果暂时仍显示 CPU/NPU，可以先保留，但要在 notes 中说明后续可继续泛化。

4. 改造结果 JSON。
   - 新增顶层字段 `reference` 和 `target`。
   - 每条结果新增 `reference` 和 `target` 统计字段。
   - 为兼容旧脚本，可以继续写出 `cpu` 和 `npu` 字段，其中 `cpu` 指向 reference，`npu` 指向 target。

5. 保持现有 benchmark 核心逻辑。
   - `DEFAULT_OPS` 不删。
   - `DEFAULT_SCAN_CASES` 不删。
   - warmup、repeat、synchronize、precision check、save-json 父目录创建逻辑保留。

本轮不要做的事情：
- 不运行 CUDA 测试。
- 不跑真实 benchmark。
- 不生成实验数据。

交付物：
- 修改后的 `run_benchmark.py`
- 如新增，`src/device_registry.py`
- `workspace/parallel_work_cuda/01_device_cli/device_cli_notes.md`

交付说明中需要写清楚：
- 新旧 CLI 参数如何对应。
- 结果 JSON 哪些字段是新 schema，哪些是兼容旧 schema。
- 当前没有 CUDA 环境，因此未运行 CUDA 性能验证。
```

---

## 工程师 2：CUDA GPU 后端代码实现

```text
你负责新增 CUDA GPU 后端代码，只写实现，不运行 CUDA 测试。

你主要负责的文件：
- 新增 src/cuda/__init__.py
- 可新增 src/cuda/README.md
- 可新增 workspace/parallel_work_cuda/02_cuda_backend/cuda_backend_notes.md

不要修改：
- run_benchmark.py，主流程由工程师 1 负责。
- src/cpu/__init__.py 和 src/npu/__init__.py，除非只是阅读参考。
- 图表和导出脚本。

任务目标：
实现一个与 `src.cpu`、`src.npu` 接口兼容的 PyTorch CUDA 后端，让后续主流程可以通过 `src.cuda` 调用 CUDA GPU。

必须实现的接口：
- `init()`
- `backend_info()`
- `synchronize()`
- `materialize_input(inp)`
- `generate_input_<op>(...)`
- `compute_<op>(inp)`
- `run_once_<op>(inp)`
- `workload_size_<op>(inp)`

CUDA 设备策略：
1. 优先使用 PyTorch CUDA。
2. 如果 `torch.cuda.is_available()` 为 True，设备为 `cuda:0`。
3. 如果 CUDA 不可用，不要崩溃；给出 warning，并回退到 CPU tensor。
4. 回退路径只用于代码导入和无 GPU 环境开发，不能作为 CUDA 性能数据。

`backend_info()` 至少包含：
- `torch_available`
- `cuda_available`
- `device`
- `cuda_device_count`
- `cuda_device_name`
- `torch_version`
- `cuda_version`

`synchronize()` 要求：
- CUDA 可用时调用 `torch.cuda.synchronize()`。
- CUDA 不可用时空操作。

`materialize_input(inp)` 要求：
- 支持 numpy array。
- 支持 torch Tensor。
- 支持 tuple/list 递归转换。
- 支持 int/float/bool 标量原样处理。

建议实现算子：
第一优先级必须完成：
- `matmul`
- `bmm`
- `add`
- `sub`
- `mul`
- `div`
- `relu`
- `gelu`
- `silu`
- `sum`
- `mean`
- `max`
- `transpose_contiguous`
- `softmax`
- `layernorm`
- `rmsnorm`

第二优先级尽量完成：
- `conv2d`
- `maxpool2d`
- `avgpool2d`
- `embedding`
- `where_mask`
- `sdpa`

实现建议：
- 可以大量参考 `src/npu/__init__.py` 的 PyTorch 写法，把 NPU 设备逻辑替换成 CUDA。
- 不要手写 CUDA C++ kernel。
- `compute_<op>()` 应返回完整 tensor，方便精度校验。
- `run_once_<op>()` 可以返回输出 sum 的 Python float，沿用现有 benchmark 风格。
- `workload_size_<op>()` 可复用 CPU/NPU 的估算公式。

本轮不要做的事情：
- 不运行 CUDA 测试。
- 不写真实 GPU 性能结论。
- 不生成 JSON/CSV/PNG 结果。

交付物：
- `src/cuda/__init__.py`
- `src/cuda/README.md`
- `workspace/parallel_work_cuda/02_cuda_backend/cuda_backend_notes.md`

交付说明中需要写清楚：
- 哪些算子已实现。
- 哪些算子只是 fallback 或暂未实现。
- CUDA 不可用时的行为。
- 当前环境无 CUDA GPU，因此未运行真实 CUDA 验证。
```

---

## 工程师 3：结果 Schema 兼容与分析脚本改造

```text
你负责让结果导出和图表脚本兼容新 device schema。只写代码，不运行 CUDA 测试。

你主要负责的文件：
- export_benchmark_results.py
- make_thesis_figures.py
- 可新增 workspace/parallel_work_cuda/03_schema_tools/schema_notes.md

不要修改：
- run_benchmark.py，主流程由工程师 1 负责。
- src/cuda/__init__.py，CUDA 后端由工程师 2 负责。
- README_BENCH.md，文档由工程师 4 负责。

任务目标：
当前分析脚本可能默认结果字段是 `cpu` 和 `npu`。你需要让它们同时兼容：

旧 schema：
```json
{
  "cpu_module": "src.cpu",
  "npu_module": "src.npu",
  "results": [
    {"cpu": {...}, "npu": {...}}
  ]
}
```

新 schema：
```json
{
  "reference": {"device": "cpu", "module": "src.cpu"},
  "target": {"device": "cuda", "module": "src.cuda"},
  "results": [
    {"reference": {...}, "target": {...}}
  ]
}
```

具体任务：
1. 在 `export_benchmark_results.py` 中新增通用读取函数。
   - 优先读取 `reference` / `target`。
   - 如果不存在，则回退读取 `cpu` / `npu`。
   - 导出字段建议使用：
     - `reference_device`
     - `target_device`
     - `reference_mean_ms`
     - `target_mean_ms`
     - `speedup_reference_div_target`

2. 在 `make_thesis_figures.py` 中新增兼容逻辑。
   - 图表标题不再固定写 CPU/NPU。
   - 根据 JSON 顶层信息动态生成标题，如 `CPU/CUDA Speedup by Operator`。
   - 如果没有设备信息，则使用 `Reference/Target`。

3. 保持旧图表可用。
   - 旧 CPU/NPU JSON 仍能被脚本读取。
   - 旧字段名如 `speedup_cpu_div_npu` 如果下游依赖较多，可以暂时保留，同时新增通用字段。

4. 图表可读性。
   - operator 多时使用横向柱状图，避免标签重叠。
   - 不要把所有 target 都写成 NPU。

本轮不要做的事情：
- 不运行 CUDA 测试。
- 不跑 benchmark。
- 不生成真实 CUDA 图表。
- 不修改 `workspace/thesis_materials/` 正式论文材料。

交付物：
- 修改后的 `export_benchmark_results.py`
- 修改后的 `make_thesis_figures.py`
- `workspace/parallel_work_cuda/03_schema_tools/schema_notes.md`

交付说明中需要写清楚：
- 新旧 JSON schema 的兼容策略。
- 新增 CSV 字段名称。
- 图表标题如何根据 device 动态生成。
- 当前没有 CUDA 环境，因此只完成代码适配，未生成真实 CUDA 结果。
```

---

## 工程师 4：README 与 CUDA/CPU 论文素材整理

```text
你负责文档和论文素材整理，不修改核心代码。

你主要负责的文件：
- README_BENCH.md
- 新增 README_CUDA_CPU.md
- workspace/parallel_work_cuda/04_docs/cuda_cpu_thesis_notes.md
- workspace/parallel_work_cuda/04_docs/migration_checklist.md

不要修改：
- run_benchmark.py
- src/cuda/__init__.py
- export_benchmark_results.py
- make_thesis_figures.py

任务目标：
把项目说明从“CPU/NPU benchmark”扩展成“多 device benchmark 框架”，并为 CUDA GPU/CPU 章节准备文字材料。

具体任务：
1. 更新 README 结构。
   - 说明当前项目支持 CPU、NPU，并计划支持 CUDA。
   - 说明 device-based framework 的设计目标。
   - 说明 `reference-device` 和 `target-device` 的含义。

2. 写 CUDA/CPU 使用说明。
   - CPU-only 环境如何跑代码导入或 CPU fallback。
   - 有 CUDA GPU 的环境应如何运行 CPU/CUDA benchmark。
   - 当前 NPU 环境不能运行 CUDA 性能测试，不能产生 CUDA 性能结论。

3. 写典型命令模板。
   - CPU vs CPU：
     ```bash
     python run_benchmark.py --reference-device cpu --target-device cpu --ops matmul --repeat 5 --warmup 2
     ```
   - CPU vs CUDA：
     ```bash
     python run_benchmark.py --reference-device cpu --target-device cuda --ops all --scan --check-precision
     ```
   - CPU vs NPU：
     ```bash
     python run_benchmark.py --reference-device cpu --target-device npu --ops all --scan --check-precision
     ```
   - 自定义模块：
     ```bash
     python run_benchmark.py --reference-module src.cpu --target-module src.cuda --ops matmul
     ```

4. 写结果解释。
   - `Speedup = reference 平均耗时 / target 平均耗时`
   - CPU/CUDA 场景中 target 是 CUDA GPU。
   - CPU/NPU 场景中 target 是 NPU。
   - 加速比是时间比，不是理论峰值算力比。

5. 写 CUDA/CPU 论文素材。
   - CUDA 编程模型简介。
   - PyTorch CUDA 张量迁移和算子调用。
   - CUDA 异步执行与 `torch.cuda.synchronize()`。
   - warmup 的必要性。
   - CPU/CUDA 精度校验方法。
   - 由于当前环境没有 GPU，所有性能数字用 `[待 GPU 环境运行后补充]`。

6. 写迁移清单。
   - 主流程由 CPU/NPU 改为 reference/target。
   - 新增 `src.cuda`。
   - 导出和图表脚本兼容新 schema。
   - README 更新命令。
   - 后续在 GPU 机器上运行真实 benchmark。

本轮不要做的事情：
- 不运行 CUDA 测试。
- 不写真实 CUDA 性能数字。
- 不修改论文 Word 文件。
- 不生成图片、CSV 或实验 JSON。

交付物：
- 修改后的 `README_BENCH.md`
- 新增 `README_CUDA_CPU.md`
- `workspace/parallel_work_cuda/04_docs/cuda_cpu_thesis_notes.md`
- `workspace/parallel_work_cuda/04_docs/migration_checklist.md`

交付说明中需要写清楚：
- 当前只是代码和文档准备。
- 真实 CUDA 性能结果需要转移到有 NVIDIA GPU 的环境再跑。
```

---

## 最终合并建议

四个工程师完成后，建议按以下顺序合并：

1. 合并工程师 2 的 `src/cuda`，确保模块文件存在且接口完整。
2. 合并工程师 1 的 device registry 和 CLI 改造。
3. 合并工程师 3 的 JSON schema 与图表兼容改造。
4. 合并工程师 4 的 README 与论文材料。

最终在有 CUDA GPU 的环境中再运行真实验证。当前 NPU 环境下不要把 CPU fallback 结果当成 CUDA GPU 性能结果。
