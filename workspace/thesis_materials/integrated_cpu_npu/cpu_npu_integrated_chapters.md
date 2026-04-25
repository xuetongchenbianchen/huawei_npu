# CPU/NPU 方向论文章节整合版

本文档整合 `workspace/parallel_work/01_experiment_protocol/`、`02_data_analysis/`、`03_npu_fusion_precision/` 和 `04_thesis_text/` 的交付结果，供论文 Word 合并使用。图号和表号仍需在 Word 中统一调整。

## 第 2 章 相关技术基础

### 2.2 NPU 硬件架构与编程模型

NPU（Neural Processing Unit）是面向神经网络计算负载设计的专用处理器。与通用 CPU 相比，NPU 更强调矩阵乘法、卷积、归一化、Softmax 等深度学习常见算子的批量并行执行能力；与传统通用并行处理器相比，NPU 通常在计算单元、片上存储、算子库和图优化编译链路上针对 AI 模型推理与训练场景进行优化。本文 CPU/NPU 实验部分选取华为昇腾 NPU 作为异构加速设备，以 Python、PyTorch 和 torch_npu 作为上层调用接口，以 CANN 软件栈作为 NPU 运行时和算子库支撑。

#### 2.2.1 昇腾 NPU 硬件架构

昇腾 NPU 的计算体系面向张量计算和神经网络算子执行设计，其核心思想是通过专用计算单元和层次化存储结构提高常见 AI 算子的吞吐能力。典型深度学习算子中，矩阵乘法、卷积和注意力计算包含大量乘加操作，适合由面向矩阵或张量运算的计算单元执行；激活、归一化、逐元素加法等算子则更多依赖向量化计算、规约操作和访存效率。因而，NPU 的性能不只取决于峰值算力，还取决于算子形状、数据类型、数据搬运、算子融合和运行时调度开销等因素。

在本文实验中，`matmul`、`bmm`、`conv2d` 和 `sdpa` 用于覆盖矩阵乘法、批量矩阵乘法、卷积和注意力类计算负载；`add`、`sub`、`mul`、`div`、`relu`、`gelu`、`silu` 用于覆盖逐元素和激活函数负载；`sum`、`mean`、`max`、`softmax`、`layernorm`、`rmsnorm` 用于覆盖规约、归一化和 Transformer 常用算子；`transpose_contiguous`、`embedding`、`where_mask`、`maxpool2d`、`avgpool2d` 用于补充访存、布局变换、掩码和 CV 场景。该算子集合既包含高计算密度算子，也包含受内存访问和调度开销影响较明显的算子，可用于观察 CPU 与 NPU 在不同负载类型下的差异。具体测试用例由 `run_benchmark.py` 中的 `DEFAULT_SCAN_CASES` 给出，实验结果来自真实 Ascend NPU extended benchmark。

#### 2.2.2 昇腾 NPU 计算与存储单元

昇腾 NPU 面向 AI 计算提供多类计算能力。矩阵乘法和卷积类负载通常具有较高的乘加密度，能够更充分利用面向矩阵计算的硬件单元；Softmax、LayerNorm、RMSNorm 等算子除了需要数值计算，还涉及规约、指数、平方根、均值方差等操作，其性能更容易受到算子实现、数据访问模式和运行时同步的影响。因此，在论文分析中不应只用单一峰值算力解释全部实验结果，而应结合算子类型分别讨论。

从存储角度看，NPU 执行过程涉及主机侧内存、设备侧 HBM、片上缓存以及算子内部临时缓冲。输入规模较小时，数据搬运和运行时调度开销可能占比较高，NPU 加速效果可能不稳定；输入规模增大后，计算密度提高，NPU 计算单元利用率通常更高。本文在 `matmul`、`conv2d`、`sdpa` 等算子上设计不同输入规模的扫描用例，正是为了观察规模变化对 CPU/NPU 延迟和加速比的影响。

#### 2.2.3 CANN 软件栈与 torch_npu 编程模型

CANN 是昇腾 AI 处理器的软件基础设施，承担算子库、运行时、图优化和设备管理等功能。本文实验没有直接编写底层 Ascend C 或自定义算子，而是采用 PyTorch 与 torch_npu 进行上层调用。其基本编程流程为：在 CPU 侧构造输入数据；通过 NPU 后端将张量迁移到 `npu:0` 设备；调用 PyTorch/torch_npu 暴露的算子接口；在计时前后进行必要的设备同步；最后将输出取回 CPU 侧进行精度校验。

本项目中，NPU 后端位于 `src/npu/__init__.py`。该模块在导入阶段尝试加载 `torch_npu`，若 `torch.npu.is_available()` 返回真，则使用 `npu:0` 作为执行设备；若环境缺少 NPU 或 torch_npu 不可用，则回退到 CPU 路径用于代码开发和导入检查。论文正式实验必须以真实 NPU 环境下的结果为准，不能把回退路径结果作为 NPU 性能数据。已有环境快照显示实验运行于 Python 3.10.19、PyTorch 2.1.0、torch_npu 2.1.0.post13，`torch.npu.is_available()` 为 True，设备数量为 16，NPU 型号由 `npu-smi info` 记录为 Ascend910。

#### 2.2.4 NPU 编程模型的特点

基于 torch_npu 的 NPU 编程模型具有较强的框架复用性。对于已有 PyTorch 算子，开发者通常只需完成张量设备迁移和少量接口适配，即可调用 NPU 后端算子库。这降低了异构设备实验平台的实现成本，也便于在同一 benchmark 框架中复用 CPU 与 NPU 的输入生成、计时统计和精度校验逻辑。

同时，该编程模型也带来若干需要在实验中控制的因素。第一，NPU 执行具有异步特征，若不在计时前后同步设备，测得的时间可能只包含任务提交开销，不能代表实际执行延迟。第二，首次调用可能包含算子选择、运行时初始化、缓存建立等开销，因此正式计时前需要进行预热。第三，不同数据类型和算子实现可能导致数值路径不同，CPU 与 NPU 输出存在小幅差异时，应结合绝对误差、相对误差和余弦相似度综合判断，不能仅凭某个阈值下的 `allclose` 结果给出结论。

### 2.3 性能测试与评估技术

#### 2.3.1 基准测试框架设计

本文 CPU/NPU benchmark 采用统一入口 `run_benchmark.py`。该脚本通过命令行参数指定 CPU 后端模块、NPU 后端模块、被测算子、预热次数、重复次数、是否开启参数扫描以及是否执行精度校验。框架启动后动态加载 `src.cpu` 与 `src.npu`，调用各后端的 `init()` 完成初始化，再根据算子名称查找输入生成函数、计算函数和工作量估算函数。

统一 benchmark 的设计目标是减少 CPU 与 NPU 之间除设备差异外的其他变量。输入数据由 CPU 侧参考后端生成，然后分别物化为 CPU 输入和 NPU 输入；每个算子在正式计时前执行若干次 warmup；每次计时前后调用后端同步函数，避免异步提交影响统计结果；最终记录平均值、中位数、最小值、最大值、标准差、单次耗时序列和工作量估计。

#### 2.3.2 性能评估指标体系

本文主要使用延迟、吞吐量和加速比三个指标评价 CPU/NPU 性能差异。延迟以毫秒为单位，表示一次算子执行的平均耗时；吞吐量由脚本中估算的 `work_units` 除以平均耗时得到，单位可表示为 GFLOPS 或等效吞吐；加速比定义如下：

```text
Speedup = CPU 平均耗时 / NPU 平均耗时
```

当 Speedup 大于 1 时，表示 NPU 在该测试用例中平均耗时低于 CPU；当 Speedup 接近 1 或小于 1 时，表示 NPU 加速效果不明显或存在额外开销。需要注意的是，Speedup 只反映当前输入规模、当前软件环境和当前算子实现下的相对结果，不应外推为所有场景下的结论。不同算子的 `work_units` 估算口径并不完全等价，跨算子比较时应称为“GFLOPS 或等效吞吐”。

#### 2.3.3 精度校验与数值稳定性评估

性能测试之外，本文还对 CPU 与 NPU 输出进行一致性校验。`bench_utils.py` 中的 `validate_outputs()` 会将输出转换为 `float32` 的 NumPy 数组，并计算 `allclose` 判定结果、最大绝对误差、平均绝对误差、余弦相似度以及输出形状。默认阈值为 `atol=1e-4, rtol=1e-3`，`conv2d` 在新版脚本中设置了算子级阈值覆盖，独立精度分析还给出了 `atol=1e-2, rtol=1e-2` 下的校验结果。

精度结果的表述应保持客观。若某一算子在严格阈值下 `allclose` 未通过，但最大绝对误差较小且余弦相似度接近 1，则说明两个输出整体方向和分布高度一致，但局部元素存在可观测数值差异。该现象可能来自 CPU 与 NPU 使用不同算子库、累加顺序、数据布局、底层优化策略或数据类型路径，不宜直接表述为算法错误。

#### 2.3.4 算子融合与图优化评估

算子融合是异构加速中的重要优化方式。对于连续的逐元素、规约或矩阵相关操作，若运行时或编译器能够将多个小算子合并为更少的执行单元，就可能减少中间张量读写、主机到设备的调度次数以及设备同步开销。本文项目提供 `analyze_fusion.py` 用于解析 `fusion_result.json` 中的图融合统计字段，包括匹配次数、实际生效次数和生效率。

当前仓库中的 `fusion_result.json` 内容为 `null`，无法直接得出有效图融合统计。因此，论文中只能说明图融合分析方法，不能写入具体融合 pass 生效次数或“融合提升了多少性能”的结论。若后续重新导出有效统计文件，可按 pass 名称列出 `match_times`、`effect_times` 和 `effect_ratio`，并与 benchmark 时延结果联合分析。

## 第 3 章 需求分析

### 3.3 需求分析

本文 CPU/NPU 对比实验平台的目标，是在统一输入、统一计时、统一统计和统一精度校验流程下，对比 CPU 参考实现与昇腾 NPU 后端在基础算子和典型神经网络算子上的执行差异。系统不追求覆盖完整深度学习框架，而是围绕论文实验所需的可复现 benchmark、结构化结果输出和论文图表生成能力展开。

#### 3.3.1 异构算子对比模块

异构算子对比模块需要支持同一算子在 CPU 与 NPU 两类后端上的执行。CPU 后端作为参考实现，主要用于生成输入、计算参考输出和提供基线延迟；NPU 后端通过 torch_npu 将张量迁移到昇腾设备并调用对应算子。当前已覆盖的算子包括 `matmul`、`bmm`、`add`、`sub`、`mul`、`div`、`relu`、`gelu`、`silu`、`sum`、`mean`、`max`、`transpose_contiguous`、`conv2d`、`maxpool2d`、`avgpool2d`、`embedding`、`where_mask`、`sdpa`、`softmax`、`layernorm` 和 `rmsnorm`，能够分别代表矩阵计算、逐元素计算、激活函数、规约、布局变换、卷积/池化、查表、掩码选择、注意力计算和归一化计算等常见负载。

该模块应满足以下功能需求：能够按算子名称选择测试范围；能够在单一输入规模和参数扫描两种模式下运行；能够对每个算子生成 CPU 与 NPU 两份输入；能够在运行结束后输出平均耗时、吞吐量估计和 CPU/NPU 加速比；能够把结构化结果保存为 JSON，供后续表格统计和图表生成使用。

#### 3.3.2 精度校验与结果记录模块

CPU/NPU 性能对比不能只记录耗时，还必须确认输出是否具有可接受的一致性。精度校验模块需要对 CPU 输出和 NPU 输出进行形状检查，并计算 `allclose`、最大绝对误差、平均绝对误差和余弦相似度。对于不同算子，可以设置不同阈值；例如 `conv2d` 因底层实现差异和累加路径差异，可在独立分析中使用 `atol=1e-2, rtol=1e-2` 观察误差是否处于可接受范围。论文制表时应以结果 JSON 中记录的 `atol`、`rtol` 字段为准。

结果记录模块需要保证实验可追溯。每轮实验至少应保存命令、日志、环境快照和结构化结果。已有实验目录采用 `logs/`、`results/`、`env/`、`notes/` 的组织方式，其中 `benchmark_results.json` 用于论文统计，`benchmark.log` 用于异常追踪，`python_env.txt` 和 `npu_smi.txt` 用于记录软件与硬件环境。

#### 3.3.3 非功能需求

系统需要满足可复现性、可扩展性和结果可信性要求。可复现性要求同一实验命令能够明确指定算子范围、预热次数、重复次数和输出路径；可扩展性要求新增算子时只需在后端模块补充 `generate_input_<op>`、`compute_<op>`、`run_once_<op>` 和 `workload_size_<op>` 等接口；结果可信性要求正式计时前执行预热，计时前后执行设备同步，并保存原始耗时序列。

系统还需要考虑无 NPU 开发环境。`src/npu/__init__.py` 在 torch_npu 或设备不可用时会回退到 CPU 路径，用于保证代码导入和开发调试，但该路径不能作为 NPU 性能实验依据。论文正式结果必须来自 `torch.npu.is_available()` 为 True 的环境，并附带 `npu-smi info` 设备快照。

#### 3.3.4 实验环境需求

CPU/NPU 实验环境应包含 Python、PyTorch、torch_npu、CANN/Ascend Toolkit、NumPy、Matplotlib 等依赖。当前项目通过 `pyproject.toml` 和 `uv` 管理依赖，要求 Python 版本为 `>=3.10,<3.11`，Ascend 额外依赖中固定 PyTorch 2.1.0 和 torch-npu 2.1.0.post13。环境初始化脚本为 `setup_ascend_env.sh`，其流程包括检查 `uv`、加载 Ascend Toolkit 环境变量、同步 Python 依赖和校验 torch_npu 可用性。

已有 benchmark 环境快照记录了 Python 3.10.19、Linux aarch64、PyTorch 2.1.0、torch_npu 2.1.0.post13、`npu_available=True`、`device_count=16`，并通过 `npu-smi info` 记录 Ascend910 设备状态。

## 第 4 章 系统设计

### 4.1 系统架构设计

CPU/NPU benchmark 系统采用“主控脚本 + 后端模块 + 公共工具 + 实验产物”的结构。主控脚本 `run_benchmark.py` 负责解析命令行参数、加载 CPU/NPU 后端、调度测试流程和写出 JSON；后端模块 `src.cpu` 与 `src.npu` 分别封装设备相关的输入物化和算子执行；公共工具 `bench_utils.py` 负责统计格式化、加速比计算和精度校验；实验产物保存于 `workspace/experiments/` 和 `workspace/thesis_materials/`。

系统运行流程可概括为：参数解析、模块加载、环境信息输出、测试用例枚举、输入构造、CPU 计时、NPU 计时、结果对比、可选精度校验、汇总输出和 JSON 保存。该流程使每个算子的测试过程保持一致，有助于减少手工测试带来的偏差。

【图 4-x：CPU/NPU benchmark 系统总体流程图，建议根据 `run_benchmark.py` 绘制】

#### 4.1.1 初始化与模块加载

框架启动时读取命令行参数，动态导入 CPU 模块和 NPU 模块，默认路径分别为 `src.cpu` 和 `src.npu`。导入后，框架调用模块的 `init()` 接口完成一次性初始化，并输出后端环境信息。NPU 后端会报告 `torch_available`、`torch_npu_available`、`device`、`npu_available`、`npu_device_count` 和 `torch_version` 等字段，用于确认当前测试是否运行在真实 NPU 设备上。

#### 4.1.2 输入构造阶段

输入构造由 CPU 后端统一生成，然后分别交给 CPU 与 NPU 后端物化。CPU 输入通常保持为 NumPy 数组或 CPU Tensor，NPU 输入通过 `materialize_input()` 转换到 NPU 设备。采用共享输入生成策略，可以保证 CPU 与 NPU 处理的是同一组随机数据，避免因输入差异影响性能和精度对比。

#### 4.1.3 基准执行阶段

基准执行阶段包括预热和正式计时两部分。预热阶段执行指定次数的算子调用，并在每次调用后进行设备同步；正式计时阶段记录每次执行的起止时间，统计平均值、中位数、最小值、最大值和标准差。NPU 后端在 `synchronize()` 中调用 `torch.npu.synchronize()`，用于保证计时覆盖实际设备执行时间。

#### 4.1.4 统计输出与对比分析

每个测试用例结束后，系统输出 CPU 与 NPU 的统计信息，并计算 Speedup。若算子实现提供 `workload_size_<op>()`，系统还会根据工作量估计吞吐量。开启 `--check-precision` 时，系统额外计算输出一致性指标，并将精度报告写入 JSON。最终 JSON 包含实验参数、算子列表和每个 case 的 CPU/NPU 统计结果，是第 6 章制表和绘图的主要数据来源。

### 4.2 核心算子测试任务设计

核心算子测试任务分为矩阵计算、逐元素/激活、规约/归一化、布局变换和 CV/NLP 场景算子五类。矩阵计算包括 `matmul` 和 `bmm`；逐元素与激活函数包括 `add`、`sub`、`mul`、`div`、`relu`、`gelu`、`silu`；规约与归一化包括 `sum`、`mean`、`max`、`softmax`、`layernorm`、`rmsnorm`；布局与条件选择包括 `transpose_contiguous`、`where_mask`；CV/NLP 场景算子包括 `conv2d`、`maxpool2d`、`avgpool2d`、`embedding` 和 `sdpa`。这些算子覆盖了计算密集、访存密集、规约、查表和注意力等不同负载类型。

参数扫描设计覆盖小规模和较大规模输入。以 `run_benchmark.py` 当前配置为例，`matmul` 扫描 256、512、1024 的方阵规模；`bmm` 扫描不同 batch 与矩阵尺寸；`conv2d`、`maxpool2d`、`avgpool2d` 扫描不同 batch、通道数和特征图大小；`sdpa` 扫描不同 batch、head 数和序列长度；`softmax`、`sum`、`mean`、`max` 等规约类算子扫描不同张量规模；`layernorm` 与 `rmsnorm` 扫描不同 batch、序列长度和 hidden size；`embedding`、`where_mask` 和 `transpose_contiguous` 用于补充索引查表、条件选择和布局转换场景。

【表 4-x：CPU/NPU benchmark 测试用例设计，数据来源为 `run_benchmark.py::DEFAULT_SCAN_CASES`】

### 4.3 功能设计

系统层次结构可分为四层：命令行交互层、benchmark 调度层、设备后端层和结果分析层。命令行交互层负责接收实验参数；benchmark 调度层负责统一测试流程；设备后端层屏蔽 CPU 与 NPU 的实现差异；结果分析层负责将 JSON 转换为表格和图表。

主控模块由 `run_benchmark.py` 实现。其关键职责包括：解析 `--ops`、`--scan`、`--repeat`、`--warmup`、`--check-precision`、`--save-json` 等参数；根据算子名称枚举测试用例；调用后端函数执行预热和计时；调用公共工具输出统计结果；在实验结束后将所有结果序列化为 JSON。该模块不直接包含设备专有实现，便于后续扩展其他后端。

CPU 后端位于 `src/cpu/__init__.py`，主要提供 NumPy/PyTorch 参考实现。NPU 后端位于 `src/npu/__init__.py`，主要通过 torch_npu 调用昇腾设备。两个后端均遵循相同接口命名规范，包括输入生成、完整输出计算、单次运行和工作量估计函数。统一接口使主控模块可以通过动态函数查找调用不同算子，而不需要为每个后端编写独立调度逻辑。

### 4.4 数据结构与接口设计

单个测试用例的统计结果由 CPU 统计、NPU 统计、精度报告和 case 参数组成。CPU 与 NPU 统计字段包括 `count`、`mean`、`median`、`min`、`max`、`stdev`、`durations` 和 `work_units`。精度报告字段包括 `ok`、`max_abs_err`、`mean_abs_err`、`cosine_similarity`、`reference_shape`、`candidate_shape`、`atol` 和 `rtol`。

结构化 JSON 的顶层字段包括 `cpu_module`、`npu_module`、`ops`、`repeat`、`warmup`、`scan`、`check_precision` 和 `results`。该结构既能支撑论文正文表格，也能保留每次重复测量的原始耗时序列，便于后续分析稳定性。

系统通过命令行参数控制实验。典型命令如下：

```bash
.venv/bin/python run_benchmark.py \
  --ops all \
  --scan \
  --check-precision \
  --repeat 20 \
  --warmup 10 \
  --save-json workspace/experiments/2026-04-25-real-npu-extended/results/benchmark_results.json
```

其中，`--ops` 指定算子集合，`--scan` 开启内置参数扫描，`--check-precision` 开启 CPU/NPU 输出校验，`--repeat` 和 `--warmup` 分别控制正式测量次数与预热次数，`--save-json` 指定结构化结果保存路径。

## 第 5 章 系统实现

### 5.1 开发工具与环境

本文 CPU/NPU 实验平台采用 Python 实现，依赖 NumPy、PyTorch、torch_npu、Matplotlib 等工具。项目通过 `uv` 和 `pyproject.toml` 管理依赖，Python 版本约束为 `>=3.10,<3.11`。Ascend NPU 相关依赖以可选依赖形式声明，其中 PyTorch 固定为 2.1.0，torch-npu 固定为 2.1.0.post13，以匹配已有实验环境。

昇腾环境初始化由 `setup_ascend_env.sh` 完成。脚本首先检查 `uv` 是否可用，然后加载 Ascend Toolkit 的 `set_env.sh`，再执行 `uv sync --extra ascend` 安装依赖，最后通过一段 Python 检查 torch 与 torch_npu 是否能够导入，并输出 `torch.npu.is_available()` 与设备数量。该脚本的作用是将环境配置过程显式化，减少人工配置差异对实验结果的影响。

本次真实 NPU extended benchmark 的环境快照保存在 `workspace/experiments/2026-04-25-real-npu-extended/env/`。其中 `python_env.txt` 记录 Python 3.10.19、PyTorch 2.1.0、torch_npu 2.1.0.post13、NPU 可用状态和设备数量；`npu_smi_before.txt` 与 `npu_smi_after.txt` 记录 Ascend910 设备状态。论文排版时可将主要环境信息整理为表 6-x。

### 5.2 公共模块实现

公共模块 `bench_utils.py` 负责统计输出、CPU/NPU 对比和精度校验。`format_stats()` 根据每个算子的耗时序列输出次数、平均耗时、中位耗时、最短耗时、最长耗时和标准差，并在提供工作量估计时计算 GFLOPS。`compare_and_print()` 根据 CPU 与 NPU 平均耗时计算 Speedup、绝对耗时差和相对耗时差，并输出 CPU/NPU 吞吐量对比。

精度校验由 `validate_outputs()` 实现。该函数将 CPU 参考输出和 NPU 候选输出转换为 NumPy 数组，并统一转为 `float32` 进行比较。若输出形状不一致，则返回 shape mismatch；若形状一致，则计算最大绝对误差、平均绝对误差、余弦相似度和 `np.allclose()` 判定结果。`format_precision_report()` 将这些字段格式化为终端可读文本。

公共模块的设计使统计和精度逻辑独立于具体设备后端。后续若新增其他设备或算子，只要保持输出对象可转换为 NumPy 数组，就能复用同一套统计与校验逻辑。

### 5.3 核心模块实现

#### 5.3.1 调度与执行主流程实现

主控脚本 `run_benchmark.py` 是 CPU/NPU 对比实验的统一入口。脚本首先解析命令行参数，并根据 `--ops` 得到待测算子列表；若 `--scan` 开启，则从 `DEFAULT_SCAN_CASES` 中读取每个算子的参数组合；若未开启扫描，则每个算子只运行默认输入规模。

对每个算子和每个 case，脚本先调用 CPU 后端生成共享输入，再分别交给 CPU 与 NPU 后端进行输入物化。随后，`benchmark()` 函数先执行 warmup，再执行指定次数的正式计时。每次正式计时前后都会调用 `maybe_sync()`，在 NPU 后端中对应 `torch.npu.synchronize()`。这种实现方式避免了异步执行导致的计时偏差。

当开启 `--check-precision` 时，脚本会在性能测试后再次计算 CPU 输出和 NPU 输出，并调用 `validate_outputs()` 生成精度报告。脚本最后将实验配置和所有 case 的结果写入 JSON，其中不保存完整输出张量，只保存耗时统计、工作量、case 参数和精度指标，以控制结果文件大小。

#### 5.3.2 CPU 参考模块实现

CPU 参考模块 `src/cpu/__init__.py` 使用 NumPy 和 PyTorch 提供算子参考实现。`matmul`、`bmm` 使用矩阵乘法接口，`add`、`sub`、`mul`、`div`、`relu`、`gelu`、`silu` 使用 NumPy 或 PyTorch 的逐元素实现，`sum`、`mean`、`max`、`softmax`、`layernorm`、`rmsnorm` 使用对应公式或框架算子实现。CV 与 NLP 相关算子中，`conv2d`、`maxpool2d`、`avgpool2d` 优先调用 PyTorch 函数，`embedding`、`where_mask`、`transpose_contiguous` 分别用于覆盖查表、条件选择和布局转换，`sdpa` 使用缩放点积注意力公式或 PyTorch 的 `scaled_dot_product_attention`。

CPU 模块还为每个算子提供工作量估算函数。例如，矩阵乘法工作量按 `2*m*k*n` 估算，卷积工作量按 batch、输出通道、空间尺寸、输入通道和卷积核大小估算，Softmax 与归一化算子按元素数量乘以经验操作系数估算。这些工作量不是硬件级精确指令计数，但能够为论文中的吞吐量比较提供统一口径。

#### 5.3.3 NPU 对比模块实现

NPU 后端 `src/npu/__init__.py` 在导入时尝试加载 PyTorch 和 torch_npu。若 torch_npu 已安装且 `torch.npu.is_available()` 为真，则设备设置为 `npu:0`；否则发出警告并回退到 CPU。该回退机制便于在无 NPU 环境下完成代码检查，但正式论文实验只能采用真实 NPU 路径的数据。

NPU 后端的 `materialize_input()` 负责将共享输入转换为 NPU 张量。对于 NumPy 数组，模块使用 `torch.from_numpy(inp).to(DEVICE)` 将其迁移到目标设备；对于已存在的 Tensor，则调用 `.to(device=DEVICE)`。算子执行方面，矩阵、逐元素、激活、规约、卷积、池化、Embedding、条件选择、布局转换、注意力和归一化算子均通过 PyTorch/torch_npu 支持的张量操作调用昇腾后端执行。

NPU 后端还实现了 `synchronize()`。在真实 NPU 环境下，该函数调用 `torch.npu.synchronize()`，确保主机侧计时器等待设备任务完成。该函数是 NPU benchmark 可信度的重要保证，应在实验方法中明确说明。

#### 5.3.4 论文图表生成实现

图表生成脚本 `make_thesis_figures.py` 和 `workspace/parallel_work/02_data_analysis/generate_data_analysis.py` 负责从 benchmark JSON、`conv2d` 精度分析 JSON 和 NPU 专项实验 JSON 中读取数据，并输出论文可用 PNG 图。当前整合版图表位于 `workspace/thesis_materials/integrated_cpu_npu/figures/`。

论文中建议按如下方式引用图表：第 6.4 节使用 `operator_average_speedup.png` 或 `speedup_summary.png` 展示算子平均加速比，使用 `representative_case_speedup.png` 或 `speedup_by_case.png` 展示不同 case 的加速比分布；第 6.3.2 节使用 `precision_vs_speedup.png` 和 `conv2d_precision_breakdown.png` 解释性能与精度之间的关系。

## 第 6 章 系统测试与对比分析

### 6.1 测试环境与部署

本文 CPU/NPU 实验在已配置昇腾 NPU 软件栈的服务器上完成。实验环境通过 `setup_ascend_env.sh` 初始化，该脚本加载 Ascend Toolkit 环境变量，并通过 `uv sync --extra ascend` 安装 Python 依赖。实验运行前，使用 Python 环境快照和 `npu-smi info` 记录软件版本与设备状态，以保证结果具有可追溯性。

本次真实 NPU extended benchmark 的环境记录显示，实验使用 Python 3.10.19、PyTorch 2.1.0、torch_npu 2.1.0.post13，系统平台为 Linux aarch64，`torch.npu.is_available()` 为 True，设备数量为 16。`npu-smi info` 输出中记录的设备型号为 Ascend910。上述信息来自 `workspace/experiments/2026-04-25-real-npu-extended/env/python_env.txt`、`npu_smi_before.txt` 和 `npu_smi_after.txt`。

【表 6-x：CPU/NPU 实验环境配置，数据来源为 `env/python_env.txt`、`env/npu_smi_before.txt` 与 `env/npu_smi_after.txt`】

实验结果目录采用统一结构保存。`logs/benchmark.log` 保存终端原始输出，`results/benchmark_results.json` 保存结构化结果，`results/benchmark_results.csv` 保存扁平化表格，`env/` 保存环境快照，`notes/command.txt` 保存实验命令。该结构使论文中的每个结论都可以回溯到命令、日志或结构化结果。

### 6.2 测试方法与用例

本实验采用统一 benchmark 脚本 `run_benchmark.py`，对 CPU 后端和 NPU 后端执行同一组算子。测试命令开启参数扫描和精度校验，并在正式计时前执行 warmup。本次真实 NPU extended benchmark 的命令如下，原始命令见 `workspace/experiments/2026-04-25-real-npu-extended/notes/command.txt`：

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

为提高结果稳定性，本次实验设置 `repeat=20,warmup=10`，即每个 case 在正式统计前预热 10 次，并重复计时 20 次。实验结果统一保存在 `workspace/experiments/2026-04-25-real-npu-extended/`，其中 `results/benchmark_results.json` 和 `results/benchmark_results.csv` 是第 6 章表格与图表的主要数据来源。

测试用例覆盖 `matmul`、`bmm`、`add`、`sub`、`mul`、`div`、`relu`、`gelu`、`silu`、`sum`、`mean`、`max`、`transpose_contiguous`、`conv2d`、`maxpool2d`、`avgpool2d`、`embedding`、`where_mask`、`sdpa`、`softmax`、`layernorm` 和 `rmsnorm`，参数扫描覆盖矩阵规模、batch、通道数、特征图大小、序列长度、特征维度和 hidden size。具体用例见表 6-x，数据来源为 `run_benchmark.py::DEFAULT_SCAN_CASES` 和真实 NPU extended benchmark 结果 JSON。

【表 6-x：CPU/NPU benchmark 测试用例列表】

### 6.3 性能优化策略验证

#### 6.3.1 图融合策略及其效果

算子融合和图优化能够减少中间结果读写与运行时调度开销，是 NPU 软件栈提高执行效率的重要手段。本文项目提供 `analyze_fusion.py` 用于读取 `fusion_result.json` 中的图融合统计，并计算 `match_times`、`effect_times` 和 `effect_ratio`。其中，`match_times` 表示某类融合 pass 匹配到可优化模式的次数，`effect_times` 表示实际生效次数，`effect_ratio` 表示生效次数与匹配次数之比。

当前仓库根目录的 `fusion_result.json` 内容为 `null`，运行 `analyze_fusion.py fusion_result.json` 无法得到有效融合统计。因此，本节只能说明图融合分析方法，不能写入具体“融合提升了多少性能”的结论。若后续在真实图模式运行后导出有效统计文件，可补充 pass 级统计表，并结合 benchmark 时延或 profiling 结果判断融合对端到端性能的影响。

【表 6-x：图融合 pass 统计表，待有效 `fusion_result.json` 补充】

#### 6.3.2 数据精度与预热策略分析

预热是 NPU benchmark 的必要步骤。NPU 首次执行可能包含运行时初始化、算子选择、缓存建立和设备调度等额外开销，若直接把首次执行计入正式统计，可能放大离群值并影响平均耗时。本文在真实 NPU extended benchmark 中设置 `warmup=10`，正式统计重复 20 次。该设置可减少初始化阶段对结果的干扰。

精度方面，本文使用 `allclose`、最大绝对误差、平均绝对误差和余弦相似度共同评价 CPU/NPU 输出差异。按 `tables/precision_overview.csv` 统计，本次真实 NPU extended benchmark 的 45 条 case 均通过 allclose 校验。`conv2d` 在当前脚本中使用算子级阈值 `atol=1e-2, rtol=1e-2`，两个 case 均通过；其最大绝对误差最大值为 `0.009488106`，平均绝对误差均值为 `0.000971284`，最低余弦相似度为 `0.999999907`。

`conv2d` 的结果需要单独说明。卷积算子在 CPU 与 NPU 上可能使用不同底层库、累加顺序和数据布局，严格阈值下容易放大局部元素差异。当前结果采用 `atol=1e-2, rtol=1e-2` 后通过校验，且余弦相似度接近 1，因此更适合解释为 CPU/NPU 数值路径差异，而不是卷积算法实现错误。论文中仍应明确阈值设置，避免把旧实验中的严格阈值失败结论继续写入新版结果。

【表 6-x：CPU/NPU 输出精度校验结果，数据来源 `tables/precision_overview.csv`】

【图 6-x：性能与精度关系散点图，图片来源 `figures/precision_vs_speedup.png`】

【图 6-x：Conv2d 绝对误差拆解图，图片来源 `figures/conv2d_precision_breakdown.png`】

### 6.4 测试结果分析与系统评价

真实 NPU extended benchmark 结果表明，NPU 在多类深度学习算子上相对 CPU 参考实现具有明显延迟优势，但不同算子和输入规模下的加速幅度存在差异。本次实验共统计 45 条 case 运行记录，覆盖 `matmul`、`bmm`、`add`、`sub`、`mul`、`div`、`relu`、`gelu`、`silu`、`sum`、`mean`、`max`、`transpose_contiguous`、`conv2d`、`maxpool2d`、`avgpool2d`、`embedding`、`where_mask`、`sdpa`、`softmax`、`layernorm` 和 `rmsnorm`。完整逐 case 数据见 `tables/all_cases.csv`。

按算子聚合后，`gelu`、`silu` 和 `conv2d` 的 NPU 加速最明显，平均加速比分别为 `829.039x`、`211.132x` 和 `203.089x`；`layernorm`、`sdpa` 和 `transpose_contiguous` 的平均加速比分别为 `77.802x`、`73.610x` 和 `68.649x`。`matmul` 的平均加速比为 `15.526x`，但不同规模差异较大，说明矩阵规模和固定调度开销会显著影响相对收益。`max`、`bmm`、`mean`、`sum` 等算子的平均加速比相对较低，但 NPU 平均延迟仍低于 CPU。

【表 6-x：各算子平均 CPU/NPU 延迟与加速比，数据来源 `tables/operator_overview.csv`】

| 算子 | 运行数 | CPU 平均延迟 ms | NPU 平均延迟 ms | 平均加速比 | 精度通过/失败 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gelu | 2 | 40.823925 | 0.049981 | 829.039x | 2/0 |
| silu | 2 | 11.394455 | 0.049833 | 211.132x | 2/0 |
| conv2d | 2 | 13.114698 | 0.062686 | 203.089x | 2/0 |
| layernorm | 2 | 11.279231 | 0.127042 | 77.802x | 2/0 |
| sdpa | 2 | 10.474680 | 0.126520 | 73.610x | 2/0 |
| transpose_contiguous | 2 | 5.878202 | 0.085884 | 68.649x | 2/0 |
| rmsnorm | 2 | 4.823217 | 0.117268 | 39.612x | 2/0 |
| matmul | 3 | 1.255562 | 0.070928 | 15.526x | 3/0 |
| bmm | 2 | 0.564451 | 0.094739 | 7.981x | 2/0 |

代表性 case 中，`gelu` 的 `[2048,2048]` 输入达到 `1333.090x`，`conv2d` 的较大输入 case 达到 `323.710x`，`silu` 的 `[2048,2048]` 输入达到 `302.681x`，`sdpa` 的较大序列 case 达到 `128.692x`。这些 case 适合放在论文正文中说明激活函数、卷积、注意力和归一化等典型深度学习算子在 NPU 上的延迟收益。

【表 6-x：代表性测试 case 结果，数据来源 `tables/representative_cases.csv`】

| 数据集 | 算子 | case | CPU ms | NPU ms | 加速比 | 精度 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| real_npu_extended | gelu | shape=[2048,2048] | 65.009924 | 0.048766 | 1333.090x | 通过 |
| real_npu_extended | gelu | shape=[1024,1024] | 16.637926 | 0.051196 | 324.988x | 通过 |
| real_npu_extended | conv2d | batch=4, in_c=16, h=64, w=64, out_c=32, k=3 | 21.322377 | 0.065869 | 323.710x | 通过 |
| real_npu_extended | silu | shape=[2048,2048] | 17.970141 | 0.059370 | 302.681x | 通过 |
| real_npu_extended | sdpa | batch=4, seq_len=256, heads=8, head_dim=64 | 18.995949 | 0.147608 | 128.692x | 通过 |
| real_npu_extended | layernorm | batch=32, seq_len=256, hidden=1024 | 18.977903 | 0.163489 | 116.081x | 通过 |
| real_npu_extended | transpose_contiguous | shape=[2048,1024] | 8.355564 | 0.085282 | 97.975x | 通过 |
| real_npu_extended | rmsnorm | batch=32, seq_len=256, hidden=1024 | 8.022239 | 0.124426 | 64.474x | 通过 |

【图 6-x：各算子平均 Speedup，图片来源 `figures/operator_average_speedup.png`】

【图 6-x：代表性 case Speedup，图片来源 `figures/representative_case_speedup.png`】

系统评价方面，本文 CPU/NPU benchmark 平台实现了统一算子接口、参数扫描、预热、同步计时、精度校验、JSON/CSV 结果保存和论文图表生成，能够支撑 CPU 与昇腾 NPU 基础算子的对比研究。其不足在于当前仍以 Python 层 benchmark 为主，尚未覆盖更底层的自定义算子开发和完整模型端到端测试；图融合分析也需要有效的运行时导出文件支撑。后续可在相同框架下继续扩展模型级 workload、更多数据类型和更完整的图优化日志分析。

## 第 7 章 总结与展望

### 7.1 开发总结

本文 CPU/NPU 部分围绕昇腾 NPU 与 CPU 参考实现的基础算子对比，完成了统一 benchmark 框架、设备后端模块、精度校验工具、实验结果保存和论文图表生成等工作。系统以 `run_benchmark.py` 为统一入口，以 `src.cpu` 和 `src.npu` 为后端实现，覆盖 `matmul`、`bmm`、`add`、`sub`、`mul`、`div`、`relu`、`gelu`、`silu`、`sum`、`mean`、`max`、`transpose_contiguous`、`conv2d`、`maxpool2d`、`avgpool2d`、`embedding`、`where_mask`、`sdpa`、`softmax`、`layernorm` 和 `rmsnorm` 等算子，能够在同一测试流程下输出 CPU/NPU 延迟、吞吐量估计、加速比和精度指标。

实验流程方面，系统引入预热、设备同步和重复测量机制，减少 NPU 初始化和异步执行对计时结果的影响；结果记录方面，系统将命令、日志、环境快照、JSON/CSV 结果和论文图表分目录保存，保证结论能够回溯到原始数据。本次真实 NPU extended benchmark 结果保存在 `workspace/experiments/2026-04-25-real-npu-extended/`，能够支撑第 6 章 CPU/NPU 性能对比分析。

从实现效果看，该平台达到了本科论文实验部分对可复现性和可验证性的基本要求。CPU 后端提供参考输出，NPU 后端调用 torch_npu 与昇腾设备，公共工具负责统计与精度校验，图表脚本负责从结构化结果生成论文图片。该设计使新增算子或新增实验轮次时不需要重写整体流程。

### 7.2 存在的问题与不足

首先，当前 CPU/NPU 对比仍以 Python 层算子调用为主，实验结果反映的是当前 PyTorch/torch_npu/CANN 软件栈下的端到端算子表现，不能完全等同于硬件理论峰值能力。对于需要解释底层瓶颈的场景，还需要结合 CANN profiler、算子级 trace 或更详细的运行时日志。

其次，当前测试用例主要覆盖基础算子和部分 Transformer 常见算子，尚未扩展到完整模型推理或训练流程。算子级 benchmark 有利于定位局部差异，但无法完全反映真实模型中数据加载、算子调度、内存复用和多算子融合带来的综合影响。

再次，图融合分析当前缺少有效的 `fusion_result.json` 数据。仓库中的该文件内容为 `null`，只能说明分析脚本的设计方法，不能得出具体融合生效次数或性能收益结论。论文最终版本应使用真实导出的融合统计，或删去无法验证的具体结论。

最后，精度分析仍需进一步细化。本次 45 条 case 均通过 allclose 校验，`conv2d` 在 `atol=1e-2, rtol=1e-2` 阈值下通过且余弦相似度接近 1；但对于更多数据类型、更多输入分布和更大规模 case，仍需要补充测试，避免把单轮实验结果推广到所有应用场景。

### 7.3 后续研究工作展望

后续工作可以从三个方向扩展。第一，扩展测试对象，从算子级 benchmark 扩展到端到端模型推理与训练，包括 CNN、Transformer 和多模态模型中的典型子图，以观察 NPU 在真实模型工作负载下的综合表现。

第二，完善 NPU 机制分析。后续可接入 CANN profiler、图优化日志和算子级 trace，系统记录算子融合、生效 pass、设备利用率、内存带宽和同步开销，并与当前 JSON 结果建立对应关系，从而提高性能归因的可信度。

第三，扩展数据类型和部署场景。当前实验主要围绕 float32 路径和部分 float16 专项实验设计展开，后续可补充 float16、bfloat16、混合精度和量化场景，并分析不同精度设置对延迟、吞吐量和误差指标的影响。对于工程部署场景，还可以增加批量大小、并发请求和长时间稳定性测试，使实验结论更接近实际应用需求。
