# CPU/NPU 表格与图表索引

## 表格

### 表 6-x CPU/NPU 实验环境配置

数据来源：

- `workspace/experiments/2026-04-25-real-npu-extended/env/python_env.txt`
- `workspace/experiments/2026-04-25-real-npu-extended/env/npu_smi_before.txt`

| 项目 | 内容 |
| --- | --- |
| Python | 3.10.19 |
| PyTorch | 2.1.0 |
| torch_npu | 2.1.0.post13 |
| NPU 可用状态 | True |
| NPU 数量 | 16 |
| 实际执行设备 | npu:0 |
| NPU 型号 | Ascend910 |

### 表 6-x 各算子平均 CPU/NPU 延迟与加速比

数据来源：`tables/operator_overview.csv`

建议字段：算子、case 数、CPU 平均延迟、NPU 平均延迟、平均加速比、NPU 平均 GFLOPS/等效吞吐、精度通过/失败。

### 表 6-x CPU/NPU 输出精度校验结果

数据来源：`tables/precision_overview.csv`

建议字段：算子、运行数、allclose 通过数、allclose 失败数、最大绝对误差最大值、平均绝对误差均值、最低余弦相似度。

### 表 6-x 代表性测试 Case 结果

数据来源：`tables/representative_cases.csv`

建议字段：算子、case、CPU 平均耗时、NPU 平均耗时、加速比、精度。

### 表 6-x 图融合 Pass 统计表

当前无有效数据。`fusion_result.json` 内容为 `null`，本表不能填写 pass 生效次数或生效率。若后续重新导出有效文件，建议字段如下：

| pass 名称 | match_times | effect_times | effect_ratio | 可能影响的算子链 | 数据来源 |
| --- | ---: | ---: | ---: | --- | --- |
| 待补 | 待补 | 待补 | 待补 | 待补 | 有效 fusion_result.json |

## 图表

| 建议图号 | 图片路径 | 用途 |
| --- | --- | --- |
| 图 6-x | `figures/operator_average_speedup.png` | 展示真实 NPU 各算子平均加速比 |
| 图 6-x | `figures/operator_latency_logscale.png` | 展示 CPU/NPU 平均延迟，对数坐标 |
| 图 6-x | `figures/representative_case_speedup.png` | 展示 Top 10 代表性 case 加速比 |
| 图 6-x | `figures/speedup_summary.png` | 平均加速比汇总图 |
| 图 6-x | `figures/speedup_by_case.png` | 全 case 加速比分布图 |
| 图 6-x | `figures/precision_vs_speedup.png` | 展示性能与精度关系 |
