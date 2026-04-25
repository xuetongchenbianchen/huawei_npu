# CPU/NPU 论文整合交付说明

本目录是 CPU/NPU 方向论文材料的整合版产物，已同步到真实 Ascend NPU 扩展算子 benchmark 结果。原始实验结果位于 `workspace/experiments/2026-04-25-real-npu-extended/`。

## 目录内容

- `cpu_npu_integrated_chapters.md`：CPU/NPU 方向第 2 至第 7 章整合版正文草稿。正文仍需按 Word 格式统一图号、表号和章节编号。
- `integration_summary.md`：真实 NPU 数据来源、核心结论、待补事项和写作边界。
- `table_figure_index.md`：论文可用表格和图表索引。
- `tables/`：论文可用 CSV 表格，来源于真实 NPU benchmark CSV。
- `figures/`：论文可用 PNG 图表，已使用真实 NPU extended 数据重生成。

## 建议使用顺序

1. 先查看 `integration_summary.md`，确认真实 NPU 数据来源和核心结论。
2. 从 `tables/operator_overview.csv`、`tables/precision_overview.csv`、`tables/representative_cases.csv` 生成 Word 表格。
3. 按 `table_figure_index.md` 插入 PNG 图表。
4. 再把 `cpu_npu_integrated_chapters.md` 中对应章节复制进 Word，并把其中旧数据表述替换为本目录表格中的新结果。
5. 最后统一 Word 中的表号、图号、引用格式和目录编号。

## 重要边界

- 本目录当前数据来自真实 Ascend NPU，环境为 PyTorch 2.1.0、torch_npu 2.1.0.post13，实际执行设备为 `npu:0`。
- 当前 45 个 case 的精度校验全部通过。
- 当前 `fusion_result.json` 为 `null`，不能写入图融合 pass 生效次数或性能收益数字。
- 本目录只支撑 CPU/NPU 对比，不支撑 GPU/NPU 结论。
