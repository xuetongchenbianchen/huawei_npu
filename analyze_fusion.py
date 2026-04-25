#!/usr/bin/env python3
"""分析 Ascend 图融合结果。"""
from __future__ import annotations

import argparse
import json


def load_graph_fusion(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} 不是有效的 fusion 统计对象，当前顶层类型为 {type(data).__name__}")

    fusion = {}
    for _, session in data.items():
        if not isinstance(session, dict):
            continue
        graph_fusion = session.get("graph_fusion", {})
        if not isinstance(graph_fusion, dict):
            continue
        for pass_name, stats in graph_fusion.items():
            if not isinstance(stats, dict):
                continue
            fusion[pass_name] = {
                "match_times": int(stats.get("match_times", 0)),
                "effect_times": int(stats.get("effect_times", 0)),
            }
    return fusion


def summarize(fusion):
    if not fusion:
        return (
            "未发现可分析的图融合统计。\n"
            "当前输入文件可能为空、内容为 null，或没有 graph_fusion 字段。\n"
            "需要先通过 CANN/torch_npu 的图模式或 profiler 导出包含 graph_fusion 的 fusion_result.json。"
        )

    total_match = sum(item["match_times"] for item in fusion.values())
    total_effect = sum(item["effect_times"] for item in fusion.values())
    effect_ratio = total_effect / total_match if total_match else 0.0

    lines = [
        f"融合 Pass 总数: {len(fusion)}",
        f"总匹配次数: {total_match}",
        f"总生效次数: {total_effect}",
        f"整体生效率: {effect_ratio:.2%}",
        "",
        "各 Pass 详情:",
    ]

    sorted_items = sorted(
        fusion.items(),
        key=lambda item: (item[1]["effect_times"], item[1]["match_times"]),
        reverse=True,
    )
    for pass_name, stats in sorted_items:
        match_times = stats["match_times"]
        effect_times = stats["effect_times"]
        ratio = effect_times / match_times if match_times else 0.0
        lines.append(
            f"- {pass_name}: match={match_times}, effect={effect_times}, effect_ratio={ratio:.2%}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="分析 fusion_result.json")
    parser.add_argument("path", nargs="?", default="fusion_result.json", help="fusion_result.json 路径")
    args = parser.parse_args()

    fusion = load_graph_fusion(args.path)
    print(summarize(fusion))


if __name__ == "__main__":
    main()
