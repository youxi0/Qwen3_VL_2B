#!/usr/bin/env python3
"""合并按阶段采集的 Nsight Systems 摘要。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGE_ORDER = ("prefill", "decode", "visual")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--stages", default=",".join(STAGE_ORDER))
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    stages: dict[str, dict[str, Any]] = {}
    requested_stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    unknown_stages = set(requested_stages) - set(STAGE_ORDER)
    if unknown_stages:
        raise RuntimeError(f"不支持的 NSYS stage：{sorted(unknown_stages)}")
    for stage in requested_stages:
        path = args.input_dir / stage / "summary.json"
        if path.exists():
            stages[stage] = json.loads(path.read_text(encoding="utf-8"))
    if not stages:
        raise RuntimeError(f"{args.input_dir} 中没有可合并的 NSYS stage 摘要")

    span_ms = sum(float(item.get("trace_gpu_span_ms", 0.0)) for item in stages.values())
    busy_ms = sum(float(item.get("gpu_busy_union_ms", 0.0)) for item in stages.values())
    result = {
        "collection_mode": "isolated_stages",
        "trace_gpu_span_ms": span_ms,
        "gpu_busy_union_ms": busy_ms,
        "gpu_busy_percent_within_gpu_span": busy_ms * 100.0 / span_ms if span_ms else 0.0,
        "kernel_launches": sum(int(item.get("kernel_launches", 0)) for item in stages.values()),
        "memcpy_calls": sum(int(item.get("memcpy_calls", 0)) for item in stages.values()),
        "stages": stages,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Nsight Systems 分阶段摘要",
        "",
        "为降低 4 GiB Jetson 上的采集内存，prefill、decode、visual 使用独立进程采集。",
        "汇总时间是各阶段之和，不表示一次端到端请求的墙钟时间；端到端口径请看 E2E 报告。",
        "",
        "| 阶段 | GPU activity span/ms | GPU union busy/ms | Busy/span | Kernel launches | Memcpy calls |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage, item in stages.items():
        lines.append(
            f"| {stage} | {item.get('trace_gpu_span_ms', 0.0):.3f} | "
            f"{item.get('gpu_busy_union_ms', 0.0):.3f} | "
            f"{item.get('gpu_busy_percent_within_gpu_span', 0.0):.2f}% | "
            f"{item.get('kernel_launches', 0)} | {item.get('memcpy_calls', 0)} |"
        )
    lines.extend(["", "阶段详情："])
    for stage in stages:
        lines.append(f"- [{stage} 摘要]({stage}/summary.md)")
        lines.append(f"- [{stage} NSYS 报告]({stage}/{stage}.nsys-rep)")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
