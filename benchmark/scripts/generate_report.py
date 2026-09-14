#!/usr/bin/env python3
"""合并工具链各阶段输出，生成统一 JSON 与 Markdown 报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_optional(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "未测量"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    e2e = load_optional(run_dir / "e2e/summary.json")
    capacity = load_optional(run_dir / "capacity/summary.json")
    nsys = load_optional(run_dir / "nsys/summary.json")
    quality = load_optional(run_dir / "quality/summary.json")
    ncu_summaries = [load_optional(path) for path in sorted((run_dir / "ncu").glob("*.summary.json"))]

    cold = e2e.get("cold", {})
    steady = e2e.get("steady", {})
    memory = e2e.get("memory", {})
    quality_text = "未配置原模型基线/业务标注"
    if quality.get("status") == "measured":
        quality_text = f"pass={quality.get('pass_rate', 0.0):.2%}, similarity={quality.get('mean_text_similarity')}"

    overall = {
        "peak_ram_system_used_mb": memory.get("system_ram_peak_used_mb"),
        "peak_ram_delta_mb": memory.get("system_ram_peak_delta_mb"),
        "process_peak_rss_mb": memory.get("process_peak_rss_mb"),
        "cold_ttft_ms": cold.get("median_ttft_ms"),
        "steady_ttft_ms": steady.get("median_ttft_ms"),
        "tpot_ms": steady.get("median_tpot_ms"),
        "decode_tokens_per_second": steady.get("median_decode_tokens_per_second"),
        "vision_latency_ms": cold.get("median_vision_latency_ms"),
        "prefill_latency_ms": steady.get("median_prefill_latency_ms"),
        "decode_latency_ms": steady.get("median_decode_latency_ms"),
        "max_stable_context_tokens": capacity.get("max_stable_context_tokens"),
        "max_stable_kv_capacity": capacity.get("max_stable_kv_capacity"),
        "quality": quality_text,
    }
    document = {
        "overall": overall,
        "e2e": e2e,
        "capacity": capacity,
        "nsys": nsys,
        "quality": quality,
        "ncu": ncu_summaries,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Qwen3-VL TensorRT 性能与质量报告",
        "",
        "## 整体指标",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| Peak RAM（系统已用） | {fmt(overall['peak_ram_system_used_mb'], ' MiB')} |",
        f"| Peak RAM（相对采样起点） | {fmt(overall['peak_ram_delta_mb'], ' MiB')} |",
        f"| TTFT（冷请求） | {fmt(overall['cold_ttft_ms'], ' ms')} |",
        f"| TTFT（稳定态） | {fmt(overall['steady_ttft_ms'], ' ms')} |",
        f"| TPOT | {fmt(overall['tpot_ms'], ' ms/token')} |",
        f"| Decode tokens/s | {fmt(overall['decode_tokens_per_second'])} |",
        f"| Vision latency | {fmt(overall['vision_latency_ms'], ' ms')} |",
        f"| Prefill latency | {fmt(overall['prefill_latency_ms'], ' ms')} |",
        f"| 最大稳定 Context | {fmt(overall['max_stable_context_tokens'], ' tokens')} |",
        f"| 最大稳定 KV Capacity | {fmt(overall['max_stable_kv_capacity'], ' tokens')} |",
        f"| 精度/任务质量 | {overall['quality']} |",
        "",
        "## 逐层结果",
        "",
        "逐层 CSV 均按 `time_ms_mean` 降序：",
        "",
    ]
    for stage in ("prefill", "decode", "visual"):
        relative = Path("layers") / f"layers_{stage}_sorted.csv"
        if (run_dir / relative).exists():
            lines.append(f"- [{stage} CSV]({relative.as_posix()})")
    if nsys:
        lines.extend(
            [
                "",
                "## Nsight Systems",
                "",
                f"GPU union busy：{nsys.get('gpu_busy_union_ms', 0.0):.3f} ms；"
                f"activity span 内忙碌率：{nsys.get('gpu_busy_percent_within_gpu_span', 0.0):.2f}%。",
                "",
                "详细结果见 [nsys/summary.md](nsys/summary.md)。",
            ]
        )
    if ncu_summaries:
        lines.extend(
            [
                "",
                "## Nsight Compute",
                "",
                f"已合并 {len(ncu_summaries)} 个按需目标；NCU 不属于默认运行项。",
            ]
        )
    lines.extend(
        [
            "",
            "## 口径",
            "",
            "- TTFT：项目 runtime 收到请求到首个 token callback。",
            "- TPOT：首末 token callback 之间的平均 token 间隔。",
            "- Vision latency：首次端到端请求中的视觉阶段 CUDA event 时间。",
            "- Prefill latency：稳定态端到端请求中的 prefill 阶段 CUDA event 时间。",
            "- 最大稳定容量：在当前 engine 配置上限 shape 处实测通过，不代表长上下文任务质量不下降。",
            "- 精度/任务质量只有提供原模型参考输出或业务关键词标注后才会判定通过。",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
