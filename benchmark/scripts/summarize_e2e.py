#!/usr/bin/env python3
"""汇总冷启动请求和稳定态请求的运行时指标。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def median(runs: list[dict[str, Any]], key: str) -> float:
    values = [float(run[key]) for run in runs if run.get(key) is not None]
    return statistics.median(values) if values else 0.0


def aggregate(document: dict[str, Any]) -> dict[str, Any]:
    runs = document.get("runs", [])
    keys = (
        "latency_ms",
        "tokens_per_second",
        "ttft_ms",
        "tpot_ms",
        "decode_tokens_per_second",
        "vision_latency_ms",
        "prefill_latency_ms",
        "decode_latency_ms",
    )
    result = {f"median_{key}": median(runs, key) for key in keys}
    result["runs"] = len(runs)
    result["cuda_graph_captured"] = bool(document.get("cuda_graph_captured", False))
    result["generated_tokens"] = [int(run.get("generated_tokens", 0)) for run in runs]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold-runtime", required=True, type=Path)
    parser.add_argument("--cold-memory", required=True, type=Path)
    parser.add_argument("--steady-runtime", required=True, type=Path)
    parser.add_argument("--steady-memory", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    cold_memory = load(args.cold_memory)
    steady_memory = load(args.steady_memory)
    result = {
        "cold": aggregate(load(args.cold_runtime)),
        "steady": aggregate(load(args.steady_runtime)),
        "memory": {
            "process_peak_rss_mb": max(
                float(cold_memory.get("process_peak_rss_mb", 0.0)),
                float(steady_memory.get("process_peak_rss_mb", 0.0)),
            ),
            "system_ram_peak_used_mb": max(
                int(cold_memory.get("system_ram_peak_used_mb", 0)),
                int(steady_memory.get("system_ram_peak_used_mb", 0)),
            ),
            "system_ram_peak_delta_mb": max(
                int(cold_memory.get("system_ram_peak_delta_mb", 0)),
                int(steady_memory.get("system_ram_peak_delta_mb", 0)),
            ),
            "temperature_peak_c": max(
                float(cold_memory.get("temperature_peak_c", 0.0)),
                float(steady_memory.get("temperature_peak_c", 0.0)),
            ),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cold = result["cold"]
    steady = result["steady"]
    memory = result["memory"]
    lines = [
        "# 端到端指标",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| Peak RAM（系统已用） | {memory['system_ram_peak_used_mb']:.0f} MiB |",
        f"| Peak RAM（相对采样起点） | {memory['system_ram_peak_delta_mb']:.0f} MiB |",
        f"| 进程 Peak RSS | {memory['process_peak_rss_mb']:.1f} MiB |",
        f"| 冷请求 TTFT | {cold['median_ttft_ms']:.3f} ms |",
        f"| 稳定态 TTFT | {steady['median_ttft_ms']:.3f} ms |",
        f"| 稳定态 TPOT | {steady['median_tpot_ms']:.3f} ms/token |",
        f"| Decode tokens/s | {steady['median_decode_tokens_per_second']:.3f} |",
        f"| 冷请求 Vision latency | {cold['median_vision_latency_ms']:.3f} ms |",
        f"| 稳定态 Prefill latency | {steady['median_prefill_latency_ms']:.3f} ms |",
        f"| 稳定态完整请求 | {steady['median_latency_ms']:.3f} ms |",
        "",
        "TTFT 从 `generate()` 请求开始计到首个 token callback；TPOT 为首末 token callback 间隔除以间隔数。",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
