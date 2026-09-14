#!/usr/bin/env python3
"""将 NCU raw CSV 压缩成目标 kernel 的主要指标。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DISPLAY_NAMES = {
    "gpu__time_duration.sum": "Duration",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM Throughput",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM Throughput",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "L2 Throughput",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "Active Warps",
    "launch__occupancy_limit_blocks": "Occupancy Limit: Blocks",
    "launch__occupancy_limit_registers": "Occupancy Limit: Registers",
    "launch__occupancy_limit_shared_mem": "Occupancy Limit: Shared Memory",
    "launch__occupancy_limit_warps": "Occupancy Limit: Warps",
    "launch__waves_per_multiprocessor": "Waves/SM",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if "Metric Name" in line and "Kernel Name" in line),
        None,
    )
    if header_index is None:
        raise RuntimeError(f"无法在 {path} 中找到 NCU CSV 表头")
    return list(csv.DictReader(lines[header_index:]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    rows = load_rows(args.raw_csv)
    kernels: dict[str, dict[str, Any]] = {}
    for row in rows:
        kernel = row.get("Kernel Name", "[unknown]")
        metric = row.get("Metric Name", "")
        if metric not in DISPLAY_NAMES:
            continue
        entry = kernels.setdefault(kernel, {"kernel_name": kernel, "metrics": {}})
        entry["metrics"][metric] = {
            "label": DISPLAY_NAMES[metric],
            "value": row.get("Metric Value", ""),
            "unit": row.get("Metric Unit", ""),
        }

    result = {
        "selector": args.selector,
        "report": str(args.report),
        "kernels": list(kernels.values()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Nsight Compute 主要指标",
        "",
        f"目标：`{args.selector}`",
        "",
        "| Kernel | 指标 | 数值 | 单位 |",
        "|---|---|---:|---|",
    ]
    for kernel in result["kernels"]:
        name = str(kernel["kernel_name"]).replace("|", "\\|")
        for metric in kernel["metrics"].values():
            lines.append(f"| {name} | {metric['label']} | {metric['value']} | {metric['unit']} |")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
