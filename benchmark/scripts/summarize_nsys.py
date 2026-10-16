#!/usr/bin/env python3
"""从 Nsight Systems SQLite 中提取面向优化的简明统计。"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


def union_duration(intervals: Iterable[tuple[int, int]]) -> tuple[int, list[int]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0, []
    merged: list[tuple[int, int]] = []
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    gaps = [merged[index + 1][0] - merged[index][1] for index in range(len(merged) - 1)]
    return sum(end - start for start, end in merged), gaps


def has_table(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def top_named_durations(
    connection: sqlite3.Connection, table: str, name_column: str, limit: int
) -> list[dict[str, Any]]:
    if not has_table(connection, table) or not has_table(connection, "StringIds"):
        return []
    query = f"""
        SELECT COALESCE(s.value, '[unknown]') AS name,
               COUNT(*) AS calls,
               SUM(t.end - t.start) AS duration_ns,
               AVG(t.end - t.start) AS average_ns
        FROM {table} AS t
        LEFT JOIN StringIds AS s ON s.id = t.{name_column}
        GROUP BY t.{name_column}
        ORDER BY duration_ns DESC
        LIMIT ?
    """
    return [
        {
            "name": row[0],
            "calls": int(row[1]),
            "total_ms": float(row[2]) / 1e6,
            "average_us": float(row[3]) / 1e3,
        }
        for row in connection.execute(query, (limit,))
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--runtime-json", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    connection = sqlite3.connect(args.sqlite)
    activity_tables = [
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
    ]
    intervals: list[tuple[int, int]] = []
    counts: dict[str, int] = {}
    for table in activity_tables:
        if not has_table(connection, table):
            continue
        rows = list(connection.execute(f"SELECT start, end FROM {table}"))
        intervals.extend((int(row[0]), int(row[1])) for row in rows)
        counts[table] = len(rows)

    busy_ns, gaps = union_duration(intervals)
    span_ns = max((end for _, end in intervals), default=0) - min((start for start, _ in intervals), default=0)
    top_kernels = top_named_durations(
        connection, "CUPTI_ACTIVITY_KIND_KERNEL", "shortName", args.top
    )
    top_cuda_apis = top_named_durations(
        connection, "CUPTI_ACTIVITY_KIND_RUNTIME", "nameId", args.top
    )

    memcpy: list[dict[str, Any]] = []
    if has_table(connection, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        enum_join = ""
        label = "CAST(m.copyKind AS TEXT)"
        if has_table(connection, "ENUM_CUDA_MEMCPY_OPER"):
            enum_join = "LEFT JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id = m.copyKind"
            label = "COALESCE(e.label, e.name, CAST(m.copyKind AS TEXT))"
        query = f"""
            SELECT {label}, COUNT(*), SUM(m.bytes), SUM(m.end-m.start)
            FROM CUPTI_ACTIVITY_KIND_MEMCPY m
            {enum_join}
            GROUP BY m.copyKind
            ORDER BY SUM(m.end-m.start) DESC
        """
        memcpy = [
            {
                "kind": row[0],
                "calls": int(row[1]),
                "bytes": int(row[2]),
                "total_ms": float(row[3]) / 1e6,
            }
            for row in connection.execute(query)
        ]

    runtime_metrics: dict[str, Any] = {}
    if args.runtime_json and args.runtime_json.exists():
        runtime_metrics = json.loads(args.runtime_json.read_text(encoding="utf-8"))

    result = {
        "trace_gpu_span_ms": span_ns / 1e6,
        "gpu_busy_union_ms": busy_ns / 1e6,
        "gpu_busy_percent_within_gpu_span": busy_ns * 100.0 / span_ns if span_ns else 0.0,
        "gpu_idle_gap_total_ms": sum(gaps) / 1e6,
        "gpu_idle_gap_mean_us": (sum(gaps) / len(gaps) / 1e3) if gaps else 0.0,
        "kernel_launches": counts.get("CUPTI_ACTIVITY_KIND_KERNEL", 0),
        "memcpy_calls": counts.get("CUPTI_ACTIVITY_KIND_MEMCPY", 0),
        "memset_calls": counts.get("CUPTI_ACTIVITY_KIND_MEMSET", 0),
        "top_kernels": top_kernels,
        "top_cuda_apis": top_cuda_apis,
        "memcpy": memcpy,
        "runtime": runtime_metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Nsight Systems 摘要",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| GPU activity span | {result['trace_gpu_span_ms']:.3f} ms |",
        f"| GPU union busy | {result['gpu_busy_union_ms']:.3f} ms |",
        f"| GPU busy/span | {result['gpu_busy_percent_within_gpu_span']:.2f}% |",
        f"| Kernel launches | {result['kernel_launches']} |",
        f"| Memcpy calls | {result['memcpy_calls']} |",
        "",
        "## GPU kernel 总时间 Top",
        "",
        "| 排名 | Kernel | 调用次数 | 总时间/ms | 平均/us |",
        "|---:|---|---:|---:|---:|",
    ]
    for index, kernel in enumerate(top_kernels, start=1):
        name = str(kernel["name"]).replace("|", "\\|")
        lines.append(
            f"| {index} | {name} | {kernel['calls']} | {kernel['total_ms']:.4f} | {kernel['average_us']:.3f} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
