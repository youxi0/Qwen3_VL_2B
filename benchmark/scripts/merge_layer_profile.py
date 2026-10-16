#!/usr/bin/env python3
"""合并 layer timing 与 TensorRT inspector 元数据，并按耗时降序输出。"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PROFILE_SUFFIX = re.compile(r"\s+\[profile\s+\d+\]$")


def base_layer_name(name: str) -> str:
    return PROFILE_SUFFIX.sub("", name)


def join_values(items: list[dict[str, Any]], key: str) -> str:
    return "; ".join(str(item.get(key, "")) for item in items if item.get(key, "") != "")


def shape(dimensions: Any) -> str:
    if not isinstance(dimensions, list):
        return ""
    return "[" + "x".join(str(value) for value in dimensions) + "]"


def shapes(items: list[dict[str, Any]]) -> str:
    return "; ".join(shape(item.get("Dimensions")) for item in items if "Dimensions" in item)


def metadata_text(layer: dict[str, Any]) -> str:
    for key in ("Metadata", "Origin", "ParameterType"):
        value = layer.get(key)
        if isinstance(value, str):
            return value
        if value not in (None, "", [], {}):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return ""


def classify_layer(layer: dict[str, Any], timing: dict[str, str]) -> str:
    layer_type = str(layer.get("LayerType", ""))
    plugin_type = str(layer.get("PluginType", ""))
    tactic = str(layer.get("TacticName", ""))
    name = str(timing.get("layer_name", ""))
    text = " ".join((layer_type, plugin_type, tactic, name)).lower()
    if any(token in text for token in ("attention", "fmha", "flash")):
        return "attention"
    if any(token in text for token in ("gemm", "matmul", "convolution", "conv3d")):
        return "gemm"
    if any(token in text for token in ("reformat", "noop", "copy")):
        return "reformat"
    return timing.get("category", "other")


def load_inspector(path: Path) -> dict[str, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    layers = document.get("Layers", document if isinstance(document, list) else [])
    result: dict[str, dict[str, Any]] = {}
    for layer in layers:
        if isinstance(layer, dict) and isinstance(layer.get("Name"), str):
            result.setdefault(layer["Name"], layer)
    return result


def markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-csv", required=True, type=Path)
    parser.add_argument("--inspector-json", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    inspector = load_inspector(args.inspector_json)
    with args.timing_csv.open("r", encoding="utf-8", newline="") as source:
        timing_rows = list(csv.DictReader(source))

    rows: list[dict[str, Any]] = []
    matched = 0
    for timing in timing_rows:
        timing_name = timing.get("layer_name", "")
        engine_name = base_layer_name(timing_name)
        layer = inspector.get(engine_name, {})
        if layer:
            matched += 1
        inputs = layer.get("Inputs", []) if isinstance(layer.get("Inputs", []), list) else []
        outputs = layer.get("Outputs", []) if isinstance(layer.get("Outputs", []), list) else []
        input_layouts = join_values(inputs, "Format/Datatype")
        output_layouts = join_values(outputs, "Format/Datatype")
        mean_ms = float(timing.get("time_ms_mean", 0.0) or 0.0)
        row: dict[str, Any] = {
            "rank": 0,
            "stage": args.stage,
            "layer_name": timing_name,
            "engine_layer_name": engine_name,
            "layer_type": layer.get("LayerType", layer.get("ParameterType", "")),
            "plugin_type": layer.get("PluginType", ""),
            "plugin_version": layer.get("PluginVersion", ""),
            "stream_id": layer.get("StreamId", ""),
            "layer_layout": f"input({input_layouts}) -> output({output_layouts})",
            "input_tensor_names": join_values(inputs, "Name"),
            "output_tensor_names": join_values(outputs, "Name"),
            "input_shapes": timing.get("input_shapes") or shapes(inputs),
            "output_shapes": timing.get("output_shapes") or shapes(outputs),
            "input_data_types": timing.get("input_data_types") or input_layouts,
            "output_data_types": timing.get("output_data_types") or output_layouts,
            "onnx_op": timing.get("onnx_op") or metadata_text(layer),
            "tactic_name": timing.get("tactic_name") or layer.get("TacticName", layer.get("TacticValue", "")),
            "category": classify_layer(layer, timing),
            "raw_category": timing.get("category", ""),
            "time_ms_mean": mean_ms,
            "time_percent": 0.0,
            "time_ms_std": float(timing.get("time_ms_std", 0.0) or 0.0),
            "time_ms_min": float(timing.get("time_ms_min", 0.0) or 0.0),
            "time_ms_max": float(timing.get("time_ms_max", 0.0) or 0.0),
        }
        for key, value in timing.items():
            if key not in row and key not in {"layer_name"}:
                row[key] = value
        rows.append(row)

    rows.sort(key=lambda row: float(row["time_ms_mean"]), reverse=True)
    total_ms = sum(float(row["time_ms_mean"]) for row in rows)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
        row["time_percent"] = float(row["time_ms_mean"]) * 100.0 / total_ms if total_ms else 0.0

    base_fields = [
        "rank",
        "stage",
        "layer_name",
        "engine_layer_name",
        "layer_type",
        "plugin_type",
        "plugin_version",
        "stream_id",
        "layer_layout",
        "input_tensor_names",
        "output_tensor_names",
        "input_shapes",
        "output_shapes",
        "input_data_types",
        "output_data_types",
        "onnx_op",
        "tactic_name",
        "category",
        "raw_category",
        "time_ms_mean",
        "time_percent",
        "time_ms_std",
        "time_ms_min",
        "time_ms_max",
    ]
    extra_fields = sorted({key for row in rows for key in row if key not in base_fields})
    fieldnames = base_fields + extra_fields

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "stage": args.stage,
        "source_timing_csv": str(args.timing_csv),
        "source_inspector_json": str(args.inspector_json),
        "layers": len(rows),
        "metadata_matches": matched,
        "metadata_match_rate": matched / len(rows) if rows else 0.0,
        "total_layer_time_ms": total_ms,
        "top_layers": rows[:20],
    }
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# {args.stage} Layer Profile（按平均耗时降序）",
        "",
        f"总层数：{len(rows)}；元数据匹配：{matched}/{len(rows)}；层耗时合计：{total_ms:.4f} ms。",
        "",
        "| 排名 | 平均耗时/ms | 占比 | 层名 | 层类型 | 层布局 | 输入 tensor | 输出 tensor | 输入 shape | 输出 shape |",
        "|---:|---:|---:|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {float(row['time_ms_mean']):.4f} | {float(row['time_percent']):.2f}% | "
            f"{markdown(row['layer_name'])} | {markdown(row['layer_type'])} | {markdown(row['layer_layout'])} | "
            f"{markdown(row['input_tensor_names'])} | {markdown(row['output_tensor_names'])} | "
            f"{markdown(row['input_shapes'])} | {markdown(row['output_shapes'])} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
