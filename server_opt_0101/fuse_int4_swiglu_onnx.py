#!/usr/bin/env python3
"""把 LLM ONNX 中的 gate/up INT4 投影与 SwiGLU 改写为单个插件节点。"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import helper


PLUGIN_DOMAIN = "trt_edgellm"
GEMM_OP = "Int4GroupwiseGemmPluginV2"
FUSED_OP = "Int4GroupwiseSwiGluPlugin"
FICLONE = 0x40049409


def _attributes(node: onnx.NodeProto) -> dict[str, int]:
    return {
        attr.name: int(helper.get_attribute_value(attr))
        for attr in node.attribute
    }


def _copy_tree_with_links(source: Path, destination: Path) -> None:
    """优先硬链接大权重文件；跨文件系统时退回普通复制。"""
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
            continue
        if relative == Path("llm/model.onnx"):
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.name.endswith(".onnx.data"):
            # ONNX Python 会拒绝读取硬链接的外置数据；CoW 克隆兼顾安全检查与磁盘占用。
            try:
                with (
                    source_path.open("rb") as source_file,
                    destination_path.open("wb") as destination_file,
                ):
                    fcntl.ioctl(destination_file.fileno(), FICLONE, source_file.fileno())
                shutil.copystat(source_path, destination_path)
                continue
            except OSError:
                destination_path.unlink(missing_ok=True)
                shutil.copy2(source_path, destination_path)
                continue
        try:
            os.link(source_path, destination_path)
        except OSError:
            shutil.copy2(source_path, destination_path)


def _single_consumer(
    consumers: dict[str, list[onnx.NodeProto]], tensor: str, op_type: str
) -> onnx.NodeProto | None:
    nodes = consumers.get(tensor, [])
    if len(nodes) != 1 or nodes[0].op_type != op_type:
        return None
    return nodes[0]


def fuse_graph(model: onnx.ModelProto) -> list[dict[str, object]]:
    """识别标准 SiLU(gate) * up 子图，保持原输出张量名不变。"""
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    producer: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)
        for name in node.output:
            producer[name] = node

    remove_ids: set[int] = set()
    replacements: dict[int, onnx.NodeProto] = {}
    records: list[dict[str, object]] = []

    for gate in model.graph.node:
        if gate.domain != PLUGIN_DOMAIN or gate.op_type != GEMM_OP:
            continue
        if len(gate.input) != 3 or len(gate.output) != 1:
            continue
        if ".mlp.gate_proj." not in gate.input[1]:
            continue

        gate_output = gate.output[0]
        gate_consumers = consumers.get(gate_output, [])
        sigmoid = next((node for node in gate_consumers if node.op_type == "Sigmoid"), None)
        silu = next(
            (
                node
                for node in gate_consumers
                if node.op_type == "Mul" and gate_output in node.input
            ),
            None,
        )
        if sigmoid is None or silu is None or len(gate_consumers) != 2:
            continue
        if len(sigmoid.output) != 1 or sigmoid.output[0] not in silu.input:
            continue

        swiglu = _single_consumer(consumers, silu.output[0], "Mul")
        if swiglu is None:
            continue
        up_tensor = next((name for name in swiglu.input if name != silu.output[0]), None)
        up = producer.get(up_tensor or "")
        if (
            up is None
            or up.domain != PLUGIN_DOMAIN
            or up.op_type != GEMM_OP
            or len(up.input) != 3
            or len(up.output) != 1
            or ".mlp.up_proj." not in up.input[1]
            or len(consumers.get(up.output[0], [])) != 1
        ):
            continue
        gate_attrs = _attributes(gate)
        up_attrs = _attributes(up)
        if gate_attrs != up_attrs:
            raise RuntimeError(f"gate/up 属性不一致：{gate.name} 与 {up.name}")

        fused = helper.make_node(
            FUSED_OP,
            inputs=[
                gate.input[0],
                up.input[0],
                gate.input[1],
                gate.input[2],
                up.input[1],
                up.input[2],
            ],
            outputs=list(swiglu.output),
            name=f"{gate.name}_swiglu",
            domain=PLUGIN_DOMAIN,
            **gate_attrs,
        )
        replacements[id(gate)] = fused
        remove_ids.update(map(id, (gate, up, sigmoid, silu, swiglu)))
        records.append(
            {
                "gate_node": gate.name,
                "up_node": up.name,
                "fused_node": fused.name,
                "output": fused.output[0],
                **gate_attrs,
            }
        )

    rewritten: list[onnx.NodeProto] = []
    for node in model.graph.node:
        node_id = id(node)
        if node_id in replacements:
            rewritten.append(replacements[node_id])
        elif node_id not in remove_ids:
            rewritten.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="原始 ONNX 导出目录")
    parser.add_argument("destination", type=Path, help="融合后的新目录")
    parser.add_argument("--expected-layers", type=int, default=28)
    args = parser.parse_args()

    source_model = args.source / "llm/model.onnx"
    destination_model = args.destination / "llm/model.onnx"
    if not source_model.is_file():
        raise FileNotFoundError(source_model)
    if args.destination.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{args.destination}")

    _copy_tree_with_links(args.source, args.destination)
    model = onnx.load(source_model, load_external_data=False)
    original_nodes = len(model.graph.node)
    records = fuse_graph(model)
    if len(records) != args.expected_layers:
        shutil.rmtree(args.destination)
        raise RuntimeError(
            f"应融合 {args.expected_layers} 层，实际只识别到 "
            f"{len(records)} 层；输出目录已清理"
        )

    destination_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, destination_model)
    reloaded = onnx.load(destination_model, load_external_data=False)
    fused_count = sum(node.op_type == FUSED_OP for node in reloaded.graph.node)
    if fused_count != args.expected_layers:
        raise RuntimeError(f"保存后融合节点数量异常：{fused_count}")

    report = {
        "source": str(args.source.resolve()),
        "destination": str(args.destination.resolve()),
        "original_nodes": original_nodes,
        "rewritten_nodes": len(reloaded.graph.node),
        "fused_layers": fused_count,
        "removed_nodes_per_layer": 5,
        "inserted_nodes_per_layer": 1,
        "layers": records,
    }
    report_path = args.destination / "reports/swiglu_fusion.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
