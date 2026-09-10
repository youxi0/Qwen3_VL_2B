#!/usr/bin/env python3
"""Restore selected AWQ decoder Linear modules to FP16 without recalibration."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from common import (AWQ_DECODER_LAYER_COUNT, AWQ_LINEAR_SUFFIXES,
                    checkpoint_index, inspect_awq_modules, read_json,
                    write_json)
from llm_layer_sensitivity import expand_module_group


def tensor_nbytes(entry):
    return entry["data_offsets"][1] - entry["data_offsets"][0]


def allowed_decoder_modules():
    return {
        f"model.language_model.layers.{layer}.{suffix}"
        for layer in range(AWQ_DECODER_LAYER_COUNT)
        for suffix in AWQ_LINEAR_SUFFIXES
    }


def tensor_belongs_to_any_module(tensor_name, modules):
    return any(
        tensor_name == module or tensor_name.startswith(module + ".")
        for module in modules
    )


def add_quantization_exclusions(config, modules):
    quant = config.get("quantization")
    if not isinstance(quant, dict):
        raise ValueError("hf_quant_config.json has no quantization object")
    exclusions = quant.setdefault("exclude_modules", [])
    if not isinstance(exclusions, list):
        raise ValueError("quantization.exclude_modules must be a list")
    for module in modules:
        if module not in exclusions:
            exclusions.append(module)
    return config


def copy_auxiliary_files(source, destination):
    for path in Path(source).iterdir():
        if path.name == "model.safetensors.index.json" or path.suffix == ".safetensors":
            continue
        target = destination / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def load_selected_fp16_tensors(index, modules):
    import torch
    from safetensors import safe_open

    selected_names = sorted(
        name for name in index if tensor_belongs_to_any_module(name, modules)
    )
    by_file = defaultdict(list)
    for name in selected_names:
        by_file[index[name][0]].append(name)
    tensors = {}
    for path, names in by_file.items():
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in names:
                value = handle.get_tensor(name)
                if value.is_floating_point():
                    value = value.to(torch.float16)
                tensors[name] = value.contiguous()
    return tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--awq", type=Path, required=True,
                        help="Validated AWQ checkpoint to preserve")
    parser.add_argument("--fp16", type=Path, required=True,
                        help="Original FP16/BF16 checkpoint")
    parser.add_argument("--module-group", action="append", default=[],
                        help="LAYER:GROUP from sensitivity report")
    parser.add_argument("--fp16-module", action="append", default=[],
                        help="Exact decoder Linear module name")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = args.awq.expanduser().resolve()
    original = args.fp16.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")

    modules = set(args.fp16_module)
    for group in args.module_group:
        modules.update(expand_module_group(group))
    modules = sorted(modules)
    if not modules:
        raise ValueError("Select at least one --module-group or --fp16-module")
    invalid = sorted(set(modules) - allowed_decoder_modules())
    if invalid:
        raise ValueError(f"Invalid decoder Linear modules: {invalid}")

    source_index = checkpoint_index(source)
    original_index = checkpoint_index(original)
    source_layout = inspect_awq_modules(source_index)
    missing_quantized = sorted(set(modules) - source_layout["packed_modules"])
    if missing_quantized:
        raise ValueError(
            "Selected modules are not packed INT4 in --awq: "
            f"{missing_quantized}"
        )
    if not source_layout["lm_head_int4"]:
        raise ValueError("--awq must retain the validated packed INT4 lm_head")
    missing_original = sorted(
        module for module in modules
        if module + ".weight" not in original_index
    )
    if missing_original:
        raise ValueError(f"Original FP16 weights missing: {missing_original}")

    quant_path = source / "hf_quant_config.json"
    quant_config = read_json(quant_path)
    quant = quant_config.get("quantization", {})
    expected = {
        "quant_algo": "W4A16_AWQ",
        "group_size": 128,
        "has_zero_point": False,
        "pre_quant_scale": True,
    }
    mismatch = {
        key: (quant.get(key), value)
        for key, value in expected.items() if quant.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Unexpected AWQ metadata: {mismatch}")

    output.mkdir(parents=True)
    copy_auxiliary_files(source, output)

    # Reuse every immutable source shard. Selected packed tensors remain as
    # unreferenced bytes in those hard-linked shards; the new index points to
    # the small FP16 fallback file instead.
    source_files = sorted({path for path, _ in source_index.values()})
    source_file_map = {}
    link_modes = []
    for number, source_path in enumerate(source_files, 1):
        target_name = (
            f"awq-source-{number:05d}-of-{len(source_files):05d}.safetensors"
        )
        mode = link_or_copy(source_path, output / target_name)
        source_file_map[source_path] = target_name
        link_modes.append({
            "source": str(source_path), "target": target_name, "mode": mode
        })

    weight_map = {
        name: source_file_map[path]
        for name, (path, _) in source_index.items()
        if not tensor_belongs_to_any_module(name, modules)
    }

    fp16_tensors = load_selected_fp16_tensors(original_index, modules)
    fallback_name = "selected-fp16-modules.safetensors"
    from safetensors.torch import save_file
    save_file(fp16_tensors, str(output / fallback_name))
    weight_map.update({name: fallback_name for name in fp16_tensors})

    total_size = sum(
        tensor_nbytes(source_index[name][1])
        for name in source_index
        if not tensor_belongs_to_any_module(name, modules)
    ) + sum(value.numel() * value.element_size()
            for value in fp16_tensors.values())
    (output / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    add_quantization_exclusions(quant_config, modules)
    write_json(output / "hf_quant_config.json", quant_config)

    output_index = checkpoint_index(output)
    output_layout = inspect_awq_modules(output_index)
    expected_backbone = source_layout["backbone_linears"] - len(modules)
    expected_total = source_layout["total_packed_linears"] - len(modules)
    if (output_layout["backbone_linears"] != expected_backbone
            or output_layout["total_packed_linears"] != expected_total
            or not output_layout["lm_head_int4"]
            or not set(modules).issubset(
                output_layout["fp16_backbone_modules"]
            )):
        raise RuntimeError(f"Mixed checkpoint validation failed: {output_layout}")
    for module in modules:
        entry = output_index[module + ".weight"][1]
        if entry["dtype"] != "F16":
            raise RuntimeError(f"FP16 fallback was not saved as F16: {module}")

    write_json(output / "fp16_module_splice_report.json", {
        "awq_source": str(source),
        "fp16_source": str(original),
        "output": str(output),
        "fp16_modules": modules,
        "fp16_tensor_names": sorted(fp16_tensors),
        "source_layout": {
            key: value for key, value in source_layout.items()
            if key != "packed_modules"
        },
        "output_layout": {
            key: value for key, value in output_layout.items()
            if key != "packed_modules"
        },
        "logical_total_mib": total_size / 2**20,
        "estimated_logical_increase_mib": (
            total_size - sum(tensor_nbytes(item)
                             for _, item in source_index.values())
        ) / 2**20,
        "source_files": link_modes,
        "note": (
            "No AWQ calibration was rerun. Every unselected tensor remains "
            "byte-for-byte from the validated AWQ checkpoint; selected "
            "decoder Linear tensors come from the original model converted "
            "to FP16. Hard-linked source shards are shared immutable files."
        ),
    })
    print(
        f"Validated mixed checkpoint: {output}\n"
        f"FP16 modules ({len(modules)}): {modules}\n"
        f"Packed INT4 linears: {output_layout['total_packed_linears']}"
    )


if __name__ == "__main__":
    main()
