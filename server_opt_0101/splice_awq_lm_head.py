#!/usr/bin/env python3
"""Combine a proven AWQ decoder checkpoint with an INT4 AWQ LM head."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path

from common import (AWQ_BACKBONE_LINEAR_COUNT, checkpoint_index,
                    inspect_awq_modules, read_json, write_json)


def tensor_nbytes(entry):
    return entry["data_offsets"][1] - entry["data_offsets"][0]


def quantization_metadata(folder):
    path = Path(folder) / "hf_quant_config.json"
    quant = read_json(path)["quantization"]
    expected = {
        "quant_algo": "W4A16_AWQ",
        "group_size": 128,
        "has_zero_point": False,
        "pre_quant_scale": True,
    }
    mismatch = {
        key: (quant.get(key), value)
        for key, value in expected.items()
        if quant.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Unexpected AWQ metadata in {path}: {mismatch}")
    return quant


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


def load_selected_tensors(index, names):
    from safetensors import safe_open

    by_file = defaultdict(list)
    for name in names:
        by_file[index[name][0]].append(name)
    tensors = {}
    for path, keys in by_file.items():
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key)
    return tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, required=True,
                        help="Original 196-Linear AWQ checkpoint")
    parser.add_argument("--lm-head-source", type=Path, required=True,
                        help="Checkpoint containing packed INT4 lm_head")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    backbone = args.backbone.expanduser().resolve()
    head_source = args.lm_head_source.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")

    backbone_quant = quantization_metadata(backbone)
    head_quant = quantization_metadata(head_source)
    backbone_index = checkpoint_index(backbone)
    head_index = checkpoint_index(head_source)
    backbone_layout = inspect_awq_modules(backbone_index)
    head_layout = inspect_awq_modules(head_index)
    if (backbone_layout["backbone_linears"] != AWQ_BACKBONE_LINEAR_COUNT
            or backbone_layout["lm_head_int4"]):
        raise ValueError(
            "--backbone must contain exactly 196 INT4 decoder linears and no "
            "packed lm_head"
        )
    if not head_layout["lm_head_int4"]:
        raise ValueError("--lm-head-source has no packed INT4 lm_head")
    for key in ("quant_algo", "group_size", "has_zero_point", "pre_quant_scale"):
        if backbone_quant.get(key) != head_quant.get(key):
            raise ValueError(f"Incompatible AWQ metadata field: {key}")

    head_keys = sorted(
        name for name in head_index
        if name == "lm_head" or name.startswith("lm_head.")
    )
    required_head = {
        "lm_head.weight", "lm_head.weight_scale", "lm_head.pre_quant_scale"
    }
    if not required_head.issubset(head_keys):
        raise ValueError(
            f"INT4 LM head tensors missing: {sorted(required_head - set(head_keys))}"
        )

    output.mkdir(parents=True)
    try:
        # Use the LM-head source metadata because it declares lm_head quantized.
        copy_auxiliary_files(head_source, output)

        old_weight_map = {}
        link_modes = []
        source_files = sorted({path for path, _ in backbone_index.values()})
        for number, source_path in enumerate(source_files, 1):
            target_name = f"backbone-{number:05d}-of-{len(source_files):05d}.safetensors"
            mode = link_or_copy(source_path, output / target_name)
            link_modes.append({
                "source": str(source_path), "target": target_name, "mode": mode
            })
            for name, (path, _) in backbone_index.items():
                if path == source_path and not name.startswith("lm_head."):
                    old_weight_map[name] = target_name

        from safetensors.torch import save_file
        head_tensors = load_selected_tensors(head_index, head_keys)
        head_filename = "lm-head-int4.safetensors"
        save_file(head_tensors, str(output / head_filename))

        weight_map = dict(old_weight_map)
        weight_map.update({name: head_filename for name in head_keys})
        total_size = sum(
            tensor_nbytes(backbone_index[name][1])
            for name in old_weight_map
        ) + sum(tensor_nbytes(head_index[name][1]) for name in head_keys)
        (output / "model.safetensors.index.json").write_text(
            json.dumps({
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        merged_index = checkpoint_index(output)
        merged_layout = inspect_awq_modules(merged_index)
        if (merged_layout["backbone_linears"] != AWQ_BACKBONE_LINEAR_COUNT
                or not merged_layout["lm_head_int4"]
                or merged_layout["total_packed_linears"] != 197):
            raise RuntimeError(f"Merged checkpoint validation failed: {merged_layout}")
        output_quant = quantization_metadata(output)
        if any(fnmatch("lm_head", pattern)
               for pattern in output_quant.get("exclude_modules", [])):
            raise ValueError(
                "Merged hf_quant_config.json still excludes lm_head; use the "
                "quantized-head checkpoint as --lm-head-source"
            )

        write_json(output / "splice_report.json", {
            "backbone": str(backbone),
            "lm_head_source": str(head_source),
            "output": str(output),
            "backbone_tensor_count": len(old_weight_map),
            "lm_head_tensors": head_keys,
            "layout": {k: v for k, v in merged_layout.items()
                       if k != "packed_modules"},
            "backbone_files": link_modes,
            "logical_total_mib": total_size / 2**20,
            "note": (
                "All decoder and non-lm_head tensors come byte-for-byte from "
                "the proven backbone checkpoint; only packed lm_head tensors "
                "come from lm_head_source. Hardlinks do not duplicate backbone "
                "disk blocks but must be treated as shared immutable files."
            ),
        })
    except Exception:
        # Preserve a failed output for inspection rather than deleting data.
        raise

    print(f"Validated old AWQ decoder + INT4 lm_head checkpoint: {output}")


if __name__ == "__main__":
    main()
