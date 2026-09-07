#!/usr/bin/env python3
"""Create Qwen3-VL W4A16-AWQ with an additional INT4 AWQ LM head."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import (checkpoint_index, inspect_awq_modules, read_json,
                    read_manifest, versions, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    if not 1 <= args.num_samples <= 128:
        raise ValueError("Use 1..128 multimodal AWQ calibration samples")
    rows = read_manifest(args.calib, require_image=True)
    rows = rows[:args.num_samples]
    versions(strict=True)

    import numpy as np
    import torch
    from tensorrt_edgellm.quantization import quantize_and_export

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def image_dataset():
        for row in rows:
            yield row["image"], row["question"]

    quantize_and_export(
        model_dir=str(model_dir),
        output_dir=str(output),
        quantization="int4_awq",
        lm_head_quantization="int4_awq",
        visual_quantization=None,
        dtype="fp16",
        device="cuda",
        image_dataset=image_dataset,
        num_samples=len(rows),
    )

    index = checkpoint_index(output)
    layout = inspect_awq_modules(index)
    if not layout["lm_head_int4"] or layout["total_packed_linears"] != 197:
        raise RuntimeError(f"LM-head INT4 export validation failed: {layout}")
    quant = read_json(output / "hf_quant_config.json")["quantization"]
    if quant.get("quant_algo") != "W4A16_AWQ" or quant.get("group_size") != 128:
        raise RuntimeError("Expected symmetric group-128 W4A16_AWQ metadata")
    report = {
        "model_dir": str(model_dir),
        "output_dir": str(output),
        "calibration_manifest": str(args.calib.resolve()),
        "calibration_samples": len(rows),
        "seed": args.seed,
        "layout": {k: v for k, v in layout.items() if k != "packed_modules"},
        "note": (
            "Backbone AWQ was recalibrated from FP16 together with the LM head; "
            "the visual tower remains FP16 in this checkpoint."
        ),
    }
    write_json(output / "lm_head_int4_report.json", report)
    print(f"Validated 196 backbone AWQ linears + INT4 lm_head: {output}")


if __name__ == "__main__":
    main()
