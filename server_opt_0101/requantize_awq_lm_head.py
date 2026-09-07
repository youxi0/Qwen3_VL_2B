#!/usr/bin/env python3
"""Create Qwen3-VL W4A16-AWQ with an additional INT4 AWQ LM head."""
from __future__ import annotations

import argparse
import copy
import importlib
import random
from collections import Counter
from pathlib import Path

from common import (CALIBRATION_IMAGE_PROMPTS, checkpoint_index,
                    inspect_awq_modules, read_json, read_manifest, versions,
                    write_json)


def diversify_repeated_questions(rows):
    """Replace only repeated, unlabeled generic prompts with a prompt mix."""
    rows = [dict(row) for row in rows]
    counts = Counter(row["question"].strip() for row in rows)
    changed = 0
    occurrences = Counter()
    for row in rows:
        question = row["question"].strip()
        occurrence = occurrences[question]
        occurrences[question] += 1
        # Never invalidate a supplied answer by changing its question. Four
        # repetitions indicate a generated generic prompt rather than useful
        # task-specific variation.
        if (row.get("image") and not row.get("answer", "").strip()
                and counts[question] >= 4):
            row["question"] = CALIBRATION_IMAGE_PROMPTS[
                occurrence % len(CALIBRATION_IMAGE_PROMPTS)]
            changed += int(row["question"] != question)
    return rows, changed


def select_calibration_rows(image_rows, text_rows, total, text_fraction, seed):
    """Choose a deterministic image/text mix with at least one image."""
    if not image_rows:
        raise ValueError("AWQ VLM calibration requires at least one image sample")
    if not 0.0 <= text_fraction < 1.0:
        raise ValueError("--text-fraction must be in [0, 1)")

    rng = random.Random(seed)
    image_pool = [dict(row) for row in image_rows]
    text_pool = [dict(row) for row in text_rows]
    rng.shuffle(image_pool)
    rng.shuffle(text_pool)

    desired_text = round(total * text_fraction) if text_pool else 0
    text_count = min(len(text_pool), desired_text, max(0, total - 1))
    image_count = min(len(image_pool), total - text_count)
    remaining = total - image_count - text_count
    if remaining:
        extra_text = min(remaining, len(text_pool) - text_count)
        text_count += extra_text
        remaining -= extra_text
    if remaining:
        image_count += min(remaining, len(image_pool) - image_count)

    selected = image_pool[:image_count] + text_pool[:text_count]
    if not selected or not any(row.get("image") for row in selected):
        raise ValueError("Calibration selection produced no multimodal sample")
    rng.shuffle(selected)
    return selected


def build_calibration_batches(processor, rows, max_seq_len):
    """Materialize full user/assistant conversations for ModelOpt AWQ."""
    import torch
    from PIL import Image

    batches = []
    lengths = []
    for row in rows:
        content = []
        if row.get("image"):
            with Image.open(row["image"]) as source:
                image = source.convert("RGB")
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": row["question"]})
        messages = [{"role": "user", "content": content}]

        answer = row.get("answer", "").strip()
        if answer:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            })
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=not bool(answer),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        length = int(inputs["input_ids"].shape[-1])
        if length > max_seq_len:
            raise ValueError(
                f"{row['id']}: calibration sequence has {length} tokens, "
                f"above --max-seq-len={max_seq_len}; shorten this record"
            )
        lengths.append(length)
        batches.append({
            key: value
            for key, value in inputs.items()
            if value is not None
            and not (isinstance(value, torch.Tensor) and value.numel() == 0)
        })
    return batches, lengths


def configure_image_token_range(processor, min_image_tokens, max_image_tokens):
    """Match AWQ calibration image preprocessing to the deployment profile."""
    image_processor = processor.image_processor
    patch_size = int(image_processor.patch_size)
    merge_size = int(image_processor.merge_size)
    if patch_size != 16 or merge_size != 2:
        raise ValueError(
            "This calibration path is specific to Qwen3-VL patch=16, merge=2"
        )
    pixels_per_merged_token = (patch_size * merge_size) ** 2
    image_processor.size = {
        "shortest_edge": min_image_tokens * pixels_per_merged_token,
        "longest_edge": max_image_tokens * pixels_per_merged_token,
    }
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = image_processor.size["shortest_edge"]
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = image_processor.size["longest_edge"]
    return pixels_per_merged_token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True,
                        help="Image JSONL with question and optional answer")
    parser.add_argument("--text-calib", type=Path,
                        help="Optional text-only JSONL with question/answer")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Total image + text samples; Edge 0.10.1 caps VLM AWQ at 128")
    parser.add_argument("--text-fraction", type=float, default=0.25,
                        help="Fraction reserved for --text-calib when supplied")
    parser.add_argument("--alpha-step", type=float, default=0.05,
                        help="AWQ alpha search step; ModelOpt default is 0.1")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--min-image-tokens", type=int, default=16)
    parser.add_argument("--max-image-tokens", type=int, default=64)
    parser.add_argument("--keep-image-questions", action="store_true",
                        help="Do not diversify repeated unlabeled image questions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    if not 1 <= args.num_samples <= 128:
        raise ValueError("Use 1..128 total AWQ calibration samples")
    if not 0.0 < args.alpha_step <= 1.0:
        raise ValueError("--alpha-step must be in (0, 1]")
    if args.max_seq_len < 32:
        raise ValueError("--max-seq-len must be at least 32")
    if not 1 <= args.min_image_tokens <= args.max_image_tokens:
        raise ValueError(
            "Expected 1 <= --min-image-tokens <= --max-image-tokens"
        )

    image_rows = read_manifest(args.calib, require_image=True)

    text_rows = []
    if args.text_calib:
        text_rows = read_manifest(args.text_calib, require_image=False)
        with_images = [row["id"] for row in text_rows if row.get("image")]
        if with_images:
            raise ValueError(
                "--text-calib must contain text-only rows; image rows found: "
                + ", ".join(with_images[:5])
            )

    rows = select_calibration_rows(
        image_rows, text_rows, args.num_samples, args.text_fraction, args.seed
    )
    if args.keep_image_questions:
        diversified = 0
    else:
        rows, diversified = diversify_repeated_questions(rows)
    versions(strict=True)

    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Edge-LLM 0.10.1 exposes neither AWQ alpha_step nor a mixed VLM/text
    # loader. Patch its in-process configuration and loader, restore both in
    # finally, and fail if its private API no longer has the expected shape.
    quant_module = importlib.import_module(
        "tensorrt_edgellm.quantization.quantize"
    )
    config_module = importlib.import_module(
        "tensorrt_edgellm.quantization.quantization_configs"
    )
    original_cfg = config_module._BACKBONE_CFG_MAP["int4_awq"]
    custom_cfg = copy.deepcopy(original_cfg)
    algorithm = custom_cfg.get("algorithm")
    if not isinstance(algorithm, dict) or algorithm.get("method") != "awq_lite":
        raise RuntimeError(
            f"Unexpected Edge-LLM/ModelOpt INT4 AWQ config: {algorithm!r}"
        )
    algorithm["alpha_step"] = args.alpha_step

    original_loader = quant_module._multimodal_calib_dataloader
    batch_metadata = {}

    def mixed_loader(processor, image_dataset, num_samples=128,
                     max_length=512, is_phi4mm=False):
        del image_dataset, num_samples, max_length, is_phi4mm
        configure_image_token_range(
            processor, args.min_image_tokens, args.max_image_tokens
        )
        batches, lengths = build_calibration_batches(
            processor, rows, args.max_seq_len
        )
        observed_image_tokens = []
        for row, batch in zip(rows, batches):
            if not row.get("image"):
                continue
            grid = batch.get("image_grid_thw")
            if grid is None:
                raise ValueError(f"{row['id']}: processor returned no image_grid_thw")
            token_count = int(grid.prod(dim=-1).sum().item()) // 4
            if not args.min_image_tokens <= token_count <= args.max_image_tokens:
                raise ValueError(
                    f"{row['id']}: processed image has {token_count} visual "
                    f"tokens, outside {args.min_image_tokens}..{args.max_image_tokens}"
                )
            observed_image_tokens.append(token_count)
        batch_metadata.update({
            "sequence_token_min": min(lengths),
            "sequence_token_max": max(lengths),
            "sequence_token_mean": sum(lengths) / len(lengths),
            "observed_image_token_min": min(observed_image_tokens),
            "observed_image_token_max": max(observed_image_tokens),
        })
        return batches

    def image_dataset():
        # resolve_dataset requires the registered two-field callable even
        # though mixed_loader consumes the richer rows directly.
        for row in rows:
            if row.get("image"):
                yield row["image"], row["question"]

    image_count = sum(bool(row.get("image")) for row in rows)
    text_count = len(rows) - image_count
    answer_count = sum(bool(row.get("answer", "").strip()) for row in rows)
    print(
        f"Full-model AWQ calibration: alpha_step={args.alpha_step}, "
        f"images={image_count}, text={text_count}, "
        f"assistant_answers={answer_count}"
    )

    config_module._BACKBONE_CFG_MAP["int4_awq"] = custom_cfg
    quant_module._multimodal_calib_dataloader = mixed_loader
    try:
        quant_module.quantize_and_export(
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
    finally:
        quant_module._multimodal_calib_dataloader = original_loader
        config_module._BACKBONE_CFG_MAP["int4_awq"] = original_cfg

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
        "text_calibration_manifest": (
            str(args.text_calib.resolve()) if args.text_calib else None
        ),
        "calibration_samples": len(rows),
        "image_samples": image_count,
        "text_samples": text_count,
        "assistant_answer_samples": answer_count,
        "unique_questions": len({row["question"] for row in rows}),
        "diversified_repeated_image_questions": diversified,
        "awq_alpha_step": args.alpha_step,
        "max_sequence_tokens": args.max_seq_len,
        "min_image_tokens": args.min_image_tokens,
        "max_image_tokens": args.max_image_tokens,
        **batch_metadata,
        "seed": args.seed,
        "layout": {k: v for k, v in layout.items() if k != "packed_modules"},
        "note": (
            "Image and optional text conversations were forwarded through the "
            "complete VLM; decoder AWQ and LM-head AWQ were recalibrated "
            "together from FP16, while the visual tower remained FP16."
        ),
    }
    write_json(output / "lm_head_int4_report.json", report)
    print(f"Validated 196 backbone AWQ linears + INT4 lm_head: {output}")


if __name__ == "__main__":
    main()
