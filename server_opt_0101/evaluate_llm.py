#!/usr/bin/env python3
"""Evaluate an existing mixed AWQ checkpoint without training or calibration.

Vision remains FP16 to isolate LLM quantization. This does not validate an INT8
Vision artifact, TensorRT kernels, or Jetson memory. Supports PTQ and QAT configs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import (checkpoint_index, inspect_awq_modules, load_config,
                    read_manifest, versions, write_json)
from quality_metrics import NORMALIZATION, evaluation_fingerprint, normalize_answer


def generation_summary(samples):
    annotated = [row for row in samples if "ground_truth_answer" in row]
    return {
        "sample_count": len(samples),
        "generation_exact_to_fp16": sum(row["generation_exact_match_to_fp16"] for row in samples),
        "annotated_sample_count": len(annotated),
        "fp16_single_answer_exact": sum(
            row["reference_text"].strip() == row["ground_truth_answer"].strip()
            for row in annotated),
        "candidate_single_answer_exact": sum(row["ground_truth_exact_match"] for row in annotated),
        "fp16_normalized_exact": sum(
            normalize_answer(row["reference_text"]) == normalize_answer(row["ground_truth_answer"])
            for row in annotated),
        "candidate_normalized_exact": sum(
            normalize_answer(row["candidate_text"]) == normalize_answer(row["ground_truth_answer"])
            for row in annotated),
        "normalization": NORMALIZATION,
        "note": "Single-answer, case-sensitive stripped-text exact match only; "
                "NOT official VQAv2 multi-annotator accuracy. Inspect generated text too.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, args.root)
    rows = read_manifest(args.eval)
    if not rows:
        raise ValueError("Evaluation manifest is empty")
    output = args.out.expanduser().resolve()
    protected = [Path(cfg[key]).resolve() for key in ("original", "awq", "awq_onnx", "images")]
    if any(output == path or output.is_relative_to(path) or path.is_relative_to(output)
           for path in protected):
        raise ValueError("Output must be separate from model and data input directories")
    if output.exists():
        raise FileExistsError(f"Preserving previous reports; choose a fresh directory: {output}")
    environment = versions(strict=True)
    layout = inspect_awq_modules(checkpoint_index(cfg["awq"]))

    import numpy as np
    import torch
    from torch_work import (create_references, evaluate_candidate, load_awq_reference,
                            load_fp16, make_processor, release_cuda, tokenizer_check)
    if not torch.cuda.is_available():
        raise RuntimeError("Run on the Linux CUDA server")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True)
    write_json(output / "evaluation.json", {
        "config": cfg, "evaluation": rows, "environment": environment,
        "layout": {**layout, "packed_modules": sorted(layout["packed_modules"])},
        "calibration_performed": False, "training_performed": False,
        "vision_precision": "FP16", "engine_verified": False, "jetson_fit_verified": False,
    })
    teacher = candidate = references = None
    try:
        processor = make_processor(cfg)
        tokenizer_check(cfg, processor)
        teacher = load_fp16(cfg["original"])
        references = create_references(teacher, processor, rows, cfg)
        # References are CPU tensors. Avoid holding both models during generation.
        teacher = None
        release_cuda()
        candidate, _ = load_awq_reference(cfg)
        scores = evaluate_candidate(candidate, processor, references, cfg,
                                    "Existing mixed LLM + FP16 Vision")
        scores["evaluation_fingerprint"] = evaluation_fingerprint(rows)
        scores["evaluation_settings"] = {key: cfg[key] for key in (
            "original", "min_image_tokens", "max_image_tokens", "max_input_tokens", "max_new_tokens")}
        manifest_by_id = {row["id"]: row for row in rows}
        for sample in scores["samples"]:
            sample["source_subset"] = manifest_by_id[sample["id"]].get("source_subset", "unspecified")
        summary = generation_summary(scores["samples"])
        write_json(output / "alignment.json", scores)
        write_json(output / "summary.json", {"metrics": scores["metrics"], **summary})
        print("\n[Existing mixed LLM + FP16 Vision]")
        for name, value in scores["metrics"].items():
            print(f"{name}: {value}")
        print(f"generation exact to FP16: {summary['generation_exact_to_fp16']}/{len(rows)}")
        print("Single-annotation exact (NOT official VQA accuracy): "
              f"FP16={summary['fp16_single_answer_exact']}/{summary['annotated_sample_count']}, "
              f"candidate={summary['candidate_single_answer_exact']}/{summary['annotated_sample_count']}")
        print("Normalized single-answer exact (proxy): "
              f"FP16={summary['fp16_normalized_exact']}/{summary['annotated_sample_count']}, "
              f"candidate={summary['candidate_normalized_exact']}/{summary['annotated_sample_count']}")
        print(f"Reports: {output}")
    except Exception as exc:
        try:
            write_json(output / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        except OSError:
            pass
        raise
    finally:
        teacher = candidate = references = None
        release_cuda()


if __name__ == "__main__":
    main()
