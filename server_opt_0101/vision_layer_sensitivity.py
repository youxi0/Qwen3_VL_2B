#!/usr/bin/env python3
"""Rank INT8 Vision groups by output recovery when restored to FP16."""
from __future__ import annotations

import argparse
import random
from contextlib import contextmanager
from pathlib import Path

from common import load_config, read_manifest, versions, write_json


OUTPUT_NAMES = (
    "output",
    "deepstack_features_0",
    "deepstack_features_1",
    "deepstack_features_2",
)


def summarize_vision_rows(rows, threshold):
    outputs = {}
    for name in OUTPUT_NAMES:
        selected = [row for row in rows if row["output"] == name]
        if not selected:
            raise ValueError(f"Missing Vision measurements for {name}")
        outputs[name] = {
            "mean_cosine": sum(row["mean_token_cosine"] for row in selected)
            / len(selected),
            "worst_cosine": min(row["mean_token_cosine"] for row in selected),
            "worst_relative_l2": max(row["relative_l2"] for row in selected),
            "below_gate": sum(
                row["mean_token_cosine"] < threshold for row in selected
            ),
            "samples": len(selected),
        }
    worst = min(value["worst_cosine"] for value in outputs.values())
    deficit = sum(
        max(0.0, threshold - value["worst_cosine"])
        for value in outputs.values()
    )
    return {
        "gate": threshold,
        "worst_cosine": worst,
        "total_output_gate_deficit": deficit,
        "outputs": outputs,
    }


def target_groups(target_names):
    remaining = set(target_names)
    groups = {}
    for block in range(24):
        prefix = f"model.visual.blocks.{block}."
        names = sorted(name for name in remaining if name.startswith(prefix))
        if names:
            groups[f"block_{block}"] = names
            remaining.difference_update(names)

    merger_prefixes = (
        ("merger", "model.visual.merger."),
        ("deepstack_merger_0", "model.visual.deepstack_merger_list.0."),
        ("deepstack_merger_1", "model.visual.deepstack_merger_list.1."),
        ("deepstack_merger_2", "model.visual.deepstack_merger_list.2."),
    )
    for label, prefix in merger_prefixes:
        names = sorted(name for name in remaining if name.startswith(prefix))
        if names:
            groups[label] = names
            remaining.difference_update(names)
    if remaining:
        raise ValueError(f"Unclassified Vision INT8 modules: {sorted(remaining)}")
    return groups


def estimated_fp16_extra_mib(model, names):
    """Estimate INT8-to-FP16 weight growth, including common SQ sidecars."""
    fp16_bytes = 0
    int8_bytes = 0
    for name in names:
        weight = model.get_submodule(name).weight
        if weight.ndim != 2:
            raise ValueError(f"Expected a 2-D Linear weight: {name}")
        out_features, in_features = weight.shape
        fp16_bytes += weight.numel() * 2
        # Packed checkpoint observed layout: I8 weight, FP32 per-output weight
        # scale, FP32 per-input pre-quant scale, and scalar input scale.
        int8_bytes += weight.numel() + 4 * (out_features + in_features + 1)
    return max(0, fp16_bytes - int8_bytes) / 2**20


def recovery_row(name, kind, metrics, baseline, extra_mib, exact_modules):
    deficit_reduction = (
        baseline["total_output_gate_deficit"]
        - metrics["total_output_gate_deficit"]
    )
    worst_gain = metrics["worst_cosine"] - baseline["worst_cosine"]
    output_gain = (
        metrics["outputs"]["output"]["worst_cosine"]
        - baseline["outputs"]["output"]["worst_cosine"]
    )
    return {
        "name": name,
        "kind": kind,
        "estimated_extra_mib": extra_mib,
        "exact_modules": list(exact_modules),
        "metrics": metrics,
        "worst_cosine_gain": worst_gain,
        "output_worst_cosine_gain": output_gain,
        "gate_deficit_reduction": deficit_reduction,
        "gate_deficit_reduction_per_extra_mib": (
            deficit_reduction / extra_mib if extra_mib > 0 else 0.0
        ),
    }


@contextmanager
def restore_fp16_modules(candidate, teacher, names):
    from torch_work import replace_named_submodule

    saved = {}
    try:
        for name in names:
            saved[name] = candidate.get_submodule(name)
            replace_named_submodule(candidate, name, teacher.get_submodule(name))
        yield
    finally:
        for name, module in saved.items():
            replace_named_submodule(candidate, name, module)


def collect_references(model, processor, records, cfg):
    from torch_work import prepare_inputs, vision_outputs

    references = []
    for index, row in enumerate(records):
        print(f"[FP16 Vision {index + 1}/{len(records)}] {row['id']}", flush=True)
        inputs = prepare_inputs(processor, row, cfg)
        references.append({
            "row": row,
            "inputs": inputs,
            "features": vision_outputs(model, inputs),
        })
    return references


def score_vision(model, references, threshold, label):
    import torch.nn.functional as functional
    from torch_work import vision_outputs

    rows = []
    for index, reference in enumerate(references):
        print(
            f"[{label} {index + 1}/{len(references)}] "
            f"{reference['row']['id']}",
            flush=True,
        )
        values = vision_outputs(model, reference["inputs"])
        for name, expected, actual in zip(
                OUTPUT_NAMES, reference["features"], values):
            expected = expected.float()
            actual = actual.float()
            if expected.shape != actual.shape:
                raise ValueError(f"Vision output shape changed: {name}")
            cosine = functional.cosine_similarity(expected, actual, dim=-1)
            rows.append({
                "id": reference["row"]["id"],
                "output": name,
                "mean_token_cosine": cosine.mean().item(),
                "min_token_cosine": cosine.min().item(),
                "relative_l2": (
                    (actual - expected).norm()
                    / expected.norm().clamp_min(1e-12)
                ).item(),
                "max_abs_error": (actual - expected).abs().max().item(),
            })
    result = summarize_vision_rows(rows, threshold)
    result["samples_detail"] = rows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True,
                        help="Full Vision calibration manifest")
    parser.add_argument("--probe", type=Path,
                        help="Screening manifest; defaults to --calib")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--blocks", type=int, nargs="*",
                        help="Optional block subset; default is all 24")
    parser.add_argument("--module-blocks", type=int, nargs="*", default=[],
                        help="Also scan each INT8 Linear inside these blocks")
    parser.add_argument("--skip-mergers", action="store_true")
    parser.add_argument("--joint-group", action="append", default=[],
                        help="Joint group such as block_3 or deepstack_merger_1")
    parser.add_argument("--joint-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not 1 <= args.max_samples <= 32:
        raise ValueError("Use 1..32 probe samples")
    for block in list(args.blocks or []) + list(args.module_blocks):
        if not 0 <= block < 24:
            raise ValueError(f"Invalid Vision block: {block}")
    if args.joint_only and not args.joint_group:
        raise ValueError("--joint-only requires at least one --joint-group")
    output = args.out.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh report path: {output}")

    cfg = load_config(args.config, args.root)
    versions(strict=True)
    calibration = read_manifest(args.calib, require_image=True)
    probe_path = args.probe or args.calib
    probe = read_manifest(probe_path, require_image=True)
    rng = random.Random(cfg["seed"])
    if len(probe) > args.max_samples:
        probe = rng.sample(probe, args.max_samples)

    import numpy as np
    import torch
    from torch_work import (load_fp16, make_processor, quantize_vision,
                            release_cuda, tokenizer_check)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    teacher = candidate = references = None
    report = None
    try:
        processor = make_processor(cfg)
        tokenizer_check(cfg, processor)
        print("Loading two FP16 models for teacher and Vision candidate...", flush=True)
        teacher = load_fp16(cfg["original"])
        candidate = load_fp16(cfg["original"])
        references = collect_references(teacher, processor, probe, cfg)
        quant_info = quantize_vision(candidate, processor, calibration, cfg)
        targets = [row["module"] for row in quant_info["quantized_modules"]]
        groups = target_groups(targets)
        threshold = cfg["gates"]["vision_mean_token_cosine_min"]
        baseline = score_vision(
            candidate, references, threshold, "all-INT8 Vision baseline"
        )

        requested_blocks = (
            list(range(24)) if args.blocks is None
            else sorted(set(args.blocks))
        )
        requested_groups = []
        if not args.joint_only:
            requested_groups.extend(f"block_{block}" for block in requested_blocks)
            if not args.skip_mergers:
                requested_groups.extend(
                    name for name in groups if not name.startswith("block_")
                )

        report = {
            "status": "running",
            "config": str(args.config.resolve()),
            "calibration_manifest": str(args.calib.resolve()),
            "probe_manifest": str(Path(probe_path).resolve()),
            "probe_ids": [row["id"] for row in probe],
            "probe_samples": len(probe),
            "vision_recipe": cfg["vision_recipe"],
            "smoothquant_alpha": cfg["smoothquant_alpha"],
            "quantized_module_count": len(targets),
            "groups": groups,
            "baseline": baseline,
            "results": [],
            "module_results": [],
            "selection_warning": (
                "One-at-a-time gains are not additive. Select on this probe "
                "set, test the joint rollback, then run the untouched final "
                "evaluation set and rebuild the memory report."
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)

        for group_name in requested_groups:
            if group_name not in groups:
                raise ValueError(f"Group is not INT8 in this recipe: {group_name}")
            names = groups[group_name]
            with restore_fp16_modules(candidate, teacher, names):
                metrics = score_vision(
                    candidate, references, threshold, f"FP16 {group_name}"
                )
            report["results"].append(recovery_row(
                group_name, "vision_group", metrics, baseline,
                estimated_fp16_extra_mib(teacher, names), names,
            ))
            write_json(output, report)

        for block in sorted(set(args.module_blocks)):
            group_name = f"block_{block}"
            if group_name not in groups:
                raise ValueError(f"Block is not INT8 in this recipe: {block}")
            for name in groups[group_name]:
                with restore_fp16_modules(candidate, teacher, [name]):
                    metrics = score_vision(
                        candidate, references, threshold, f"FP16 {name}"
                    )
                report["module_results"].append(recovery_row(
                    name, "vision_linear", metrics, baseline,
                    estimated_fp16_extra_mib(teacher, [name]), [name],
                ))
                write_json(output, report)

        if args.joint_group:
            selected = list(dict.fromkeys(args.joint_group))
            unknown = [name for name in selected if name not in groups]
            if unknown:
                raise ValueError(
                    f"Unknown/non-INT8 joint groups {unknown}; choices={list(groups)}"
                )
            exact_names = sorted({
                module for group_name in selected for module in groups[group_name]
            })
            with restore_fp16_modules(candidate, teacher, exact_names):
                metrics = score_vision(
                    candidate, references, threshold,
                    "joint FP16 " + ",".join(selected),
                )
            report["joint_result"] = recovery_row(
                ",".join(selected), "joint_vision_groups", metrics, baseline,
                estimated_fp16_extra_mib(teacher, exact_names), exact_names,
            )

        report["results_by_deficit_reduction"] = sorted(
            report["results"],
            key=lambda row: row["gate_deficit_reduction"], reverse=True,
        )
        report["results_by_deficit_per_mib"] = sorted(
            report["results"],
            key=lambda row: row["gate_deficit_reduction_per_extra_mib"],
            reverse=True,
        )
        report["results_by_worst_cosine_gain"] = sorted(
            report["results"],
            key=lambda row: row["worst_cosine_gain"], reverse=True,
        )
        report["module_results_by_deficit_reduction"] = sorted(
            report["module_results"],
            key=lambda row: row["gate_deficit_reduction"], reverse=True,
        )
        report["status"] = "complete"
        write_json(output, report)

        print("\nTop Vision FP16 rollback candidates:", flush=True)
        for row in report["results_by_deficit_reduction"][:10]:
            print(
                f"{row['name']:>20}  +{row['estimated_extra_mib']:.1f} MiB  "
                f"worst={row['worst_cosine_gain']:+.6f}  "
                f"output={row['output_worst_cosine_gain']:+.6f}  "
                f"deficit={row['gate_deficit_reduction']:+.6f}",
                flush=True,
            )
    except Exception as exc:
        if report is not None:
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
            write_json(output, report)
        raise
    finally:
        teacher = candidate = references = None
        release_cuda()

    print(f"Vision layer sensitivity report: {output}", flush=True)


if __name__ == "__main__":
    main()
