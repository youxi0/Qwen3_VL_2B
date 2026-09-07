#!/usr/bin/env python3
"""Rank Qwen3-VL decoder blocks by final-logit recovery when restored to FP16."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import (AWQ_DECODER_LAYER_COUNT, checkpoint_index,
                    inspect_awq_modules, load_config, read_json, read_manifest,
                    versions, write_json)


METRIC_KEYS = (
    "teacher_top1_agreement",
    "teacher_kl",
    "teacher_token_nll_reference",
    "teacher_token_nll_candidate",
)

MODULE_GROUPS = {
    "attention_qkv": (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"
    ),
    "attention_o": ("self_attn.o_proj",),
    "mlp_gate_up": ("mlp.gate_proj", "mlp.up_proj"),
    "mlp_down": ("mlp.down_proj",),
}


def expand_module_group(spec):
    """Expand ``LAYER:GROUP`` into exact decoder Linear module names."""
    try:
        layer_text, group = spec.split(":", 1)
        layer = int(layer_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid module group {spec!r}; use e.g. 3:mlp_down"
        ) from exc
    if not 0 <= layer < AWQ_DECODER_LAYER_COUNT or group not in MODULE_GROUPS:
        raise ValueError(
            f"Invalid module group {spec!r}; groups={list(MODULE_GROUPS)}"
        )
    return [
        f"model.language_model.layers.{layer}.{suffix}"
        for suffix in MODULE_GROUPS[group]
    ]


def aggregate(rows):
    total = sum(row["tokens"] for row in rows)
    return {
        key: sum(row[key] * row["tokens"] for row in rows) / total
        for key in METRIC_KEYS
    }


def score_logits_only(model, processor, references, cfg, label):
    """Evaluate teacher-forced logits without expensive generation."""
    from torch_work import (continuation_logits, logit_metrics,
                            prepare_inputs)

    rows = []
    for index, reference in enumerate(references):
        print(
            f"[{label} {index + 1}/{len(references)}] "
            f"{reference['row']['id']}",
            flush=True,
        )
        inputs = prepare_inputs(processor, reference["row"], cfg)
        tokens = reference["tokens"].to("cuda")
        logits = continuation_logits(model, inputs, tokens)
        rows.append(logit_metrics(reference["logits"].to("cuda"), logits,
                                  tokens))
    return aggregate(rows)


def tensor_bytes(index, prefix):
    return sum(
        item["data_offsets"][1] - item["data_offsets"][0]
        for name, (_, item) in index.items()
        if name == prefix or name.startswith(prefix + ".")
    )


def recovery_row(name, kind, metrics, baseline, extra_mib):
    kl_reduction = baseline["teacher_kl"] - metrics["teacher_kl"]
    nll_reduction = (
        baseline["teacher_token_nll_candidate"]
        - metrics["teacher_token_nll_candidate"]
    )
    top1_gain = (
        metrics["teacher_top1_agreement"]
        - baseline["teacher_top1_agreement"]
    )
    return {
        "name": name,
        "kind": kind,
        "estimated_extra_mib": extra_mib,
        "metrics": metrics,
        "top1_gain": top1_gain,
        "kl_reduction": kl_reduction,
        "candidate_nll_reduction": nll_reduction,
        "kl_reduction_per_extra_mib": (
            kl_reduction / extra_mib if extra_mib > 0 else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True,
                        help="Screening JSONL; use calibration, not final eval")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--layers", type=int, nargs="*",
                        help="Optional subset; default scans every INT4 block")
    parser.add_argument(
        "--module-layers", type=int, nargs="*", default=[],
        help="Scan qkv/o/gate+up/down groups inside these decoder layers",
    )
    parser.add_argument("--skip-block-scan", action="store_true")
    parser.add_argument(
        "--joint-layers", type=int, nargs="*", default=[],
        help="Also evaluate these FP16 blocks together in one forward sweep",
    )
    parser.add_argument(
        "--joint-module", action="append", default=[],
        help="Jointly restore LAYER:GROUP, e.g. 0:attention_qkv; repeat as needed",
    )
    parser.add_argument("--joint-only", action="store_true",
                        help="Skip individual scans and run only joint recovery")
    parser.add_argument("--skip-lm-head", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.max_samples <= 32:
        raise ValueError("Use 1..32 probe samples")
    output = args.out.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh report path: {output}")

    cfg = load_config(args.config, args.root)
    versions(strict=True)
    records = read_manifest(args.probe)
    rng = random.Random(cfg["seed"])
    if len(records) > args.max_samples:
        records = rng.sample(records, args.max_samples)

    awq_index = checkpoint_index(cfg["awq"])
    original_index = checkpoint_index(cfg["original"])
    layout = inspect_awq_modules(awq_index)
    already_fp16 = set(layout["fp16_backbone_layers"])
    requested = [] if (args.joint_only or args.skip_block_scan) else (
        list(range(AWQ_DECODER_LAYER_COUNT))
        if args.layers is None else sorted(set(args.layers))
    )
    if args.joint_only and not (args.joint_layers or args.joint_module):
        raise ValueError("--joint-only requires --joint-layers or --joint-module")
    invalid = [layer for layer in requested
               if not 0 <= layer < AWQ_DECODER_LAYER_COUNT]
    invalid += [layer for layer in args.joint_layers
                if not 0 <= layer < AWQ_DECODER_LAYER_COUNT]
    invalid += [layer for layer in args.module_layers
                if not 0 <= layer < AWQ_DECODER_LAYER_COUNT]
    if invalid:
        raise ValueError(f"Invalid layer indices: {invalid}")
    layers = [layer for layer in requested if layer not in already_fp16]

    import numpy as np
    import torch
    from torch_work import (create_references, load_awq_reference, load_fp16,
                            make_processor, release_cuda,
                            replace_named_submodule, tokenizer_check)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    processor = make_processor(cfg)
    tokenizer_check(cfg, processor)
    print("Loading FP16 and dequantized AWQ models...", flush=True)
    base = load_fp16(cfg["original"])
    candidate, _ = load_awq_reference(cfg)
    references = create_references(base, processor, records, cfg)
    baseline = score_logits_only(
        candidate, processor, references, cfg, "all-INT4 baseline"
    )

    report = {
        "status": "running",
        "config": str(args.config.resolve()),
        "probe_manifest": str(args.probe.resolve()),
        "probe_ids": [reference["row"]["id"] for reference in references],
        "probe_samples": len(references),
        "baseline": baseline,
        "already_fp16_layers": sorted(already_fp16),
        "results": [],
        "module_results": [],
        "selection_warning": (
            "Independent one-at-a-time recoveries are not additive. Select on "
            "this probe set, then validate the resulting mixed checkpoint on "
            "the untouched final evaluation set."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)

    try:
        for layer in layers:
            name = f"model.language_model.layers.{layer}"
            quantized_module = candidate.get_submodule(name)
            replace_named_submodule(candidate, name, base.get_submodule(name))
            try:
                metrics = score_logits_only(
                    candidate, processor, references, cfg, f"FP16 layer {layer}"
                )
            finally:
                replace_named_submodule(candidate, name, quantized_module)
            extra = (
                tensor_bytes(original_index, name)
                - tensor_bytes(awq_index, name)
            ) / 2**20
            report["results"].append(
                recovery_row(str(layer), "decoder_block", metrics, baseline,
                             extra)
            )
            write_json(output, report)

        already_fp16_modules = set(layout["fp16_backbone_modules"])
        for layer in sorted(set(args.module_layers)):
            for group, suffixes in MODULE_GROUPS.items():
                names = [
                    f"model.language_model.layers.{layer}.{suffix}"
                    for suffix in suffixes
                ]
                names = [name for name in names
                         if name not in already_fp16_modules]
                if not names:
                    continue
                saved = {}
                try:
                    for name in names:
                        saved[name] = candidate.get_submodule(name)
                        replace_named_submodule(candidate, name,
                                                base.get_submodule(name))
                    metrics = score_logits_only(
                        candidate, processor, references, cfg,
                        f"FP16 {layer}:{group}",
                    )
                finally:
                    for name, module in saved.items():
                        replace_named_submodule(candidate, name, module)
                extra = sum(
                    tensor_bytes(original_index, name)
                    - tensor_bytes(awq_index, name)
                    for name in names
                ) / 2**20
                report["module_results"].append(
                    recovery_row(f"{layer}:{group}", "module_group",
                                 metrics, baseline, extra)
                )
                write_json(output, report)

        joint_layers = sorted(set(args.joint_layers) - already_fp16)
        if joint_layers:
            saved = {}
            try:
                for layer in joint_layers:
                    name = f"model.language_model.layers.{layer}"
                    saved[name] = candidate.get_submodule(name)
                    replace_named_submodule(candidate, name,
                                            base.get_submodule(name))
                metrics = score_logits_only(
                    candidate, processor, references, cfg,
                    "joint FP16 layers " + ",".join(map(str, joint_layers)),
                )
            finally:
                for name, module in saved.items():
                    replace_named_submodule(candidate, name, module)
            extra = sum(
                tensor_bytes(original_index,
                             f"model.language_model.layers.{layer}")
                - tensor_bytes(awq_index,
                               f"model.language_model.layers.{layer}")
                for layer in joint_layers
            ) / 2**20
            report["joint_result"] = recovery_row(
                ",".join(map(str, joint_layers)), "joint_decoder_blocks",
                metrics, baseline, extra,
            )
            write_json(output, report)

        if args.joint_module:
            group_names = list(dict.fromkeys(args.joint_module))
            exact_names = []
            for spec in group_names:
                exact_names.extend(expand_module_group(spec))
            exact_names = sorted(set(exact_names) - already_fp16_modules)
            if not exact_names:
                raise ValueError("All requested joint module groups are already FP16")
            saved = {}
            try:
                for name in exact_names:
                    saved[name] = candidate.get_submodule(name)
                    replace_named_submodule(candidate, name,
                                            base.get_submodule(name))
                metrics = score_logits_only(
                    candidate, processor, references, cfg,
                    "joint FP16 modules " + ",".join(group_names),
                )
            finally:
                for name, module in saved.items():
                    replace_named_submodule(candidate, name, module)
            extra = sum(
                tensor_bytes(original_index, name)
                - tensor_bytes(awq_index, name)
                for name in exact_names
            ) / 2**20
            report["joint_module_result"] = recovery_row(
                ",".join(group_names), "joint_module_groups", metrics,
                baseline, extra,
            )
            report["joint_module_result"]["exact_modules"] = exact_names
            write_json(output, report)

        if not args.skip_lm_head and layout["lm_head_int4"]:
            quantized_head = candidate.lm_head
            candidate.lm_head = base.lm_head
            try:
                metrics = score_logits_only(
                    candidate, processor, references, cfg, "FP16 lm_head"
                )
            finally:
                candidate.lm_head = quantized_head
            text_cfg = read_json(Path(cfg["original"]) / "config.json")[
                "text_config"
            ]
            fp16_head = text_cfg["vocab_size"] * text_cfg["hidden_size"] * 2
            extra = (fp16_head - tensor_bytes(awq_index, "lm_head")) / 2**20
            report["results"].append(
                recovery_row("lm_head", "lm_head", metrics, baseline, extra)
            )

        report["results_by_kl_per_mib"] = sorted(
            report["results"],
            key=lambda row: row["kl_reduction_per_extra_mib"],
            reverse=True,
        )
        report["results_by_kl_reduction"] = sorted(
            report["results"],
            key=lambda row: row["kl_reduction"],
            reverse=True,
        )
        report["module_results_by_kl_per_mib"] = sorted(
            report["module_results"],
            key=lambda row: row["kl_reduction_per_extra_mib"],
            reverse=True,
        )
        report["status"] = "complete"
        write_json(output, report)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_json(output, report)
        raise
    finally:
        del candidate, base, references
        release_cuda()

    print(f"Layer sensitivity report: {output}")


if __name__ == "__main__":
    main()
