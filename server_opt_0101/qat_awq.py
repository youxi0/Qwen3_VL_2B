#!/usr/bin/env python3
"""Experimental fixed-grid W4A16 QAT/distillation of an existing AWQ model.

The existing AWQ group scales and input pre-scales remain frozen. FP32 shadow
weights are optimized with STE, then repacked using ModelOpt 0.45.0. No new
inference adapters, formats, or FP16 fallback modules are introduced.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from common import (checkpoint_index, inspect_awq_modules, load_config,
                    read_json, read_manifest, versions, write_json)


METRIC_KEYS = ("teacher_top1_agreement", "teacher_kl",
               "teacher_token_nll_reference", "teacher_token_nll_candidate")


def split_training_rows(rows, evaluation, holdout_samples, train_samples, seed):
    """Split by image content/text prompt, never by record ID alone."""
    def identity(row):
        if row.get("image"):
            return ("image", row["image_sha256"])
        return ("text", row["question"].strip())

    forbidden = {identity(row) for row in evaluation}
    if any(identity(row) in forbidden for row in rows):
        raise ValueError("QAT data overlaps the final evaluation images/text prompts")
    groups = {}
    for row in rows:
        groups.setdefault(identity(row), []).append(dict(row))
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    if holdout_samples < 1 or len(keys) <= holdout_samples:
        raise ValueError("Need more distinct training items than --holdout-samples")
    held_keys = keys[:holdout_samples]
    # One record per held-out content item keeps the comparison inexpensive.
    holdout = [groups[key][0] for key in held_keys]
    train = [row for key in keys[holdout_samples:] for row in groups[key]]
    random.Random(seed + 1).shuffle(train)
    if train_samples:
        train = train[:train_samples]
    if not train:
        raise ValueError("No training rows remain after splitting")
    return train, holdout


def select_trainable_modules(packed_names, scope):
    if scope == "head":
        names = [name for name in packed_names if name == "lm_head"]
    elif scope == "decoder":
        names = [name for name in packed_names
                 if name.startswith("model.language_model.layers.")]
    elif scope == "all":
        names = list(packed_names)
    else:
        raise ValueError(f"Unknown QAT scope: {scope}")
    if not names:
        raise ValueError(f"No packed INT4 modules found for scope={scope}")
    return sorted(names)


def packed_change_summary(original, current):
    """Count actual changed INT4 values, including two values in each byte."""
    import numpy as np
    if set(original) != set(current):
        raise ValueError("Packed module sets differ")
    changed_bytes, changed_codes = {}, {}
    total_codes = 0
    for name in sorted(original):
        before, after = np.asarray(original[name]), np.asarray(current[name])
        if (before.dtype != np.uint8 or after.dtype != np.uint8
                or before.shape != after.shape):
            raise ValueError(f"Packed shape/dtype mismatch: {name}")
        left, right = before.reshape(-1), after.reshape(-1)
        byte_count = code_count = 0
        for start in range(0, left.size, 2**20):
            delta = np.bitwise_xor(left[start:start + 2**20],
                                   right[start:start + 2**20])
            byte_count += int(np.count_nonzero(delta))
            code_count += int(np.count_nonzero(delta & 15))
            code_count += int(np.count_nonzero(delta >> 4))
        changed_bytes[name] = byte_count
        changed_codes[name] = code_count
        total_codes += 2 * left.size
    total_changed = sum(changed_codes.values())
    return {
        "changed_packed_bytes": changed_bytes,
        "changed_int4_values": changed_codes,
        "total_changed_int4_values": total_changed,
        "total_int4_values": total_codes,
        "changed_fraction": total_changed / total_codes if total_codes else 0.0,
    }


def make_qat_linear_class():
    import torch
    from torch import nn
    from torch.nn import functional as functional

    class FixedGridSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, master, scales, group_size):
            grouped = master.reshape(master.shape[0], -1, group_size)
            codes = (grouped / scales[:, :, None]).round().clamp(-8, 7)
            return (codes * scales[:, :, None]).reshape_as(master).half()

        @staticmethod
        def backward(ctx, grad_output):
            # Identity STE. Project shadow weights back into the signed INT4
            # range after each update; scales/pre-scales are never optimized.
            return grad_output.float(), None, None

    class QATLinear(nn.Module):
        def __init__(self, packed, scales, pre_scale, bias, group_size):
            super().__init__()
            from torch_work import unpack_awq_numpy
            integers = torch.from_numpy(unpack_awq_numpy(packed.cpu().numpy()))
            scales = scales.detach().float()
            self.out_features = integers.shape[0]
            self.in_features = integers.shape[1]
            self.group_size = group_size
            if (self.in_features % group_size or
                    tuple(scales.shape) != (
                        self.out_features, self.in_features // group_size)):
                raise ValueError("AWQ scale/group shape mismatch")
            if not torch.isfinite(scales).all() or not (scales > 0).all():
                raise ValueError("AWQ scales must be finite and positive")
            integers = integers.to(scales.device).float()
            master = (integers.reshape(self.out_features, -1, group_size)
                      * scales[:, :, None]).reshape_as(integers)
            self.master_weight = nn.Parameter(master)
            self.register_buffer("weight_scale", scales)
            self.register_buffer("pre_quant_scale", pre_scale.detach().half().reshape(-1))
            self.register_buffer("bias", None if bias is None else bias.detach().half())

        def forward(self, inputs):
            weight = FixedGridSTE.apply(
                self.master_weight, self.weight_scale, self.group_size)
            return functional.linear(inputs * self.pre_quant_scale, weight, self.bias)

        @torch.no_grad()
        def initialize_within_bins(self, original_weight, margin=0.49):
            """Recover FP16 residuals while strictly preserving every INT4 code.

            Original W / AWQ input pre-scale is in the shadow-weight domain.
            Clip it inside the existing code's bin rather than requantizing
            the original model and losing the validated AWQ assignment.
            """
            if tuple(original_weight.shape) != tuple(self.master_weight.shape):
                raise ValueError("FP16 source/QAT weight shape mismatch")
            if not 0 < margin < 0.5:
                raise ValueError("Initialization margin must be in (0, 0.5)")
            for start in range(0, self.out_features, 256):
                stop = min(start + 256, self.out_features)
                weight = self.master_weight[start:stop]
                scale = self.weight_scale[start:stop, :, None]
                center = weight.reshape(stop - start, -1, self.group_size)
                original = (original_weight[start:stop].float()
                            / self.pre_quant_scale.float()[None, :])
                original = original.reshape_as(center)
                if not torch.isfinite(original).all():
                    raise ValueError("Non-finite FP16 within-bin initialization")
                center.copy_(torch.maximum(
                    torch.minimum(original, center + margin * scale),
                    center - margin * scale))
            self.project()

        @torch.no_grad()
        def shadow_diagnostics(self):
            total = maximum = 0.0
            for start in range(0, self.out_features, 256):
                grouped = self.master_weight[start:start + 256].reshape(
                    -1, self.in_features // self.group_size, self.group_size)
                normalized = grouped / self.weight_scale[start:start + 256, :, None]
                offset = (normalized - normalized.round()).abs()
                total += offset.sum().item()
                maximum = max(maximum, offset.max().item())
            return {"mean_abs_subbin_offset": total / self.master_weight.numel(),
                    "max_abs_subbin_offset": maximum}

        @torch.no_grad()
        def project(self):
            grouped = self.master_weight.reshape(self.out_features, -1, self.group_size)
            bound = self.weight_scale[:, :, None]
            grouped.copy_(torch.maximum(torch.minimum(grouped, 7 * bound), -8 * bound))

        @torch.no_grad()
        def packed_weight(self):
            from modelopt.torch.export.quant_utils import pack_int4_in_uint8
            # Pack the FP32 shadow directly, avoiding a second rounding via
            # FP16 weights at INT4 bin boundaries.
            return pack_int4_in_uint8(self.master_weight, self.weight_scale).cpu()

    return QATLinear


def install_qat_modules(model, cfg, names, teacher=None, init_mode="centers"):
    import torch
    from torch_work import TensorStore, replace_named_submodule
    QATLinear = make_qat_linear_class()
    group = read_json(Path(cfg["awq"]) / "hf_quant_config.json")["quantization"]["group_size"]
    modules = {}
    source = TensorStore(cfg["awq"])
    try:
        for name in names:
            old = model.get_submodule(name)
            packed = source.get(name + ".weight")
            scales = source.get(name + ".weight_scale").to("cuda", dtype=torch.float32)
            pre_scale = source.get(name + ".pre_quant_scale").to("cuda")
            layer = QATLinear(packed, scales, pre_scale, old.bias, group)
            if init_mode == "fp16-in-bin":
                if teacher is None:
                    raise ValueError("FP16 within-bin initialization requires the teacher")
                layer.initialize_within_bins(teacher.get_submodule(name).weight)
            elif init_mode != "centers":
                raise ValueError(f"Unknown QAT initialization: {init_mode}")
            if not torch.equal(layer.packed_weight(), packed.cpu()):
                raise ValueError(f"QAT initialization does not reproduce AWQ bytes: {name}")
            replace_named_submodule(model, name, layer)
            modules[name] = layer
    finally:
        source.close()
    return modules


def answer_logits(model, inputs, tokens, require_grad=False):
    """Use only answer positions, with the same M-RoPE/mask handling as eval."""
    import torch
    from torch_work import continuation_inputs, reset_rope
    kwargs = continuation_inputs(inputs, tokens)
    start = inputs["input_ids"].shape[1] - 1
    positions = torch.arange(start, start + tokens.shape[1], device=tokens.device)
    reset_rope(model)
    # Unlike continuation_logits(), this path explicitly permits autograd.
    with torch.set_grad_enabled(require_grad):
        result = model(**kwargs, use_cache=False, return_dict=True,
                       logits_to_keep=positions)
        logits = result.logits
    if logits.shape[1] != tokens.shape[1] or not torch.isfinite(logits).all():
        raise ValueError("QAT answer logits are invalid")
    return logits


def distillation_loss(student, teacher, tokens, temperature, ce_weight):
    import torch.nn.functional as functional
    log_student = functional.log_softmax(student.float() / temperature, dim=-1)
    log_teacher = functional.log_softmax(teacher.float() / temperature, dim=-1)
    kl = (log_teacher.exp() * (log_teacher - log_student)).sum(-1).mean()
    kl = kl * temperature ** 2
    ce = functional.cross_entropy(student.float().reshape(-1, student.shape[-1]),
                                  tokens.reshape(-1))
    return (1 - ce_weight) * kl + ce_weight * ce, kl.detach(), ce.detach()


def get_answer_tokens(teacher, processor, inputs, row, cfg, answer_source="dataset"):
    """Choose the continuation without changing the original manifest/answers.

    Teacher mode deliberately ignores annotated answers for recovery training.
    Dataset mode retains the old behavior, generating only when no answer exists.
    """
    from torch_work import greedy
    if answer_source not in ("dataset", "teacher"):
        raise ValueError(f"Unknown answer source: {answer_source}")
    if answer_source == "dataset" and row.get("answer", "").strip():
        import torch
        ids = processor.tokenizer.encode(row["answer"], add_special_tokens=False)
        if not ids or len(ids) > cfg["max_new_tokens"]:
            raise ValueError(
                f"{row['id']}: answer length {len(ids)} exceeds the configured "
                f"1..{cfg['max_new_tokens']}; shorten it or raise max_new_tokens")
        return torch.tensor([ids], dtype=torch.long, device="cuda")
    tokens = greedy(teacher, inputs, cfg).clone()
    if tokens.ndim != 2 or tokens.shape[0] != 1 or not 1 <= tokens.shape[1] <= cfg["max_new_tokens"]:
        raise ValueError(f"{row['id']}: teacher returned an invalid continuation")
    return tokens


def build_holdout_references(teacher, processor, rows, cfg,
                             answer_source="dataset", label="holdout"):
    from torch_work import prepare_inputs
    references = []
    for index, row in enumerate(rows):
        print(f"[QAT {label} teacher {index + 1}/{len(rows)}] {row['id']}", flush=True)
        inputs = prepare_inputs(processor, row, cfg)
        tokens = get_answer_tokens(teacher, processor, inputs, row, cfg, answer_source)
        logits = answer_logits(teacher, inputs, tokens)
        references.append({"row": row, "tokens": tokens.cpu(), "logits": logits.cpu()})
    return references


def reference_token_summary(references):
    lengths = [int(reference["tokens"].shape[1]) for reference in references]
    return {"samples": len(lengths), "tokens": sum(lengths),
            "mean_answer_tokens": sum(lengths) / len(lengths) if lengths else 0,
            "min_answer_tokens": min(lengths) if lengths else 0,
            "max_answer_tokens": max(lengths) if lengths else 0}


def score_holdout(model, processor, references, cfg):
    from torch_work import logit_metrics, prepare_inputs
    model.eval()
    rows = []
    for reference in references:
        inputs = prepare_inputs(processor, reference["row"], cfg)
        tokens = reference["tokens"].to("cuda")
        logits = answer_logits(model, inputs, tokens)
        rows.append(logit_metrics(reference["logits"].to("cuda"), logits, tokens))
    total = sum(row["tokens"] for row in rows)
    return {key: sum(row[key] * row["tokens"] for row in rows) / total
            for key in METRIC_KEYS}


def snapshot_packed(modules):
    return {name: module.packed_weight() for name, module in modules.items()}


def export_checkpoint(cfg, packed_weights, destination):
    """Index unchanged source shards and store only updated packed tensors."""
    from safetensors.torch import save_file
    from splice_awq_fp16_modules import copy_auxiliary_files, link_or_copy, tensor_nbytes
    from torch_work import TensorStore

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    copy_auxiliary_files(cfg["awq"], destination)
    source = checkpoint_index(cfg["awq"])
    overwritten = {name + ".weight" for name in packed_weights}
    kept = {name: value for name, value in source.items() if name not in overwritten}
    paths = sorted({path for path, _ in kept.values()})
    filenames = {}
    for number, path in enumerate(paths, 1):
        filename = f"qat-source-{number:05d}-of-{len(paths):05d}.safetensors"
        link_or_copy(path, destination / filename)
        filenames[path] = filename
    weight_map = {name: filenames[path] for name, (path, _) in kept.items()}
    # One shard per changed Linear bounds host memory, even with --scope all.
    for number, (module_name, packed) in enumerate(sorted(packed_weights.items()), 1):
        key = module_name + ".weight"
        if tuple(packed.shape) != tuple(source[key][1]["shape"]):
            raise ValueError(f"QAT packed weight shape changed: {module_name}")
        filename = f"qat-weight-{number:05d}.safetensors"
        save_file({key: packed.contiguous()}, str(destination / filename))
        weight_map[key] = filename
    write_json(destination / "model.safetensors.index.json", {
        "metadata": {"total_size": sum(tensor_nbytes(item)
                                        for _, item in source.values())},
        "weight_map": weight_map,
    })
    layout = inspect_awq_modules(checkpoint_index(destination))
    original_layout = inspect_awq_modules(source)
    if layout != original_layout:
        raise ValueError("QAT export changed the source INT4/FP16 layout")
    # Read the actual exported bytes back through the same checkpoint reader.
    exported = TensorStore(destination)
    try:
        import torch
        for name, expected in packed_weights.items():
            if not torch.equal(exported.get(name + ".weight"), expected):
                raise ValueError(f"Exported QAT bytes mismatch: {name}")
    finally:
        exported.close()
    return layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True,
                        help="Calibration/training JSONL; image and text rows supported")
    parser.add_argument("--eval", type=Path, required=True,
                        help="Final eval JSONL, used only to reject training overlap")
    parser.add_argument("--scope", choices=["head", "decoder", "all"], default="head")
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--holdout-samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--init", choices=["centers", "fp16-in-bin"],
                        default="fp16-in-bin",
                        help="Shadow initialization; both modes preserve AWQ step-zero bytes")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.0,
                        help="Optional answer CE mixture; default is teacher KL only")
    parser.add_argument("--answer-source", choices=["dataset", "teacher"], default="dataset",
                        help="Continuation for both train/holdout; teacher ignores annotated answers")
    parser.add_argument("--train-probe-samples", type=int, default=0,
                        help="Fixed training subset to rescore at baseline/every eval; never selects checkpoints")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--diversify-prompts", action="store_true")
    parser.add_argument("--out", type=Path, required=True,
                        help="Fresh experiment directory; checkpoint is written inside it")
    args = parser.parse_args()
    if (min(args.steps, args.grad_accum, args.eval_every, args.train_samples) < 1
            or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.temperature) or args.temperature <= 0
            or not 0 <= args.ce_weight <= 1 or args.train_probe_samples < 0):
        parser.error("Invalid steps/accumulation/LR/temperature/CE settings")
    output = args.out.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh experiment directory: {output}")
    cfg = load_config(args.config, args.root)
    environment = versions(strict=True)
    rows = read_manifest(args.train)
    evaluation = read_manifest(args.eval)
    train, holdout = split_training_rows(
        rows, evaluation, args.holdout_samples, args.train_samples, cfg["seed"])
    if args.train_probe_samples > len(train):
        parser.error("--train-probe-samples exceeds the actual training row count")
    diversified = 0
    if args.diversify_prompts:
        from requantize_awq_lm_head import diversify_repeated_questions
        train, diversified = diversify_repeated_questions(train)
    index = checkpoint_index(cfg["awq"])
    layout = inspect_awq_modules(index)
    names = select_trainable_modules(layout["packed_modules"], args.scope)
    qcfg = read_json(Path(cfg["awq"]) / "hf_quant_config.json")["quantization"]
    if (qcfg.get("quant_algo") != "W4A16_AWQ" or qcfg.get("group_size") != 128
            or qcfg.get("has_zero_point", False)
            or not layout["lm_head_int4"]):
        raise ValueError("Expected group-128 symmetric W4A16 AWQ with INT4 LM head")

    import numpy as np
    import torch
    from torch_work import (load_awq_reference, load_fp16, make_processor,
                            prepare_inputs, release_cuda, tokenizer_check)
    if not torch.cuda.is_available():
        raise RuntimeError("Run QAT on the Linux CUDA server")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True)
    report = {
        "status": "initializing", "experimental": True,
        "method": "Fixed-grid W4A16 STE quantization-aware distillation",
        "scope": args.scope, "source_awq": cfg["awq"],
        "source_config": dict(cfg),
        "source_layout": {**layout, "packed_modules": sorted(layout["packed_modules"])},
        "answer_source": args.answer_source,
        "metric_note": "Teacher consistency on the selected continuation, not VQA task accuracy. "
                       "Compare only runs with identical data, split and answer_source.",
        "train_rows": train, "holdout_rows": holdout,
        "diversified_prompts": diversified, "trained_modules": names,
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
        "environment": environment, "history": [], "best_step": 0,
        "selection": "Lowest KL on calibration-internal holdout, including baseline step 0",
        "final_eval_used_for_selection": False,
        "engine_verified": False,
    }
    report_path = output / "qat_report.json"
    write_json(report_path, report)
    teacher = student = references = probe_references = optimizer = modules = None
    try:
        processor = make_processor(cfg)
        tokenizer_check(cfg, processor)
        print("Loading frozen FP16 teacher and validated AWQ student...", flush=True)
        teacher = load_fp16(cfg["original"])
        student, _ = load_awq_reference(cfg)
        references = build_holdout_references(
            teacher, processor, holdout, cfg, args.answer_source)
        probe_references = build_holdout_references(
            teacher, processor, train[:args.train_probe_samples], cfg,
            args.answer_source, label="train probe")
        baseline = score_holdout(student, processor, references, cfg)
        baseline_probe = (score_holdout(student, processor, probe_references, cfg)
                          if probe_references else None)
        modules = install_qat_modules(student, cfg, names, teacher, args.init)
        # Both the packed bytes and held-out forward must match at step zero.
        initialized = score_holdout(student, processor, references, cfg)
        if any(abs(initialized[key] - baseline[key]) > 1e-6 for key in METRIC_KEYS):
            raise ValueError(f"QAT step-zero mismatch: before={baseline}, after={initialized}")
        parameters = [module.master_weight for module in modules.values()]
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0,
                                      foreach=False)
        scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
        if args.scope != "head":
            student.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            student.enable_input_require_grads()
        student.config.use_cache = False
        best_kl = baseline["teacher_kl"]
        original_packed = snapshot_packed(modules)
        best_packed = original_packed
        best_metrics = baseline
        report.update({"status": "training", "baseline": baseline,
                       "step_zero": initialized,
                       "baseline_train_probe": baseline_probe,
                       "holdout_token_summary": reference_token_summary(references),
                       "train_probe_token_summary": reference_token_summary(probe_references),
                       "train_probe_ids": [reference["row"]["id"] for reference in probe_references],
                       "trainable_parameters": sum(p.numel() for p in parameters)})
        write_json(report_path, report)
        print(f"QAT baseline: {baseline}", flush=True)
        print(f"QAT answer source: {args.answer_source}; "
              f"holdout tokens: {report['holdout_token_summary']}", flush=True)
        if baseline_probe is not None:
            print(f"[QAT train probe 0] {baseline_probe}", flush=True)
        print(f"Trainable INT4 modules: {len(names)}; parameters: "
              f"{report['trainable_parameters']:,}", flush=True)
        # Teacher continuations of probe rows are identical in training/scoring.
        token_cache = {index: reference["tokens"]
                       for index, reference in enumerate(probe_references)}
        rng = random.Random(cfg["seed"])
        schedule = []
        consecutive_overflows = 0
        successful_updates = 0
        torch.cuda.reset_peak_memory_stats()
        for step in range(1, args.steps + 1):
            if args.scope == "head":
                student.eval()  # Gradients remain enabled for the head.
            else:
                student.train()
                student.model.visual.eval()
                # Train-mode exists to activate checkpointing, not dropout.
                for module in student.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.eval()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = kl_sum = ce_sum = 0.0
            for _ in range(args.grad_accum):
                if not schedule:
                    schedule = list(range(len(train)))
                    rng.shuffle(schedule)
                row_index = schedule.pop()
                row = train[row_index]
                inputs = prepare_inputs(processor, row, cfg)
                if row_index not in token_cache:
                    tokens = get_answer_tokens(
                        teacher, processor, inputs, row, cfg, args.answer_source)
                    token_cache[row_index] = tokens.cpu()
                tokens = token_cache[row_index].to("cuda")
                teacher_logits = answer_logits(teacher, inputs, tokens)
                student_logits = answer_logits(student, inputs, tokens, require_grad=True)
                loss, kl, ce = distillation_loss(
                    student_logits, teacher_logits, tokens,
                    args.temperature, args.ce_weight)
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite training loss at step {step}")
                scaler.scale(loss / args.grad_accum).backward()
                loss_sum += loss.item() / args.grad_accum
                kl_sum += kl.item() / args.grad_accum
                ce_sum += ce.item() / args.grad_accum
                del student_logits, teacher_logits, loss, kl, ce, inputs, tokens
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            finite_gradient = bool(torch.isfinite(grad_norm))
            scaler.step(optimizer)  # Automatically skips an overflowing update.
            scaler.update()
            if finite_gradient:
                consecutive_overflows = 0
                successful_updates += 1
                for module in modules.values():
                    module.project()
            else:
                consecutive_overflows += 1
                if consecutive_overflows >= 5:
                    raise ValueError("Five consecutive FP16 gradient overflows; reduce LR")
            record = {"step": step, "loss": loss_sum, "train_kl": kl_sum,
                      "train_ce": ce_sum,
                      "gradient_norm": grad_norm.item() if finite_gradient else None,
                      "skipped_overflow": not finite_gradient,
                      "loss_scale": scaler.get_scale()}
            print(f"[QAT {step}/{args.steps}] loss={loss_sum:.6f} "
                  f"KL={kl_sum:.6f} CE={ce_sum:.6f}", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                metrics = score_holdout(student, processor, references, cfg)
                record["holdout"] = metrics
                if probe_references:
                    probe_metrics = score_holdout(student, processor, probe_references, cfg)
                    record["train_probe"] = probe_metrics
                    report["last_train_probe"] = probe_metrics
                    print(f"[QAT train probe {step}] {probe_metrics}", flush=True)
                current_packed = snapshot_packed(modules)
                changes = packed_change_summary(original_packed, current_packed)
                record["quantization"] = changes
                record["shadow"] = {
                    name: module.shadow_diagnostics()
                    for name, module in modules.items()
                }
                print(f"[QAT holdout {step}] {metrics}", flush=True)
                print(f"[QAT INT4 {step}] changed values="
                      f"{changes['total_changed_int4_values']}/"
                      f"{changes['total_int4_values']} "
                      f"({changes['changed_fraction']:.6%})", flush=True)
                if metrics["teacher_kl"] < best_kl:
                    best_kl = metrics["teacher_kl"]
                    best_metrics = metrics
                    best_packed = current_packed
                    report["best_step"] = step
                report["last_quantization"] = changes
                report["last_metrics"] = metrics
                del current_packed
            report["history"].append(record)
            write_json(report_path, report)

        report.update({"best_metrics": best_metrics,
                       "holdout_kl_reduction": baseline["teacher_kl"] - best_kl,
                       "improved_on_holdout": report["best_step"] > 0,
                       "successful_updates": successful_updates,
                       "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})
        best_changes = packed_change_summary(original_packed, best_packed)
        report["changed_packed_bytes"] = best_changes["changed_packed_bytes"]
        report["best_quantization"] = best_changes
        checkpoint = output / "checkpoint"
        export_checkpoint(cfg, best_packed, checkpoint)
        next_cfg = dict(cfg)
        next_cfg["awq"] = str(checkpoint)
        next_cfg["awq_onnx"] = str(output / "llm_onnx")
        write_json(output / "config.qat.json", next_cfg)
        report.update({"status": "complete", "checkpoint": str(checkpoint),
                       "note": "Final eval and TensorRT engine checks are still required. "
                       "Only the best calibration-holdout checkpoint is exported; "
                       "step 0 is retained if no training checkpoint improves KL."})
        write_json(report_path, report)
        print(f"QAT best step={report['best_step']}, metrics={best_metrics}", flush=True)
        print(f"Exported checkpoint: {checkpoint}", flush=True)
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        try:
            write_json(report_path, report)
        except OSError:
            pass
        raise
    finally:
        teacher = student = references = probe_references = optimizer = modules = None
        release_cuda()


if __name__ == "__main__":
    main()
