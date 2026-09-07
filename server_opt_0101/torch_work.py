"""Server GPU work. AWQ reference arithmetic is deliberately distinct from a TRT engine."""
from __future__ import annotations

import copy
import gc
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path

from common import (VISION_RECIPE_EXPECTED, checkpoint_index, read_json,
                    write_json)


def vision_linear_is_target(name, recipe):
    """Return whether a Qwen3-VL Vision Linear belongs to a named recipe.

    ``residual_fp16`` keeps the two projections that write directly into each
    transformer residual stream (attention proj and MLP fc2) in FP16.  It
    still quantizes qkv, MLP fc1, and the four merger fc1 layers.
    """
    if recipe not in VISION_RECIPE_EXPECTED:
        raise ValueError(
            f"Unknown vision_recipe {recipe!r}; choose from "
            f"{sorted(VISION_RECIPE_EXPECTED)}")
    if not name.startswith("model.visual."):
        return False
    is_block = name.startswith("model.visual.blocks.")
    if recipe == "all_linears":
        return True
    if recipe == "blocks":
        return is_block
    if recipe == "conservative":
        return is_block or name.endswith(".linear_fc1")
    return ((is_block and
             (name.endswith(".attn.qkv") or
              name.endswith(".mlp.linear_fc1"))) or
            (not is_block and name.endswith(".linear_fc1")))


def unpack_awq_numpy(packed):
    """ModelOpt W4A16: byte low/high nibbles are adjacent OUTPUT rows."""
    import numpy as np
    if packed.dtype != np.uint8 or packed.ndim != 2:
        raise ValueError("Expected ModelOpt uint8 [out/2, in], not AutoAWQ qweight")
    pair = np.stack((packed & 15, packed >> 4), axis=1).astype(np.int8)
    pair[pair >= 8] -= 16
    return pair.reshape(packed.shape[0] * 2, packed.shape[1])


class TensorStore:
    def __init__(self, folder):
        from safetensors import safe_open
        self.index = checkpoint_index(folder)
        self.stack = ExitStack()
        self.files = {path: self.stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
                      for path in {p for p, _ in self.index.values()}}

    def __contains__(self, key):
        return key in self.index

    def get(self, key):
        return self.files[self.index[key][0]].get_tensor(key)

    def close(self):
        self.stack.close()


def load_fp16(folder):
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(folder), dtype=torch.float16, attn_implementation="sdpa", local_files_only=True,
    ).eval().to("cuda")
    model.requires_grad_(False)
    return model


def make_processor(cfg):
    from transformers import AutoProcessor
    # 5.14.1 uses size pixel-area bounds. Set both compatibility spellings.
    processor = AutoProcessor.from_pretrained(cfg["original"], local_files_only=True)
    ip = processor.image_processor
    ip.size = {"shortest_edge": cfg["min_image_tokens"] * 1024,
               "longest_edge": cfg["max_image_tokens"] * 1024}
    if hasattr(ip, "min_pixels"):
        ip.min_pixels = ip.size["shortest_edge"]
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = ip.size["longest_edge"]
    if ip.patch_size != 16 or ip.merge_size != 2:
        raise ValueError("This package is specific to Qwen3-VL patch=16, merge=2")
    return processor


def tokenizer_check(cfg, processor):
    from transformers import AutoTokenizer
    other = AutoTokenizer.from_pretrained(cfg["awq"], local_files_only=True)
    first, second = processor.tokenizer.get_vocab(), other.get_vocab()
    mismatch = [token for token, idx in first.items() if second.get(token) != idx]
    probes = ["描述图像中的内容。", "What is in this image?", "0 1 23 456", "<|im_start|>user\n"]
    probe_mismatches = [p for p in probes if processor.tokenizer.encode(p, add_special_tokens=False) != other.encode(p, add_special_tokens=False)]
    report = {"base_vocab": len(first), "awq_vocab": len(second),
              "token_id_mismatch_count": len(mismatch), "token_id_mismatch_examples": mismatch[:20],
              "encoding_probe_mismatches": probe_mismatches,
              "chat_template_equal": processor.tokenizer.chat_template == other.chat_template,
              "comparison_policy": "Both models receive identical original-processor input_ids and pixel_values; template differences are reported, never mixed."}
    if mismatch or probe_mismatches:
        raise ValueError("Tokenizer mismatch; numerical comparisons would be misleading: " + json.dumps(report, ensure_ascii=False))
    return report


def prepare_inputs(processor, row, cfg):
    import torch
    from PIL import Image
    content = []
    if row.get("image"):
        with Image.open(row["image"]) as im:
            image = im.convert("RGB")
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": row["question"]})
    result = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )
    result.pop("token_type_ids", None)
    if result["input_ids"].shape[1] > cfg["max_input_tokens"]:
        raise ValueError(f"{row['id']}: prompt including visual tokens exceeds {cfg['max_input_tokens']}; shorten the question or choose a larger profile. Image tokens are never truncated.")
    if row.get("image"):
        grids = result["image_grid_thw"]
        if tuple(grids.shape) != (1, 3) or grids[0, 0].item() != 1:
            raise ValueError("First baseline supports one static image per request")
        count = int(grids.prod(dim=-1).sum().item()) // 4
        if not cfg["min_image_tokens"] <= count <= cfg["max_image_tokens"]:
            raise ValueError(f"{row['id']}: actual image tokens={count}, outside configured profile")
    return {k: v.to("cuda", dtype=torch.float16 if v.is_floating_point() else v.dtype)
            if isinstance(v, torch.Tensor) else v for k, v in result.items()}


def reset_rope(model):
    model.model.rope_deltas = None


def load_awq_reference(cfg):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from modelopt.torch.export.quant_utils import pack_int4_in_uint8
    qcfg = read_json(Path(cfg["awq"]) / "hf_quant_config.json")["quantization"]
    if qcfg.get("quant_algo") != "W4A16_AWQ" or qcfg.get("has_zero_point", False):
        raise ValueError("Only symmetric ModelOpt W4A16_AWQ checkpoints are supported")
    group = int(qcfg["group_size"])
    model = load_fp16(cfg["original"])
    source = TensorStore(cfg["awq"])
    stats = []

    class AWQReferenceLinear(torch.nn.Module):
        def __init__(self, weight, scale, bias):
            super().__init__()
            self.register_buffer("weight", weight)
            self.register_buffer("pre_quant_scale", scale)
            self.register_buffer("bias", bias)

        def forward(self, x):
            # Preserve the FP16 activation multiply. Folding this into weights
            # changes rounding and can hide missing pre_quant_scale bugs.
            return F.linear(x * self.pre_quant_scale, self.weight, self.bias)

    try:
        quant_names = {key[:-7] for key, (_, entry) in source.index.items()
                       if key.endswith(".weight") and entry["dtype"] == "U8"}
        expected = cfg.get("expected_awq_linears", 196)
        if len(quant_names) != expected:
            raise ValueError(f"Expected {expected} packed LLM linears; found {len(quant_names)}")
        with torch.no_grad():
            # Copy norms, embedding, visual tower and all unquantized tensors
            # from AWQ, rather than silently substituting original weights.
            for key, dest in model.state_dict().items():
                if key.endswith(".weight") and key[:-7] in quant_names:
                    continue
                if key not in source:
                    if key == "lm_head.weight" and model.config.tie_word_embeddings:
                        continue
                    raise KeyError(f"AWQ checkpoint lacks unquantized tensor {key}")
                value = source.get(key)
                if tuple(value.shape) != tuple(dest.shape):
                    raise ValueError(f"Shape mismatch for {key}")
                dest.copy_(value.to(device=dest.device, dtype=dest.dtype))

            for name in sorted(quant_names):
                if not name.startswith("model.language_model.layers."):
                    raise ValueError(f"Unexpected quantized module {name}")
                old = model.get_submodule(name)
                q = source.get(name + ".weight").numpy()
                integers = unpack_awq_numpy(q)
                scale = source.get(name + ".weight_scale").float().numpy()
                pqs = source.get(name + ".pre_quant_scale").half().reshape(-1)
                if scale.shape != (old.out_features, old.in_features // group):
                    raise ValueError(f"Invalid scale shape: {name}: {scale.shape}")
                if pqs.numel() != old.in_features or not torch.isfinite(pqs).all() or not (pqs > 0).all():
                    raise ValueError(f"Invalid Smooth/AWQ pre_quant_scale: {name}")
                if not np.isfinite(scale).all() or not (scale > 0).all():
                    raise ValueError(f"Invalid group scale: {name}")
                dense_np = (integers.reshape(old.out_features, -1, group).astype(np.float32)
                            * scale[:, :, None]).reshape(old.out_features, old.in_features)
                # Independently check this installed ModelOpt version's
                # packing convention against actual checkpoint bytes.
                if not stats:
                    repacked = pack_int4_in_uint8(torch.from_numpy(dense_np), torch.from_numpy(scale))
                    if not np.array_equal(repacked.numpy(), q):
                        raise ValueError("INT4 unpacking disagrees with installed ModelOpt packing")
                dense = torch.from_numpy(dense_np).to("cuda", dtype=torch.float16)
                pqs = pqs.to("cuda")
                original = old.weight.float()
                effective = dense.float() * pqs.float()[None, :]
                rel = (effective - original).norm() / original.norm().clamp_min(1e-12)
                cosine = F.cosine_similarity(effective.flatten(), original.flatten(), dim=0)
                stats.append({"module": name, "relative_weight_l2": rel.item(), "weight_cosine": cosine.item(),
                              "modelopt_repack_checked": len(stats) == 0})
                parent_name, child = name.rsplit(".", 1)
                setattr(model.get_submodule(parent_name), child, AWQReferenceLinear(dense, pqs, old.bias))
    finally:
        source.close()
    return model.eval(), sorted(stats, key=lambda r: r["relative_weight_l2"], reverse=True)


def greedy(model, inputs, cfg):
    import torch
    reset_rope(model)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, num_beams=1,
                                   max_new_tokens=cfg["max_new_tokens"], use_cache=True)
    return generated[:, inputs["input_ids"].shape[1]:].detach()


def continuation_inputs(inputs, continuation):
    """Append a text answer without leaving prompt-length multimodal metadata.

    Transformers 5.14.1 uses mm_token_type_ids to build M-RoPE positions:
    0=text, 1=image, 2=video. Preserve the prompt types and append text zeros.
    This is a fresh full-sequence forward, not a cached generation step.
    """
    import torch
    prompt_ids = inputs["input_ids"]
    if len(prompt_ids.shape) != 2 or len(continuation.shape) != 2:
        raise ValueError("Prompt and continuation must both have shape [batch, sequence]")
    if prompt_ids.shape[0] != continuation.shape[0] or continuation.shape[1] == 0:
        raise ValueError("Continuation must be non-empty and match the prompt batch size")
    attention_mask = inputs["attention_mask"]
    if tuple(attention_mask.shape) != tuple(prompt_ids.shape):
        raise ValueError("Prompt attention_mask must match input_ids before appending text")
    kwargs = dict(inputs)
    kwargs["input_ids"] = torch.cat([prompt_ids, continuation], dim=1)
    kwargs["attention_mask"] = torch.cat(
        [attention_mask, torch.ones_like(continuation, dtype=attention_mask.dtype)], dim=1)
    mm_types = inputs.get("mm_token_type_ids")
    if mm_types is not None:
        if tuple(mm_types.shape) != tuple(prompt_ids.shape):
            raise ValueError("Prompt mm_token_type_ids must match input_ids before appending text")
        kwargs["mm_token_type_ids"] = torch.cat(
            [mm_types, torch.zeros_like(continuation, dtype=mm_types.dtype)], dim=1)
    elif inputs.get("image_grid_thw") is not None or inputs.get("video_grid_thw") is not None:
        raise ValueError("Multimodal teacher forcing requires processor-provided mm_token_type_ids")
    # Let Qwen3-VL regenerate positions for the entire extended sequence.
    kwargs.pop("position_ids", None)
    kwargs.pop("cache_position", None)
    kwargs.pop("past_key_values", None)
    return kwargs


def continuation_logits(model, inputs, continuation):
    import torch
    kwargs = continuation_inputs(inputs, continuation)
    prompt_length = inputs["input_ids"].shape[1]
    reset_rope(model)
    with torch.inference_mode():
        out = model(**kwargs, use_cache=False, return_dict=True)
        logits = out.logits[:, prompt_length - 1:prompt_length - 1 + continuation.shape[1], :].detach().clone()
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite logits")
    return logits


def create_references(model, processor, records, cfg):
    refs = []
    for i, row in enumerate(records):
        print(f"[FP16 reference {i+1}/{len(records)}] {row['id']}", flush=True)
        inputs = prepare_inputs(processor, row, cfg)
        tokens = greedy(model, inputs, cfg)
        if tokens.numel() == 0:
            raise ValueError(f"Empty reference continuation: {row['id']}")
        logits = continuation_logits(model, inputs, tokens)
        entry = {"row": row, "tokens": tokens.cpu(), "logits": logits.cpu(),
                 "text": processor.tokenizer.decode(tokens[0], skip_special_tokens=True),
                 "prompt_ids": inputs["input_ids"].cpu().tolist()[0]}
        if row.get("answer"):
            import torch
            answer_ids = processor.tokenizer.encode(row["answer"], add_special_tokens=False)
            if len(answer_ids) > cfg["max_new_tokens"]:
                raise ValueError("Reference answer exceeds max_new_tokens; increase limit without truncating labels")
            answer = torch.tensor([answer_ids], dtype=torch.long, device="cuda")
            entry["answer_ids"] = answer.cpu()
            entry["answer_logits"] = continuation_logits(model, inputs, answer).cpu()
        refs.append(entry)
    return refs


def logit_metrics(reference, candidate, tokens):
    import torch
    import torch.nn.functional as F
    reference, candidate = reference.float(), candidate.float()
    if reference.shape != candidate.shape or not torch.isfinite(candidate).all():
        raise ValueError("Invalid candidate logits")
    lp, lq = F.log_softmax(reference, dim=-1), F.log_softmax(candidate, dim=-1)
    count = tokens.numel()
    return {"tokens": count,
            "teacher_top1_agreement": (reference.argmax(-1) == candidate.argmax(-1)).float().mean().item(),
            "teacher_kl": (lp.exp() * (lp - lq)).sum(-1).mean().item(),
            "teacher_token_nll_reference": -lp.gather(-1, tokens[..., None]).mean().item(),
            "teacher_token_nll_candidate": -lq.gather(-1, tokens[..., None]).mean().item()}


def evaluate_candidate(model, processor, refs, cfg, label):
    rows = []
    for i, ref in enumerate(refs):
        print(f"[{label} {i+1}/{len(refs)}] {ref['row']['id']}", flush=True)
        inputs = prepare_inputs(processor, ref["row"], cfg)
        if inputs["input_ids"].cpu().tolist()[0] != ref["prompt_ids"]:
            raise ValueError("Processor changed input IDs between comparison passes")
        tokens = ref["tokens"].to("cuda")
        actual = continuation_logits(model, inputs, tokens)
        scores = logit_metrics(ref["logits"].to("cuda"), actual, tokens)
        generated = greedy(model, inputs, cfg).cpu()
        text = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
        row = {"id": ref["row"]["id"], "image": ref["row"].get("image"),
               "question": ref["row"]["question"], **scores,
               "reference_text": ref["text"], "candidate_text": text,
               "reference_token_ids": ref["tokens"][0].tolist(),
               "candidate_token_ids": generated[0].tolist(),
               "generation_exact_match_to_fp16": generated.tolist() == ref["tokens"].tolist()}
        if "answer_ids" in ref:
            answer = ref["answer_ids"].to("cuda")
            logits = continuation_logits(model, inputs, answer)
            metrics = logit_metrics(ref["answer_logits"].to("cuda"), logits, answer)
            row["ground_truth_answer"] = ref["row"]["answer"]
            row["ground_truth_nll_reference"] = metrics["teacher_token_nll_reference"]
            row["ground_truth_nll_candidate"] = metrics["teacher_token_nll_candidate"]
            row["ground_truth_exact_match"] = text.strip() == ref["row"]["answer"].strip()
        rows.append(row)
    total = sum(r["tokens"] for r in rows)
    mean = {key: sum(r[key] * r["tokens"] for r in rows) / total
            for key in ("teacher_top1_agreement", "teacher_kl", "teacher_token_nll_reference", "teacher_token_nll_candidate")}
    return {"backend": "PyTorch FP16 dequantized AWQ reference; not the TensorRT INT4 kernel",
            "label": label, "sample_count": len(rows), "teacher_token_count": total,
            "metrics": mean, "ground_truth_sample_count": sum("ground_truth_answer" in r for r in rows),
            "interpretation": "Teacher agreement/NLL is numerical consistency, not task accuracy or ground-truth perplexity.",
            "samples": rows}


def vision_outputs(model, inputs):
    import torch
    with torch.inference_mode():
        output = model.model.visual(hidden_states=inputs["pixel_values"], grid_thw=inputs["image_grid_thw"], return_dict=True)
    result = [output.pooler_output] + list(output.deepstack_features)
    if len(result) != 4 or any(not torch.isfinite(x).all() for x in result):
        raise ValueError("Vision must produce four finite tensors")
    return [x.detach().clone() for x in result]


def trt_inputs_from_hf(model, inputs):
    """Exactly the six inputs in the Edge-LLM 0.10.1 TRT-10 visual ABI."""
    import torch
    from transformers.vision_utils import get_vision_bilinear_indices_and_weights, get_vision_position_ids
    vision = model.model.visual
    grid = inputs["image_grid_thw"]
    idx, weight = get_vision_bilinear_indices_and_weights(grid, num_grid_per_side=vision.num_grid_per_side, spatial_merge_size=2)
    position_ids = get_vision_position_ids(grid, 2)
    rope = vision.rotary_pos_emb(position_ids).reshape(-1, 32)
    n = inputs["pixel_values"].shape[0]
    result = {"input": inputs["pixel_values"].half(), "rotary_pos_emb": rope.float(),
              "cu_seqlens": torch.tensor([0, n], device="cuda", dtype=torch.int32),
              "fast_pos_embed_idx": idx.to(device="cuda", dtype=torch.int64),
              "fast_pos_embed_weight": weight.to(device="cuda", dtype=torch.float16),
              "max_seqlen_carrier": torch.zeros(n, device="cuda", dtype=torch.int32)}
    return {k: v.contiguous().cpu().numpy() for k, v in result.items()}


def collect_vision_reference(model, processor, records, cfg, output):
    import numpy as np
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    refs = []
    for row in records:
        if not row.get("image"):
            continue
        inputs = prepare_inputs(processor, row, cfg)
        features = [x.cpu() for x in vision_outputs(model, inputs)]
        case = output / f"case_{len(refs):04d}"
        case.mkdir()
        np.savez(case / "inputs.npz", **trt_inputs_from_hf(model, inputs))
        names = ["output"] + [f"deepstack_features_{i}" for i in range(3)]
        np.savez(case / "fp16_outputs.npz", **{k: v.numpy() for k, v in zip(names, features)})
        write_json(case / "sample.json", {"id": row["id"], "image": row["image"], "grid_thw": inputs["image_grid_thw"].cpu().tolist()})
        refs.append({"row": row, "features": features, "case": case})
    if not refs:
        raise ValueError("Evaluation set needs at least one image for visual accuracy validation")
    return refs


def repair_zero_channel_sq_scale(scale, activation_amax, weight_amax, module_name):
    """Repair only proven 0/0 channels; never hide invalid live-channel data.

    ModelOpt 0.45.0 clamps its scale before calling the smoothing helper, but
    clamp leaves NaN unchanged. For an exactly zero weight column AND exactly
    zero observed activation channel, a neutral multiplier of 1 is equivalent.
    Arrays are small FP32 per-input-channel vectors, not full model weights.
    """
    import numpy as np
    values = np.asarray(scale, dtype=np.float32).reshape(-1).copy()
    act = np.asarray(activation_amax, dtype=np.float32).reshape(-1)
    weight = np.asarray(weight_amax, dtype=np.float32).reshape(-1)
    if not values.size or values.shape != act.shape or values.shape != weight.shape:
        raise ValueError(f"SmoothQuant {module_name}: scale/activation/weight channel shapes disagree")
    if not np.isfinite(act).all() or not np.isfinite(weight).all() or (act < 0).any() or (weight < 0).any():
        raise ValueError(f"SmoothQuant {module_name}: non-finite or negative amax; not a safe zero-channel repair")
    zero_pair = (act == 0) & (weight == 0)
    repair = np.isnan(values) & zero_pair
    invalid = ~np.isfinite(values) | (values <= 0)
    unsafe = invalid & ~repair
    if unsafe.any():
        channels = np.flatnonzero(unsafe)[:16].tolist()
        raise ValueError(f"SmoothQuant {module_name}: invalid scales on channels {channels}; "
                         "only NaN from confirmed zero activation AND zero weight may be repaired")
    values[repair] = 1.0
    details = {"module": module_name, "channels": int(values.size),
               "zero_activation_channels": int((act == 0).sum()),
               "zero_weight_columns": int((weight == 0).sum()),
               "repaired_zero_over_zero_count": int(repair.sum()),
               "repaired_channel_indices": np.flatnonzero(repair).tolist()}
    return values, details


@contextmanager
def modelopt_smoothquant_zero_guard(model, targets):
    """Scoped workaround for ModelOpt 0.45.0; package files stay untouched.

    Keep official calibration, smoothing, weight recalibration and export state.
    Intercept only the point immediately before applying a calculated scale.
    This process-local hook is intended for this single-model CLI, not concurrent
    quantization threads. Restore it even if calibration raises an exception.
    """
    import torch
    from modelopt.torch.quantization import model_calib
    original = model_calib.apply_pre_quant_scale_and_smooth
    allowed = set(targets)
    module_names = None
    checked = []

    def guarded_apply(linear, pre_quant_scale=None):
        nonlocal module_names
        # Resolve after mtq.quantize has converted/replaced modules.
        if module_names is None:
            module_names = {id(module): name for name, module in model.named_modules()}
        name = module_names.get(id(linear), "<unknown>")
        if name not in allowed:
            raise ValueError(f"SmoothQuant tried to smooth non-whitelisted layer {name}")
        act = getattr(linear.input_quantizer, "_amax_for_smoothing", None)
        if pre_quant_scale is None or act is None:
            raise ValueError(f"SmoothQuant {name}: expected ModelOpt 0.45.0 scale and channel statistics")
        with torch.no_grad():
            weight_amax = linear.weight.detach().float().abs().amax(dim=0)
            fixed, details = repair_zero_channel_sq_scale(
                pre_quant_scale.detach().float().cpu().numpy(),
                act.detach().float().cpu().numpy(),
                weight_amax.cpu().numpy(), name)
            checked.append(details)
            if details["repaired_zero_over_zero_count"]:
                print("[SmoothQuant zero-channel repair] " + json.dumps(details), flush=True)
                pre_quant_scale = torch.from_numpy(fixed).to(
                    device=pre_quant_scale.device, dtype=pre_quant_scale.dtype).reshape(pre_quant_scale.shape)
            try:
                return original(linear, pre_quant_scale)
            except Exception as exc:
                raise RuntimeError(f"SmoothQuant apply failed in {name}: {exc}") from exc

    model_calib.apply_pre_quant_scale_and_smooth = guarded_apply
    try:
        yield checked
    finally:
        model_calib.apply_pre_quant_scale_and_smooth = original


def quantize_vision(model, processor, calibration, cfg):
    import torch
    import modelopt.torch.quantization as mtq
    recipe_name = cfg["vision_recipe"]
    if recipe_name not in VISION_RECIPE_EXPECTED:
        raise ValueError(
            f"Unknown vision_recipe {recipe_name!r}; choose from "
            f"{sorted(VISION_RECIPE_EXPECTED)}")
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or not name.startswith("model.visual."):
            continue
        if vision_linear_is_target(name, recipe_name):
            targets.append(name)
    expected = VISION_RECIPE_EXPECTED[recipe_name]
    if len(targets) != expected:
        raise ValueError(
            f"Vision {recipe_name} whitelist mismatch: expected {expected}, "
            f"found {len(targets)}")
    recipe = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
    recipe["quant_cfg"] = [{"quantizer_name": "*", "enable": False}]
    for name in targets:
        recipe["quant_cfg"].extend([
            {"quantizer_name": name + ".weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}, "enable": True},
            {"quantizer_name": name + ".input_quantizer", "cfg": {"num_bits": 8, "axis": None}, "enable": True},
        ])
    recipe["algorithm"] = {"method": "smoothquant", "alpha": cfg["smoothquant_alpha"]}

    def forward_loop(quant_model):
        # A replayable generator: ModelOpt can call this more than once.
        with torch.no_grad():
            for i, row in enumerate(calibration):
                print(f"[Vision calibration {i+1}/{len(calibration)}] {row['id']}", flush=True)
                data = prepare_inputs(processor, row, cfg)
                output = quant_model.model.visual(hidden_states=data["pixel_values"], grid_thw=data["image_grid_thw"], return_dict=True)
                if not torch.isfinite(output.pooler_output).all():
                    raise ValueError(f"Non-finite calibration sample: {row['id']}")

    with modelopt_smoothquant_zero_guard(model, targets) as smoother_checks:
        mtq.quantize(model, recipe, forward_loop=forward_loop)
    if {row["module"] for row in smoother_checks} != set(targets) or len(smoother_checks) != len(targets):
        raise ValueError("SmoothQuant did not process exactly the whitelisted Vision layers")
    from modelopt.torch.quantization.nn import TensorQuantizer
    enabled = {name for name, quantizer in model.named_modules()
               if isinstance(quantizer, TensorQuantizer) and quantizer.is_enabled}
    expected_enabled = {name + suffix for name in targets for suffix in (".input_quantizer", ".weight_quantizer")}
    if enabled != expected_enabled:
        raise ValueError(f"Vision quantizer whitelist mismatch: extra={sorted(enabled-expected_enabled)}, "
                         f"missing={sorted(expected_enabled-enabled)}")
    summary = []
    for name in targets:
        layer = model.get_submodule(name)
        iq, wq = layer.input_quantizer, layer.weight_quantizer
        for quantizer in (iq, wq):
            if not quantizer.is_enabled or quantizer.amax is None or not torch.isfinite(quantizer.amax).all():
                raise ValueError(f"Uncalibrated quantizer: {name}")
        pqs = iq.pre_quant_scale
        if pqs is None or not torch.isfinite(pqs).all() or not (pqs > 0).all():
            raise ValueError(f"SmoothQuant did not produce a valid pre_quant_scale for {name}")
        summary.append({"module": name, "shape": list(layer.weight.shape),
                        "activation_amax": iq.amax.max().item(),
                        "smoother_min": pqs.min().item(), "smoother_max": pqs.max().item()})
    mtq.print_quant_summary(model)
    return {"recipe": recipe, "quantized_modules": summary, "calibration_samples": len(calibration),
            "enabled_quantizer_count": len(enabled), "smoothquant_scale_checks": smoother_checks,
            "zero_channel_repair_policy": "Only NaN from exactly zero activation range AND zero weight column uses neutral scale 1; all other invalid statistics fail."}


def compare_vision(model, processor, references, cfg):
    import numpy as np
    import torch.nn.functional as F
    rows = []
    names = ["output"] + [f"deepstack_features_{i}" for i in range(3)]
    for ref in references:
        values = vision_outputs(model, prepare_inputs(processor, ref["row"], cfg))
        np.savez(ref["case"] / "int8_fake_outputs.npz", **{k: v.cpu().numpy() for k, v in zip(names, values)})
        for name, expected, actual in zip(names, ref["features"], values):
            expected, actual = expected.float().to("cuda"), actual.float()
            if expected.shape != actual.shape:
                raise ValueError(f"Vision output shape changed: {name}")
            cosine = F.cosine_similarity(expected, actual, dim=-1)
            rows.append({"id": ref["row"]["id"], "output": name,
                         "mean_token_cosine": cosine.mean().item(), "min_token_cosine": cosine.min().item(),
                         "relative_l2": ((actual-expected).norm()/expected.norm().clamp_min(1e-12)).item(),
                         "max_abs_error": (actual-expected).abs().max().item()})
    return {"backend": "PyTorch ModelOpt W8A8 fake quantization", "samples": rows,
            "worst_sample_mean_token_cosine": min(r["mean_token_cosine"] for r in rows)}


def export_vision_checkpoint(model, processor, directory):
    from modelopt.torch.export import export_hf_checkpoint
    from tensorrt_edgellm.quantization.quantize import _fix_generation_config_for_strict_validate, _normalize_tied_weights_keys
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    _fix_generation_config_for_strict_validate(model)
    _normalize_tied_weights_keys(model)
    # This is the original FP16 LLM + SQ vision, not the AWQ candidate.
    # No AWQ re-quantization or AWQ weight merge is performed here.
    export_hf_checkpoint(model, export_dir=str(directory))
    processor.save_pretrained(directory)
    header = checkpoint_index(directory)
    quant = [key for key, (_, value) in header.items() if key.startswith("model.visual.") and key.endswith(".weight") and value["dtype"] == "I8"]
    if not quant:
        raise ValueError("Export did not preserve INT8 vision weights")
    for key in quant:
        for suffix in (".weight_scale", ".input_scale", ".pre_quant_scale"):
            if key[:-7] + suffix not in header:
                raise ValueError(f"Exported INT8 layer lacks required SQ scale: {key[:-7] + suffix}")
    return quant


def release_cuda():
    import torch
    gc.collect()
    torch.cuda.empty_cache()
