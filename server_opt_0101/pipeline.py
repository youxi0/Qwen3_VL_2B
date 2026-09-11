#!/usr/bin/env python3
"""Create a server-tested Qwen3-VL AWQ + Vision INT8 ONNX candidate bundle."""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

from common import (PACKAGE, check_splits, checkpoint_index,
                    inspect_awq_modules, load_config, prepare_manifests,
                    read_json, read_manifest, versions, write_json)


def preflight(cfg, strict=False):
    original = checkpoint_index(cfg["original"])
    awq = checkpoint_index(cfg["awq"])
    oc = read_json(Path(cfg["original"]) / "config.json")
    if oc["model_type"] != "qwen3_vl" or oc["vision_config"]["depth"] != 24 or oc["text_config"]["hidden_size"] != 2048:
        raise ValueError("Expected Qwen3-VL-2B-Instruct")
    qcfg = read_json(Path(cfg["awq"]) / "hf_quant_config.json")["quantization"]
    awq_layout = inspect_awq_modules(awq)
    if qcfg.get("quant_algo") != "W4A16_AWQ" or qcfg.get("has_zero_point", False):
        raise ValueError("Expected symmetric ModelOpt W4A16_AWQ")
    excluded = set(qcfg.get("exclude_modules", []))
    if awq_layout["lm_head_int4"] and "lm_head" in excluded:
        raise ValueError("Packed lm_head conflicts with hf_quant_config exclusion")
    llm = Path(cfg["awq_onnx"]) / "llm"
    config = read_json(llm / "config.json")
    if config.get("edgellm_version") != "0.10.1":
        raise ValueError("LLM ONNX must be exported by Edge-LLM 0.10.1")
    for name in ["model.onnx", "model.onnx.data", "embedding.safetensors"] + [r["file"] for r in config.get("external_weight_files", [])]:
        if not (llm / name).is_file():
            raise FileNotFoundError(llm / name)
    return {"paths": {k: cfg[k] for k in ("original", "awq", "awq_onnx", "images")},
            "vision_recipe": cfg["vision_recipe"],
            "vision_fp16_blocks": cfg.get("vision_fp16_blocks", []),
            "original_tensors": len(original), "awq_tensors": len(awq),
            "awq_packed_linears": awq_layout["total_packed_linears"],
            "awq_backbone_linears": awq_layout["backbone_linears"],
            "lm_head_int4": awq_layout["lm_head_int4"],
            "awq_group_size": qcfg["group_size"],
            "llm_onnx_version": config["edgellm_version"],
            "required_llm_sidecars": [{"file": r["file"], "tensor_count": len(r["tensors"])} for r in config.get("external_weight_files", [])],
            "environment": versions(strict), "host": platform.platform()}


def run(cfg, args):
    if platform.system() != "Linux":
        raise RuntimeError("Run GPU stages on the Linux server; local Windows is for preparing files only")
    check = preflight(cfg, strict=True)
    import torch
    import numpy as np
    from artifacts import audit_onnx, make_bundle, memory_budget
    from torch_work import (collect_vision_reference, compare_vision, create_references,
                            evaluate_candidate, export_vision_checkpoint, load_awq_reference,
                            load_fp16, make_processor, quantize_vision, release_cuda, tokenizer_check)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check the server driver / CUDA PyTorch wheel.")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    check["cuda"] = {"torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                     "total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20}
    calibration = read_manifest(args.calib, require_image=True)
    evaluation = read_manifest(args.eval)
    check_splits(calibration, evaluation)
    if not args.smoke and (len(calibration) < cfg["calibration_samples"] or len(evaluation) < cfg["evaluation_samples"]):
        raise ValueError("Dataset smaller than configured calibration/evaluation minimum. --smoke is explicitly non-validating.")
    out = Path(args.out).resolve()
    protected = [Path(cfg[k]).resolve() for k in ("original", "awq", "awq_onnx", "images")]
    if any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in protected):
        raise ValueError("Output must be separate from input directories")
    if out.exists():
        raise FileExistsError(f"Choose a fresh run directory; preserving previous results: {out}")
    reports = out / "reports"
    reports.mkdir(parents=True)
    args._created_output = out
    write_json(out / "config.resolved.json", cfg)
    write_json(reports / "preflight.json", check)
    write_json(reports / "dataset.json", {"smoke_only": args.smoke, "calibration": calibration, "evaluation": evaluation})
    write_json(reports / "existing_llm_onnx.json", audit_onnx(Path(cfg["awq_onnx"]) / "llm"))
    processor = make_processor(cfg)
    write_json(reports / "tokenizer_alignment.json", tokenizer_check(cfg, processor))

    print("Loading FP16 original and dequantized AWQ reference on the server...", flush=True)
    base = load_fp16(cfg["original"])
    candidate, weight_stats = load_awq_reference(cfg)
    write_json(reports / "awq_weight_alignment.json", {"packed_linears": len(weight_stats), "layers": weight_stats})
    references = create_references(base, processor, evaluation, cfg)
    write_json(reports / "fp16_references.json", [{"id": r["row"]["id"], "image": r["row"].get("image"), "question": r["row"]["question"], "prompt_ids": r["prompt_ids"], "generated_ids": r["tokens"][0].tolist(), "generated_text": r["text"]} for r in references])
    awq_scores = evaluate_candidate(candidate, processor, references, cfg, "AWQ + FP16 vision")
    write_json(reports / "awq_alignment.json", awq_scores)
    visual_refs = collect_vision_reference(base, processor, evaluation, cfg, out / "vision_cases")

    quant_info = quantize_vision(base, processor, calibration, cfg)
    write_json(reports / "vision_quantization.json", quant_info)
    visual_scores = compare_vision(base, processor, visual_refs, cfg)
    write_json(reports / "vision_alignment.json", visual_scores)
    # Runtime combination, without merging or re-quantizing the AWQ weights.
    candidate.model.visual = base.model.visual
    combined = evaluate_candidate(candidate, processor, references, cfg, "AWQ + W8A8 vision")
    write_json(reports / "combined_alignment.json", combined)
    del candidate, references, visual_refs
    release_cuda()

    vision_checkpoint = out / "vision_sq_checkpoint"
    quantized_keys = export_vision_checkpoint(base, processor, vision_checkpoint)
    if len(quantized_keys) != len(quant_info["quantized_modules"]):
        raise ValueError("Saved checkpoint INT8 layer count differs from calibrated layer count")
    del base
    release_cuda()
    exported = out / "vision_onnx"
    print("Exporting only Vision with the installed Edge-LLM 0.10.1 exporter...", flush=True)
    env = os.environ.copy()
    # Do not enable TRT 11 native attention for a Jetson/TRT-10 ONNX bundle.
    env["USE_TRT_NATIVE_ATTN"] = "0"
    command = [sys.executable, "-m", "tensorrt_edgellm.scripts.export", str(vision_checkpoint), str(exported), "--components", "visual", "--dtype", "float16"]
    subprocess.run(command, check=True, env=env)
    audit = audit_onnx(exported / "visual", vision=True, expect_int8=len(quantized_keys))
    write_json(reports / "vision_onnx_audit.json", audit)
    # Ensure deployment carries the exact calibration pixel budget.
    write_json(reports / "preprocessor_exported.json", read_json(exported / "visual" / "preprocessor_config.json"))
    budget = memory_budget(cfg, vision_checkpoint)
    write_json(reports / "memory_budget.json", budget)
    g = cfg["gates"]
    passed = {
        "awq_top1": awq_scores["metrics"]["teacher_top1_agreement"] >= g["awq_teacher_top1_min"],
        "awq_kl": awq_scores["metrics"]["teacher_kl"] <= g["awq_teacher_kl_max"],
        "vision_cosine": visual_scores["worst_sample_mean_token_cosine"] >= g["vision_mean_token_cosine_min"],
        "combined_top1": combined["metrics"]["teacher_top1_agreement"] >= g["combined_teacher_top1_min"],
        "combined_kl": combined["metrics"]["teacher_kl"] <= g["combined_teacher_kl_max"],
        "estimated_memory": budget["within_estimated_budget"],
    }
    status = {"edgellm_version": "0.10.1", "smoke_only": args.smoke,
              "experimental_gate_results": passed,
              "candidate_passes_configured_gates": all(passed.values()) and not args.smoke,
              "server_trt_vision_verified": False, "server_trt_llm_verified": False,
              "jetson_fit_verified": False,
              "next": "Run trt_vision_check.py on the server, then validate target-built engines and task accuracy on Jetson. No GPU-specific .engine is included in the transferable ONNX bundle."}
    write_json(reports / "status.json", status)
    make_bundle(cfg, exported, out / "jetson_onnx_candidate", reports, status)
    print(f"Candidate bundle: {out / 'jetson_onnx_candidate'}", flush=True)
    print(f"Estimated memory: {budget['estimated_total_mib']:.1f} MiB; headroom {budget['estimated_headroom_mib']:.1f} MiB. Target fit is NOT verified.", flush=True)
    if not all(passed.values()) and not args.smoke:
        print("One or more configured gates failed. Artifacts and reports were retained as a candidate, not accepted deployment.", file=sys.stderr)
        return 2
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "prepare", "run"])
    parser.add_argument("--root", type=Path, default=PACKAGE.parent)
    parser.add_argument("--config", type=Path, default=PACKAGE / "config.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--calib", type=Path)
    parser.add_argument("--eval", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.root)
    if args.command == "inspect":
        import json
        print(json.dumps(preflight(cfg), ensure_ascii=False, indent=2))
        return 0
    if args.out is None:
        parser.error("--out is required")
    if args.command == "prepare":
        print(prepare_manifests(cfg, args.out, args.smoke))
        return 0
    if args.calib is None or args.eval is None:
        parser.error("run requires --calib and --eval")
    try:
        return run(cfg, args)
    except Exception as exc:
        if getattr(args, "_created_output", None) is not None:
            import traceback
            try:
                write_json(args._created_output / "reports" / "failure.json",
                           {"error": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
                            "note": "Run failed; no acceptance or Jetson fit is claimed. Preserve this report and the console log."})
            except OSError as report_error:
                # Do not hide the original exception when the quota is already
                # exhausted and even the small failure report cannot be saved.
                print(f"Could not write failure report: {report_error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
