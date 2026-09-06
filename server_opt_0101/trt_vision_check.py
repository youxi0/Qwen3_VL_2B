#!/usr/bin/env python3
"""Build/run only the visual engines on Linux; never claim Jetson portability."""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import platform
from pathlib import Path

from common import read_json, write_json

INPUT_NAMES = {"input", "rotary_pos_emb", "cu_seqlens", "fast_pos_embed_idx",
               "fast_pos_embed_weight", "max_seqlen_carrier"}
OUTPUT_NAMES = {"output", "deepstack_features_0", "deepstack_features_1", "deepstack_features_2"}


def profile_shapes(n):
    return {"input": (n, 1536), "rotary_pos_emb": (n, 32), "cu_seqlens": (2,),
            "fast_pos_embed_idx": (4, n), "fast_pos_embed_weight": (4, n),
            "max_seqlen_carrier": (n,)}


def feature_metrics(reference, actual):
    import numpy as np
    if reference.shape != actual.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Non-finite or mismatched visual output")
    ref, value = reference.astype(np.float64), actual.astype(np.float64)
    nr, nv = np.linalg.norm(ref, axis=-1), np.linalg.norm(value, axis=-1)
    cosine = (ref * value).sum(-1) / np.maximum(nr * nv, 1e-12)
    cosine = np.where((nr == 0) & (nv == 0), 1.0, cosine)
    return {"mean_token_cosine": float(cosine.mean()), "min_token_cosine": float(cosine.min()),
            "relative_l2": float(np.linalg.norm(value-ref) / max(np.linalg.norm(ref), 1e-12)),
            "max_abs_error": float(np.abs(value-ref).max())}


def build_engine(trt, logger, onnx_path, destination, cfg, workspace_mib):
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed: {onnx_path}\n{errors}")
    if {network.get_input(i).name for i in range(network.num_inputs)} != INPUT_NAMES:
        raise ValueError("Wrong visual input ABI; expected Edge-LLM 0.10.1 TRT-10 export")
    if {network.get_output(i).name for i in range(network.num_outputs)} != OUTPUT_NAMES:
        raise ValueError("Missing final / DeepStack outputs")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 2**20)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.TF32)
    # Explicit Q/DQ contains calibrated scales. No legacy INT8 calibrator/flag.
    profile = builder.create_optimization_profile()
    minimum = profile_shapes(cfg["min_image_tokens"] * 4)
    maximum = profile_shapes(cfg["max_image_tokens"] * 4)
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        if any(d < 0 for d in tensor.shape):
            if not profile.set_shape(tensor.name, minimum[tensor.name], maximum[tensor.name], maximum[tensor.name]):
                raise RuntimeError(f"Invalid profile for {tensor.name}")
    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("Could not add visual optimization profile")
    print(f"Building server-only {destination.name}...", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed. Check plugin ABI / TensorRT version and builder log.")
    destination.write_bytes(bytes(serialized))


def execute_case(trt, torch, engine, context, arrays, repeat):
    buffers = {}
    for name, array in arrays.items():
        if name not in INPUT_NAMES:
            raise ValueError(f"Unexpected input {name}")
        value = torch.from_numpy(array.copy()).contiguous().to("cuda")
        if trt.nptype(engine.get_tensor_dtype(name)) != array.dtype:
            raise ValueError(f"Wrong dtype for {name}: {array.dtype}")
        if not context.set_input_shape(name, tuple(value.shape)):
            raise ValueError(f"Input outside engine profile: {name}: {tuple(value.shape)}")
        buffers[name] = value
    if set(buffers) != INPUT_NAMES:
        raise ValueError("A saved test case is missing visual inputs")
    for name in OUTPUT_NAMES:
        shape = tuple(context.get_tensor_shape(name))
        if any(d <= 0 for d in shape):
            raise ValueError(f"Unresolved output shape: {name}: {shape}")
        if engine.get_tensor_dtype(name) != trt.float16:
            raise ValueError(f"Expected FP16 output for {name}")
        buffers[name] = torch.empty(shape, device="cuda", dtype=torch.float16)
    for name, value in buffers.items():
        if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
            raise ValueError(f"Unexpected host I/O: {name}")
        if not context.set_tensor_address(name, value.data_ptr()):
            raise RuntimeError(f"Cannot bind {name}")
    stream = torch.cuda.current_stream()

    def invoke():
        if not context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("TensorRT visual execution failed")

    for _ in range(3):
        invoke()
    stream.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(repeat):
        invoke()
    end.record(stream)
    end.synchronize()
    values = {name: buffers[name].cpu().numpy() for name in OUTPUT_NAMES}
    return values, start.elapsed_time(end) / repeat


def check_engine(trt, torch, logger, path, cases, label, repeat, cosine_min, l2_max, quant_cosine_min, report_dir):
    import numpy as np
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Cannot deserialize {path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Cannot create visual execution context")
    rows = []
    for case in cases:
        with np.load(case / "inputs.npz", allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        outputs, ms = execute_case(trt, torch, engine, context, arrays, repeat)
        gold_name = "fp16_outputs.npz" if label == "fp16" else "int8_fake_outputs.npz"
        with np.load(case / gold_name, allow_pickle=False) as gold, np.load(case / "fp16_outputs.npz", allow_pickle=False) as original:
            for name in sorted(OUTPUT_NAMES):
                metrics = feature_metrics(gold[name], outputs[name])
                vs_original = feature_metrics(original[name], outputs[name])
                passed = metrics["mean_token_cosine"] >= cosine_min and metrics["relative_l2"] <= l2_max
                if label == "int8":
                    passed &= vs_original["mean_token_cosine"] >= quant_cosine_min
                rows.append({"case": case.name, "output": name, "versus_matching_pytorch": metrics,
                             "versus_original_fp16": vs_original, "engine_only_latency_ms": ms, "pass": bool(passed)})
        np.savez(report_dir / f"{label}_{case.name}_outputs.npz", **outputs)
        print(f"[{label}] {case.name}: {ms:.3f} ms (server, excludes preprocessing/transfers)", flush=True)
    inspector = engine.create_engine_inspector()
    inspector.execution_context = context
    info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    try:
        layer_info = json.loads(info)
    except json.JSONDecodeError:
        layer_info = {"raw": info}
    write_json(report_dir / f"{label}_engine_inspector.json", layer_info)
    # This is evidence for inspection, not proof every GEMM was selected as INT8.
    formats = []

    def gather(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if "format" in key.lower() or "datatype" in key.lower():
                    if isinstance(child, (str, int, float)):
                        formats.append(str(child))
                gather(child)
        elif isinstance(value, list):
            for child in value:
                gather(child)

    gather(layer_info)
    int8_formats = sorted({s for s in formats if "int8" in s.lower() or "i8" in s.lower()})
    result = {"precision": label, "engine_mib": path.stat().st_size / 2**20,
              "context_device_memory_mib": engine.device_memory_size / 2**20,
              "numerical_pass": all(row["pass"] for row in rows), "samples": rows,
              "int8_tensor_formats_observed": int8_formats,
              "int8_execution_evidence_found": bool(int8_formats),
              "note": "Inspector tensor formats are diagnostic; review per-layer tactics for INT8 GEMM coverage. These are RTX server engine numbers, not Jetson measurements."}
    del inspector, context, engine, runtime
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plugin-lib", type=Path, required=True, help="Server-built Edge-LLM v0.10.1 shared plugin library")
    parser.add_argument("--workspace-mib", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--cosine-min", type=float, default=0.999)
    parser.add_argument("--relative-l2-max", type=float, default=0.03)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run TensorRT only on the Linux server")
    if args.repeat < 1 or args.workspace_mib < 1:
        parser.error("repeat and workspace-mib must be positive")
    import numpy as np
    import torch
    import tensorrt as trt
    if trt.__version__.split(".")[0] != "10":
        raise RuntimeError("This validator requires TensorRT 10.x + matching Edge-LLM 0.10.1 server plugins, not TensorRT 11")
    if not args.plugin_lib.is_file():
        raise FileNotFoundError(args.plugin_lib)
    plugin_handle = ctypes.CDLL(str(args.plugin_lib.resolve()), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    run_dir = args.run_dir.resolve()
    cfg = read_json(run_dir / "config.resolved.json")
    cases = sorted((run_dir / "vision_cases").glob("case_*"))
    if not cases:
        raise ValueError("No real visual input cases; run pipeline.py first")
    output = run_dir / "server_engines"
    output.mkdir()  # Refuse silent overwriting of a previous engine check.
    results = {}
    for label, source in (("fp16", Path(cfg["awq_onnx"]) / "visual" / "model.onnx"),
                          ("int8", run_dir / "vision_onnx" / "visual" / "model.onnx")):
        engine_path = output / f"vision_{label}_SERVER_ONLY.engine"
        build_engine(trt, logger, source, engine_path, cfg, args.workspace_mib)
        results[label] = check_engine(trt, torch, logger, engine_path, cases, label, args.repeat,
                                     args.cosine_min, args.relative_l2_max,
                                     cfg["gates"]["vision_mean_token_cosine_min"], output)
    numerical_pass = all(r["numerical_pass"] for r in results.values())
    passed = numerical_pass and results["int8"]["int8_execution_evidence_found"]
    report = {"server_trt_vision_verified": passed, "tensorrt": trt.__version__,
              "numerical_pass": numerical_pass,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "plugin_lib": str(args.plugin_lib.resolve()), "workspace_limit_mib": args.workspace_mib,
              "cosine_min": args.cosine_min, "relative_l2_max": args.relative_l2_max,
              "quantization_cosine_min": cfg["gates"]["vision_mean_token_cosine_min"],
              "engines": results, "server_trt_llm_verified": False, "jetson_fit_verified": False}
    write_json(run_dir / "reports" / "server_trt_vision.json", report)
    write_json(run_dir / "jetson_onnx_candidate" / "reports" / "server_trt_vision.json", report)
    status = read_json(run_dir / "reports" / "status.json")
    status["server_trt_vision_verified"] = passed
    status["server_trt_int8_tensor_evidence"] = results["int8"]["int8_execution_evidence_found"]
    for relative in ("reports/status.json", "jetson_onnx_candidate/STATUS.json", "jetson_onnx_candidate/reports/status.json"):
        write_json(run_dir / relative, status)
    print(f"Server vision numerical verification: {'PASS' if passed else 'FAIL'}; Jetson fit remains unverified.")
    if not results["int8"]["int8_execution_evidence_found"]:
        print("No INT8 tensor formats found in engine inspector. Check the detailed inspector file before accepting the engine.")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
