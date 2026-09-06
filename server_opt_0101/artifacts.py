"""ONNX structure checks and conservative deployment-weight accounting."""
from __future__ import annotations

import collections
import shutil
from pathlib import Path

from common import checkpoint_index, read_json, write_json


def audit_onnx(folder, vision=False, expect_int8=None):
    import onnx
    folder = Path(folder)
    model = onnx.load(str(folder / "model.onnx"), load_external_data=False)
    counts = collections.Counter(n.op_type for n in model.graph.node)
    used = {name for node in model.graph.node for name in node.input}
    external_files = set()
    bytes_by_type = collections.Counter()
    initializers = {t.name: t for t in model.graph.initializer}
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            meta = {v.key: v.value for v in tensor.external_data}
            target = (folder / meta["location"]).resolve()
            if not target.is_relative_to(folder.resolve()):
                raise ValueError("ONNX external data must remain within its component directory")
            offset, size = int(meta.get("offset", 0)), int(meta.get("length", 0))
            if not target.is_file() or size <= 0 or offset < 0 or offset + size > target.stat().st_size:
                raise ValueError(f"Missing/truncated external tensor {tensor.name}: {target}")
            external_files.add(meta["location"])
            if tensor.name in used:
                bytes_by_type[onnx.TensorProto.DataType.Name(tensor.data_type)] += size
    # Sidecars belong to the runtime and must be copied even though ONNX
    # checker cannot validate their contents as graph initializers.
    cfg = read_json(folder / "config.json")
    for entry in cfg.get("external_weight_files", []):
        target = (folder / entry["file"]).resolve()
        if not target.is_relative_to(folder.resolve()) or not target.is_file():
            raise ValueError(f"Missing/unsafe runtime sidecar: {entry['file']}")
        checkpoint_index_for_sidecar(target)
    if vision:
        expected_inputs = {"input", "rotary_pos_emb", "cu_seqlens", "fast_pos_embed_idx", "fast_pos_embed_weight", "max_seqlen_carrier"}
        expected_outputs = {"output", "deepstack_features_0", "deepstack_features_1", "deepstack_features_2"}
        if {v.name for v in model.graph.input} != expected_inputs:
            raise ValueError("Unexpected visual ABI. Export must use the TRT-10 ViTAttentionPlugin path.")
        if {v.name for v in model.graph.output} != expected_outputs:
            raise ValueError("Vision export lost DeepStack outputs")
        if any(v.type.tensor_type.elem_type != onnx.TensorProto.FLOAT16 for v in model.graph.output):
            raise ValueError("Vision outputs must remain FP16 for the existing LLM")
        if counts["ViTAttentionPlugin"] != 24:
            raise ValueError("Expected 24 existing FP16 ViT attention plugins")
        if expect_int8 is not None:
            weight_dq = [n for n in model.graph.node if n.op_type == "DequantizeLinear" and n.input[0] in initializers
                         and initializers[n.input[0]].data_type == onnx.TensorProto.INT8]
            if len(weight_dq) != expect_int8 or counts["QuantizeLinear"] < expect_int8:
                raise ValueError(f"INT8 export coverage mismatch: {len(weight_dq)} weight DQ and {counts['QuantizeLinear']} Q; expected {expect_int8} linears")
    # Path-based validation avoids the >2 GiB in-memory protobuf limit.
    onnx.checker.check_model(str(folder / "model.onnx"), full_check=False)
    return {"folder": str(folder), "producer": [model.producer_name, model.producer_version],
            "opsets": {v.domain: v.version for v in model.opset_import}, "op_counts": dict(counts),
            "referenced_external_tensor_mib": {k: v / 2**20 for k, v in bytes_by_type.items()},
            "external_files": sorted(external_files),
            "runtime_sidecars": [r["file"] for r in cfg.get("external_weight_files", [])],
            "note": "Structural validation only; custom plugin execution requires TensorRT."}


def checkpoint_index_for_sidecar(path):
    from common import safetensor_header
    return safetensor_header(path)


def memory_budget(cfg, vision_checkpoint):
    awq = checkpoint_index(cfg["awq"])
    quant = checkpoint_index(vision_checkpoint)
    nbytes = lambda item: item["data_offsets"][1] - item["data_offsets"][0]
    awq_visual = sum(nbytes(v) for k, (_, v) in awq.items() if k.startswith("model.visual."))
    awq_non_visual = sum(nbytes(v) for k, (_, v) in awq.items() if not k.startswith("model.visual."))
    int8_visual = sum(nbytes(v) for k, (_, v) in quant.items() if k.startswith("model.visual."))
    text = read_json(Path(cfg["original"]) / "config.json")["text_config"]
    # HF tied embedding is often serialized once, but this Edge runtime has
    # separate embedding and lm_head resources. Count both; do not claim tying.
    extra_head = 0 if "lm_head.weight" in awq else text["vocab_size"] * text["hidden_size"] * 2
    kv_per_token = 2 * text["num_hidden_layers"] * text["num_key_value_heads"] * text["head_dim"] * 2
    kv = kv_per_token * cfg["kv_cache_capacity"]
    weights = awq_non_visual + int8_visual + extra_head
    reserves = (cfg["system_and_cuda_reserve_mib"] + cfg["activation_and_workspace_reserve_mib"]) * 2**20
    estimated = weights + kv + reserves
    available = cfg["jetson_total_gib"] * 2**30
    return {"awq_llm_and_embedding_mib": awq_non_visual / 2**20,
            "extra_runtime_lm_head_mib": extra_head / 2**20,
            "vision_fp16_mib": awq_visual / 2**20, "vision_int8_mib": int8_visual / 2**20,
            "vision_weight_saving_mib": (awq_visual - int8_visual) / 2**20,
            "total_logical_weight_mib": weights / 2**20,
            "fp16_kv_cache_mib": kv / 2**20, "assumed_reserve_mib": reserves / 2**20,
            "estimated_total_mib": estimated / 2**20,
            "configured_total_mib": available / 2**20,
            "estimated_headroom_mib": (available - estimated) / 2**20,
            "within_estimated_budget": estimated < available,
            "jetson_fit_verified": False,
            "limitations": "Logical weight + KV estimate with user-editable reserves. Engine weight repacking, host/device copies, CUDA, build-time peaks and actual OS consumption need target measurements. Server RTX memory does not prove Jetson fit."}


def make_bundle(cfg, visual_onnx, destination, reports, status):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    # Copy whole components: v0.10.1 FFN / lm_head sidecars are required.
    shutil.copytree(Path(cfg["awq_onnx"]) / "llm", destination / "llm")
    shutil.copytree(Path(visual_onnx) / "visual", destination / "visual")
    limits = {key: cfg[key] for key in ("min_image_tokens", "max_image_tokens", "max_input_tokens", "max_new_tokens", "kv_cache_capacity")}
    limits.update({"batch_size": 1, "max_images_per_request": 1, "kv_cache_dtype": "fp16", "edgellm_version": "0.10.1", "visual_mha": "fp16"})
    write_json(destination / "deployment_limits.json", limits)
    write_json(destination / "STATUS.json", status)
    shutil.copytree(reports, destination / "reports")
    return destination
