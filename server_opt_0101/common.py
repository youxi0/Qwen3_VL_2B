"""CPU-only manifest/path utilities; importing this file never initializes CUDA."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import struct
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
EXPECTED = {
    "tensorrt-edgellm": "0.10.1", "torch": "2.13.0",
    "transformers": "5.14.1", "nvidia-modelopt": "0.45.0",
    "onnx": "1.19.0", "safetensors": "0.8.0",
}
VISION_RECIPE_EXPECTED = {
    "blocks": 96,
    "conservative": 100,
    "residual_fp16": 52,
    "all_linears": 104,
}
AWQ_BACKBONE_LINEAR_COUNT = 196


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path, root):
    cfg = read_json(path)
    root = Path(root).expanduser().resolve()
    for key in ("original", "awq", "awq_onnx", "images"):
        p = Path(cfg[key]).expanduser()
        cfg[key] = str((root / p).resolve() if not p.is_absolute() else p.resolve())
    cfg["root"] = str(root)
    lo, hi = cfg["min_image_tokens"], cfg["max_image_tokens"]
    if not 1 <= lo <= hi:
        raise ValueError("Expected 1 <= min_image_tokens <= max_image_tokens")
    if cfg["kv_cache_capacity"] < cfg["max_input_tokens"] + cfg["max_new_tokens"]:
        raise ValueError("KV capacity must cover prompt + generated tokens")
    if not 0 <= cfg["smoothquant_alpha"] <= 1:
        raise ValueError("smoothquant_alpha must be in [0,1]")
    if cfg["vision_recipe"] not in VISION_RECIPE_EXPECTED:
        raise ValueError(
            f"Unknown vision_recipe {cfg['vision_recipe']!r}; choose from "
            f"{sorted(VISION_RECIPE_EXPECTED)}")
    return cfg


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safetensor_header(path):
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"Invalid safetensors file: {path}")
        size = struct.unpack("<Q", raw)[0]
        if size > 128 * 1024 * 1024:
            raise ValueError(f"Unreasonable safetensors header: {path}")
        header = json.loads(stream.read(size))
    result = {k: v for k, v in header.items() if k != "__metadata__"}
    for name, item in result.items():
        begin, end = item["data_offsets"]
        if not 0 <= begin <= end <= path.stat().st_size - size - 8:
            raise ValueError(f"Truncated weight: {path}: {name}")
    return result


def checkpoint_index(folder):
    folder = Path(folder)
    index = folder / "model.safetensors.index.json"
    if index.exists():
        mapping = read_json(index)["weight_map"]
        headers = {name: safetensor_header(folder / name) for name in sorted(set(mapping.values()))}
        for key, name in mapping.items():
            if key not in headers[name]:
                raise ValueError(f"Index references missing tensor {key} in {name}")
        return {key: (folder / name, headers[name][key]) for key, name in mapping.items()}
    single = folder / "model.safetensors"
    return {key: (single, entry) for key, entry in safetensor_header(single).items()}


def inspect_awq_modules(index):
    """Validate this model's packed ModelOpt AWQ module layout.

    The original checkpoint has 196 decoder linears.  A second supported
    layout adds a packed ``lm_head`` while keeping the embedding and visual
    tower unquantized.
    """
    packed = {
        key[:-7] for key, (_, entry) in index.items()
        if key.endswith(".weight") and entry["dtype"] == "U8"
    }
    backbone = {
        name for name in packed
        if name.startswith("model.language_model.layers.")
    }
    allowed = backbone | ({"lm_head"} if "lm_head" in packed else set())
    unexpected = sorted(packed - allowed)
    if len(backbone) != AWQ_BACKBONE_LINEAR_COUNT or unexpected:
        raise ValueError(
            "Expected 196 packed decoder linears plus optional packed "
            f"lm_head; backbone={len(backbone)}, unexpected={unexpected[:8]}"
        )
    return {
        "packed_modules": packed,
        "backbone_linears": len(backbone),
        "lm_head_int4": "lm_head" in packed,
        "total_packed_linears": len(packed),
    }


def versions(strict=True):
    found = {}
    errors = []
    for name, expected in EXPECTED.items():
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
        if found[name] is None or found[name].split("+")[0] != expected:
            errors.append(f"{name}: expected {expected}, found {found[name]}")
    if strict and errors:
        raise RuntimeError("Server environment mismatch:\n" + "\n".join(errors))
    return {"versions": found, "mismatches": errors}


def read_manifest(path, require_image=False):
    path = Path(path).resolve()
    records = []
    identifiers = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["id"] = str(row.get("id", f"sample_{line_number:05d}"))
        if row["id"] in identifiers:
            raise ValueError(f"Duplicate id {row['id']} in {path}")
        identifiers.add(row["id"])
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"Missing question in {path}:{line_number}")
        if row.get("image"):
            image = Path(row["image"]).expanduser()
            image = image if image.is_absolute() else path.parent / image
            if not image.is_file():
                raise FileNotFoundError(image)
            row["image"] = str(image.resolve())
            row["image_sha256"] = sha256(image)
        elif require_image:
            raise ValueError("Vision calibration requires an image for every record")
        if "answer" in row and not isinstance(row["answer"], str):
            raise ValueError("answer must be a string when supplied")
        records.append(row)
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def check_splits(calibration, evaluation):
    left = {r["image_sha256"] for r in calibration}
    right = {r["image_sha256"] for r in evaluation if r.get("image")}
    overlap = left & right
    if overlap:
        raise ValueError(f"Calibration/evaluation image leakage: {len(overlap)} identical files")
    if len(left) != len(calibration):
        raise ValueError("Calibration contains duplicate image bytes; use distinct images")
    if len(right) != sum(bool(r.get("image")) for r in evaluation):
        raise ValueError("Evaluation contains duplicate image bytes; use distinct images")


def prepare_manifests(cfg, output, smoke=False):
    from PIL import Image
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new manifest directory: {output}")
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    unique = {}
    sizes = {}
    scanned = 0
    for path in sorted(Path(cfg["images"]).rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            with Image.open(path) as image:
                image.load()
                rgb = image.convert("RGB")
                pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                shape = f"{image.width}x{image.height}"
                sizes[shape] = sizes.get(shape, 0) + 1
            scanned += 1
            unique.setdefault(pixel_hash, path.resolve())
    images = list(unique.values())
    random.Random(cfg["seed"]).shuffle(images)
    nc, ne = (1, 1) if smoke else (cfg["calibration_samples"], cfg["evaluation_samples"])
    if len(images) < nc + ne:
        raise ValueError(f"Need {nc + ne} distinct images ({nc} calibration + {ne} evaluation); found {len(images)}. Use --smoke only to test the pipeline.")
    output.mkdir(parents=True)
    for name, subset in (("calib", images[:nc]), ("eval", images[nc:nc+ne])):
        rows = [{"id": f"{name}_{i:05d}", "image": str(image), "question": "描述这张图片中可见的内容。"} for i, image in enumerate(subset)]
        (output / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    write_json(output / "manifest_info.json", {"smoke_only": smoke, "calibration": nc, "evaluation": ne,
               "seed": cfg["seed"], "ground_truth_answers": False, "images_scanned": scanned,
               "unique_decoded_images": len(images), "unused_unique_images": len(images)-nc-ne,
               "original_dimensions": sizes,
               "note": "Pixel-exact duplicates removed. Near duplicates / adjacent video frames require a manual scene-level split."})
    return output
