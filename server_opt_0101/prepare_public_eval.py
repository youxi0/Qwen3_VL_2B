#!/usr/bin/env python3
"""Prepare a fresh task-balanced evaluation set, excluding all known QAT data.

This draws from public train splits, NOT official benchmark test sets. It does
not train, calibrate, change prompts or resize the saved original RGB pixels.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from common import load_config, read_manifest, write_json
from prepare_public_qat import REPO, candidate_pairs, pixel_hash, prompt_length

TASKS = ("vqav2", "textvqa", "docvqa", "chartqa")


def excluded_pixel_hashes(manifests):
    from PIL import Image
    excluded = set()
    for manifest in manifests:
        for row in read_manifest(manifest):
            if row.get("image"):
                with Image.open(row["image"]) as image:
                    excluded.add(pixel_hash(image))
    return excluded


def materialize_eval(sources, processor, cfg, destination, per_task, excluded, seed,
                     max_examined=100000):
    from PIL import Image
    if per_task < 1 or max_examined < per_task:
        raise ValueError("Invalid evaluation sample/examination limits")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Preserving previous data; choose a new folder: {destination}")
    (destination / "images").mkdir(parents=True)
    seen = set(excluded)
    counts = {}
    with (destination / "eval.jsonl").open("x", encoding="utf-8") as handle:
        for task_index, (task, stream) in enumerate(sources):
            if task not in TASKS or task in counts:
                raise ValueError(f"Unknown or duplicate evaluation task: {task}")
            stats = Counter()
            counts[task] = stats
            rng = random.Random(seed + task_index)
            for source_index, row in enumerate(stream):
                if source_index >= max_examined:
                    break
                stats["examined"] += 1
                images = row.get("images")
                if not isinstance(images, list) or len(images) != 1:
                    stats["not_single_image"] += 1
                    continue
                if not isinstance(images[0], Image.Image):
                    raise ValueError("Expected a decoded PIL image from The Cauldron")
                image = images[0].convert("RGB")
                image.load()
                fingerprint = pixel_hash(image)
                if fingerprint in seen:
                    stats["duplicate_or_excluded"] += 1
                    continue
                selected = None
                for question, answer in candidate_pairs(
                        row.get("texts"), processor.tokenizer, cfg["max_new_tokens"], rng):
                    if prompt_length(processor, image, question) <= cfg["max_input_tokens"]:
                        selected = question, answer
                        break
                if selected is None:
                    stats["no_pair_within_token_limits"] += 1
                    continue
                identifier = f"independent_{task}_{stats['saved']:05d}"
                filename = f"images/{identifier}.png"
                image.save(destination / filename, format="PNG")
                record = {
                    "id": identifier, "image": filename, "question": selected[0],
                    "answer": selected[1], "source_dataset": REPO, "source_subset": task,
                    "source_stream_index": source_index, "rgb_sha256": fingerprint,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen.add(fingerprint)
                stats["saved"] += 1
                if stats["saved"] % 32 == 0 or stats["saved"] == per_task:
                    handle.flush()
                    print(f"[{task}] saved {stats['saved']}/{per_task}", flush=True)
                if stats["saved"] == per_task:
                    break
            if stats["saved"] != per_task:
                raise RuntimeError(f"{task}: found {stats['saved']}/{per_task} valid new images; "
                                   "partial output retained, do not evaluate an incomplete set")
    if not counts:
        raise ValueError("No evaluation sources supplied")
    return {task: dict(stats) for task, stats in counts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True,
                        help="Repeat for ALL known QAT train/holdout and previously inspected eval manifests")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--per-task", type=int, default=128)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--shuffle-buffer", type=int, default=64)
    parser.add_argument("--max-examined-per-task", type=int, default=100000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.per_task < 1 or args.shuffle_buffer < 1
            or args.max_examined_per_task < args.per_task
            or len(set(args.tasks)) != len(args.tasks)):
        parser.error("Invalid sampling limits or duplicate tasks")
    output = args.out.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    cfg = load_config(args.config, args.root)
    excluded = excluded_pixel_hashes(args.exclude)
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from torch_work import make_processor
    revision = HfApi().dataset_info(REPO, revision=args.revision).sha
    processor = make_processor(cfg)

    def sources():
        for i, task in enumerate(args.tasks):
            stream = load_dataset(REPO, task, split="train", revision=revision, streaming=True)
            yield task, stream.shuffle(seed=cfg["seed"] + 1000 + i,
                                       buffer_size=args.shuffle_buffer)

    info = {
        "status": "preparing", "dataset": REPO, "revision": revision, "split": "train",
        "tasks": args.tasks, "per_task": args.per_task, "seed": cfg["seed"],
        "shuffle_seed_offset": 1000, "shuffle_buffer": args.shuffle_buffer,
        "excluded_manifests": [str(path.resolve()) for path in args.exclude],
        "excluded_unique_rgb_images": len(excluded),
        "evaluation_note": "Fresh task-balanced holdout from public train splits; NOT official benchmark scores. "
                           "Only supplied exclusion manifests were audited; no guarantee of pretraining "
                           "decontamination or exclusion of unavailable old PTQ calibration images.",
        "deduplication": "Exact RGB pixels plus dimensions, including across tasks; not perceptual dedup",
        "preprocessing": {key: cfg[key] for key in (
            "min_image_tokens", "max_image_tokens", "max_input_tokens", "max_new_tokens")},
        "selection": "One QA per image; filter over-budget records without truncation; keep original questions",
        "license_notes": "Each original dataset/image has its own terms. Public availability is not "
                         "a blanket commercial license; retain attribution and review original sources.",
        "source": "https://huggingface.co/datasets/HuggingFaceM4/the_cauldron",
    }
    try:
        counts = materialize_eval(sources(), processor, cfg, output, args.per_task,
                                  excluded, cfg["seed"], args.max_examined_per_task)
        records = read_manifest(output / "eval.jsonl", require_image=True)
        if len(records) != len(args.tasks) * args.per_task:
            raise ValueError("Prepared record count does not match the locked task plan")
        info.update(status="complete", counts=counts, sample_count=len(records))
        write_json(output / "DATASET_INFO.json", info)
    except Exception as exc:
        if output.exists():
            info.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            try:
                write_json(output / "DATASET_INFO.json", info)
            except OSError:
                pass  # Preserve the original failure if storage is exhausted.
        raise
    print(f"Prepared {len(records)} new evaluation images: {output / 'eval.jsonl'}")
    print("No training, calibration or model weight loading performed.")


if __name__ == "__main__":
    main()
