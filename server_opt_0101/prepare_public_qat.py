#!/usr/bin/env python3
"""Stream a small VQAv2 train subset into the existing QAT JSONL format."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from common import load_config, read_manifest, write_json

REPO = "HuggingFaceM4/the_cauldron"
SUBSET = "vqav2"


def pixel_hash(image):
    rgb = image.convert("RGB")
    digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def candidate_pairs(texts, tokenizer, max_answer_tokens, rng):
    if not isinstance(texts, list):
        raise ValueError("Expected The Cauldron texts as a list of user/assistant objects")
    pairs = []
    for item in texts:
        if not isinstance(item, dict):
            raise ValueError("Unexpected The Cauldron text entry")
        question, answer = item.get("user"), item.get("assistant")
        if not isinstance(question, str) or not isinstance(answer, str):
            continue
        question = question.replace("<image>", "").strip()
        answer = answer.strip()
        if not question or not answer:
            continue
        length = len(tokenizer.encode(answer, add_special_tokens=False))
        if 1 <= length <= max_answer_tokens:
            pairs.append((question, answer))
    rng.shuffle(pairs)
    return pairs


def prompt_length(processor, image, question):
    encoded = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}], tokenize=True, add_generation_prompt=True, return_dict=True,
    )
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return len(ids)


def materialize(stream, processor, cfg, destination, samples, excluded, seed,
                eval_samples=0):
    """One question per unique decoded image; never resize/crop during saving."""
    from PIL import Image
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Choose a fresh dataset folder: {destination}")
    (destination / "images").mkdir(parents=True)
    rng = random.Random(seed)
    total_samples = samples + eval_samples
    evaluation_assignment = [True] * eval_samples + [False] * samples
    if eval_samples:
        random.Random(seed + 1).shuffle(evaluation_assignment)
    seen = set(excluded)
    stats = Counter()
    with ExitStack() as stack:
        train_handle = stack.enter_context(
            (destination / "train.jsonl").open("x", encoding="utf-8"))
        eval_handle = (stack.enter_context(
            (destination / "eval.jsonl").open("x", encoding="utf-8"))
            if eval_samples else None)
        for source_index, row in enumerate(stream):
            stats["examined"] += 1
            images = row.get("images")
            if not isinstance(images, list) or len(images) != 1:
                stats["not_single_image"] += 1
                continue
            if not isinstance(images[0], Image.Image):
                raise ValueError("The Cauldron must decode images into PIL.Image")
            image = images[0].convert("RGB")
            image.load()
            fingerprint = pixel_hash(image)
            if fingerprint in seen:
                stats["duplicate_or_excluded_image"] += 1
                continue
            pairs = candidate_pairs(row.get("texts"), processor.tokenizer,
                                    cfg["max_new_tokens"], rng)
            selected = None
            for question, answer in pairs:
                if prompt_length(processor, image, question) <= cfg["max_input_tokens"]:
                    selected = question, answer
                    break
            if selected is None:
                stats["no_pair_within_token_limits"] += 1
                continue
            filename = f"images/{stats['saved']:06d}.png"
            # PNG preserves pixels, so dedup hashes remain valid after saving.
            image.save(destination / filename, format="PNG")
            record = {
                "id": f"vqav2_{stats['saved']:06d}", "image": filename,
                "question": selected[0], "answer": selected[1],
                "source_dataset": REPO, "source_subset": SUBSET,
                "source_stream_index": source_index,
            }
            is_eval = evaluation_assignment[stats["saved"]]
            handle = eval_handle if is_eval else train_handle
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            seen.add(fingerprint)
            stats["saved"] += 1
            stats["eval_saved" if is_eval else "train_saved"] += 1
            if stats["saved"] % 100 == 0:
                train_handle.flush()
                if eval_handle is not None:
                    eval_handle.flush()
                print(f"[VQAv2] saved {stats['saved']}/{total_samples}", flush=True)
            if stats["saved"] >= total_samples:
                break
    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exclude-eval", type=Path,
                        help="Reject exact decoded-image duplicates of final eval")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=0,
                        help="Also reserve this many different images as a new fixed eval")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--shuffle-buffer", type=int, default=64,
                        help="Bounded streaming shuffle; not a global uniform sample")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 2 or args.shuffle_buffer < 1 or args.eval_samples < 0:
        parser.error("Use --samples >= 2, --shuffle-buffer >= 1 and --eval-samples >= 0")
    if args.exclude_eval is None and args.eval_samples == 0:
        parser.error("Provide --exclude-eval or create a new eval with --eval-samples N")
    cfg = load_config(args.config, args.root)
    output = args.out.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh dataset folder: {output}")
    from PIL import Image
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from torch_work import make_processor

    excluded = set()
    old_eval = read_manifest(args.exclude_eval) if args.exclude_eval else []
    for row in old_eval:
        if row.get("image"):
            with Image.open(row["image"]) as image:
                excluded.add(pixel_hash(image))
    # Resolve main once and record the exact dataset revision for reproducibility.
    revision = HfApi().dataset_info(REPO, revision=args.revision).sha
    processor = make_processor(cfg)
    stream = load_dataset(REPO, SUBSET, split="train", revision=revision,
                          streaming=True)
    stream = stream.shuffle(seed=cfg["seed"], buffer_size=args.shuffle_buffer)
    try:
        stats = materialize(stream, processor, cfg, output, args.samples,
                            excluded, cfg["seed"], eval_samples=args.eval_samples)
    except Exception as exc:
        if output.exists():
            write_json(output / "PREPARATION_FAILED.json", {
                "error": f"{type(exc).__name__}: {exc}",
                "note": "Partial images/JSONL retained; use a fresh output path on retry.",
            })
        raise
    total_requested = args.samples + args.eval_samples
    complete = stats.get("saved", 0) == total_requested
    write_json(output / "DATASET_INFO.json", {
        "status": "complete" if complete else "incomplete",
        "dataset": REPO, "subset": SUBSET, "split": "train", "revision": revision,
        "seed": cfg["seed"], "shuffle_buffer": args.shuffle_buffer,
        "selection": "Bounded streaming shuffle, one QA per unique decoded image",
        "requested_train": args.samples, "requested_eval": args.eval_samples,
        "requested": total_requested, "counts": stats,
        "excluded_eval": str(args.exclude_eval.resolve()) if args.exclude_eval else None,
        "evaluation_note": (
            "New image-disjoint holdout from VQAv2 train, not the official VQA benchmark split. "
            "Historical scores on a different eval set are not directly comparable."
            if args.eval_samples else "Existing final evaluation set retained"
        ),
        "deduplication": "Exact RGB pixel hash including image dimensions; not perceptual dedup",
        "max_answer_tokens": cfg["max_new_tokens"],
        "max_prompt_tokens_including_image": cfg["max_input_tokens"],
        "source": "https://huggingface.co/datasets/HuggingFaceM4/the_cauldron",
        "annotation_license": "VQA annotations: CC BY 4.0; preserve source attribution",
        "license_notes": "Image rights follow their original licenses; consult source terms.",
        "vqa_terms": "https://visualqa.org/terms.html",
    })
    if not complete:
        raise RuntimeError(f"Only {stats.get('saved', 0)}/{total_requested} valid images were found")
    # Use the same parser that QAT will consume to validate every saved image.
    records = read_manifest(output / "train.jsonl", require_image=True)
    print(f"Prepared {len(records)} records: {output / 'train.jsonl'}", flush=True)
    if args.eval_samples:
        validation = read_manifest(output / "eval.jsonl", require_image=True)
        from common import check_splits
        check_splits(records, validation)
        print(f"Reserved {len(validation)} independent images: "
              f"{output / 'eval.jsonl'}", flush=True)
    print("Only data preparation ran; no model weights were loaded or trained.", flush=True)


if __name__ == "__main__":
    main()
