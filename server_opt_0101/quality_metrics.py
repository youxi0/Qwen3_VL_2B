"""Small, explicit task-score proxy; not an official VQA/DocVQA/ChartQA scorer."""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata

NORMALIZATION = "nfkc-casefold-whitespace-trailing-punctuation-v1"


def normalize_answer(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip().rstrip(".!?。！？").strip()


def evaluation_fingerprint(rows):
    """Independent of absolute paths; bind labels, questions and image bytes."""
    records = [{key: row.get(key) for key in
                ("id", "question", "answer", "image_sha256", "source_subset")}
               for row in sorted(rows, key=lambda row: row["id"])]
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def exact_mcnemar_p(gained, lost):
    """Two-sided exact binomial/McNemar test on paired discordant answers."""
    if gained < 0 or lost < 0:
        raise ValueError("Counts must be nonnegative")
    count = gained + lost
    if count == 0:
        return 1.0
    tail = sum(math.comb(count, k) for k in range(min(gained, lost) + 1))
    return min(1.0, 2 * tail / 2**count)


def paired_summary(cases):
    count = len(cases)
    teacher = sum(row["fp16_correct"] for row in cases)
    baseline = sum(row["baseline_correct"] for row in cases)
    candidate = sum(row["candidate_correct"] for row in cases)
    gained = sum(not row["baseline_correct"] and row["candidate_correct"] for row in cases)
    lost = sum(row["baseline_correct"] and not row["candidate_correct"] for row in cases)
    return {
        "samples": count, "fp16_correct": teacher,
        "baseline_correct": baseline, "candidate_correct": candidate,
        "fp16_score": teacher / count if count else None,
        "baseline_score": baseline / count if count else None,
        "candidate_score": candidate / count if count else None,
        "net_correct_gain": candidate - baseline, "gained": gained, "lost": lost,
        "both_correct": sum(row["baseline_correct"] and row["candidate_correct"] for row in cases),
        "both_wrong": sum(not row["baseline_correct"] and not row["candidate_correct"] for row in cases),
        "score_delta_pp": 100 * (candidate - baseline) / count if count else None,
        "baseline_score_retention": baseline / teacher if teacher else None,
        "candidate_score_retention": candidate / teacher if teacher else None,
        "exact_mcnemar_p_two_sided": exact_mcnemar_p(gained, lost),
    }
