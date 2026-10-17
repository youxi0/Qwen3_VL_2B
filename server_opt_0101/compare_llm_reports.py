#!/usr/bin/env python3
"""Paired, per-task comparison of fixed PTQ/QAT candidates; no model execution."""
from __future__ import annotations

import argparse
from pathlib import Path

from common import read_json, write_json
from quality_metrics import NORMALIZATION, normalize_answer, paired_summary


def compare_reports(baseline, candidate):
    for key in ("evaluation_fingerprint", "evaluation_settings"):
        if key in baseline or key in candidate:
            if baseline.get(key) != candidate.get(key):
                raise ValueError(f"Reports differ in {key}; rerun with the same eval/settings")

    def indexed(report):
        rows = report["samples"]
        mapping = {row["id"]: row for row in rows}
        if not mapping or len(mapping) != len(rows):
            raise ValueError("Empty report or duplicate sample IDs")
        return mapping

    left, right = indexed(baseline), indexed(candidate)
    if left.keys() != right.keys():
        raise ValueError("Sample ID sets differ")
    cases = []
    for identifier in sorted(left):
        a, b = left[identifier], right[identifier]
        # Compare the actual frozen teacher sequences, not just aggregate scores.
        for key in ("question", "image", "reference_text", "reference_token_ids", "ground_truth_answer"):
            if key not in a or key not in b or a[key] != b[key]:
                raise ValueError(f"{identifier}: mismatched/missing {key}")
        task = a.get("source_subset", "unspecified")
        if task != b.get("source_subset", "unspecified"):
            raise ValueError(f"{identifier}: task labels differ")
        if not a["ground_truth_answer"].strip():
            raise ValueError(f"{identifier}: empty annotation")
        label = normalize_answer(a["ground_truth_answer"])
        cases.append({
            "id": identifier, "task": task, "image": a["image"], "question": a["question"],
            "answer": a["ground_truth_answer"], "fp16_text": a["reference_text"],
            "baseline_text": a["candidate_text"], "candidate_text": b["candidate_text"],
            "fp16_correct": normalize_answer(a["reference_text"]) == label,
            "baseline_correct": normalize_answer(a["candidate_text"]) == label,
            "candidate_correct": normalize_answer(b["candidate_text"]) == label,
        })
    return {
        "normalization": NORMALIZATION,
        "overall": paired_summary(cases),
        "by_task": {task: paired_summary([row for row in cases if row["task"] == task])
                    for task in sorted({row["task"] for row in cases})},
        "baseline_teacher_metrics": baseline["metrics"],
        "candidate_teacher_metrics": candidate["metrics"],
        "cases": cases,
        "limitations": "Normalized single-annotation exact-match proxy, NOT official benchmark accuracy. "
                        "P-values are descriptive paired comparisons, no multiple-testing correction. "
                        "Score retention is an aggregate ratio, not identical answers or recovered loss. "
                        "No automatic deployment acceptance; target memory/engine checks remain required.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Choose a new report file: {args.out}")
    result = compare_reports(read_json(args.baseline), read_json(args.candidate))
    write_json(args.out, result)
    print("[Normalized single-answer exact; NOT official benchmark scoring]")
    for task, summary in [("overall", result["overall"]), *result["by_task"].items()]:
        n = summary["samples"]
        retention = summary["candidate_score_retention"]
        retention_text = f"{100 * retention:.2f}%" if retention is not None else "n/a"
        print(f"{task}: FP16={summary['fp16_correct']}/{n}, "
              f"PTQ={summary['baseline_correct']}/{n}, QAT={summary['candidate_correct']}/{n}, "
              f"gain/loss={summary['gained']}/{summary['lost']}, "
              f"delta={summary['score_delta_pp']:+.2f}pp, QAT/FP16={retention_text}, "
              f"paired p={summary['exact_mcnemar_p_two_sided']:.4g}")
    print(f"Cases and summaries: {args.out}")


if __name__ == "__main__":
    main()
