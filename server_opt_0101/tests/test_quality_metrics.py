import copy
import sys
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality_metrics import (evaluation_fingerprint, exact_mcnemar_p,
                             normalize_answer, paired_summary)
from compare_llm_reports import compare_reports
import compare_llm_reports


class QualityMetricTests(unittest.TestCase):
    def test_normalization_matches_existing_proxy_without_numeric_corruption(self):
        self.assertEqual(normalize_answer(" Living  ROOM. "), "living room")
        self.assertEqual(normalize_answer("Ｙｅｓ！"), "yes")
        self.assertEqual(normalize_answer("-1.25."), "-1.25")
        self.assertNotEqual(normalize_answer("two"), normalize_answer("2"))
        self.assertNotEqual(normalize_answer("living room"), normalize_answer("a living room"))

    def test_exact_paired_test(self):
        self.assertEqual(exact_mcnemar_p(0, 0), 1)
        self.assertEqual(exact_mcnemar_p(3, 0), 0.25)
        self.assertEqual(exact_mcnemar_p(0, 3), 0.25)
        self.assertEqual(exact_mcnemar_p(4, 4), 1)
        self.assertAlmostEqual(exact_mcnemar_p(10, 0), 2 / 1024)
        self.assertTrue(0 <= exact_mcnemar_p(1000, 30) <= 1)

    def test_zero_teacher_score_has_no_fictitious_retention(self):
        result = paired_summary([{
            "fp16_correct": False, "baseline_correct": False, "candidate_correct": True}])
        self.assertIsNone(result["candidate_score_retention"])
        self.assertEqual(result["net_correct_gain"], 1)
        self.assertEqual(result["score_delta_pp"], 100)

    def test_fingerprint_ignores_paths_order_but_not_labels_or_image_content(self):
        a = [{"id": "b", "image": "/a.png", "image_sha256": "hash-b", "answer": "yes"},
             {"id": "a", "image": "/b.png", "image_sha256": "hash-a", "answer": "no"}]
        b = copy.deepcopy(list(reversed(a)))
        b[0]["image"] = "/other/disk/b.png"
        self.assertEqual(evaluation_fingerprint(a), evaluation_fingerprint(b))
        b[0]["answer"] = "changed"
        self.assertNotEqual(evaluation_fingerprint(a), evaluation_fingerprint(b))
        b = copy.deepcopy(a)
        b[0]["image_sha256"] = "changed pixels"
        self.assertNotEqual(evaluation_fingerprint(a), evaluation_fingerprint(b))

    def reports(self):
        rows = [{"id": str(i), "question": "What color?", "image": f"/{i}.png",
                 "reference_text": "Red.", "reference_token_ids": [1, 2],
                 "ground_truth_answer": "red", "candidate_text": prediction,
                 "source_subset": "vqav2" if i < 2 else "textvqa"}
                for i, prediction in enumerate(("red", "blue", "red", "blue"))]
        a = {"metrics": {"teacher_kl": 0.1}, "samples": rows}
        b = copy.deepcopy(a)
        for row, prediction in zip(b["samples"], ("red", "red", "blue", "red")):
            row["candidate_text"] = prediction
        return a, b

    def test_gains_losses_are_paired_and_grouped_not_just_net_counts(self):
        result = compare_reports(*self.reports())
        overall = result["overall"]
        self.assertEqual(overall["baseline_correct"], 2)
        self.assertEqual(overall["candidate_correct"], 3)
        self.assertEqual(overall["gained"], 2)
        self.assertEqual(overall["lost"], 1)
        self.assertEqual(overall["net_correct_gain"], 1)
        self.assertEqual(overall["candidate_score_retention"], 0.75)
        self.assertEqual(result["by_task"]["vqav2"]["candidate_correct"], 2)
        self.assertEqual(result["by_task"]["textvqa"]["candidate_correct"], 1)

    def test_refuses_mismatched_teacher_data_protocol_and_duplicate_ids(self):
        for key, value in [("reference_text", "blue"), ("reference_token_ids", [3]),
                           ("ground_truth_answer", "blue"), ("question", "Different?"),
                           ("image", "/different.png"), ("source_subset", "chartqa")]:
            a, b = self.reports()
            b["samples"][0][key] = value
            with self.assertRaises(ValueError):
                compare_reports(a, b)
        a, b = self.reports()
        a["evaluation_fingerprint"] = "one"
        b["evaluation_fingerprint"] = "two"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            compare_reports(a, b)
        a, b = self.reports()
        b["samples"][0]["id"] = b["samples"][1]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare_reports(a, b)

    def test_cli_writes_summary_and_cases_then_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate = self.reports()
            a, b, output = root / "ptq.json", root / "qat.json", root / "comparison.json"
            a.write_text(json.dumps(baseline), encoding="utf-8")
            b.write_text(json.dumps(candidate), encoding="utf-8")
            argv = ["compare_llm_reports", "--baseline", str(a),
                    "--candidate", str(b), "--out", str(output)]
            with patch.object(sys, "argv", argv):
                compare_llm_reports.main()
                first = output.read_bytes()
                with self.assertRaises(FileExistsError):
                    compare_llm_reports.main()
                self.assertEqual(first, output.read_bytes())
            result = json.loads(first)
            self.assertEqual(result["overall"]["net_correct_gain"], 1)
            self.assertEqual(len(result["cases"]), 4)


if __name__ == "__main__":
    unittest.main()
