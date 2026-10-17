import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_llm import generation_summary


class EvaluationSummaryTests(unittest.TestCase):
    def test_single_annotation_exact_is_not_teacher_agreement(self):
        samples = [
            {"reference_text": " two ", "candidate_text": "three", "ground_truth_answer": "two",
             "ground_truth_exact_match": False, "generation_exact_match_to_fp16": False},
            {"reference_text": "three", "candidate_text": " two ", "ground_truth_answer": "two",
             "ground_truth_exact_match": True, "generation_exact_match_to_fp16": False},
            {"reference_text": "unannotated", "generation_exact_match_to_fp16": True},
        ]
        result = generation_summary(samples)
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["annotated_sample_count"], 2)
        self.assertEqual(result["fp16_single_answer_exact"], 1)
        self.assertEqual(result["candidate_single_answer_exact"], 1)
        self.assertEqual(result["generation_exact_to_fp16"], 1)
        self.assertEqual(result["fp16_normalized_exact"], 1)
        self.assertEqual(result["candidate_normalized_exact"], 1)
        self.assertIn("NOT official", result["note"])

    def test_no_annotations_has_no_fake_denominator(self):
        result = generation_summary([])
        self.assertEqual(result["annotated_sample_count"], 0)
        self.assertEqual(result["candidate_single_answer_exact"], 0)


if __name__ == "__main__":
    unittest.main()
