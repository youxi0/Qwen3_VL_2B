"""Local fixture checks for public-data conversion; no Hub or CUDA needed."""
import random
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_manifest
from prepare_public_qat import candidate_pairs, materialize, pixel_hash


class FixtureProcessor:
    tokenizer = None

    def __init__(self):
        self.tokenizer = self

    def encode(self, text, **kwargs):
        return text.split()

    def apply_chat_template(self, messages, **kwargs):
        question = messages[0]["content"][1]["text"]
        return {"input_ids": [[0] * (8 + len(question.split()))]}


class PublicQATTests(unittest.TestCase):
    def test_pair_filter_preserves_questions_and_does_not_truncate_answers(self):
        pairs = candidate_pairs([
            {"user": "<image> How many?", "assistant": "two"},
            {"user": "Long?", "assistant": "one two three four five"},
            {"user": "", "assistant": "empty question"},
        ], FixtureProcessor(), 4, random.Random(42))
        self.assertEqual(pairs, [("How many?", "two")])

    def test_exact_eval_image_and_internal_duplicates_are_excluded(self):
        red = Image.new("RGB", (8, 8), "red")
        blue = Image.new("RGB", (8, 8), "blue")
        green = Image.new("RGB", (8, 8), "green")
        rows = [{"images": [image], "texts": [
            {"user": "What color?", "assistant": color}
        ]} for image, color in [(red, "red"), (blue, "blue"),
                                (blue.copy(), "blue"), (green, "green")]]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "data"
            stats = materialize(rows, FixtureProcessor(), {
                "max_new_tokens": 4, "max_input_tokens": 32,
            }, output, 2, {pixel_hash(red)}, 42)
            self.assertEqual(stats["saved"], 2)
            self.assertEqual(stats["duplicate_or_excluded_image"], 2)
            records = read_manifest(output / "train.jsonl", require_image=True)
            self.assertEqual({record["answer"] for record in records}, {"blue", "green"})
            with Image.open(records[0]["image"]) as saved:
                self.assertEqual(pixel_hash(saved), pixel_hash(blue))

    def test_prompt_limit_and_existing_directory_protection(self):
        rows = [{"images": [Image.new("RGB", (8, 8), "red")], "texts": [
            {"user": "Question is much too long", "assistant": "yes"}
        ]}]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "data"
            cfg = {"max_new_tokens": 4, "max_input_tokens": 9}
            stats = materialize(rows, FixtureProcessor(), cfg, output, 2, set(), 42)
            self.assertEqual(stats["no_pair_within_token_limits"], 1)
            self.assertFalse(list((output / "images").iterdir()))
            with self.assertRaises(FileExistsError):
                materialize(rows, FixtureProcessor(), cfg, output, 2, set(), 42)


if __name__ == "__main__":
    unittest.main()
