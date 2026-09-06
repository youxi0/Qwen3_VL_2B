"""Regression for 71 prompt tokens + 32 text tokens + stale modality IDs.

The NumPy adapter exercises shape/type handling without installing torch locally.
If torch is installed, additional tests exercise actual CPU tensors and the
continuation_logits forward call. No model weights or CUDA are loaded.
"""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_work import continuation_inputs, continuation_logits

TORCH_INSTALLED = importlib.util.find_spec("torch") is not None
NUMPY_TENSOR_OPS = SimpleNamespace(
    cat=lambda tensors, dim: np.concatenate(tensors, axis=dim),
    ones_like=np.ones_like,
    zeros_like=np.zeros_like,
)


class ContinuationArrayTests(unittest.TestCase):
    def setUp(self):
        self.adapter = patch.dict(sys.modules, {"torch": NUMPY_TENSOR_OPS})
        self.adapter.start()
        self.addCleanup(self.adapter.stop)

    def inputs(self):
        types = np.zeros((1, 71), dtype=np.int32)
        types[:, 8:56] = 1
        return {"input_ids": np.arange(71, dtype=np.int64)[None, :],
                "attention_mask": np.ones((1, 71), dtype=np.int32),
                "mm_token_type_ids": types, "image_grid_thw": np.array([[1, 12, 16]]),
                "pixel_values": np.zeros((192, 1536), dtype=np.float16)}

    def test_reported_71_plus_32_failure(self):
        inputs = self.inputs()
        continuation = np.full((1, 32), 123, dtype=np.int64)
        # Reproduce the original indexing failure without running a model.
        with self.assertRaises(IndexError):
            inputs["mm_token_type_ids"][0][np.ones(103, dtype=bool)]
        kwargs = continuation_inputs(inputs, continuation)
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            self.assertEqual(kwargs[key].shape, (1, 103))
        np.testing.assert_array_equal(kwargs["input_ids"][:, 71:], continuation)
        np.testing.assert_array_equal(kwargs["mm_token_type_ids"][:, :71], inputs["mm_token_type_ids"])
        np.testing.assert_array_equal(kwargs["mm_token_type_ids"][:, 71:], 0)
        self.assertEqual(kwargs["mm_token_type_ids"][0][kwargs["attention_mask"][0].astype(bool)].shape, (103,))
        self.assertEqual(kwargs["mm_token_type_ids"].dtype, np.int32)
        self.assertEqual(kwargs["attention_mask"].dtype, np.int32)
        self.assertEqual(inputs["input_ids"].shape, (1, 71))
        self.assertEqual(inputs["mm_token_type_ids"].shape, (1, 71))
        self.assertIs(kwargs["pixel_values"], inputs["pixel_values"])
        self.assertIs(kwargs["image_grid_thw"], inputs["image_grid_thw"])

    def test_short_labeled_answer_and_padding(self):
        inputs = self.inputs()
        inputs["attention_mask"][:, :3] = 0
        kwargs = continuation_inputs(inputs, np.array([[4, 5]], dtype=np.int64))
        self.assertEqual(kwargs["mm_token_type_ids"].shape, (1, 73))
        np.testing.assert_array_equal(kwargs["attention_mask"][:, :71], inputs["attention_mask"])
        np.testing.assert_array_equal(kwargs["attention_mask"][:, 71:], 1)

    def test_text_only_without_modality_ids(self):
        inputs = {"input_ids": np.array([[1, 2]]), "attention_mask": np.ones((1, 2), dtype=bool)}
        kwargs = continuation_inputs(inputs, np.array([[3]]))
        self.assertNotIn("mm_token_type_ids", kwargs)
        self.assertEqual(kwargs["attention_mask"].dtype, np.bool_)

    def test_reject_malformed_prompt_metadata(self):
        for key in ("attention_mask", "mm_token_type_ids"):
            with self.subTest(key=key):
                inputs = self.inputs()
                inputs[key] = inputs[key][:, :-1]
                with self.assertRaisesRegex(ValueError, key):
                    continuation_inputs(inputs, np.ones((1, 32), dtype=np.int64))

    def test_multimodal_types_must_not_be_discarded(self):
        inputs = self.inputs()
        del inputs["mm_token_type_ids"]
        with self.assertRaisesRegex(ValueError, "mm_token_type_ids"):
            continuation_inputs(inputs, np.array([[3]]))

    def test_fresh_positions_and_no_input_mutation(self):
        inputs = self.inputs()
        inputs.update(position_ids=object(), cache_position=object(), past_key_values=object())
        kwargs = continuation_inputs(inputs, np.array([[3]]))
        for key in ("position_ids", "cache_position", "past_key_values"):
            self.assertNotIn(key, kwargs)
            self.assertIn(key, inputs)

    def test_reject_empty_answer_or_wrong_batch(self):
        for shape in ((1, 0), (2, 3), (3,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                continuation_inputs(self.inputs(), np.zeros(shape, dtype=np.int64))


@unittest.skipUnless(TORCH_INSTALLED, "Real torch CPU checks run on the configured server")
class ContinuationTorchTests(unittest.TestCase):
    def test_real_tensors_and_forward_contract(self):
        import torch
        inputs = {"input_ids": torch.arange(71)[None, :],
                  "attention_mask": torch.ones((1, 71), dtype=torch.int32),
                  "mm_token_type_ids": torch.zeros((1, 71), dtype=torch.int32),
                  "image_grid_thw": torch.tensor([[1, 12, 16]])}
        inputs["mm_token_type_ids"][:, 8:56] = 1
        continuation = torch.full((1, 32), 42, dtype=torch.int64)
        test = self

        class FakeModel:
            def __init__(self):
                self.model = SimpleNamespace(rope_deltas="stale")

            def __call__(self, **kwargs):
                test.assertIsNone(self.model.rope_deltas)
                test.assertFalse(kwargs["use_cache"])
                test.assertTrue(kwargs["return_dict"])
                test.assertEqual(kwargs["input_ids"].shape, (1, 103))
                test.assertEqual(kwargs["mm_token_type_ids"].shape, (1, 103))
                selected = kwargs["mm_token_type_ids"][0][kwargs["attention_mask"][0].bool()]
                test.assertEqual(selected.shape, (103,))
                test.assertEqual((selected == 1).sum().item(), 48)
                test.assertEqual(selected[71:].sum().item(), 0)
                return SimpleNamespace(logits=torch.arange(103).view(1, 103, 1).expand(1, 103, 8).float())

        logits = continuation_logits(FakeModel(), inputs, continuation)
        self.assertEqual(logits.shape, (1, 32, 8))
        self.assertEqual(logits[0, 0, 0].item(), 70)
        self.assertEqual(logits[0, -1, 0].item(), 101)
        self.assertEqual(inputs["mm_token_type_ids"].shape, (1, 71))


if __name__ == "__main__":
    unittest.main()
