"""QAT split leakage, training gradients, and exact INT4 export semantics."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qat_awq import (make_qat_linear_class, select_trainable_modules,
                     split_training_rows, answer_logits, distillation_loss,
                     packed_change_summary, get_answer_tokens,
                     build_holdout_references, reference_token_summary)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
MODELOPT_AVAILABLE = TORCH_AVAILABLE and importlib.util.find_spec("modelopt") is not None


class QATDataTests(unittest.TestCase):
    def test_default_answer_mode_preserves_annotation_behavior(self):
        processor = Mock()
        processor.tokenizer.encode.return_value = [2, 3]
        fake_torch = SimpleNamespace(tensor=Mock(), long="long")
        with patch.dict(sys.modules, {"torch": fake_torch}), \
                patch("torch_work.greedy") as greedy:
            result = get_answer_tokens("teacher", processor, {},
                                       {"id": "a", "answer": "two"}, {"max_new_tokens": 32})
        processor.tokenizer.encode.assert_called_once_with("two", add_special_tokens=False)
        fake_torch.tensor.assert_called_once_with([[2, 3]], dtype="long", device="cuda")
        self.assertIs(result, fake_torch.tensor.return_value)
        greedy.assert_not_called()

    def test_teacher_answer_mode_ignores_annotations_without_modifying_row(self):
        row = {"id": "a", "answer": "human answer"}
        tokens = SimpleNamespace(ndim=2, shape=(1, 3))
        generated = Mock()
        generated.clone.return_value = tokens
        processor = Mock()
        with patch("torch_work.greedy", return_value=generated) as greedy:
            result = get_answer_tokens("teacher", processor, {"prompt": 1}, row,
                                       {"max_new_tokens": 32}, "teacher")
        self.assertIs(result, tokens)
        greedy.assert_called_once()
        processor.tokenizer.encode.assert_not_called()
        self.assertEqual(row["answer"], "human answer")

    def test_invalid_answer_source_and_empty_teacher_response_rejected(self):
        cfg = {"max_new_tokens": 32}
        with self.assertRaisesRegex(ValueError, "Unknown answer source"):
            get_answer_tokens(None, None, {}, {"id": "a"}, cfg, "invalid")
        generated = Mock()
        generated.clone.return_value = SimpleNamespace(ndim=2, shape=(1, 0))
        with patch("torch_work.greedy", return_value=generated):
            with self.assertRaisesRegex(ValueError, "invalid continuation"):
                get_answer_tokens(None, None, {}, {"id": "a"}, cfg, "teacher")

    def test_reference_builder_passes_answer_policy_and_keeps_fixed_tokens(self):
        tokens, logits = Mock(), Mock()
        row, cfg = {"id": "a", "answer": "annotation"}, {"max_new_tokens": 32}
        with patch("torch_work.prepare_inputs", return_value={"prompt": 1}), \
                patch("qat_awq.get_answer_tokens", return_value=tokens) as choose, \
                patch("qat_awq.answer_logits", return_value=logits):
            result = build_holdout_references(
                "teacher", "processor", [row], cfg, "teacher", "train probe")
        choose.assert_called_once_with("teacher", "processor", {"prompt": 1},
                                       row, cfg, "teacher")
        self.assertIs(result[0]["tokens"], tokens.cpu.return_value)
        self.assertIs(result[0]["logits"], logits.cpu.return_value)

    def test_answer_token_summary_includes_lengths(self):
        references = [{"tokens": SimpleNamespace(shape=(1, count))} for count in (2, 6)]
        self.assertEqual(reference_token_summary(references), {
            "samples": 2, "tokens": 8, "mean_answer_tokens": 4,
            "min_answer_tokens": 2, "max_answer_tokens": 6})
        self.assertEqual(reference_token_summary([])["tokens"], 0)

    def test_actual_changed_values_are_counted_per_nibble(self):
        import numpy as np
        before = {"head": np.array([[0x01, 0x72, 0xFF]], dtype=np.uint8)}
        after = {"head": np.array([[0x02, 0x82, 0x00]], dtype=np.uint8)}
        result = packed_change_summary(before, after)
        self.assertEqual(result["changed_packed_bytes"]["head"], 3)
        self.assertEqual(result["changed_int4_values"]["head"], 4)
        self.assertEqual(result["total_int4_values"], 6)
        self.assertAlmostEqual(result["changed_fraction"], 4 / 6)
        unchanged = packed_change_summary(before, before)
        self.assertEqual(unchanged["total_changed_int4_values"], 0)

    def test_same_image_different_questions_cannot_cross_holdout(self):
        rows = []
        for i in range(10):
            for question in ("describe", "count"):
                rows.append({"id": f"{i}-{question}", "image": f"{i}.jpg",
                             "image_sha256": str(i), "question": question})
        train, held = split_training_rows(rows, [], 3, 100, 42)
        train_images = {row["image_sha256"] for row in train}
        held_images = {row["image_sha256"] for row in held}
        self.assertFalse(train_images & held_images)
        self.assertEqual(len(train), 14)
        self.assertEqual(len(held), 3)

    def test_final_eval_overlap_rejected_even_with_different_id(self):
        rows = [{"id": str(i), "image": f"{i}.jpg",
                 "image_sha256": str(i), "question": "describe"}
                for i in range(6)]
        evaluation = [dict(rows[0], id="final-eval", question="other question")]
        with self.assertRaisesRegex(ValueError, "overlaps"):
            split_training_rows(rows, evaluation, 2, 100, 42)

    def test_text_prompt_overlap_and_empty_training_rejected(self):
        rows = [{"id": str(i), "question": f"question {i}"} for i in range(6)]
        with self.assertRaisesRegex(ValueError, "overlaps"):
            split_training_rows(rows, [dict(rows[0], answer="different")], 2, 10, 42)
        with self.assertRaises(ValueError):
            split_training_rows(rows[:2], [], 2, 10, 42)
        self.assertEqual(split_training_rows(rows, [], 2, 3, 42),
                         split_training_rows(rows, [], 2, 3, 42))

    def test_scope_selection_cannot_train_fp16_fallbacks(self):
        names = {"lm_head", "model.language_model.layers.1.mlp.down_proj"}
        self.assertEqual(select_trainable_modules(names, "head"), ["lm_head"])
        self.assertEqual(select_trainable_modules(names, "decoder"),
                         ["model.language_model.layers.1.mlp.down_proj"])
        self.assertEqual(set(select_trainable_modules(names, "all")), names)
        with self.assertRaises(ValueError):
            select_trainable_modules(names - {"lm_head"}, "head")


@unittest.skipUnless(TORCH_AVAILABLE, "Torch checks run on the server")
class QATTorchTests(unittest.TestCase):
    def toy(self):
        import torch
        cls = make_qat_linear_class()
        # Low nibbles are output row 0, high nibbles are output row 1.
        packed = torch.tensor([[0x81, 0x72, 0xF3, 0x04]], dtype=torch.uint8)
        scales = torch.tensor([[0.25, 0.5], [0.125, 0.25]])
        return cls(packed, scales, torch.tensor([1, 2, 1, 0.5]), None, 2)

    def test_step_zero_forward_and_nonzero_ste_gradient(self):
        import torch
        import torch.nn.functional as functional
        layer = self.toy()
        integers = torch.tensor([[1, 2, 3, 4], [-8, 7, -1, 0]])
        weight = (integers.reshape(2, 2, 2).float()
                  * layer.weight_scale[:, :, None]).reshape(2, 4).half()
        x = torch.tensor([[1, 0.5, -1, 2]], dtype=torch.float16)
        expected = functional.linear(x * layer.pre_quant_scale, weight)
        value = layer(x)
        self.assertTrue(torch.equal(value, expected))
        value.float().sum().backward()
        self.assertIsNotNone(layer.master_weight.grad)
        self.assertGreater(layer.master_weight.grad.abs().sum().item(), 0)
        self.assertIsNone(layer.weight_scale.grad)

    def test_update_can_change_quantized_forward(self):
        import torch
        layer = self.toy()
        x = torch.ones((1, 4), dtype=torch.float16)
        before = layer(x).detach()
        optimizer = torch.optim.SGD([layer.master_weight], lr=0.2)
        layer(x).float().sum().backward()
        optimizer.step()
        layer.project()
        self.assertFalse(torch.equal(before, layer(x).detach()))

    def test_fp16_in_bin_init_preserves_forward_but_reduces_rounding_barrier(self):
        import torch
        layer = self.toy()
        x = torch.ones((1, 4), dtype=torch.float16)
        before = layer(x).detach()
        centers = layer.master_weight.detach().clone()
        scales = layer.weight_scale.repeat_interleave(layer.group_size, dim=1)
        original = (centers - 0.49 * scales) * layer.pre_quant_scale
        layer.initialize_within_bins(original)
        self.assertTrue(torch.equal(before, layer(x).detach()))
        self.assertFalse(torch.equal(centers, layer.master_weight.detach()))
        optimizer = torch.optim.SGD([layer.master_weight], lr=0.01)
        layer(x).float().sum().backward()
        optimizer.step()
        layer.project()
        self.assertFalse(torch.equal(before, layer(x).detach()))
        diagnostics = layer.shadow_diagnostics()
        self.assertGreater(diagnostics["mean_abs_subbin_offset"], 0)
        self.assertLessEqual(diagnostics["max_abs_subbin_offset"], 0.50001)

    def test_teacher_no_grad_but_student_distillation_has_gradients(self):
        import torch
        from types import SimpleNamespace

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(5, 7))
                self.model = SimpleNamespace(rope_deltas="stale")

            def forward(self, input_ids, logits_to_keep, **kwargs):
                self.last_positions = logits_to_keep.tolist()
                return SimpleNamespace(logits=self.weight[logits_to_keep][None])

        inputs = {"input_ids": torch.tensor([[0, 1, 2]]),
                  "attention_mask": torch.ones((1, 3), dtype=torch.long)}
        tokens = torch.tensor([[3, 4]])
        model = Toy()
        teacher = answer_logits(model, inputs, tokens)
        self.assertFalse(teacher.requires_grad)
        student = answer_logits(model, inputs, tokens, require_grad=True)
        self.assertTrue(student.requires_grad)
        self.assertEqual(model.last_positions, [2, 3])
        self.assertIsNone(model.model.rope_deltas)
        teacher = teacher + torch.arange(7).float()
        loss, _, _ = distillation_loss(student, teacher, tokens, 1.0, 0.0)
        loss.backward()
        self.assertGreater(model.weight.grad.abs().sum().item(), 0)


@unittest.skipUnless(MODELOPT_AVAILABLE, "Packing parity requires server ModelOpt")
class QATModelOptTests(unittest.TestCase):
    def test_roundtrip_matches_fake_quantized_forward(self):
        import torch
        import torch.nn.functional as functional
        from torch_work import unpack_awq_numpy
        layer = QATTorchTests().toy()
        # Move across bin boundaries and test a negative clipping extreme.
        with torch.no_grad():
            layer.master_weight.add_(0.17)
            layer.master_weight[0, 0] = -100
        layer.project()
        packed = layer.packed_weight()
        integers = torch.from_numpy(unpack_awq_numpy(packed.numpy()))
        weights = (integers.reshape(2, 2, 2).float()
                   * layer.weight_scale[:, :, None]).reshape(2, 4).half()
        x = torch.tensor([[1, 2, -1, 0.5]], dtype=torch.float16)
        self.assertTrue(torch.equal(layer(x), functional.linear(
            x * layer.pre_quant_scale, weights)))


if __name__ == "__main__":
    unittest.main()
