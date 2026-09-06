"""Check the ModelOpt 0/0 repair without weights or GPU; optional real CPU test."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_work import modelopt_smoothquant_zero_guard, repair_zero_channel_sq_scale

MODELOPT_AVAILABLE = (importlib.util.find_spec("torch") is not None
                      and importlib.util.find_spec("modelopt") is not None)


class SmoothQuantGuardTests(unittest.TestCase):
    def test_fp16_underflow_and_nan_survives_clamp(self):
        # Read-only inspection found these magnitudes at block0 MLP channel1590.
        self.assertEqual(np.float16(4.5e-12), 0)
        x = np.float64(-5.375)
        gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3)))
        self.assertEqual(np.float16(gelu), 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.clip(np.sqrt(np.array([0, 4], np.float32)) /
                            np.sqrt(np.array([0, 16], np.float32)), 1e-4, 1e4)
        self.assertTrue(np.isnan(scale[0]))
        repaired, details = repair_zero_channel_sq_scale(scale, [0, 16], [0, 4], "block0.fc2")
        np.testing.assert_array_equal(repaired, [1, 0.5])
        self.assertEqual(details["repaired_channel_indices"], [0])

    def test_healthy_scales_unchanged(self):
        scales = np.array([1e-4, 0.25, 2, 1e4], dtype=np.float32)
        fixed, details = repair_zero_channel_sq_scale(scales, [1, 4, 1, 0], [0, 1, 4, 2], "ok")
        np.testing.assert_array_equal(fixed, scales)
        self.assertEqual(details["repaired_zero_over_zero_count"], 0)

    def test_only_zero_over_zero_can_be_repaired(self):
        for act, weight in (([0, 2], [1, 3]), ([1, 2], [0, 3]), ([1, 2], [1, 3])):
            with self.subTest(act=act, weight=weight), self.assertRaisesRegex(ValueError, "only NaN"):
                repair_zero_channel_sq_scale([np.nan, 1], act, weight, "live")

    def test_infinite_or_negative_statistics_rejected(self):
        for values in ([np.inf, 1], [np.nan, 1], [-1, 1]):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "amax"):
                repair_zero_channel_sq_scale([1, 1], values, [0, 1], "bad-activation")
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "amax"):
                repair_zero_channel_sq_scale([1, 1], [0, 1], values, "bad-weight")

    def test_not_a_generic_nan_to_num(self):
        for value in (np.inf, -np.inf, -1, 0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                repair_zero_channel_sq_scale([value, 1], [0, 1], [0, 1], "bad-scale")

    def test_channel_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shapes"):
            repair_zero_channel_sq_scale([1, 2], [1], [1, 2], "bad-shape")

    def test_inputs_not_modified(self):
        scales = np.array([np.nan, 0.5], dtype=np.float32)
        repaired, details = repair_zero_channel_sq_scale(scales, [0, 4], [0, 1], "zero")
        self.assertTrue(np.isnan(scales[0]))
        self.assertEqual(repaired[0], 1)
        self.assertEqual(details["repaired_zero_over_zero_count"], 1)

    def test_zero_column_output_equivalence(self):
        weights = np.array([[0, 2], [0, -1]], dtype=np.float32)
        # Even an unseen, nonzero activation cannot affect an exactly zero column.
        inputs = np.array([[100, 3]], dtype=np.float32)
        scale, _ = repair_zero_channel_sq_scale([np.nan, 0.5], [0, 4], [0, 2], "zero")
        np.testing.assert_allclose((inputs * scale) @ (weights / scale).T, inputs @ weights.T)


@unittest.skipUnless(MODELOPT_AVAILABLE, "Real ModelOpt CPU test requires the server dependencies")
class ModelOptIntegrationTests(unittest.TestCase):
    def test_official_smoothing_with_dead_channel_and_hook_restoration(self):
        import torch
        import modelopt.torch.quantization as mtq
        from modelopt.torch.quantization import model_calib

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = torch.nn.Sequential(
                    torch.nn.Linear(4, 6), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(6, 4))
                self.unused_llm = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.visual(x)

        model = Toy().half().eval()
        with torch.no_grad():
            for linear in (model.visual[0], model.visual[2]):
                linear.weight.fill_(0.2)
                linear.bias.fill_(0.1)
            model.visual[0].weight[3].zero_()
            model.visual[0].bias[3] = -5.375
            model.visual[2].weight[:, 3].zero_()
        x = torch.tensor([[1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]], dtype=torch.float16)
        targets = ["visual.0", "visual.2"]
        config = {"quant_cfg": [{"quantizer_name": "*", "enable": False}],
                  "algorithm": {"method": "smoothquant", "alpha": 0.5}}
        for name in targets:
            config["quant_cfg"].extend([
                {"quantizer_name": name + ".input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
                {"quantizer_name": name + ".weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
            ])
        original_helper = model_calib.apply_pre_quant_scale_and_smooth
        with modelopt_smoothquant_zero_guard(model, targets) as checks:
            mtq.quantize(model, config, forward_loop=lambda m: m(x))
        self.assertIs(model_calib.apply_pre_quant_scale_and_smooth, original_helper)
        self.assertEqual({row["module"] for row in checks}, set(targets))
        self.assertEqual(checks[-1]["repaired_channel_indices"], [3])
        self.assertEqual(model.visual[2].input_quantizer.pre_quant_scale[3].item(), 1)
        self.assertTrue(torch.isfinite(model(x)).all())
        self.assertFalse(model.unused_llm.weight_quantizer.is_enabled)
        self.assertFalse(model.unused_llm.input_quantizer.is_enabled)
        for name in targets:
            layer = model.get_submodule(name)
            self.assertIsNone(layer.input_quantizer.axis)
            self.assertTrue(torch.isfinite(layer.input_quantizer.amax).all())
            self.assertTrue((layer.input_quantizer.pre_quant_scale > 0).all())
        with self.assertRaisesRegex(RuntimeError, "test restoration"):
            with modelopt_smoothquant_zero_guard(model, targets):
                raise RuntimeError("test restoration")
        self.assertIs(model_calib.apply_pre_quant_scale_and_smooth, original_helper)


if __name__ == "__main__":
    unittest.main()
