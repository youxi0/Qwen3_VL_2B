"""CPU-only checks for the exact mixed-precision Vision whitelist."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_work import VISION_RECIPE_EXPECTED, vision_linear_is_target


def qwen3_vl_linear_names():
    names = []
    for block in range(24):
        prefix = f"model.visual.blocks.{block}"
        names.extend([
            prefix + ".attn.qkv",
            prefix + ".attn.proj",
            prefix + ".mlp.linear_fc1",
            prefix + ".mlp.linear_fc2",
        ])
    for prefix in [
        "model.visual.merger",
        "model.visual.deepstack_merger_list.0",
        "model.visual.deepstack_merger_list.1",
        "model.visual.deepstack_merger_list.2",
    ]:
        names.extend([prefix + ".linear_fc1", prefix + ".linear_fc2"])
    return names


class VisionRecipeTests(unittest.TestCase):
    def test_expected_counts(self):
        names = qwen3_vl_linear_names()
        for recipe, expected in VISION_RECIPE_EXPECTED.items():
            selected = [n for n in names if vision_linear_is_target(n, recipe)]
            self.assertEqual(len(selected), expected, recipe)

    def test_residual_outputs_stay_fp16(self):
        recipe = "residual_fp16"
        self.assertTrue(vision_linear_is_target(
            "model.visual.blocks.0.attn.qkv", recipe))
        self.assertTrue(vision_linear_is_target(
            "model.visual.blocks.23.mlp.linear_fc1", recipe))
        self.assertTrue(vision_linear_is_target(
            "model.visual.deepstack_merger_list.2.linear_fc1", recipe))
        self.assertFalse(vision_linear_is_target(
            "model.visual.blocks.0.attn.proj", recipe))
        self.assertFalse(vision_linear_is_target(
            "model.visual.blocks.23.mlp.linear_fc2", recipe))
        self.assertFalse(vision_linear_is_target(
            "model.language_model.layers.0.mlp.down_proj", recipe))

    def test_unknown_recipe_fails(self):
        with self.assertRaises(ValueError):
            vision_linear_is_target("model.visual.blocks.0.attn.qkv", "typo")


if __name__ == "__main__":
    unittest.main()
