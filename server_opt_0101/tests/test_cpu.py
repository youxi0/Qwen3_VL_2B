"""No torch/ONNX/CUDA imports or model execution: packing, paths, splits, metrics."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (check_splits, inspect_awq_modules, load_config,
                    AWQ_LINEAR_SUFFIXES, prepare_manifests, read_json, read_manifest,
                    safetensor_header)
from requantize_awq_lm_head import (configure_image_token_range,
                                    diversify_repeated_questions,
                                    select_calibration_rows)
from llm_layer_sensitivity import aggregate, recovery_row
from torch_work import replace_named_submodule, unpack_awq_numpy
from trt_vision_check import (classify_feature_metrics, feature_metrics,
                              profile_shapes)
from vision_layer_sensitivity import (recovery_row as vision_recovery_row,
                                      summarize_vision_rows, target_groups)


class PackingTests(unittest.TestCase):
    def test_all_signed_nibbles_and_output_axis(self):
        values = np.arange(-8, 8, dtype=np.int8).reshape(4, 4)
        packed = ((values[0::2].astype(np.int16) & 15) |
                  ((values[1::2].astype(np.int16) & 15) << 4)).astype(np.uint8)
        np.testing.assert_array_equal(unpack_awq_numpy(packed), values)

    def test_known_byte(self):
        actual = unpack_awq_numpy(np.array([[0x87, 0xF1]], dtype=np.uint8))
        np.testing.assert_array_equal(actual, [[7, 1], [-8, -1]])

    def test_reject_autoawq_int32(self):
        with self.assertRaises(ValueError):
            unpack_awq_numpy(np.ones((2, 8), dtype=np.int32))

    def test_reject_wrong_rank(self):
        with self.assertRaises(ValueError):
            unpack_awq_numpy(np.ones(8, dtype=np.uint8))

    def test_awq_layout_accepts_optional_int4_lm_head(self):
        def item(dtype="U8"):
            return Path("weights.safetensors"), {
                "dtype": dtype, "shape": [1], "data_offsets": [0, 1]
            }

        index = {
            f"model.language_model.layers.{i}.{suffix}.weight": item()
            for i in range(28) for suffix in AWQ_LINEAR_SUFFIXES
        }
        base = inspect_awq_modules(index)
        self.assertEqual(base["total_packed_linears"], 196)
        self.assertFalse(base["lm_head_int4"])
        index["lm_head.weight"] = item()
        with_head = inspect_awq_modules(index)
        self.assertEqual(with_head["total_packed_linears"], 197)
        self.assertTrue(with_head["lm_head_int4"])

    def test_awq_layout_rejects_quantized_embedding(self):
        def item():
            return Path("weights.safetensors"), {
                "dtype": "U8", "shape": [1], "data_offsets": [0, 1]
            }

        index = {
            f"model.language_model.layers.{i}.{suffix}.weight": item()
            for i in range(28) for suffix in AWQ_LINEAR_SUFFIXES
        }
        index["model.language_model.embed_tokens.weight"] = item()
        with self.assertRaises(ValueError):
            inspect_awq_modules(index)

    def test_awq_layout_accepts_fp16_blocks_and_modules(self):
        def item(dtype="U8"):
            return Path("weights.safetensors"), {
                "dtype": dtype, "shape": [1], "data_offsets": [0, 1]
            }

        index = {
            f"model.language_model.layers.{i}.{suffix}.weight": item()
            for i in range(28) for suffix in AWQ_LINEAR_SUFFIXES
            if i not in (3, 17)
        }
        for i in (3, 17):
            for suffix in AWQ_LINEAR_SUFFIXES:
                index[f"model.language_model.layers.{i}.{suffix}.weight"] = item("F16")
        index["lm_head.weight"] = item()
        layout = inspect_awq_modules(index)
        self.assertEqual(layout["backbone_linears"], 182)
        self.assertEqual(layout["fp16_backbone_layers"], [3, 17])
        index["model.language_model.layers.4.mlp.down_proj.weight"] = item("F16")
        partial = inspect_awq_modules(index)
        self.assertEqual(partial["partial_fp16_backbone_layers"], [4])
        self.assertIn("model.language_model.layers.4.mlp.down_proj",
                      partial["fp16_backbone_modules"])

    def test_replace_root_and_nested_quantized_modules(self):
        class Node:
            def get_submodule(self, name):
                current = self
                for part in name.split("."):
                    current = getattr(current, part)
                return current

        model = Node()
        model.lm_head = "fp16-head"
        model.model = Node()
        model.model.proj = "fp16-proj"
        replace_named_submodule(model, "lm_head", "int4-head")
        replace_named_submodule(model, "model.proj", "int4-proj")
        self.assertEqual(model.lm_head, "int4-head")
        self.assertEqual(model.model.proj, "int4-proj")


class FileTests(unittest.TestCase):
    def test_safetensors_valid_and_truncated(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            header = json.dumps({"x": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}}).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x12\x34")
            self.assertEqual(safetensor_header(path)["x"]["shape"], [2])
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x12")
            with self.assertRaises(ValueError):
                safetensor_header(path)

    def test_relative_manifest_and_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "image.png").write_bytes(b"image fixture")
            path = root / "samples.jsonl"
            path.write_text(json.dumps({"id": "a", "image": "image.png", "question": "describe"}) + "\n", encoding="utf-8")
            records = read_manifest(path, require_image=True)
            self.assertEqual(records[0]["image"], str((root / "image.png").resolve()))
            with self.assertRaises(ValueError):
                check_splits(records, records)
            check_splits(records, [{"id": "text", "question": "hello"}])

    def test_dedup_and_reproducible_prepare(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "images"
            source.mkdir()
            for i in range(7):
                Image.new("RGB", (32+i, 32), (i*20, 1, 2)).save(source / f"{i}.png")
            Image.open(source / "0.png").save(source / "duplicate.png")
            cfg = {"images": str(source), "seed": 42, "calibration_samples": 4, "evaluation_samples": 2}
            first = prepare_manifests(cfg, root / "first")
            second = prepare_manifests(cfg, root / "second")
            self.assertEqual((first / "calib.jsonl").read_bytes(), (second / "calib.jsonl").read_bytes())
            self.assertEqual(read_json(first / "manifest_info.json")["unique_decoded_images"], 7)
            calib = read_manifest(first / "calib.jsonl", True)
            self.assertGreater(len({row["question"] for row in calib}), 1)
            check_splits(calib, read_manifest(first / "eval.jsonl"))
            with self.assertRaises(FileExistsError):
                prepare_manifests(cfg, first)

    def test_full_model_awq_calibration_mix(self):
        images = [{"id": f"i{i}", "image": f"{i}.jpg", "question": "same"}
                  for i in range(8)]
        texts = [{"id": f"t{i}", "question": f"text {i}", "answer": "ok"}
                 for i in range(8)]
        diversified, changed = diversify_repeated_questions(images)
        self.assertEqual(changed, 8)
        self.assertGreater(len({row["question"] for row in diversified}), 1)
        selected = select_calibration_rows(diversified, texts, 8, 0.25, 42)
        self.assertEqual(sum(bool(row.get("image")) for row in selected), 6)
        self.assertEqual(sum(not row.get("image") for row in selected), 2)

    def test_labeled_questions_are_not_rewritten(self):
        rows = [{"id": str(i), "image": f"{i}.jpg", "question": "same",
                 "answer": "truth"} for i in range(4)]
        diversified, changed = diversify_repeated_questions(rows)
        self.assertEqual(changed, 0)
        self.assertEqual({row["question"] for row in diversified}, {"same"})

    def test_awq_image_range_matches_deployment_profile(self):
        class ImageProcessor:
            patch_size = 16
            merge_size = 2
            min_pixels = None
            max_pixels = None

        class Processor:
            image_processor = ImageProcessor()

        pixels = configure_image_token_range(Processor(), 16, 64)
        self.assertEqual(pixels, 1024)
        self.assertEqual(Processor.image_processor.size, {
            "shortest_edge": 16384,
            "longest_edge": 65536,
        })
        self.assertEqual(Processor.image_processor.max_pixels, 65536)

    def test_split_duplicates_rejected(self):
        calib = [{"image": "a", "image_sha256": "one"}]
        evaluation = [{"image": "b", "image_sha256": "two"}] * 2
        with self.assertRaises(ValueError):
            check_splits(calib, evaluation)
        with self.assertRaises(ValueError):
            check_splits(calib * 2, [])

    def test_config_resolves_root_and_limits(self):
        config = Path(__file__).resolve().parents[1] / "config.json"
        with tempfile.TemporaryDirectory() as temporary:
            cfg = load_config(config, temporary)
            self.assertTrue(Path(cfg["original"]).is_absolute())
            self.assertEqual(cfg["calibration_samples"], 256)
            self.assertLessEqual(cfg["max_input_tokens"] + cfg["max_new_tokens"], cfg["kv_cache_capacity"])

    def test_residual_fp16_recipe_passes_config_validation(self):
        source = Path(__file__).resolve().parents[1] / "config.json"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            data = read_json(source)
            data["vision_recipe"] = "residual_fp16"
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(
                load_config(path, temporary)["vision_recipe"],
                "residual_fp16")


class MetricTests(unittest.TestCase):
    def test_layer_recovery_metrics_are_token_weighted(self):
        rows = [
            {"tokens": 1, "teacher_top1_agreement": 1.0,
             "teacher_kl": 0.1, "teacher_token_nll_reference": 0.2,
             "teacher_token_nll_candidate": 0.3},
            {"tokens": 3, "teacher_top1_agreement": 0.0,
             "teacher_kl": 0.5, "teacher_token_nll_reference": 0.6,
             "teacher_token_nll_candidate": 0.7},
        ]
        metrics = aggregate(rows)
        self.assertAlmostEqual(metrics["teacher_top1_agreement"], 0.25)
        self.assertAlmostEqual(metrics["teacher_kl"], 0.4)
        baseline = dict(metrics)
        baseline["teacher_kl"] = 0.6
        baseline["teacher_token_nll_candidate"] = 0.8
        row = recovery_row("3", "decoder_block", metrics, baseline, 10.0)
        self.assertAlmostEqual(row["kl_reduction"], 0.2)
        self.assertAlmostEqual(row["kl_reduction_per_extra_mib"], 0.02)

    def test_vision_grouping_and_recovery(self):
        targets = [
            "model.visual.blocks.0.attn.qkv",
            "model.visual.blocks.0.mlp.linear_fc1",
            "model.visual.deepstack_merger_list.1.linear_fc1",
        ]
        groups = target_groups(targets)
        self.assertEqual(len(groups["block_0"]), 2)
        self.assertEqual(
            groups["deepstack_merger_1"],
            ["model.visual.deepstack_merger_list.1.linear_fc1"],
        )

        rows = []
        for name in ("output", "deepstack_features_0",
                     "deepstack_features_1", "deepstack_features_2"):
            rows.extend([
                {"output": name, "mean_token_cosine": 0.98,
                 "relative_l2": 0.2},
                {"output": name, "mean_token_cosine": 0.96,
                 "relative_l2": 0.3},
            ])
        baseline = summarize_vision_rows(rows, 0.99)
        improved_rows = [dict(row, mean_token_cosine=0.985) for row in rows]
        improved = summarize_vision_rows(improved_rows, 0.99)
        recovery = vision_recovery_row(
            "block_0", "vision_group", improved, baseline, 10.0,
            groups["block_0"],
        )
        self.assertAlmostEqual(baseline["worst_cosine"], 0.96)
        self.assertEqual(baseline["outputs"]["output"]["below_gate"], 2)
        self.assertGreater(recovery["gate_deficit_reduction"], 0)
        self.assertGreater(recovery["worst_cosine_gain"], 0)

    def test_identical(self):
        values = np.array([[1, 2, 3], [-1, 3, 1]], dtype=np.float16)
        result = feature_metrics(values, values)
        self.assertAlmostEqual(result["mean_token_cosine"], 1)
        self.assertEqual(result["relative_l2"], 0)

    def test_nonfinite_and_shapes(self):
        with self.assertRaises(ValueError):
            feature_metrics(np.zeros((2, 3)), np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            feature_metrics(np.ones((2, 3)), np.full((2, 3), np.nan))

    def test_profile_contract(self):
        shapes = profile_shapes(256)
        self.assertEqual(shapes["input"], (256, 1536))
        self.assertEqual(shapes["fast_pos_embed_idx"], (4, 256))
        self.assertEqual(shapes["cu_seqlens"], (2,))
        self.assertEqual(shapes["rotary_pos_emb"], (256, 32))

    def test_engine_fidelity_is_independent_of_quantization_gate(self):
        matching = {"mean_token_cosine": 0.9999, "relative_l2": 0.01}
        original = {"mean_token_cosine": 0.95, "relative_l2": 0.30}
        fidelity, quantization = classify_feature_metrics(
            matching, original, "int8", 0.999, 0.03, 0.99)
        self.assertTrue(fidelity)
        self.assertFalse(quantization)


if __name__ == "__main__":
    unittest.main()
