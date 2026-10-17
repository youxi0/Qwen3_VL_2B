"""Exercise the real report-writing startup paths without CUDA/model weights."""
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import AWQ_LINEAR_SUFFIXES, inspect_awq_modules
import evaluate_llm
import qat_awq


class ReportSerializationTests(unittest.TestCase):
    def fixture(self, root):
        cfg = {key: str(root / key) for key in ("original", "awq", "awq_onnx", "images")}
        cfg["seed"] = 42
        fp16 = {
            "model.language_model.layers.13.self_attn.o_proj",
            "model.language_model.layers.16.self_attn.q_proj",
            "model.language_model.layers.16.self_attn.k_proj",
            "model.language_model.layers.16.self_attn.v_proj",
            "model.language_model.layers.20.mlp.down_proj",
            "model.language_model.layers.20.self_attn.o_proj",
        }
        names = [f"model.language_model.layers.{i}.{suffix}"
                 for i in range(28) for suffix in AWQ_LINEAR_SUFFIXES] + ["lm_head"]
        index = {name + ".weight": (root / "weights.safetensors", {
            "dtype": "F16" if name in fp16 else "U8", "shape": [1],
            "data_offsets": [0, 1],
        }) for name in names}
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True), manual_seed=Mock(),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace()),
                                     cudnn=SimpleNamespace()))
        return cfg, index, torch

    def run_until_model_loading(self, module, root, extra_args):
        cfg, index, torch = self.fixture(root)
        rows = [{"id": str(i), "question": f"Question {i}"} for i in range(4)]
        output = root / "reports"
        argv = [module.__name__, "--config", "config.json", "--eval", "eval.jsonl",
                "--out", str(output)] + extra_args
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch.dict(sys.modules, {"torch": torch}))
            stack.enter_context(patch.object(module, "load_config", return_value=cfg))
            stack.enter_context(patch.object(module, "read_manifest", side_effect=lambda path:
                rows if Path(path).name == "train.jsonl" else [
                    {"id": "eval", "question": "Disjoint final evaluation question"}]))
            stack.enter_context(patch.object(module, "versions", return_value={}))
            stack.enter_context(patch.object(module, "checkpoint_index", return_value=index))
            stack.enter_context(patch("torch_work.make_processor", return_value=Mock()))
            stack.enter_context(patch("torch_work.tokenizer_check"))
            stack.enter_context(patch("torch_work.release_cuda"))
            load = stack.enter_context(patch("torch_work.load_fp16",
                                              side_effect=RuntimeError("model-load sentinel")))
            if module is qat_awq:
                stack.enter_context(patch.object(module, "read_json", return_value={
                    "quantization": {"quant_algo": "W4A16_AWQ", "group_size": 128}}))
            # The former bug failed with TypeError before reaching this sentinel.
            with self.assertRaisesRegex(RuntimeError, "model-load sentinel"):
                module.main()
            load.assert_called_once()
        return output, inspect_awq_modules(index)

    def assert_layout(self, actual, original):
        self.assertIsInstance(original["packed_modules"], set)
        self.assertEqual(actual["packed_modules"], sorted(original["packed_modules"]))
        self.assertEqual(actual["total_packed_linears"], 191)
        self.assertEqual(len(actual["fp16_backbone_modules"]), 6)
        self.assertTrue(actual["lm_head_int4"])

    def test_evaluation_preflight_serializes_real_mixed_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            output, original = self.run_until_model_loading(evaluate_llm, Path(directory), [])
            report = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
            self.assert_layout(report["layout"], original)
            self.assertFalse(report["training_performed"])
            self.assertFalse(report["calibration_performed"])

    def test_qat_initial_report_serializes_real_mixed_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            output, original = self.run_until_model_loading(qat_awq, Path(directory), [
                "--train", "train.jsonl", "--train-samples", "1", "--holdout-samples", "1"])
            report = json.loads((output / "qat_report.json").read_text(encoding="utf-8"))
            self.assert_layout(report["source_layout"], original)
            self.assertEqual(report["status"], "failed")
            self.assertIn("model-load sentinel", report["error"])


if __name__ == "__main__":
    unittest.main()
