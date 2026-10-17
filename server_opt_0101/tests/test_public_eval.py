import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from contextlib import ExitStack

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_manifest
from prepare_public_eval import excluded_pixel_hashes, materialize_eval
import prepare_public_eval
from prepare_public_qat import pixel_hash
from test_public_qat import FixtureProcessor


class IndependentEvalTests(unittest.TestCase):
    cfg = {"max_new_tokens": 4, "max_input_tokens": 32}

    def row(self, color, answer="red", question="What color?"):
        return {"images": [Image.new("RGB", (8, 8), color)],
                "texts": [{"user": question, "assistant": answer}]}

    def test_tasks_are_balanced_and_exclude_training_and_cross_task_duplicates(self):
        red = self.row("red")
        blue, green, yellow, black = [self.row(color) for color in ("blue", "green", "yellow", "black")]
        sources = [("vqav2", [red, blue, green]), ("textvqa", [blue, yellow, black])]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite"
            result = materialize_eval(sources, FixtureProcessor(), self.cfg,
                                      output, 2, {pixel_hash(red["images"][0])}, 42)
            rows = read_manifest(output / "eval.jsonl", require_image=True)
            self.assertEqual(len(rows), 4)
            self.assertEqual(result["vqav2"]["saved"], 2)
            self.assertEqual(result["textvqa"]["saved"], 2)
            self.assertEqual(result["vqav2"]["duplicate_or_excluded"], 1)
            self.assertEqual(result["textvqa"]["duplicate_or_excluded"], 1)
            self.assertEqual(len({row["rgb_sha256"] for row in rows}), 4)
            self.assertFalse((output / "train.jsonl").exists())
            with self.assertRaises(FileExistsError):
                materialize_eval(sources, FixtureProcessor(), self.cfg, output, 2, set(), 42)

    def test_insufficient_valid_records_fail_and_do_not_silently_truncate_labels(self):
        rows = [self.row("red", answer="one two three four five"), self.row("blue")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite"
            with self.assertRaisesRegex(RuntimeError, "1/2"):
                materialize_eval([("vqav2", rows)], FixtureProcessor(), self.cfg,
                                 output, 2, set(), 42)
            saved = read_manifest(output / "eval.jsonl")
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["answer"], "red")

    def test_pixel_exclusion_reads_manifests_and_rejects_missing_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (8, 8), "red")
            image.save(root / "old.png")
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps({"id": "old", "image": "old.png", "question": "Q?"}),
                                encoding="utf-8")
            self.assertEqual(excluded_pixel_hashes([manifest]), {pixel_hash(image)})
            manifest.write_text(json.dumps({"image": "missing.png", "question": "Q?"}), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                excluded_pixel_hashes([manifest])

    def test_main_pins_revision_and_writes_complete_or_failed_metadata(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "suite"
                cfg = dict(self.cfg, seed=42, min_image_tokens=16, max_image_tokens=64)
                stream = Mock()
                stream.shuffle.return_value = [] if fail else [self.row("blue")]
                load = Mock(return_value=stream)
                hub = Mock()
                hub.dataset_info.return_value.sha = "pinned-source-revision"
                with ExitStack() as stack:
                    stack.enter_context(patch.object(sys, "argv", ["prepare_public_eval", "--config", "c.json",
                        "--exclude", "old.jsonl", "--tasks", "vqav2", "--per-task", "1", "--out", str(output)]))
                    stack.enter_context(patch.object(prepare_public_eval, "load_config", return_value=cfg))
                    stack.enter_context(patch.object(prepare_public_eval, "excluded_pixel_hashes", return_value=set()))
                    stack.enter_context(patch("torch_work.make_processor", return_value=FixtureProcessor()))
                    stack.enter_context(patch.dict(sys.modules, {
                        "datasets": SimpleNamespace(load_dataset=load),
                        "huggingface_hub": SimpleNamespace(HfApi=Mock(return_value=hub))}))
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "0/1"):
                            prepare_public_eval.main()
                    else:
                        prepare_public_eval.main()
                info = json.loads((output / "DATASET_INFO.json").read_text())
                self.assertEqual(info["revision"], "pinned-source-revision")
                self.assertEqual(info["status"], "failed" if fail else "complete")
                self.assertEqual(load.call_args.kwargs["revision"], "pinned-source-revision")
                self.assertTrue(load.call_args.kwargs["streaming"])
                if not fail:
                    self.assertEqual(info["sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
