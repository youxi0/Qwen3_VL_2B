#!/usr/bin/env python3
"""执行小型任务集，并与原模型文本或业务关键词进行回归比较。"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    project_root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    cases_path = resolve(project_root, config["quality"]["cases_file"])
    cases = [
        json.loads(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    runtime = resolve(project_root, paths["runtime_cli"])
    llm_dir = resolve(project_root, paths["llm_engine_dir"])
    multimodal_dir = resolve(project_root, paths["multimodal_engine_dir"])
    plugin = resolve(project_root, paths["plugin"])
    max_tokens = int(config["request"]["max_new_tokens"])
    results: list[dict[str, Any]] = []

    for case in cases:
        image = resolve(project_root, case["image"])
        with tempfile.TemporaryDirectory(prefix="qwen3-vl-quality-") as temp_dir:
            runtime_json = Path(temp_dir) / "runtime.json"
            command = [
                str(runtime),
                "--engine-dir",
                str(llm_dir),
                "--multimodal-engine-dir",
                str(multimodal_dir),
                "--plugin",
                str(plugin),
                "--image",
                str(image),
                "--prompt",
                case["prompt"],
                "--max-new-tokens",
                str(int(case.get("max_new_tokens", max_tokens))),
                "--json-output",
                str(runtime_json),
            ]
            completed = subprocess.run(command, env=os.environ.copy(), capture_output=True, text=True)
            if completed.returncode != 0 or not runtime_json.exists():
                results.append(
                    {
                        "id": case.get("id", image.name),
                        "passed": False,
                        "error": completed.stderr[-2000:],
                    }
                )
                continue
            run = json.loads(runtime_json.read_text(encoding="utf-8"))["runs"][0]

        candidate = str(run.get("text", ""))
        reference = str(case.get("reference_text", ""))
        keywords = [str(value) for value in case.get("required_keywords", [])]
        has_reference = bool(reference)
        has_keywords = bool(keywords)
        exact = candidate == reference if has_reference else None
        similarity = difflib.SequenceMatcher(None, reference, candidate).ratio() if has_reference else None
        keyword_pass = all(keyword in candidate for keyword in keywords) if has_keywords else None
        scored = has_reference or has_keywords
        passed = (exact if has_reference else True) and (keyword_pass if has_keywords else True)
        results.append(
            {
                "id": case.get("id", image.name),
                "candidate_text": candidate,
                "reference_text": reference,
                "required_keywords": keywords,
                "scored": scored,
                "exact_match": exact,
                "text_similarity": similarity,
                "keyword_pass": keyword_pass,
                "passed": passed if scored else None,
            }
        )

    scored = [result for result in results if result.get("scored")]
    referenced = [result for result in scored if result.get("text_similarity") is not None]
    summary = {
        "cases": len(results),
        "scored_cases": len(scored),
        "status": "measured" if scored else "reference_required",
        "pass_rate": sum(bool(result.get("passed")) for result in scored) / len(scored) if scored else None,
        "exact_match_rate": sum(bool(result.get("exact_match")) for result in referenced) / len(referenced)
        if referenced
        else None,
        "mean_text_similarity": sum(float(result["text_similarity"]) for result in referenced) / len(referenced)
        if referenced
        else None,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 精度/任务质量回归",
        "",
        "| Case | 是否评分 | 通过 | Exact | 文本相似度 | 关键词通过 |",
        "|---|---|---|---|---:|---|",
    ]
    for result in results:
        similarity = result.get("text_similarity")
        similarity_text = f"{float(similarity):.4f}" if similarity is not None else "N/A"
        lines.append(
            f"| {result['id']} | {result.get('scored', False)} | {result.get('passed', 'N/A')} | "
            f"{result.get('exact_match', 'N/A')} | {similarity_text} | {result.get('keyword_pass', 'N/A')} |"
        )
    if not scored:
        lines.extend(
            [
                "",
                "当前 case 没有填写 `reference_text` 或 `required_keywords`，因此只生成候选输出，不声称质量通过。",
            ]
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
