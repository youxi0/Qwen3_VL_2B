#!/usr/bin/env python3
"""汇总 engine 上限处的 prefill 与 KV 稳定性检查。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_first_csv(directory: Path) -> dict[str, Any]:
    files = sorted(directory.glob("e2e_*.csv"))
    if not files:
        return {}
    with files[0].open("r", encoding="utf-8", newline="") as source:
        return next(csv.DictReader(source), {})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-config", required=True, type=Path)
    parser.add_argument("--prefill-dir", required=True, type=Path)
    parser.add_argument("--decode-dir", required=True, type=Path)
    parser.add_argument("--prefill-exit", required=True, type=int)
    parser.add_argument("--decode-exit", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    engine = json.loads(args.engine_config.read_text(encoding="utf-8"))
    builder = engine["builder_config"]
    max_input = int(builder["max_input_len"])
    max_kv = int(builder["max_kv_cache_capacity"])
    prefill = read_first_csv(args.prefill_dir)
    decode = read_first_csv(args.decode_dir)

    result = {
        "configured_max_input_len": max_input,
        "configured_max_kv_cache_capacity": max_kv,
        "prefill_test": {
            "input_len": max_input,
            "passed": args.prefill_exit == 0,
            "exit_code": args.prefill_exit,
            "e2e_time_ms": float(prefill.get("e2e_time_ms", 0.0) or 0.0),
        },
        "decode_test": {
            "past_kv_len": max(max_kv - 1, 0),
            "resulting_kv_capacity": max_kv,
            "passed": args.decode_exit == 0,
            "exit_code": args.decode_exit,
            "per_token_ms": float(decode.get("per_token_ms", 0.0) or 0.0),
        },
        "max_stable_context_tokens": max_input if args.prefill_exit == 0 else None,
        "max_stable_kv_capacity": max_kv if args.decode_exit == 0 else None,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Context/KV Capacity 检查",
        "",
        "| 项目 | 配置上限 | 实测位置 | 结果 |",
        "|---|---:|---:|---|",
        f"| Prefill context | {max_input} | input_len={max_input} | {'通过' if args.prefill_exit == 0 else '失败'} |",
        f"| KV capacity | {max_kv} | past_kv_len={max(max_kv - 1, 0)}，执行下一 token | "
        f"{'通过' if args.decode_exit == 0 else '失败'} |",
        "",
        "这里验证的是当前 engine contract 的最大 shape，不代表模型训练时的长期上下文任务质量。",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if args.prefill_exit == 0 and args.decode_exit == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
