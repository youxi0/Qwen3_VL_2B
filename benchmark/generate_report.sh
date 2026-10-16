#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/common.sh"
parse_common_args "$@"
if ((${#REMAINING_ARGS[@]})); then
  echo "未知参数：${REMAINING_ARGS[*]}" >&2
  exit 2
fi
python3 "${BENCHMARK_DIR}/scripts/generate_report.py" --run-dir "${BENCH_OUTPUT}"
echo "报告已重新生成：${BENCH_OUTPUT}/report.md"
