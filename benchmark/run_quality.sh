#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/common.sh"
parse_common_args "$@"

FORCE=0
for argument in "${REMAINING_ARGS[@]}"; do
  case "${argument}" in
    --force) FORCE=1 ;;
    *)
      echo "未知参数：${argument}" >&2
      exit 2
      ;;
  esac
done

ENABLED="$(config_get quality.enabled)"
if [[ "${ENABLED}" != "true" && "${FORCE}" != "1" ]]; then
  echo "质量回归默认关闭；填好 reference_text/required_keywords 后设置 quality.enabled=true，或使用 --force。"
  exit 0
fi

prepare_runtime_env
CASES="$(config_get_path quality.cases_file)"
require_file "${CASES}"
OUTPUT="${BENCH_OUTPUT}/quality"
mkdir -p "${OUTPUT}"

python3 "${BENCHMARK_DIR}/scripts/evaluate_quality.py" \
  --config "${BENCH_CONFIG}" \
  --output-json "${OUTPUT}/summary.json" \
  --output-md "${OUTPUT}/summary.md"

echo "精度/任务质量结果已写入：${OUTPUT}/summary.json"
