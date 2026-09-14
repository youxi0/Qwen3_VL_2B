#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/common.sh"
parse_common_args "$@"

SKIP_NSYS=0
WITH_QUALITY=0
for argument in "${REMAINING_ARGS[@]}"; do
  case "${argument}" in
    --skip-nsys) SKIP_NSYS=1 ;;
    --with-quality) WITH_QUALITY=1 ;;
    *)
      echo "未知参数：${argument}" >&2
      exit 2
      ;;
  esac
done

CHILD_ARGS=(--config "${BENCH_CONFIG}" --output-dir "${BENCH_OUTPUT}")
if [[ "${BENCH_QUICK}" == "1" ]]; then
  CHILD_ARGS+=(--quick)
fi

{
  date --iso-8601=seconds
  uname -a
  nvpmodel -q 2>&1 || true
  jetson_clocks --show 2>&1 || true
  "$(config_get_path paths.nsys)" --version 2>&1 || true
  "$(config_get_path paths.ncu)" --version 2>&1 || true
  trtexec --version 2>&1 || true
} >"${BENCH_OUTPUT}/environment.txt"

"${BENCHMARK_DIR}/run_e2e.sh" "${CHILD_ARGS[@]}"
"${BENCHMARK_DIR}/run_layer_profile.sh" "${CHILD_ARGS[@]}"
"${BENCHMARK_DIR}/run_capacity.sh" "${CHILD_ARGS[@]}"
if [[ "${SKIP_NSYS}" != "1" ]]; then
  "${BENCHMARK_DIR}/run_nsys.sh" "${CHILD_ARGS[@]}"
fi
if [[ "${WITH_QUALITY}" == "1" ]]; then
  "${BENCHMARK_DIR}/run_quality.sh" "${CHILD_ARGS[@]}" --force
fi

python3 "${BENCHMARK_DIR}/scripts/generate_report.py" --run-dir "${BENCH_OUTPUT}"
echo "完整报告已写入：${BENCH_OUTPUT}/report.md"
