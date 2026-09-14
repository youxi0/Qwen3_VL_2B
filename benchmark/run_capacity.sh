#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/common.sh"
parse_common_args "$@"
if ((${#REMAINING_ARGS[@]})); then
  echo "未知参数：${REMAINING_ARGS[*]}" >&2
  exit 2
fi

prepare_runtime_env

LLM_BENCH="$(config_get_path paths.llm_bench)"
LLM_DIR="$(config_get_path paths.llm_engine_dir)"
ENGINE_CONFIG="${LLM_DIR}/config.json"
ITERATIONS="$(config_get capacity.iterations)"
WARMUP="$(config_get capacity.warmup)"

if [[ "${BENCH_QUICK}" == "1" ]]; then
  ITERATIONS=1
  WARMUP=1
fi

require_file "${LLM_BENCH}"
require_file "${ENGINE_CONFIG}"

readarray -t LIMITS < <(python3 - "${ENGINE_CONFIG}" <<'PY'
import json
import sys
config = json.load(open(sys.argv[1], encoding="utf-8"))["builder_config"]
print(config["max_input_len"])
print(config["max_kv_cache_capacity"])
PY
)
MAX_INPUT="${LIMITS[0]}"
MAX_KV="${LIMITS[1]}"
PAST_KV=$((MAX_KV - 1))

OUTPUT="${BENCH_OUTPUT}/capacity"
PREFILL_DIR="${OUTPUT}/prefill_max"
DECODE_DIR="${OUTPUT}/decode_max"
mkdir -p "${PREFILL_DIR}" "${DECODE_DIR}"

set +e
"${LLM_BENCH}" --engineDir "${LLM_DIR}" --mode prefill --inputLen "${MAX_INPUT}" --batchSize 1 \
  --iterations "${ITERATIONS}" --warmup "${WARMUP}" --outputDir "${PREFILL_DIR}" \
  >"${PREFILL_DIR}/llm_bench.log" 2>&1
PREFILL_EXIT=$?

"${LLM_BENCH}" --engineDir "${LLM_DIR}" --mode decode --pastKVLen "${PAST_KV}" --batchSize 1 \
  --iterations "${ITERATIONS}" --warmup "${WARMUP}" --outputDir "${DECODE_DIR}" \
  >"${DECODE_DIR}/llm_bench.log" 2>&1
DECODE_EXIT=$?
set -e

python3 "${BENCHMARK_DIR}/scripts/summarize_capacity.py" \
  --engine-config "${ENGINE_CONFIG}" \
  --prefill-dir "${PREFILL_DIR}" \
  --decode-dir "${DECODE_DIR}" \
  --prefill-exit "${PREFILL_EXIT}" \
  --decode-exit "${DECODE_EXIT}" \
  --output-json "${OUTPUT}/summary.json" \
  --output-md "${OUTPUT}/summary.md"

echo "Context/KV Capacity 结果已写入：${OUTPUT}/summary.json"
