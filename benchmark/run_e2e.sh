#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/common.sh"
parse_common_args "$@"
if ((${#REMAINING_ARGS[@]})); then
  echo "未知参数：${REMAINING_ARGS[*]}" >&2
  exit 2
fi

prepare_runtime_env

RUNTIME_CLI="$(config_get_path paths.runtime_cli)"
LLM_DIR="$(config_get_path paths.llm_engine_dir)"
MULTIMODAL_DIR="$(config_get_path paths.multimodal_engine_dir)"
PLUGIN="$(config_get_path paths.plugin)"
IMAGE="$(config_get_path request.image)"
PROMPT="$(config_get request.prompt)"
MAX_NEW_TOKENS="$(config_get request.max_new_tokens)"
STEADY_WARMUP="$(config_get e2e.steady_warmup)"
STEADY_RUNS="$(config_get e2e.steady_runs)"
SAMPLE_INTERVAL="$(config_get e2e.memory_sample_interval_ms)"

if [[ "${BENCH_QUICK}" == "1" ]]; then
  MAX_NEW_TOKENS=8
  STEADY_WARMUP=1
  STEADY_RUNS=2
fi

require_file "${RUNTIME_CLI}"
require_dir "${LLM_DIR}"
require_dir "${MULTIMODAL_DIR}"
require_file "${PLUGIN}"
require_file "${IMAGE}"

OUTPUT="${BENCH_OUTPUT}/e2e"
mkdir -p "${OUTPUT}"

COMMON_ARGS=(
  "${RUNTIME_CLI}"
  --engine-dir "${LLM_DIR}"
  --multimodal-engine-dir "${MULTIMODAL_DIR}"
  --plugin "${PLUGIN}"
  --image "${IMAGE}"
  --prompt "${PROMPT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)

python3 "${BENCHMARK_DIR}/scripts/monitor_process.py" \
  --metrics "${OUTPUT}/cold_memory.json" \
  --stdout "${OUTPUT}/cold_stdout.log" \
  --stderr "${OUTPUT}/cold_stderr.log" \
  --tegrastats "${OUTPUT}/cold_tegrastats.log" \
  --interval-ms "${SAMPLE_INTERVAL}" \
  -- "${COMMON_ARGS[@]}" --warmup 0 --repeat 1 --json-output "${OUTPUT}/cold_runtime.json"

python3 "${BENCHMARK_DIR}/scripts/monitor_process.py" \
  --metrics "${OUTPUT}/steady_memory.json" \
  --stdout "${OUTPUT}/steady_stdout.log" \
  --stderr "${OUTPUT}/steady_stderr.log" \
  --tegrastats "${OUTPUT}/steady_tegrastats.log" \
  --interval-ms "${SAMPLE_INTERVAL}" \
  -- "${COMMON_ARGS[@]}" --warmup "${STEADY_WARMUP}" --repeat "${STEADY_RUNS}" \
  --json-output "${OUTPUT}/steady_runtime.json"

python3 "${BENCHMARK_DIR}/scripts/summarize_e2e.py" \
  --cold-runtime "${OUTPUT}/cold_runtime.json" \
  --cold-memory "${OUTPUT}/cold_memory.json" \
  --steady-runtime "${OUTPUT}/steady_runtime.json" \
  --steady-memory "${OUTPUT}/steady_memory.json" \
  --output-json "${OUTPUT}/summary.json" \
  --output-md "${OUTPUT}/summary.md"

echo "端到端指标已写入：${OUTPUT}/summary.json"
