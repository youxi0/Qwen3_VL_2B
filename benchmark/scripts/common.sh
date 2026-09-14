#!/usr/bin/env bash

set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${BENCHMARK_DIR}/.." && pwd)"
DEFAULT_CONFIG="${BENCHMARK_DIR}/config.json"
CONFIG_VALUE="${BENCHMARK_DIR}/scripts/config_value.py"

config_get() {
  python3 "${CONFIG_VALUE}" "${BENCH_CONFIG}" "$1"
}

config_get_json() {
  python3 "${CONFIG_VALUE}" "${BENCH_CONFIG}" "$1" --json
}

config_get_path() {
  python3 "${CONFIG_VALUE}" "${BENCH_CONFIG}" "$1" --path
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "缺少文件：$1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "缺少目录：$1" >&2
    exit 1
  fi
}

prepare_runtime_env() {
  local cuda_lib="/usr/local/cuda/targets/sbsa-linux/lib"
  export LD_LIBRARY_PATH="/usr/lib:/usr/lib/aarch64-linux-gnu:${cuda_lib}:${LD_LIBRARY_PATH:-}"
  export EDGELLM_PLUGIN_PATH
  EDGELLM_PLUGIN_PATH="$(config_get_path paths.plugin)"
}

default_run_dir() {
  date "+${PROJECT_ROOT}/output/benchmark-%Y%m%d-%H%M%S"
}

parse_common_args() {
  BENCH_CONFIG="${DEFAULT_CONFIG}"
  BENCH_OUTPUT=""
  BENCH_QUICK="${BENCH_QUICK:-0}"
  REMAINING_ARGS=()

  while (($#)); do
    case "$1" in
      --config)
        BENCH_CONFIG="$2"
        shift 2
        ;;
      --output-dir)
        BENCH_OUTPUT="$2"
        shift 2
        ;;
      --quick)
        BENCH_QUICK=1
        shift
        ;;
      *)
        REMAINING_ARGS+=("$1")
        shift
        ;;
    esac
  done

  BENCH_CONFIG="$(realpath "${BENCH_CONFIG}")"
  require_file "${BENCH_CONFIG}"
  if [[ -z "${BENCH_OUTPUT}" ]]; then
    BENCH_OUTPUT="$(default_run_dir)"
  elif [[ "${BENCH_OUTPUT}" != /* ]]; then
    BENCH_OUTPUT="${PROJECT_ROOT}/${BENCH_OUTPUT}"
  fi
  mkdir -p "${BENCH_OUTPUT}"
  export BENCH_CONFIG BENCH_OUTPUT BENCH_QUICK PROJECT_ROOT BENCHMARK_DIR
}
