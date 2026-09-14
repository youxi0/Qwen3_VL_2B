#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/common.sh"
parse_common_args "$@"

STAGE="$(config_get ncu.default_stage)"
KERNEL_REGEX=""
LAYER_NVTX=""
LAUNCH_COUNT="$(config_get ncu.launch_count)"
index=0
while ((index < ${#REMAINING_ARGS[@]})); do
  case "${REMAINING_ARGS[index]}" in
    --stage)
      STAGE="${REMAINING_ARGS[index + 1]}"
      index=$((index + 2))
      ;;
    --kernel-regex)
      KERNEL_REGEX="${REMAINING_ARGS[index + 1]}"
      index=$((index + 2))
      ;;
    --layer-nvtx)
      LAYER_NVTX="${REMAINING_ARGS[index + 1]}"
      index=$((index + 2))
      ;;
    --launch-count)
      LAUNCH_COUNT="${REMAINING_ARGS[index + 1]}"
      index=$((index + 2))
      ;;
    *)
      echo "未知参数：${REMAINING_ARGS[index]}" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${KERNEL_REGEX}" && -z "${LAYER_NVTX}" ]]; then
  echo "NCU 必须显式指定 --kernel-regex REGEX 或 --layer-nvtx RANGE。" >&2
  exit 2
fi
if [[ -n "${KERNEL_REGEX}" && -n "${LAYER_NVTX}" ]]; then
  echo "--kernel-regex 与 --layer-nvtx 只能选择一个。" >&2
  exit 2
fi

prepare_runtime_env
NCU="$(config_get_path paths.ncu)"
LLM_BENCH="$(config_get_path paths.llm_bench)"
LLM_DIR="$(config_get_path paths.llm_engine_dir)"
MULTIMODAL_DIR="$(config_get_path paths.multimodal_engine_dir)"
VISUAL_DIR="${MULTIMODAL_DIR}/visual"
PREFILL_LEN="$(config_get layer_profile.prefill_input_len)"
PAST_KV_LEN="$(config_get layer_profile.decode_past_kv_len)"
IMAGE_HEIGHT="$(config_get layer_profile.visual_height)"
IMAGE_WIDTH="$(config_get layer_profile.visual_width)"

require_file "${NCU}"
require_file "${LLM_BENCH}"

OUTPUT="${BENCH_OUTPUT}/ncu"
mkdir -p "${OUTPUT}"
set +e
"${NCU}" --query-metrics >"${OUTPUT}/metric_query.log" 2>&1
QUERY_RC=$?
set -e
if ((QUERY_RC != 0)) || grep -q "==ERROR==" "${OUTPUT}/metric_query.log"; then
  echo "NCU 无法访问 GPU performance counters，详情见 ${OUTPUT}/metric_query.log" >&2
  echo "Jetson 上通常需要管理员启用计数器权限后再运行这个独立接口。" >&2
  exit 1
fi

readarray -t METRICS < <(python3 - "${BENCH_CONFIG}" <<'PY'
import json
import sys
for metric in json.load(open(sys.argv[1], encoding="utf-8"))["ncu"]["metrics"]:
    print(metric)
PY
)
METRIC_CSV="$(IFS=,; echo "${METRICS[*]}")"

case "${STAGE}" in
  prefill)
    TARGET_ARGS=(--engineDir "${LLM_DIR}" --mode prefill --inputLen "${PREFILL_LEN}")
    ;;
  decode)
    TARGET_ARGS=(--engineDir "${LLM_DIR}" --mode decode --pastKVLen "${PAST_KV_LEN}" --noCudaGraph)
    ;;
  visual)
    TARGET_ARGS=(--engineDir "${VISUAL_DIR}" --mode visual --imageSize "${IMAGE_HEIGHT}x${IMAGE_WIDTH}")
    ;;
  *)
    echo "不支持的 NCU stage：${STAGE}" >&2
    exit 2
    ;;
esac

FILTER_ARGS=()
SELECTOR=""
if [[ -n "${KERNEL_REGEX}" ]]; then
  FILTER_ARGS=(--kernel-name "regex:${KERNEL_REGEX}" --kernel-name-base demangled)
  SELECTOR="kernel:${KERNEL_REGEX}"
else
  FILTER_ARGS=(--nvtx --nvtx-include "${LAYER_NVTX}")
  SELECTOR="nvtx:${LAYER_NVTX}"
fi

SAFE_NAME="$(printf '%s' "${STAGE}_${SELECTOR}" | tr -cs 'A-Za-z0-9._-' '_')"
BASE="${OUTPUT}/${SAFE_NAME}"
"${NCU}" --target-processes all --replay-mode kernel --force-overwrite --export "${BASE}" \
  --metrics "${METRIC_CSV}" --launch-count "${LAUNCH_COUNT}" "${FILTER_ARGS[@]}" \
  "${LLM_BENCH}" "${TARGET_ARGS[@]}" --batchSize 1 --iterations 1 --warmup 1 \
  >"${BASE}.stdout.log" 2>"${BASE}.stderr.log"

REPORT="${BASE}.ncu-rep"
RAW_CSV="${BASE}.raw.csv"
require_file "${REPORT}"
"${NCU}" --import "${REPORT}" --csv --page raw --print-metric-name name --print-units base \
  --log-file "${RAW_CSV}"

python3 "${BENCHMARK_DIR}/scripts/summarize_ncu.py" \
  --raw-csv "${RAW_CSV}" \
  --report "${REPORT}" \
  --selector "${SELECTOR}" \
  --output-json "${BASE}.summary.json" \
  --output-md "${BASE}.summary.md"

echo "NCU 精简结果已写入：${BASE}.summary.json"
