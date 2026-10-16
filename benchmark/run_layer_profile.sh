#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/common.sh"
parse_common_args "$@"

STAGES="prefill,decode,visual"
index=0
while ((index < ${#REMAINING_ARGS[@]})); do
  case "${REMAINING_ARGS[index]}" in
    --stages)
      STAGES="${REMAINING_ARGS[index + 1]}"
      index=$((index + 2))
      ;;
    *)
      echo "未知参数：${REMAINING_ARGS[index]}" >&2
      exit 2
      ;;
  esac
done

prepare_runtime_env

LLM_BENCH="$(config_get_path paths.llm_bench)"
TRTEXEC="$(config_get_path paths.trtexec)"
LLM_DIR="$(config_get_path paths.llm_engine_dir)"
MULTIMODAL_DIR="$(config_get_path paths.multimodal_engine_dir)"
VISUAL_DIR="${MULTIMODAL_DIR}/visual"
PLUGIN="$(config_get_path paths.plugin)"
ITERATIONS="$(config_get layer_profile.iterations)"
WARMUP="$(config_get layer_profile.warmup)"
BATCH_SIZE="$(config_get layer_profile.batch_size)"
PREFILL_LEN="$(config_get layer_profile.prefill_input_len)"
PAST_KV_LEN="$(config_get layer_profile.decode_past_kv_len)"
IMAGE_HEIGHT="$(config_get layer_profile.visual_height)"
IMAGE_WIDTH="$(config_get layer_profile.visual_width)"

if [[ "${BENCH_QUICK}" == "1" ]]; then
  ITERATIONS=1
  WARMUP=1
fi

require_file "${LLM_BENCH}"
require_file "${TRTEXEC}"
require_file "${PLUGIN}"
require_dir "${LLM_DIR}"
require_dir "${VISUAL_DIR}"

OUTPUT="${BENCH_OUTPUT}/layers"
RAW="${OUTPUT}/raw"
INSPECTOR="${OUTPUT}/inspector"
mkdir -p "${RAW}" "${INSPECTOR}"

contains_stage() {
  [[ ",${STAGES}," == *",$1,"* ]]
}

run_llm_stage() {
  local stage="$1"
  local raw_dir="${RAW}/${stage}"
  mkdir -p "${raw_dir}"
  local mode_args=()
  case "${stage}" in
    prefill)
      mode_args=(--engineDir "${LLM_DIR}" --mode prefill --inputLen "${PREFILL_LEN}")
      ;;
    decode)
      mode_args=(--engineDir "${LLM_DIR}" --mode decode --pastKVLen "${PAST_KV_LEN}")
      ;;
    visual)
      mode_args=(--engineDir "${VISUAL_DIR}" --mode visual --imageSize "${IMAGE_HEIGHT}x${IMAGE_WIDTH}")
      ;;
    *)
      echo "不支持的 stage：${stage}" >&2
      exit 2
      ;;
  esac

  "${LLM_BENCH}" "${mode_args[@]}" --batchSize "${BATCH_SIZE}" --iterations "${ITERATIONS}" \
    --warmup "${WARMUP}" --profile --outputDir "${raw_dir}" \
    >"${raw_dir}/llm_bench.log" 2>&1
}

if contains_stage prefill || contains_stage decode; then
  "${TRTEXEC}" --loadEngine="${LLM_DIR}/llm.engine" --staticPlugins="${PLUGIN}" --skipInference \
    --profilingVerbosity=detailed --exportLayerInfo="${INSPECTOR}/llm.json" \
    >"${INSPECTOR}/llm_trtexec.log" 2>&1
fi

if contains_stage visual; then
  "${TRTEXEC}" --loadEngine="${VISUAL_DIR}/visual.engine" --staticPlugins="${PLUGIN}" --skipInference \
    --profilingVerbosity=detailed --exportLayerInfo="${INSPECTOR}/visual.json" \
    >"${INSPECTOR}/visual_trtexec.log" 2>&1
fi

IFS=',' read -r -a stage_list <<<"${STAGES}"
for stage in "${stage_list[@]}"; do
  run_llm_stage "${stage}"
  case "${stage}" in
    prefill)
      timing_csv="${RAW}/${stage}/layer_prefill_inputlen${PREFILL_LEN}.csv"
      ;;
    decode)
      timing_csv="${RAW}/${stage}/layer_decode_pastkvlen${PAST_KV_LEN}.csv"
      ;;
    visual)
      timing_csv="${RAW}/${stage}/layer_visual_${IMAGE_HEIGHT}x${IMAGE_WIDTH}.csv"
      ;;
  esac
  require_file "${timing_csv}"
  inspector_json="${INSPECTOR}/llm.json"
  if [[ "${stage}" == "visual" ]]; then
    inspector_json="${INSPECTOR}/visual.json"
  fi
  python3 "${BENCHMARK_DIR}/scripts/merge_layer_profile.py" \
    --timing-csv "${timing_csv}" \
    --inspector-json "${inspector_json}" \
    --stage "${stage}" \
    --output-csv "${OUTPUT}/layers_${stage}_sorted.csv" \
    --output-md "${OUTPUT}/layers_${stage}_sorted.md" \
    --output-json "${OUTPUT}/layers_${stage}_summary.json"
done

echo "按耗时降序的 Layer CSV 已写入：${OUTPUT}/layers_<stage>_sorted.csv"
