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

NSYS="$(config_get_path paths.nsys)"
LLM_BENCH="$(config_get_path paths.llm_bench)"
LLM_DIR="$(config_get_path paths.llm_engine_dir)"
MULTIMODAL_DIR="$(config_get_path paths.multimodal_engine_dir)"
VISUAL_DIR="${MULTIMODAL_DIR}/visual"
ITERATIONS="$(config_get nsys.iterations)"
WARMUP="$(config_get nsys.warmup)"
PREFILL_LEN="$(config_get layer_profile.prefill_input_len)"
PAST_KV_LEN="$(config_get layer_profile.decode_past_kv_len)"
IMAGE_HEIGHT="$(config_get layer_profile.visual_height)"
IMAGE_WIDTH="$(config_get layer_profile.visual_width)"

if [[ "${BENCH_QUICK}" == "1" ]]; then
  ITERATIONS=1
  WARMUP=1
fi

require_file "${NSYS}"
require_file "${LLM_BENCH}"
require_dir "${LLM_DIR}"
require_dir "${VISUAL_DIR}"

OUTPUT="${BENCH_OUTPUT}/nsys"
mkdir -p "${OUTPUT}"

run_stage() {
  local stage="$1"
  local stage_dir="${OUTPUT}/${stage}"
  local base="${stage_dir}/${stage}"
  local target_args=()
  mkdir -p "${stage_dir}/raw"

  case "${stage}" in
    prefill)
      target_args=(--engineDir "${LLM_DIR}" --mode prefill --inputLen "${PREFILL_LEN}")
      ;;
    decode)
      target_args=(--engineDir "${LLM_DIR}" --mode decode --pastKVLen "${PAST_KV_LEN}")
      ;;
    visual)
      target_args=(--engineDir "${VISUAL_DIR}" --mode visual --imageSize "${IMAGE_HEIGHT}x${IMAGE_WIDTH}")
      ;;
    *)
      echo "不支持的 Nsight Systems stage：${stage}" >&2
      exit 2
      ;;
  esac

  # Jetson 4 GiB 上将三个 engine 阶段拆成独立进程，避免注入 NSYS 后同时驻留导致 OOM。
  # node 粒度保留 kernel 时间线；具体硬件计数器仍交给独立的 NCU 接口。
  "${NSYS}" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node --force-overwrite=true --export=sqlite --output="${base}" \
    "${LLM_BENCH}" "${target_args[@]}" \
    --batchSize 1 --iterations "${ITERATIONS}" --warmup "${WARMUP}" \
    --outputDir "${stage_dir}/raw" \
    >"${stage_dir}/nsys_stdout.log" 2>"${stage_dir}/nsys_stderr.log"

  require_file "${base}.nsys-rep"
  require_file "${base}.sqlite"
  python3 "${BENCHMARK_DIR}/scripts/summarize_nsys.py" \
    --sqlite "${base}.sqlite" \
    --output-json "${stage_dir}/summary.json" \
    --output-md "${stage_dir}/summary.md"
}

IFS=',' read -r -a stage_list <<<"${STAGES}"
for stage in "${stage_list[@]}"; do
  run_stage "${stage}"
done

python3 "${BENCHMARK_DIR}/scripts/summarize_nsys_suite.py" \
  --input-dir "${OUTPUT}" \
  --stages "${STAGES}" \
  --output-json "${OUTPUT}/summary.json" \
  --output-md "${OUTPUT}/summary.md"

echo "Nsight Systems 分阶段报告已写入：${OUTPUT}/<stage>/<stage>.nsys-rep"
