#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDGELLM_ROOT="${PROJECT_ROOT}/third_party/TensorRT-Edge-LLM"
BUILD_DIR="${EDGELLM_ROOT}/build-v0101"
ARTIFACT_DIR="${EDGELLM_ROOT}/cpp/kernels/cuteDSLArtifact/aarch64/sm_87"
ARTIFACT_LIBRARY="${ARTIFACT_DIR}/libcutedsl_aarch64.a"
ARTIFACT_HEADER="${ARTIFACT_DIR}/include/cutedsl_all.h"
TRT_ROOT="${TRT_PACKAGE_DIR:-/usr}"
BUILD_JOBS="${BUILD_JOBS:-2}"
CUDA_TOOLKIT_VERSION="${CUDA_CTK_VERSION:-13.2}"

"${PROJECT_ROOT}/scripts/setup_edgellm.sh"

if [[ ! -f "${ARTIFACT_LIBRARY}" || ! -f "${ARTIFACT_HEADER}" ]]; then
  CUTEDSL_PYTHON="${CUTEDSL_PYTHON:-${EDGELLM_ROOT}/.venv-cutedsl-cu13/bin/python3}"
  if [[ ! -x "${CUTEDSL_PYTHON}" ]]; then
    echo "缺少 CuTe DSL Python 环境。请通过 CUTEDSL_PYTHON 指定解释器。" >&2
    exit 1
  fi
  "${CUTEDSL_PYTHON}" "${EDGELLM_ROOT}/kernelSrcs/build_cutedsl.py" \
    --kernels int4_fp16_gemm \
    --gpu_arch sm_87 \
    --arch aarch64 \
    -j "${CUTEDSL_JOBS:-1}"
fi

cmake -S "${EDGELLM_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${EDGELLM_ROOT}/cmake/aarch64_linux_toolchain.cmake" \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUDA_CTK_VERSION="${CUDA_TOOLKIT_VERSION}" \
  -DTRT_PACKAGE_DIR="${TRT_ROOT}" \
  -DENABLE_CUTE_DSL=ALL
cmake --build "${BUILD_DIR}" \
  --target NvInfer_edgellm_plugin edgellmCore llm_bench \
  -j "${BUILD_JOBS}"

echo "Edge-LLM 构建完成：${BUILD_DIR}"
