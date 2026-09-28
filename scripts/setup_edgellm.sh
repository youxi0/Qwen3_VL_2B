#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDGELLM_ROOT="${PROJECT_ROOT}/third_party/TensorRT-Edge-LLM"
PATCH_FILE="${PROJECT_ROOT}/third_party/patches/tensorrt-edgellm-qwen3-vl.patch"
EXPECTED_REVISION="e8b29522938901f6df19ebeedd4b69bc8edbcd97"

if [[ ! -d "${EDGELLM_ROOT}/.git" && ! -f "${EDGELLM_ROOT}/.git" ]]; then
  git -C "${PROJECT_ROOT}" submodule update --init --recursive third_party/TensorRT-Edge-LLM
fi

git -C "${EDGELLM_ROOT}" submodule update --init --recursive

current_revision="$(git -C "${EDGELLM_ROOT}" rev-parse HEAD)"
if [[ "${current_revision}" != "${EXPECTED_REVISION}" ]]; then
  echo "TensorRT-Edge-LLM 版本不匹配：${current_revision}" >&2
  echo "期望版本：${EXPECTED_REVISION}" >&2
  exit 1
fi

if git -C "${EDGELLM_ROOT}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "Qwen3-VL Edge-LLM 补丁已经应用。"
elif git -C "${EDGELLM_ROOT}" apply --check "${PATCH_FILE}"; then
  git -C "${EDGELLM_ROOT}" apply "${PATCH_FILE}"
  echo "已应用 Qwen3-VL Edge-LLM 补丁。"
else
  echo "Edge-LLM 工作区与项目补丁不兼容，请先检查本地修改。" >&2
  exit 1
fi

echo "TensorRT-Edge-LLM 已准备完成：${EDGELLM_ROOT}"
