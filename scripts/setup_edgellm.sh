#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDGELLM_ROOT="${PROJECT_ROOT}/third_party/TensorRT-Edge-LLM"

git -C "${PROJECT_ROOT}" submodule sync --recursive third_party/TensorRT-Edge-LLM
git -C "${PROJECT_ROOT}" submodule update \
  --init \
  --recursive \
  --checkout \
  third_party/TensorRT-Edge-LLM

current_revision="$(git -C "${EDGELLM_ROOT}" rev-parse HEAD)"
echo "TensorRT-Edge-LLM 已准备完成：${EDGELLM_ROOT} (${current_revision})"
