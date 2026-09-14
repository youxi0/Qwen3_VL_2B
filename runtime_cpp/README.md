# Qwen3-VL project runtime

This directory contains the project-owned C++ runtime. It keeps the TensorRT
engines, embedding table, tokenizer, CUDA stream, KV-cache resources, and decode
CUDA Graph alive across requests while delegating model-specific execution to
TensorRT Edge-LLM's public `LLMInferenceRuntime` API.

## Build

```bash
cd /home/jetson/Qwen3_VL_2B
cmake -S runtime_cpp -B runtime_cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DEDGELLM_SOURCE_DIR=/home/jetson/TensorRT-Edge-LLM \
  -DEDGELLM_BUILD_DIR=/home/jetson/TensorRT-Edge-LLM/build-v0101
cmake --build runtime_cpp/build -j2
```

## Run

```bash
export TRT_PACKAGE_DIR=/usr
export LD_LIBRARY_PATH="$TRT_PACKAGE_DIR/lib:$TRT_PACKAGE_DIR/lib/aarch64-linux-gnu:/usr/local/cuda/targets/sbsa-linux/lib:${LD_LIBRARY_PATH}"

runtime_cpp/build/qwen3_vl_cli \
  --engine-dir models/engines-jetson-20260911-profiled/llm \
  --multimodal-engine-dir models/engines-jetson-20260911-profiled \
  --plugin /home/jetson/TensorRT-Edge-LLM/build-v0101/libNvInfer_edgellm_plugin.so.1.0 \
  --image dataset/0.jpg \
  --prompt '请简短描述这张图片。' \
  --max-new-tokens 32
```

Use `--repeat N` to verify that one initialized runtime serves multiple requests.
`--warmup N` runs full requests before the first measured request.

## API

Applications include `qwen3VlRuntime.h`, construct a `RuntimeConfig` once, and
call `generate()` for each request. Calls on one runtime instance are serialized;
create separate instances only when the additional engine and KV-cache memory is
acceptable.
