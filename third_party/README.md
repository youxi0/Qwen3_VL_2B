# Third-party dependencies

`TensorRT-Edge-LLM` 固定在官方 `v0.10.1` 基线，并通过
`patches/tensorrt-edgellm-qwen3-vl.patch` 叠加本项目的 INT4 GEMV 和
SwiGLU 融合修改。

初始化并应用补丁：

```bash
./scripts/setup_edgellm.sh
```

生成缺失的 SM87 CuTe DSL artifact 并构建插件、Core runtime 和 benchmark：

```bash
CUTEDSL_PYTHON=/path/to/cutedsl/python ./scripts/build_edgellm.sh
```

已经存在 artifact 时不需要设置 `CUTEDSL_PYTHON`。构建输出位于
`third_party/TensorRT-Edge-LLM/build-v0101`。脚本默认按照 Jetson Orin、
CUDA 13.2 配置；其他 CUDA 版本可通过 `CUDA_CTK_VERSION` 覆盖。
