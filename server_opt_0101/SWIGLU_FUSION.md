# INT4 SwiGLU 融合

本阶段把 28 个 decoder block 中的

```text
gate INT4 GEMM -> Sigmoid -> Mul --+
                                      Mul -> down_proj
up INT4 GEMM -----------------------+
```

改写为一个 `trt_edgellm::Int4GroupwiseSwiGluPlugin` 节点。原图每层删除
5 个节点并插入 1 个融合节点，ONNX 总节点数从 1131 降为 1019。

## 实现范围

- 图改写脚本：`server_opt_0101/fuse_int4_swiglu_onnx.py`
- 融合 ONNX：`models/jetson_onnx_swiglu`
- 融合 Engine：`models/engines-jetson-swiglu-20260916`
- TensorRT 插件源码：
  `third_party/TensorRT-Edge-LLM/cpp/plugins/int4GroupwiseSwiGluPlugin`
- TensorRT 生命周期与调度：`int4GroupwiseSwiGluPlugin.cpp/.h`
- CUDA 模块装载、launch 与 FP16 fallback：
  `int4GroupwiseSwiGluKernel.cu/.h`
- decode AOT kernel：
  `third_party/TensorRT-Edge-LLM/kernelSrcs/int4_fp16_gemm_cutedsl/int4_fp16_gemv_ampere.py`
- prefill 双投影 Tensor Core AOT kernel：
  `third_party/TensorRT-Edge-LLM/kernelSrcs/int4_fp16_gemm_cutedsl/int4_fp16_gemm_ampere.py`

插件接收两路 activation、两组 INT4 权重和两组 scale，共 6 个输入。gate
和 up 必须保留各自的 activation 输入，因为导出的两路 `pre_quant_scale`
不同，不能在插件内错误地共用同一个预缩放 activation。

运行时完全由 C++/CUDA 管理：TensorRT 调用插件 `.cpp` 的 `enqueue()`，再由
`.cu` 根据 token 数选择 Decode 或 Prefill kernel。两个 Python 文件只负责在
编译阶段把 CuTe DSL Tensor Core kernel 生成成静态 AOT artifact，并链接进插件
`.so`；部署后的 engine 只加载 `.so`，不会启动 Python。这样修改插件字段、输入
输出、workspace 和分派逻辑只需要阅读 `.cpp`，修改 CUDA launch 与 fallback
只需要阅读 `.cu`；只有需要重写 Tensor Core 主循环时才进入 CuTe DSL 源码。

执行路径分为：

- `M=1`：单个 CuTe DSL AOT kernel 同时完成两路 INT4 GEMV、FP32 归约与
  `SiLU(gate) * up`、FP16 写回，是真正的 decode 单 kernel 融合。
- `2 <= M <= 4`：复用两路现有 GEMV，再执行一个向量化 `half2` SwiGLU
  kernel。
- `M>4`：单个 CuTe DSL AOT kernel 顺序执行 gate/up 两路 INT4 Tensor Core
  主循环。gate tile 暂存在 shared memory，up 复用主循环 shared buffer，最后在
  register 中执行 `SiLU(gate) * up` 并只写回一次。SM87 按 M 动态选择：
  `M <= 96` 使用 `32x128x64`/2-stage，`M > 96` 使用
  `64x128x64`/3-stage，二者均为 split-K=1；不支持的配置仍保留原双
  GEMM + `half2` 路径作为回退。

## 生成与构建

输出目录必须不存在，脚本不会覆盖已有结果：

```bash
/home/jetson/.venvs/egcinet-modelopt/bin/python \
  server_opt_0101/fuse_int4_swiglu_onnx.py \
  models/jetson_onnx \
  models/jetson_onnx_swiglu
```

初始化 submodule、应用项目补丁并编译插件：

```bash
cd /home/jetson/Qwen3_VL_2B
./scripts/setup_edgellm.sh
CUTEDSL_PYTHON=/path/to/cutedsl/python ./scripts/build_edgellm.sh
```

已有 SM87 AOT artifact 时，`build_edgellm.sh` 会跳过生成步骤，不需要设置
`CUTEDSL_PYTHON`。

构建融合 LLM Engine：

```bash
cd /home/jetson/Qwen3_VL_2B
EDGELLM_PLUGIN_PATH=third_party/TensorRT-Edge-LLM/build-v0101/libNvInfer_edgellm_plugin.so.1.0 \
llm_build \
  --onnxDir models/jetson_onnx_swiglu/llm \
  --engineDir models/engines-jetson-swiglu-20260916/llm \
  --maxInputLen 256 \
  --maxKVCacheCapacity 384 \
  --maxBatchSize 1 \
  --maxKVPoolPages 3 \
  --profilingDetailed
```

这次 prefill kernel 没有修改 plugin 的输入输出、字段或版本号，因此现有
`models/engines-jetson-swiglu-20260916` 可以直接加载更新后的 `.so`；从融合
ONNX 新建部署包时仍建议按上面的命令重建 engine，确保 engine 与插件一起归档。

## 当前验证结果

逐层数据使用 warmup 3 次、测量 10 次，prefill 输入长度为 71，decode 的
past KV 长度为 96。单 kernel 数据使用原融合 engine 加载更新后的 ABI 兼容
插件；其他构建参数不变。

| 阶段 | 指标 | 原图基线 | 图融合/half2 | 双投影单 kernel | 相对原图 |
|---|---:|---:|---:|---:|---:|
| prefill | 全图逐层耗时 | 117.3645 ms | 117.7077 ms | 114.5988 ms | -2.36% |
| prefill | 28 组 gate/up/SwiGLU | 50.5428 ms | 50.8407 ms | 47.7054 ms | -5.61% |
| decode | 全图逐层耗时 | 36.3710 ms | — | 35.7313 ms | -1.76% |
| decode | 28 组 gate/up/SwiGLU | 12.6462 ms | — | 11.7546 ms | -7.05% |

对应按耗时降序的完整 CSV：

- 基线：`output/swiglu-20260916/profile-current-baseline/layers/`
- 图融合/half2：`output/swiglu-20260916/profile-fused-half2/layers/`
- prefill 双投影单 kernel：`output/swiglu-prefill-20260918-bm32/layers/`

Nsight Systems 确认 prefill 每层从“两次 GEMM + 一次 half2 SwiGLU”变为一次
`Int4Fp16SwiGluGemmAmpere` launch。M=71 路径使用 112 registers/thread、
25,088 B dynamic shared memory，grid 为 `(3, 48, 1)`。直接 kernel sweep
显示 M=128/256 时 `64x128x64`/3-stage 分别比 32-row tile 快约 20.8%/21.2%，
因此长 prefill 动态切到 64-row tile；M=71 仍由驻留率更高的 32-row tile
执行。`64x128x64`/2-stage 在测试范围内均更慢，未进入生产分派。

文本贪心回归使用“请用一句话介绍北京。”，基线与融合 Engine 均输出 26
个 token，token ID 和最终文本完全相同。该检查证明当前样例没有功能回归，
但不能替代原模型验证集上的 logits/cosine 和任务质量评测。

此外用同一组随机 `[71,2048]` activation、INT4 fragment 权重和 scale，将新
kernel 与“两次原 GEMM + FP32 SwiGLU”逐元素比较：输出 shape `[71,6144]`，
`max_abs=3.052e-5`、`cosine=1.0000001`、99.9954% FP16 元素完全相同。
