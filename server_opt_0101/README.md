# Qwen3-VL-2B：服务器 AWQ 对齐 + Vision INT8（Edge-LLM 0.10.1）

本包只提供服务器运行代码。本地未执行量化或模型推理，也未证明 Jetson 3.6 GiB 能装下。
路线：复用现有 LLM INT4 AWQ → 对齐原模型 → 原始 Vision 单独做 SmoothQuant W8A8 → 只导出 Vision → 组合原 LLM ONNX → 独立构建、检查服务器 Vision Engine。

## 1. 当前路径与上传布局

本机已确认根目录：`E:/work/cpp/Qwen3_VL_2B`。服务器真实目录未知，以下统一以 `/data/qwen` 举例；换成你实际上传目录即可。

```text
/data/qwen/
├── server_opt_0101/                         本代码包
├── Qwen3-VL-2B-Instruct/                    原始完整 HF 权重、配置、处理器
├── Qwen3-VL-2B-INT4-AWQ/                    完整 AWQ HF checkpoint
├── Qwen3-VL-2B-INT4-AWQ-ONNX-v0101/
│   ├── llm/                                原来的 INT4 导出，整目录上传
│   └── visual/                             原来的 FP16 Vision，作为 Engine 对照
└── dataset/
    └── JPEGImages/                         已加入的 528 张图片
```

不要漏掉 LLM 的 `model.onnx.data`、`embedding.safetensors`、`external_int4_ffn_weights.safetensors`、`external_lm_head_weight.safetensors` 以及 tokenizer/config。旧 `onnx/` 不参与本流程。

图片已本地检查：528 张可读取、无文件级或像素完全相同的重复；384×288 / 960×720 / 880×600 / 640×352 四种尺寸。默认选 256 张校准、32 张验证，其余 240 张保留。不检查近似重复；若来自同一段视频，需手工按场景划分，避免相邻帧泄漏。

## 2. 服务器 Python 环境

在 Linux RTX 4090 48GB 服务器操作，建议独立 Python 3.12 环境。`tools` 依赖依据官方 0.10.1 锁定，而不是沿用旧教程的 Transformers/ModelOpt 版本。需要可用 NVIDIA 驱动；不要仅根据 `nvcc` 版本判断 PyTorch 可用。

```bash
cd /data/qwen
nvidia-smi
python3.12 -m venv .venv-edge0101
source .venv-edge0101/bin/activate
python -m pip install --upgrade pip
python -m pip install -r server_opt_0101/requirements-server.txt
python -m pip check
python -c "import torch; print('torch:', torch.__version__, 'CUDA wheel:', torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
bash server_opt_0101/run_server.sh inspect --root /data/qwen
```

要求 Edge-LLM 0.10.1、PyTorch 2.13.0、Transformers 5.14.1、ModelOpt 0.45.0、ONNX 1.19.0、safetensors 0.8.0。具体来自 [官方 v0.10.1 依赖文件](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.10.1/pyproject.toml)。如当前镜像源缺包，换可用的官方源或官方环境；不要自行降级绕过版本检查。磁盘建议另外预留至少 25 GB 给中间 checkpoint、ONNX 和候选包。

注意：Python 量化/导出阶段不要求安装 TensorRT；第 5 节的 Engine 验证另外需要 TensorRT 10.x 和匹配的服务器插件。

## 3. 划分数据并运行

输出目录必须不存在，避免覆盖之前的实验。以下两条是正式 256 + 32 图片流程。

```bash
bash server_opt_0101/run_server.sh prepare \
  --root /data/qwen --out /data/qwen/manifests-64tokens-v1

bash server_opt_0101/run_server.sh run \
  --root /data/qwen \
  --calib /data/qwen/manifests-64tokens-v1/calib.jsonl \
  --eval /data/qwen/manifests-64tokens-v1/eval.jsonl \
  --out /data/qwen/run-vision-int8-v1
```

长任务可放进已有的 `tmux` 会话；请保留完整运行日志。先测试接口时，对 prepare/run 都加 `--smoke`，并换成全新的目录：只用 1 张校准 + 1 张验证，报告会标记为冒烟测试，不能作为精度验收。

默认问句是“描述这张图片中可见的内容。”，因此报告首先衡量与原模型的一致性，不等同于业务准确率。有实际任务时，编辑生成的 JSONL，或者自己提供不重叠的校准/验证 JSONL：

```json
{"id":"eval_001","image":"../dataset/JPEGImages/0.jpg","question":"图中主要的物体是什么？","answer":"这里填写人工确认的短答案"}
```

`answer` 可省略；填写时额外计算该答案的 NLL 和生成答案严格匹配。不要把示例答案当真值。图片路径相对 JSONL 文件解析；也支持服务器绝对路径。验证可加纯文本记录，但仍需至少一张独立验证图。默认短答案限制 32 个 Token，超限会报错，不静默截断。

## 4. 实际量化范围、对齐方法与尺寸

- AWQ：不重做、不覆盖。读取现有 ModelOpt 对称 INT4 打包权重，按输出维解包低/高半字节，应用 group scale 和独立的 `pre_quant_scale`。检查 196 个 LLM Linear、词表映射、同一套输入下的 teacher-forcing logits、KL、Top-1 一致率、NLL 和贪心文本。参考原模型用 FP16 运行，以接近部署数值格式。
- AWQ 检查后端是 **反量化 FP16 数学参考，不是 TensorRT INT4 CUDA kernel**。它用于定位 checkpoint/量化质量问题；现有 LLM ONNX 仅结构与 sidecar 完整性检查，尚未验证它与 checkpoint 数值一致。最终仍需目标 LLM Engine 对齐。
- Vision：从原始 FP16 模型校准，不从已量化 LLM 上再做第二次算法。100 个 Linear 使用静态 INT8 权重 + INT8 激活（W8A8）、权重按输出通道、激活按 Tensor，SmoothQuant alpha=0.5。实现依据 [ModelOpt 0.45 配置接口](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0.45.0/modelopt/torch/quantization/config.py)。
- 默认 `conservative`：24 个 block 的 qkv/proj/fc1/fc2 共 96 层，加最终 merger / 三个 DeepStack merger 的 fc1 共 4 层。保留所有 merger 的 fc2、PatchEmbed、Norm、Softmax 和 Attention 核心为 FP16；最终输出及三路 DeepStack 仍为 FP16。不是“整座 Vision 每个算子均 INT8”。
- `residual_fp16`：用于全 W8A8 误差逐层累积时的选择性混合精度。量化每个 block 的 qkv 和 MLP fc1，以及四个 merger fc1，共 52 层；把直接写入残差流的 attention proj 和 MLP fc2 保留为 FP16。这个范围预计比 `conservative` 少节省约 120 MiB，必须重新生成实际 `memory_budget.json`，不能只采用估算值。
- 导出复用 [Edge-LLM 0.10.1 的 INT8SQLinear](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.10.1/tensorrt_edgellm/models/linear.py) 和既有 ViT Attention Plugin；不写新 FC/GELU/Attention 插件，不把纯 FP16 图冒充量化图。按 recipe 严格检查 52/96/100/104 路 INT8 权重 DQ 和激活 Q/DQ。
- 会另外把 AWQ LLM 与 fake-quant Vision 组合做第三组端到端数值比较，区分 LLM AWQ 误差、新增 Vision 误差及二者叠加。

尺寸统一由原模型 processor 控制，保留长宽比并按 patch/merge 规则对齐；不是把所有图片强行拉伸成 256×256。`min_image_tokens=16`、`max_image_tokens=64` 对应处理后面积约 16,384～65,536 像素。原图可以更大，也可以尺寸各异。

单图 Vision 的实际 ONNX 输入是 `[N,1536]` patches，不是 `[1,3,H,W]`；`N = grid_h × grid_w`，合并后视觉 Token 数为 `N/4`。因此 TensorRT profile 的 N 范围是 64～256。脚本还从同一图像 grid 生成 rotary、cu_seqlens、双线性位置插值和 carrier 输入，不能随意用随机数组代替。

低分辨率有独立的信息损失，尤其 OCR/细小物体。本包的 FP16 与量化模型均用相同的 64-Token 设置，所以不会掩盖量化误差，但也不测“高分辨率原模型 → 低分辨率部署”的任务退化。若实际任务需要 128/196 Token，请修改 `config.json` 后重新校准、导出、构建及评测；不要复用本次统计假装已覆盖新尺寸。

## 5. 服务器单独构建 Vision Engine

量化/ONNX 导出完成不代表 TensorRT 已正确执行 INT8。第二阶段需要：

1. 服务器 TensorRT **10.x** Python 包及同版本运行库（版本和 Jetson 目标尽量一致）。Edge-LLM 0.10.1 是框架版本，并不是 TensorRT 的版本号。
2. 用同一 TensorRT 版本构建、支持服务器 SM89 的 Edge-LLM **v0.10.1** 插件 `.so`。不能加载 Jetson ARM64 `.so`，也不能拿另一 TensorRT 版本编译的库凑数。按 [官方安装与构建文档](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/installation.html) 和对应 v0.10.1 源码准备；本代码不会自动安装或覆盖服务器系统的 TensorRT/CUDA。

已有对应环境后，把下面的 `.so` 路径换成实际服务器插件文件：

```bash
python -c "import tensorrt as trt; print(trt.__version__); assert trt.__version__.split('.')[0] == '10'"

python server_opt_0101/trt_vision_check.py \
  --run-dir /data/qwen/run-vision-int8-v1 \
  --plugin-lib /实际的服务器路径/libattentionPlugin.so \
  --workspace-mib 256
```

它分别构建原 FP16 Vision 和新 INT8 Vision，只跑保存的真实视觉输入，比较四个输出：FP16 Engine 对原始 PyTorch、INT8 Engine 对 fake-quant PyTorch，并另外检查 INT8 Engine 对原始 FP16 的量化误差。保存每层 inspector 信息、输出、Engine 大小、context memory 和服务器推理耗时。Engine 匹配参考要求平均余弦 ≥0.999 且 relative L2 ≤0.03；INT8 对原 FP16 的 0.99 是独立的候选精度门槛，不会再把一个忠实复现 fake-quant 的 TensorRT Engine 错报为构建验证失败。这些是可配置的实验门槛，不是官方精度保证。

`server_engines/` 不会进入可迁移包。**RTX 4090 构建的 Engine 不能直接作为 Jetson Engine 使用。** 未来需在目标 Jetson 上构建，或使用厂商明确支持且匹配平台的交叉构建方案。TensorRT 的输入与执行接口见 [TensorRT 10.x Python 文档](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html)。

如果暂时没有服务器插件，先完成第 3 节，把结果标记为“量化/导出完成、Engine 未验证”，不要将它当成通过部署验收。

## 5.1 可选：把 tied LM head 也量化为 INT4 AWQ

Edge-LLM 0.10.1 的量化器不会给已经量化的 checkpoint 增量添加 LM-head
量化；它检测到已有量化后会跳过。因此从原始 FP16 checkpoint 重新校准
decoder AWQ 与 LM head，Vision 仍不在这一步量化：

```bash
python server_opt_0101/requantize_awq_lm_head.py \
  --model-dir models/Qwen3-VL-2B-Instruct \
  --calib outputs/manifests-formal-v1/calib.jsonl \
  --out outputs/Qwen3-VL-2B-INT4-AWQ-LMHEAD-v0101 \
  --num-samples 128

USE_TRT_NATIVE_ATTN=0 python -m tensorrt_edgellm.scripts.export \
  outputs/Qwen3-VL-2B-INT4-AWQ-LMHEAD-v0101 \
  outputs/Qwen3-VL-2B-INT4-AWQ-LMHEAD-ONNX-v0101 \
  --skip-visual \
  --dtype float16 \
  --externalize-weights int4_ffn
```

量化后的 LM head 必须留在 ONNX/Engine 内，不能再传
`--externalize-weights lm_head`；0.10.1 会主动拒绝把量化 LM head 当成
FP16 sidecar 外置。随后使用
`config.lmhead-int4-residual-a07.json` 跑正式流程，它会保持 decoder/LM-head
为 W4A16 AWQ，并按之前的 residual-fp16、alpha=0.7 配方重新生成 Vision
W8A8：

```bash
bash server_opt_0101/run_server.sh run \
  --root . \
  --config server_opt_0101/config.lmhead-int4-residual-a07.json \
  --calib outputs/manifests-formal-v1/calib.jsonl \
  --eval outputs/manifests-formal-v1/eval.jsonl \
  --out outputs/run-formal-lmhead-int4-residual-a07-v1
```

新 checkpoint 应有 197 个 packed INT4 Linear（196 decoder + 1 LM head）。
最终候选不再包含 593.5 MiB 的 FP16 `external_lm_head_weight.safetensors`；
实际节省量以新 `memory_budget.json` 为准。重新校准意味着 decoder AWQ
尺度也会重算，必须重新查看 AWQ 与组合端到端指标，不能沿用旧报告。

## 6. 输出与内存判断

```text
run-vision-int8-v1/
├── reports/
│   ├── awq_weight_alignment.json        解包/缩放后的各层权重误差
│   ├── awq_alignment.json               AWQ + FP16 Vision 一致性
│   ├── vision_alignment.json            INT8 Vision 四路输出误差
│   ├── combined_alignment.json          AWQ + INT8 Vision 组合一致性
│   ├── vision_onnx_audit.json            实际量化覆盖、Plugin/接口检查
│   ├── memory_budget.json               带假设的内存估算
│   ├── status.json                      各检查项状态，不把未跑的测试当成功
│   └── server_trt_vision.json            第 5 节成功执行后才产生
├── vision_sq_checkpoint/                临时原始 FP16 LLM + INT8 Vision，不用于整模型部署
├── vision_onnx/visual/                   新视觉 ONNX
├── vision_cases/                        同一预处理输入和参考输出
├── server_engines/                      仅第 5 节生成，不上传 Jetson
└── jetson_onnx_candidate/                后续可传到 Jetson 的候选资产
    ├── llm/                             完整复用现有 AWQ 导出及 sidecars
    ├── visual/                          新 INT8 Vision
    ├── deployment_limits.json           batch=1，单图，视觉最多64 Token
    ├── STATUS.json
    └── reports/
```

默认 max_input_tokens=256（包含视觉和模板 Token），max_new_tokens=32，KV 容量384；这些是本包测试和部署约束，**不会自动改写未来 Edge-LLM 的构建/运行参数**，上板时必须使用一致的限制。

估算同时计入：AWQ LLM、完整 embedding、额外的 FP16 lm_head、INT8 Vision、FP16 KV Cache，以及可调整的系统/CUDA预留900 MiB、激活/workspace预留256 MiB。不假定 tied embedding 能在当前 Edge runtime 省掉 lm_head 副本，也不通过删词表换取虚假的内存优势。完整词表的 embedding/head 是重要内存来源。

权重理论上可因本次100层 Vision INT8减少约352 MiB，但**量化后的逻辑权重字节数、磁盘 ONNX 大小、Engine 文件大小均不等于设备运行峰值内存**。若实际预留比本包假设更高，仍可能 OOM。3.6 的单位在配置中是 GiB，请按 Jetson `/proc/meminfo` 与实际空闲内存修正。

数值或估算门槛未通过时，正式 run 返回状态码2，保留报告和带状态的候选文件，不自动覆盖/接受它。优先查看失败样本；不要仅放宽阈值让结果变绿。可分别尝试不同 alpha、`vision_recipe="blocks"`（只量化96个 block Linear）、`residual_fp16`（52层，保护残差输出投影）或 `all_linears`（104层，更省权重但风险更高），每次用新输出目录、固定验证集对比。

最后部署还缺两项不可用服务器估算替代的验证：实际 TensorRT INT4 LLM Engine 与 checkpoint 的数值/任务对齐；Jetson 原生运行时的总内存峰值和业务精度。如果本次不足3.6 GiB预算，后续再评估词表方案、运行时副本/工作区优化、KV限制或换更小模型，不能保证仅 Vision INT8 就够。

## 7. 本机已做与未做的检查

已只读检查模型目录/权重头、数据集图片；另运行 CPU 单元测试检查 INT4 半字节布局、文件截断、数据分割、输出指标和 profile 形状。本机没有安装或运行 Torch/ModelOpt/TensorRT。整个 GPU 流程需要在你的服务器执行，结果尚未知。

```bash
python -B -m unittest discover -s server_opt_0101/tests -v
```

### 已修复：对齐时 mask 103 与 token type 71 长度不一致

早期 `continuation_logits()` 追加回答 Token 时只扩展了 `input_ids` 和
`attention_mask`，漏掉 Transformers 5.14.1 的 `mm_token_type_ids`。
现已通过 `continuation_inputs()` 保留原始多模态类型，并为新增文本补类型0，
同时重新计算全序列位置。该修复作用于原模型、AWQ 和组合模型的对齐。

已有服务器环境只需更新 `server_opt_0101/torch_work.py`；无需更换依赖、模型、
`config.json` 或重新生成数据清单。重跑时使用新的 `--out` 目录保留失败日志。
如上传新增的 `tests/test_continuation.py`，可以先用下面命令在服务器做无模型、
无 CUDA 的回归测试（包含真实 PyTorch CPU 张量测试）：

```bash
python -B -m unittest discover -s server_opt_0101/tests -p 'test_continuation.py' -v
```

### 已加入保护：SmoothQuant 的零通道产生 NaN

ModelOpt 0.45.0 在计算平滑系数时可能遇到 `0/0`；其限幅不会消除 NaN，
因此在 `pre_quant_scale should be positive` 处失败。只读检查本地原始权重发现，
第0层 Vision MLP 的通道1590在 FP16 下有零权重行/列，可触发这个边界情况。

`torch_work.py` 现在为官方平滑系数应用函数加一个临时、进程内的保护：仅当
实际权重列最大值和校准激活最大值均严格为0时，才把该通道的 NaN 系数设为1；
其他 NaN/Inf、非正 scale、异常统计或非白名单层仍会报错。不修改安装包文件，
退出作用域时恢复原函数；不要在同一进程的多个线程中同时执行量化。
正常通道仍使用官方 SmoothQuant 算法，保留权重重校准与导出状态。

若实际触发修复，日志显示 `[SmoothQuant zero-channel repair]` 和层名/通道，
正式报告 `vision_quantization.json` 的 `smoothquant_scale_checks` 记录详情。
另检查只有目标100层对应的200个输入/权重量化器被启用；插入1224个量化器
不等于1224个全部启用。原 AWQ 和 Vision 量化范围不变。

服务器只需更新 `torch_work.py`，继续使用原数据清单并换一个新的输出目录。
可先上传 `tests/test_smoothquant_guard.py` 并执行以下小型CPU测试，不加载模型权重：

```bash
python -B -m unittest discover -s server_opt_0101/tests -p 'test_smoothquant_guard.py' -v
```

本地仅完成 NumPy 数值/形状回归测试，真实 ModelOpt 集成测试需在服务器运行。
