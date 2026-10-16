# Qwen3-VL TensorRT 测试工具链

这套脚本以项目专用 C++ runtime 为端到端入口，并复用 TensorRT Edge-LLM 的 `llm_bench` 做独立 stage 和 layer profiling。

## 一键执行

先确认 `benchmark/config.json` 中的 engine、plugin 和工具路径，再运行：

```bash
benchmark/run_all.sh
```

默认依次执行：

1. 冷请求和稳定态端到端测试，同时采集进程 RSS 与 `tegrastats`；
2. prefill、decode、visual 的 E2E 和逐层 profile；
3. 在 engine 配置上限执行 Context/KV Capacity 稳定性检查；
4. 分别采集 prefill、decode、visual 的 Nsight Systems；
5. 生成统一的 `summary.json` 和 `report.md`。

快速验证脚本和输出格式：

```bash
benchmark/run_all.sh --quick --skip-nsys
```

指定结果目录：

```bash
benchmark/run_all.sh --output-dir output/my-benchmark
```

## 单项执行

```bash
benchmark/run_e2e.sh --output-dir output/my-benchmark
benchmark/run_layer_profile.sh --output-dir output/my-benchmark
benchmark/run_capacity.sh --output-dir output/my-benchmark
benchmark/run_nsys.sh --output-dir output/my-benchmark
benchmark/generate_report.sh --output-dir output/my-benchmark
```

NSYS 也可以只采集指定阶段：

```bash
benchmark/run_nsys.sh \
  --output-dir output/my-benchmark \
  --stages decode
```

只 profile 部分 stage：

```bash
benchmark/run_layer_profile.sh \
  --output-dir output/my-benchmark \
  --stages decode,visual
```

## Layer Profile CSV

每个 stage 都生成按 `time_ms_mean` 降序排列的 CSV：

```text
layers/layers_prefill_sorted.csv
layers/layers_decode_sorted.csv
layers/layers_visual_sorted.csv
```

主要字段包括：

- `layer_name`：实际 profile 中的层名；
- `layer_type`：TensorRT inspector 的层类型；
- `layer_layout`：输入和输出的 `Format/Datatype`；
- `input_tensor_names`、`output_tensor_names`：输入和输出 tensor 名称；
- 输入/输出 shape 和 dtype；
- tactic、算子分类、平均/标准差/最小/最大耗时及时间占比。

同名 `.md` 是完整 CSV 的可读表格镜像，原始 `llm_bench` CSV 和 inspector JSON 保存在 `layers/raw` 与 `layers/inspector`。

## NCU 独立接口

NCU 不会被 `run_all.sh` 调用。必须明确指定一个 kernel 正则或 TensorRT/NVTX 层范围：

```bash
benchmark/run_ncu.sh \
  --output-dir output/my-benchmark \
  --stage decode \
  --kernel-regex 'int4_fp16_gemv_m1_w4' \
  --launch-count 1
```

或者：

```bash
benchmark/run_ncu.sh \
  --output-dir output/my-benchmark \
  --stage decode \
  --layer-nvtx '目标 TensorRT NVTX range' \
  --launch-count 1
```

接口只汇总以下主要指标：kernel duration、SM/DRAM/L2 throughput、active warps、occupancy 限制和 waves/SM，同时保留完整 `.ncu-rep`。Jetson 默认可能禁止普通用户读取 GPU performance counters；脚本会先检查权限并给出明确错误，不会启动一次无效的全模型采集。

Nsight Systems 以独立进程采集 prefill、decode、visual，避免 4 GiB Jetson 在工具注入后同时驻留语言与视觉 engine 而 OOM；node 粒度保留调度、CUDA API、传输、时间空洞和 kernel 时间线。需要读取某个具体 kernel 的硬件计数器时，再调用独立 NCU 接口。

## 指标定义

- Peak RAM：同时报告 `tegrastats` 系统已用峰值、相对采样起点增量和进程 Peak RSS。Jetson 使用统一内存，不能把 CPU RSS 与 CUDA 分配简单相加。
- TTFT：`generate()` 开始到首个 token callback。
- TPOT：首末 token callback 的时间差除以 token 间隔数。
- Decode tokens/s：`1000 / TPOT`。
- Vision latency：首次端到端请求中的视觉阶段 CUDA event 时间；同图重复请求可能命中视觉缓存。
- Prefill latency：稳定态端到端请求中的 prefill 阶段 CUDA event 时间。
- 最大稳定 Context/KV Capacity：在当前 engine 配置允许的最大 shape 上执行测试；它是部署容量检查，不代表长上下文任务精度。

## 精度与任务质量

质量回归默认关闭，因为没有参考答案时不能诚实地给出“精度通过”。在 `quality_cases.jsonl` 中填入原模型的 `reference_text` 或业务 `required_keywords`，然后执行：

```bash
benchmark/run_quality.sh --output-dir output/my-benchmark --force
benchmark/generate_report.sh --output-dir output/my-benchmark
```

文本完全一致率适合严格回归，文本相似度和关键词通过率只作为任务质量辅助指标，不能代替 logits/layer cosine。

## 正式测试建议

正式 A/B 前固定 `nvpmodel`，使用管理员权限执行 `jetson_clocks`，确保温度稳定。`environment.txt` 会记录测试时的软件版本和频率状态。
