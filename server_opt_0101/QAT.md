# W4A16 QAT / 量化感知蒸馏试验

适用当前 Qwen3-VL-2B、ModelOpt 0.45.0 和 Edge-LLM 0.10.1。
`qat_awq.py` 是本项目的实验训练实现。NVIDIA 文档介绍的 QAT 思路是
在模拟量化前向中微调浮点权重；这里针对已经打包、未保存训练态的 AWQ
checkpoint 恢复 FP32 shadow weights，并保留原来的 group-128 scale 和
`pre_quant_scale`。

- 教师：原始 FP16 Qwen3-VL-2B。
- 学生：当前最佳的旧 AWQ + INT4 LM head + 6 个 FP16 回退投影。
- 前向：权重除 group scale、round、clamp 到 [-8, 7]、反量化到 FP16。
- 反向：STE 梯度更新 FP32 shadow weights；默认用教师全词表 KL 蒸馏。
- 导出：调用本机 ModelOpt 的 `pack_int4_in_uint8`，仍是 W4A16 AWQ。
- 原来的 FP16 回退投影和 Vision 权重不参与训练。后续 pipeline 按同一配置
  生成 Vision W8A8 + blocks 0/4/5/11 FP16。

这条路线会实际调整部署的 INT4 字节，不需要额外运行时 LoRA。
已量化文件不能直接传给普通 HF Trainer 作为可训练浮点权重。

## 1. 先跑环境/训练检查

使用原来的 base 环境，不新增训练框架、不升级 ModelOpt。

```bash
cd /home/featurize/work/qwen3vl
python -B -m unittest discover -s server_opt_0101/tests -p test_qat_awq.py -v

python server_opt_0101/qat_awq.py \
  --root . \
  --config server_opt_0101/config.old-awq-int4-lmhead-mixed-proj-vision-fp16-blocks-residual-a07.json \
  --train outputs/manifests-formal-v1/calib.jsonl \
  --eval outputs/manifests-formal-v1/eval.jsonl \
  --scope head --train-samples 8 --holdout-samples 2 \
  --steps 2 --grad-accum 1 --eval-every 1 \
  --out outputs/qat-head-smoke-v1
```

2 步检查用于验证梯度和打包流程，通常还不足以跨越量化区间、改善精度。
训练前会验证原始 packed 字节可以完全重建、QAT step 0 的指标与原 AWQ 相同。

## 2. 第一轮：只训练 INT4 LM head

```bash
python server_opt_0101/qat_awq.py \
  --root . \
  --config server_opt_0101/config.old-awq-int4-lmhead-mixed-proj-vision-fp16-blocks-residual-a07.json \
  --train outputs/manifests-formal-v1/calib.jsonl \
  --eval outputs/manifests-formal-v1/eval.jsonl \
  --scope head \
  --train-samples 256 --holdout-samples 16 \
  --steps 200 --grad-accum 4 --lr 2e-5 --eval-every 25 \
  --diversify-prompts \
  --out outputs/qat-head-v1
```

无 `answer` 的训练记录先由教师生成短回答，随后沿回答位置蒸馏教师概率分布。
同一记录的教师回答在内存中复用，教师 logits 每次在线计算，避免写入大量磁盘缓存。
存在真实 `answer` 时保留其问答，答案超过 `max_new_tokens` 会报错，避免静默截断。
默认 KL 权重为 1；可用 `--ce-weight 0.1` 加入少量回答 token CE。

脚本把训练清单内部的 16 个内容项留作选模型集。重复图片的不同问句不会跨到
两个集合。`--eval` 只用于检测最终验证集泄漏，不用其内容选训练 checkpoint。
`--diversify-prompts` 只替换训练集中重复出现且没有人工答案的图片问句。
原 256 张图仅适合验证训练是否有效；进一步恢复通用能力需要更多有代表性、
图文/文本兼具且不与最终验证集重叠的数据。

每隔 25 步比较校准集内部 holdout，导出 KL 最低的 checkpoint。
如果没有训练 checkpoint 优于 step 0，则保留 step 0，并报告
`improved_on_holdout: false`，不会把更差权重当成改进。

## 3. 查看训练报告

```bash
python - <<'PY'
import json
d = json.load(open('outputs/qat-head-v1/qat_report.json'))
print('status:', d['status'])
print('best step:', d['best_step'])
print('baseline:', d.get('baseline'))
print('best:', d.get('best_metrics'))
print('holdout KL reduction:', d.get('holdout_kl_reduction'))
print('changed packed bytes:', d.get('changed_packed_bytes'))
print('peak allocated MiB:', d.get('peak_cuda_allocated_mib'))
PY
```

只有浮点训练 loss 下降而 `changed_packed_bytes` 为 0，意味着实际 INT4 权重
还没有改变，不能据此宣称量化精度恢复。holdout 改善仍需要最终验证集复测。

## 4. 导出、完整精度复测

```bash
USE_TRT_NATIVE_ATTN=0 python -m tensorrt_edgellm.scripts.export \
  outputs/qat-head-v1/checkpoint \
  outputs/qat-head-v1/llm_onnx \
  --skip-visual --dtype float16 --externalize-weights int4_ffn

bash server_opt_0101/run_server.sh run \
  --root . \
  --config outputs/qat-head-v1/config.qat.json \
  --calib outputs/manifests-formal-v1/calib.jsonl \
  --eval outputs/manifests-formal-v1/eval.jsonl \
  --out outputs/run-formal-qat-head-v1
```

与同一 32 张验证集上的既有基线比较：LLM Top1=0.8125、KL=0.148966，
组合 Top1=0.814453、KL=0.153002。KL/Top1 是教师一致性，真实任务精度仍需
业务样本评估。Edge 导出与实际 TensorRT Engine 构建也需要在服务器/Jetson 实测。

## 5. 扩展到 decoder

确认小实验可运行后，复制第 2 步命令，把 `--scope head` 改成 `--scope all`，
把输出改成 `outputs/qat-all-v1`。这样同时训练所有仍为 INT4 的 decoder Linear
和 INT4 LM head，保留那 6 个既有 FP16 投影。
`--scope decoder` 则只训练 INT4 decoder，冻结 LM head。

默认 microbatch=1，`--grad-accum` 是累积次数。decoder/all 模式自动启用梯度
checkpoint；48 GB 服务器上先用 2 步、`--grad-accum 1` 检查实际峰值，训练
显存不等于 Jetson 推理内存。训练生成的部署 checkpoint 保持原有 packed
shape、scale 和 INT4/FP16 布局，因此逻辑部署权重体积不增加。

新的 checkpoint 对未改写的 safetensors 使用硬链接，同一文件系统下共享磁盘块。
原文件按不可变输入处理；仅删掉一个硬链接不会破坏另一个，但原地改写会影响两边。
不同磁盘时自动复制。实验输出是推理 checkpoint，不含可恢复训练的 Adam 状态。

官方参考：
[ModelOpt QAT 文档](https://github.com/NVIDIA/Model-Optimizer/blob/main/docs/source/guides/_pytorch_quantization.rst)、
[0.45.0 INT4 打包实现](https://github.com/NVIDIA/Model-Optimizer/blob/0.45.0/modelopt/torch/export/quant_utils.py)。
