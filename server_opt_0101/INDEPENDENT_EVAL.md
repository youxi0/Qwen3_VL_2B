# 固定 PTQ / QAT 候选的独立、多任务评测

本轮不训练、不重新 AWQ、不重新校准 Vision、不构建 Engine、不修改已有权重。
固定比较原混合 PTQ 与 `outputs/qat-teacher-512-v1` 由内部holdout选出的第60步。
保留之前128条作为开发集，不再把它当作最终独立测试集。

## 服务器运行

在现有 base 环境、项目根目录执行：

```bash
bash server_opt_0101/run_independent_eval.sh all
```

依次准备数据、评测PTQ、评测QAT、生成逐任务配对报告。使用现有 `datasets`、
`huggingface_hub`、Pillow 和量化环境，不升级框架。缺少 `datasets` 时才安装。

新数据默认512张：VQAv2、TextVQA、DocVQA、ChartQA各128张，均通过
HuggingFaceM4/the_cauldron 流式读取公开train子集。它们不是官方benchmark test，
也不是Qwen原始训练集。流式读取不主动下载全部169GB，但网络预取和缓存不保证
只占最终512张图片的大小。下载源/网络在服务器实测，本地测试不验证Hub连通性。

排除 `dataset/qat-vqav2-new-v1/train.jsonl` 的全部2048张和旧 `eval.jsonl` 的128张；
因此同时排除了已知QAT训练/内部holdout。按RGB像素及尺寸去重，跨任务也不重复。
不保证识别重压缩/裁切的近似图片，不保证排除模型预训练数据，也不保证排除已经
不可获取的旧PTQ校准图片；如还有其他训练数据，用 `prepare_public_eval.py --exclude`
重复追加其清单。已有排除清单的图片必须仍可读取，不能静默跳过缺失图片。

保持原图像16..64视觉token、输入256、生成最多32token。单图、每图一个QA；
不改问题，不截断人工答案，过长记录跳过。该筛选会限制结论适用范围。
OCR/文档FP16分数也可能受低分辨率影响，不能把全部低分归因于INT4。
本套主要是英文数据，中文业务和真实设备采集场景仍需另外保留独立测试集。

## 输出与判读

- 数据：`dataset/eval-independent-v1/eval.jsonl`、`images/`、`DATASET_INFO.json`。
- PTQ：`outputs/eval-independent-v1-ptq/`。
- QAT：`outputs/eval-independent-v1-qat/`。
- 对照：`outputs/eval-independent-v1-comparison.json`。

`evaluate_llm.py` 的summary现在同时输出规范化单答案匹配，不再手动补脚本。
规则与之前一致：NFKC、大小写折叠、合并空白、去句末`.!?。！？`，不做语义判分，
不把数字词与数字等同，不移除小数点或负号，不擅自加入答案前缀提取。
不是官方VQA软准确率、DocVQA ANLS或ChartQA宽松数值评分。

对照报告总体与每个任务分别给出FP16/PTQ/QAT匹配数、净提升、QAT修正/弄错数、
QAT/FP16得分比，以及配对精确McNemar检验p值。`cases`保留每条问题和三个回答，
便于核对是否只是措辞差异。p值仅作描述，未做多重检验修正，不能用一次结果
宣称“保证99%”。FP16得分为0时保留率为null，不除零，也不把它当作100%。

对照前核验同一ID集合、问题、人工答案、图片路径、FP16生成token与文本。
新报告还核验清单指纹（含图片字节hash）和处理参数，防止不同输入预算混比。
如果教师输出不同，会拒绝合并，不能靠忽略差异凑成同一评测。

Vision始终FP16，仅隔离LLM。最终仍需混合INT8 Vision、目标端Engine精度、
实际内存和业务样本验证。新评测不用于继续挑训练步数；若用结果修改方案，
它也转为开发集，需要再留最终独立测试。

## 失败后分阶段继续，不删除已有结果

如果prepare已完成而PTQ阶段失败，不要重新跑all，已有目录会拒绝覆盖。
选择尚未完成的阶段，并给失败输出使用新标签或保留后另选输出路径：

```bash
bash server_opt_0101/run_independent_eval.sh ptq
bash server_opt_0101/run_independent_eval.sh qat
bash server_opt_0101/run_independent_eval.sh compare
```

若PTQ阶段已创建不完整输出，换一个输出路径重跑该阶段，然后比较时沿用：

```bash
PTQ_EVAL_OUT=outputs/eval-independent-v1-ptq-retry \
  bash server_opt_0101/run_independent_eval.sh ptq
PTQ_EVAL_OUT=outputs/eval-independent-v1-ptq-retry \
  bash server_opt_0101/run_independent_eval.sh compare
```

QAT同理可用 `QAT_EVAL_OUT`，比较报告可用 `EVAL_COMPARISON_OUT`。
脚本不会自动删除失败目录。全新一套可用 `INDEPENDENT_TAG=v2 ... all`，但
同seed/source会得到同一套样本，不会因为改目录名就自动成为另一份独立集。

也可对已有128条的两份报告直接配对比较，无需再推理：

```bash
python server_opt_0101/compare_llm_reports.py \
  --baseline outputs/eval-ptq-mixed-vqav2-v2/alignment.json \
  --candidate outputs/eval-qat-teacher-512-best-v1/alignment.json \
  --out outputs/comparison-vqav2-dev-v1.json
```

旧报告没有清单指纹时只能检查其已记录字段，证明强度低于新报告；不能新旧各一份
混用不同协议。来源与许可应遵循原数据集/图片各自条款，没有统一的商业使用授权。

来源：[The Cauldron 数据卡](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)、
[HF Streaming 文档](https://huggingface.co/docs/datasets/stream)。
