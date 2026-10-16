# Third-party dependencies

`TensorRT-Edge-LLM` 以官方 `v0.10.1` 为基线，项目定制代码维护在
[`youxi0/TensorRT-Edge-LLM`](https://github.com/youxi0/TensorRT-Edge-LLM) 的
`qwen3-vl-int4-swiglu` 分支。父仓库通过 Git submodule 记录经过验证的精确提交，
不要在父仓库中另外维护同一组源码补丁。

克隆父仓库时推荐直接初始化所有子模块：

```bash
git clone --recurse-submodules <project-url>
```

已有工作区使用项目脚本同步并检出父仓库锁定的版本：

```bash
./scripts/setup_edgellm.sh
```

## 定制代码开发流程

修改 Edge-LLM 时，先在子模块仓库提交并推送，再回到父仓库更新 submodule 指针：

```bash
cd third_party/TensorRT-Edge-LLM
git switch qwen3-vl-int4-swiglu
git fetch origin
git pull --ff-only origin qwen3-vl-int4-swiglu

# 修改并验证代码后
git add <files>
git commit -s
git push origin qwen3-vl-int4-swiglu

cd ../..
git add third_party/TensorRT-Edge-LLM
git commit
```

官方仓库仅用来获取新版本，不直接向其推送。首次需要同步上游时添加 remote：

```bash
cd third_party/TensorRT-Edge-LLM
git remote add upstream https://github.com/NVIDIA/TensorRT-Edge-LLM.git
git fetch upstream --tags
```

更新官方基线应单独建分支完成合并、构建和精度回归，验证后再更新父仓库指针。

生成缺失的 SM87 CuTe DSL artifact 并构建插件、Core runtime 和 benchmark：

```bash
CUTEDSL_PYTHON=/path/to/cutedsl/python ./scripts/build_edgellm.sh
```

已经存在 artifact 时不需要设置 `CUTEDSL_PYTHON`。构建输出位于
`third_party/TensorRT-Edge-LLM/build-v0101`。脚本默认按照 Jetson Orin、
CUDA 13.2 配置；其他 CUDA 版本可通过 `CUDA_CTK_VERSION` 覆盖。
