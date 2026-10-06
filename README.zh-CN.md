# CTREvo

[English](README.md) · [设计](docs/design.md) · [数据集选择](docs/datasets.md)

**让 Agent 编写 CTR 模型代码，用受控实验验证假设，并从结果中积累经验。**

CTREvo 将 [ModelEvoHarness](https://github.com/CharlesXu-HQ/ModelEvoHarness) 接入具体的点击率预测任务。Agent 可以在现有 backbone 中加入交叉分支、调整融合方式、修改 loss，也可以根据实验否定一个方向。每次变化都需要说明依据、对照和证伪条件。

项目追求有效实验的产出。是否提升指标，需要实际实验回答。

## Agent 的优化空间

| 方向 | 如何执行 |
|---|---|
| 表征和交叉结构 | 修改原生 PyTorch `build_model(schema, config)` 源码，包括 embedding、交互层和并行子网络 |
| 子网络组合 | 读取 Harness 源码，记录组件身份、输入绑定、参数共享和融合方式 |
| 训练目标 | 可选 `training_loss(logits, labels)`，以及固定范围内的学习率、权重衰减 |
| 实验决策 | 根据证据排序假设、比较备选方向、在 backbone 内迭代并选择性继承经验 |
| 经验总结 | 技术经验绑定具体数据集；业务解释必须有字段语义和实验依据 |

宿主固定数据、划分、完整训练行覆盖、优化器类型、batch size、epoch 上限和评估器。实际 CUDA 执行与 Agent 的机制解释分别记录。整体指标改善并不证明每个组件都有效。

## 首个验证任务

使用 Criteo Display Advertising Challenge **全部 45,840,617 条有标签记录**，原始特征为 **13 个数值字段 + 26 个类别字段**。额外的 13 个缺失指示器属于派生输入。

- 按文件原始位置固定划分 80%/10%/10%；没有时间戳，不能宣称是时间切分。
- 数值进行带符号 log1p，标准化只拟合训练集；类别按字段独立哈希，缺失独立编码。
- 主指标为验证集 LogLoss，同时报告 AUC、Brier 和校准表现。
- 配对 LogLoss 差值给出按曝光计算的正态区间；不修正自适应多次选择，也无法处理未知用户聚类。
- 搜索结束后另行运行测试集评估，只使用已保存的 baseline 和最终候选权重。搜索期间 Agent 看不到测试结果。
- 匿名字段不支持凭空解释用户画像、行为序列或业务分群，不会因为字段多就认为 DIN 适用。

## 闭环

```text
固定任务 + 历史观察 + Harness 指导
  → 读取相关实现 → 提出假设并编写候选
  → 隔离环境中进行全量 GPU 训练 → 宿主独立评估
  → 对照指标与假设 → 总结组件经验 → 下一次决策
```

Harness 通过 `third_party/model-evo-harness` Git submodule 引入，保持原仓库不变，记录具体提交用于复现。更新依赖需要单独进行。

## 运行

需要 Linux NVIDIA GPU、NVIDIA Container Toolkit、CUDA Docker 镜像，以及与镜像兼容的 Python 环境。模型实验禁止退回 CPU。仅接口测试可以在 CPU 上运行。

```bash
git clone --recurse-submodules https://github.com/CharlesXu-HQ/CTREvo.git
cd CTREvo
docker build -f Dockerfile.gpu -t ctrevo-cuda .
python3.12 -m venv .venv
.venv/bin/pip install -e . -e third_party/model-evo-harness
python scripts/download.py /data/criteo
.venv/bin/ctrevo prepare --raw /data/criteo/train.txt --output /data/criteo/prepared

export CTR_AGENT_API_KEY='...'
.venv/bin/ctrevo search --data /data/criteo/prepared --output runs/criteo \
  --image ctrevo-cuda --venv "$PWD/.venv" \
  --provider-url https://api.deepseek.com --model deepseek-flash --steps 2

.venv/bin/ctrevo finalize --data /data/criteo/prepared --output runs/criteo \
  --image ctrevo-cuda --venv "$PWD/.venv"
```

镜像需要能够运行挂载环境中的 Python 和原生依赖。Provider URL 和 API key 分别通过参数、环境变量配置；不要提交密钥。默认开启 thinking，常规决策使用 high，需要复核时使用 max；实际模型和参数支持以 provider 为准。不支持该扩展的服务可使用 `--thinking omit`。

默认每个候选完整训练一轮，执行 baseline 和两次 Agent 尝试。超时是失败，不会把未完成的部分训练当作结果。`--resume` 只允许在任务、评估协议和 Harness 身份一致时恢复。最终测试也必须使用相同协议参数。

如果 provider 的格式修复耗尽，保留终端错误日志，使用携带最后拒绝提案与错误的恢复入口：

```bash
python scripts/resume_with_feedback.py --data /data/criteo/prepared --output runs/criteo \
  --error-log runs/search-terminal.log --image ctrevo-cuda --venv "$PWD/.venv" --steps 2
```

seed、epoch 上限、batch size 和超时必须与原运行一致。脚本把真实错误与原响应交回 Agent，不改写候选源码、不绕过校验。原始终端输出需要通过 shell 重定向保存。普通中断仍可使用 `search --resume`。

## 结果与边界

`journal.json` 保存提案、参考源码哈希、指标、反思和最优候选。每个 trial 保留实际源码、checkpoint、预测、运行日志和 GPU 元数据。最终测试写入 `final/report.json`。Provider 日志不记录密钥和思考文本。

候选容器无网络、无 API key、无验证/测试标签。训练数据与目标集特征只读挂载，宿主独立计算指标。容器和导入检查用于减少泄漏，不等于面向恶意代码的安全执行平台。梯度和输出检查无法证明机制归因，因此目前明确标记归因为 unverified。

测试中的小数据仅用于校验接口，不能作为数据集效果。实际进展见[实验记录](docs/experiments.md)。

代码采用 [Apache-2.0](LICENSE)。数据需要自行从发布方获取并遵守其条款；仓库不发布原始数据、密钥和训练权重。
