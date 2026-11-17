# 精确归因框架（Precise Attribution Framework）

**项目：** 归因驱动的知识密集型 Harness 进化

**代码版本：** 0.2.0（合成归因原型 + LLM Harness 进化实验）
**文档语言：** 中文  
**当前状态：** 研究原型；代码托管于 [GitHub](https://github.com/K1koli/precise-attribution-framework)。模块归因、定向代码变异与 BRIGHT Biology 检索评测已可运行，跨领域泛化和先验库收益仍待系统验证。

## 项目目标

本项目研究如何通过模块化归因和受控干预定位 Harness 的薄弱环节，再结合 RAG 人类先验对目标模块做定向变异，并用开发集与 held-out 评测筛选版本。目标是提升知识密集型任务上的 Harness 能力，并检验改进能否跨领域迁移。合成模块归因和 Shapley 估计仍作为独立的实验基础设施保留。

本项目独立开发。`bright-aegis` / baseline 仅作为未来外部效果对比对象，不作为架构、模块划分、提示词或实验流程模板。

## 当前实现

- 可配置的合成模块系统，支持模块主效应、两两交互、任务异质性和随机噪声。
- OpenAI-compatible LLM 接口，支持连接本地 vLLM 服务。
- 四模块知识问答 Harness 候选实现，以及同一模型承担 Solver / Proposer / Critic 的单轮 Self-Evolve MVP。
- 多题执行轨迹诊断、Critic 定向消融、先验引导的单模块 Python 变异及 train / dev / test 数据分离。
- BRIGHT Biology 预处理候选池上的 nDCG@10 评测适配器；目前只覆盖检索排序，不代表端到端问答或跨领域泛化结论。
- H200 单卡 vLLM + PBS 作业模板（48 小时上限）。
- 对小规模模块系统枚举完整 `2^n` 联盟。
- 显式记录绕过（bypass）干预语义。
- JSONL 追加式原始运行记录。
- 配对区块估计：平均主效应、两两交互效应及其正态近似 95% 区间。
- 基于联盟价值函数的精确 Shapley 贡献分配。
- 独立验证阶段读取模拟真值；估计器接口不接收真值对象。
- JSON、Markdown 报告与可复现实验配置摘要。

## 快速开始

建议使用 Python 3.12 或 3.13：

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -e '.[dev]'
pytest
paf run --config configs/experiments/demo.yaml --output data/derived/demo
```

### DeepSeek API 或本地 vLLM

`llm.provider` 支持 `deepseek`、`vllm` 和 `openai_compatible`。
旧配置未指定 provider 时仍按 vLLM 处理。

- 本地实验：`configs/experiments/qa30_attribution.yaml`。
- DeepSeek 实验：`configs/experiments/qa30_deepseek.yaml`，默认 `deepseek-flash`。

DeepSeek Key 仅从 `DEEPSEEK_API_KEY` 环境变量读取。在 macOS 默认 zsh 终端中：

```zsh
read -rs 'DEEPSEEK_API_KEY?输入 DeepSeek API Key（输入隐藏）: '
export DEEPSEEK_API_KEY
printf '\n'
```

然后在同一个终端、仓库目录运行：

```bash
paf llm-check --config configs/experiments/qa30_deepseek.yaml
paf attribute-set --config configs/experiments/qa30_deepseek.yaml --output data/derived/qa30-deepseek
```

`llm-check` 只检查模型列表接口；真实生成是否成功需通过模型请求验证。
环境变量仅对该终端及其子进程生效，不会自动传递给其他已经打开的应用。
不要把真实 Key 写入 YAML、聊天或 Git；本地服务需要认证时也可设置 `api_key_env`。

DeepSeek 默认关闭思考（`disable_thinking: true`），适配现有较小的角色输出预算，
只发送 DeepSeek 的 `thinking` 字段，不发送 vLLM 专用参数。
vLLM 保留 `PAF_DISABLE_THINKING` 行为；显式配置 `disable_thinking` 优先于环境变量。
若启用思考，需要重新评估输出预算。通用兼容 provider 不发送这两类扩展字段。
接口和模型名称参考 [DeepSeek 官方文档](https://api-docs.deepseek.com/)。

LLM Self-Evolve（需要先启动 OpenAI-compatible vLLM 服务）：

```bash
paf llm-check --config configs/self_evolve/qwen38_27b_h200.yaml
paf evolve --config configs/self_evolve/qwen38_27b_h200.yaml --output data/derived/h200-self-evolve
```

单问题模块归因（同样需要本地 LLM 服务）：

```bash
paf attribute-task --config configs/experiments/single_task_attribution.yaml --output data/derived/single-task
```

该命令运行基线、四个逐模块旁路消融和配置中列出的单模块候选，输出完整模块 Trace、token-F1 对比及 JSONL 运行记录。

多题代码级归因与变异（默认 train/dev/test = 60%/20%/20%，测试集不参与接受）：

```bash
paf attribute-set --config configs/experiments/qa30_attribution.yaml --output data/derived/qa30-code-evolve --proposals 2
```

最终版本确定后，单独运行测试（不能每轮据此选择改动）：

```bash
paf evaluate-test --config configs/experiments/qa30_attribution.yaml --experiment data/derived/qa30-code-evolve
```

默认先让 Critic 对照 train 集高分/低分执行轨迹及标准答案，提名可疑模块；系统只对被提名模块做 bypass 消融，再把诊断与消融结果交给 Proposer 生成单模块 Python 候选。候选经过源码约束检查、契约 smoke 和 dev 对比，由 dev 决定接受；test 仅在确定最终版本后通过独立命令评测。可重复传入 `--target-module` 限定 Critic 的诊断范围。候选源码及被接受的版本保存在输出目录；模块接口和限制见 `docs/code_level_mutation.md`。

实验报告会逐题记录异常和有效配对覆盖率；候选默认需在 dev 上达到至少 0.01 的平均增益且 bootstrap 区间下界大于 0。候选模块每题最多调用 LLM 8 次。

`qa30_attribution.yaml` 中包含 EvidencePreparation 去除文档元数据前缀的手写对照候选。单独验证该候选而不生成模型提案，可运行 `paf attribute-set --config configs/experiments/qa30_attribution.yaml --output data/derived/qa30-reference-control --proposals 0`。候选源码通过 `source_file` 相对配置文件目录加载。

BRIGHT 原始语料和预处理后的 Biology 任务文件未包含在仓库中。运行 `bright_biology.yaml` 前，需要准备配置 `tasks_file` 指向的 `configs/tasks/bright_biology_50.yaml`；每题需包含候选 `documents` 和 `gold_doc_ids`。`bright_ndcg10` adapter 评测给定候选池上的检索排序，不负责下载或转换 BRIGHT 原始数据。

若要四个模块各生成一个候选，可传 `--proposals 4` 并重复指定四个 `--target-module`。BRIGHT 风格预处理数据可将配置 `benchmark` 设为 `bright_ndcg10`；该适配器只归因查询理解和证据检索模块。

集群 PBS 模板见 `scripts/run_self_evolve_h200.pbs`；提交前必须设置 `MODEL_PATH` 并按集群环境检查模块加载、虚拟环境和队列资源字段。详细说明见 `docs/llm_self_evolve_mvp.md`。

不安装命令行入口时，也可运行：

```bash
PYTHONPATH=src python -m precise_attribution.cli run \
  --config configs/experiments/demo.yaml \
  --output data/derived/demo
```

输出目录包含：

- `runs.jsonl`：原始运行记录（追加式格式）；
- `estimate.json`：估计器输出；
- `validation.json`：估计冻结后由验证器产生的真值比较；
- `report.md`：人类可读报告；
- `manifest.json`：配置与运行元数据摘要。

> 当前原型的“真值隔离”是软件接口与依赖边界隔离，不是操作系统级安全沙箱。`validation/truth.json` 仅由编排流程写入验证目录；估计器只接收运行记录。若进入多用户或对抗环境，必须增加权限隔离和独立进程。

## 研究边界与限制

- 四模块 Harness 当前仍是候选分解；EvidenceRetrieval 内部采用确定性词项基线，不是完整 BM25/Dense/RRF/Cross-Encoder 实现。
- 旧 `paf evolve` 命令仍是单轮 AnswerGenerator 提示词进化，Critic 负责提案准入；独立的 `paf attribute-set` 命令实现了代码级模块变异。
- QA 示例使用轻量 token-F1；BRIGHT 适配器只评估预处理后的候选文档排序。当前实验结果不能单独证明通用知识密集型能力或跨领域迁移。

- 合成归因实验与 LLM Harness 进化是两条独立实验路径；BRIGHT Biology 已接入预处理候选池评测，但尚未实现从原始数据到完整端到端 RAG 的统一复现流程。
- 当前仅支持二元模块状态和 `bypass` 干预；空输出、中性替换、替代实现已在设计中区分，但尚未实现为可执行策略。
- 区间使用配对区块的正态近似（1.96 × 标准误），不是小样本精确区间或多重比较校正。
- Shapley 对联盟均值精确枚举；“精确”指对已观测联盟价值函数精确计算，不代表因果机制真值。
- 当前以完整全因子数据计算均匀背景分布下的主效应与两两交互；尚未支持任意背景分布 q。

## 目录

```text
configs/                 实验与场景配置
src/precise_attribution/ 核心实现
  core/                  数据模型与序列化
  simulation/            合成任务与隐藏真值
  execution/             模块系统执行
  interventions/         干预语义
  experiments/           实验编排
  recording/             JSONL 记录
  estimators/             仅依赖观测记录的估计器
  validation/             真值验证
  reporting/              报告生成
tests/                   单元、集成与验证测试
docs/                    项目设计文档
```

## 设计文档

- `docs/architecture.md`
- `docs/attribution_theory.md`
- `docs/experiment_protocol.md`
- `docs/research_decisions.md`
- `docs/module_decomposition_study.md`（七模块候选划分研究，Proposed）
- `docs/llm_self_evolve_mvp.md`（本地模型接入与 Self-Evolve MVP）
- `docs/code_level_mutation.md`（四模块接口、源码级变异与 benchmark adapter）
- `docs/human_prior_library.md`（论文来源与先验算子库设计）

设计变更遵循日期化历史原则。未获用户确认的研究方向只能标记为“待讨论”。

### PBS：本地模型与三路划分实验（2026-11-17）

`scripts/run_self_evolve_h200.pbs` 现在启动计算节点上的 vLLM 并运行
`attribute-set`，不再运行旧的提示词级 `evolve`。保留原模板的队列、项目、
单卡和 48 小时资源配置；提交前确认这些资源和账户适用于你。

在服务器的仓库根目录提交（默认 1 个自动候选，保留配置中的 2 个人工对照）：

```bash
qsub scripts/run_self_evolve_h200.pbs
```

默认使用 `configs/experiments/qa30_attribution.yaml`，train/dev/test 比例为
60%/20%/20%，seed=0。每个作业输出到 `data/derived/h200-attribution-<jobid>/`，
默认不运行 test。vLLM 日志在 `logs/vllm-<jobid>.log`；PBS 输出采用默认作业日志名。

环境要求沿用原脚本：`vllm/0.19.0` 模块、`$HOME/venvs/paf` 虚拟环境（Python
3.12/3.13 和项目依赖）、共享模型路径 `/projects_vol/gp_wei.lu/base_model/Qwen3.8-27B`。
可通过 `qsub -v` 覆盖 `PAF_VENV`、`VLLM_MODULE`、`MODEL_PATH`、`SERVED_MODEL_NAME`、
`PORT`、`CONFIG_PATH`、`EXPERIMENT_DIR`、`PROPOSALS`、`TRAIN_RATIO`、`DEV_RATIO`、`SPLIT_SEED`。
脚本使用当前仓库源码。模型名和端口自动同步到输出目录内的 `runtime_config.yaml`，
原配置不被改写。已有非空输出目录不能重用。

只有最终版本确定后，才提交测试作业，替换下面的 `<jobid>`：

```bash
qsub -v JOB_MODE=test,EXPERIMENT_DIR=data/derived/h200-attribution-<jobid> scripts/run_self_evolve_h200.pbs
```

test 模式复用该实验保存的配置，模型名必须一致；若开发时覆盖了模型路径或环境，
测试作业也需要传入相同覆盖值。测试不会改变接受状态，不要按测试结果继续调参。
脚本已做本地语法和配置测试，尚未在集群上实际提交验证。
