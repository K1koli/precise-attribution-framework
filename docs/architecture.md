# 精确归因框架——架构设计

- **文件：** `architecture.md`
- **版本：** 1.0-draft
- **状态：** 初始设计基线；后续依据证据修订
- **最后更新：** 2026-11-09
- **项目：** 进化归因

## 1. 目的与范围

本文定义 Precise Attribution Framework（精确归因框架）首个独立版本的软件架构。项目的核心工程目标是：**通过受控干预，识别并验证系统组件对观测结果的贡献**。V1 聚焦归因基础设施与方法有效性验证，不实现自主 Harness 演化。

更广泛的研究动机是：知识密集型任务跨越不同领域，所需知识内容差异很大。系统可能把领域事实或 benchmark 特定修补误当成可复用的知识处理机制。精确归因希望区分哪些机制真正带来效果，为未来定向改进提供可信证据。未来演化闭环不属于 V1 实现范围。

## 2. 明确的项目边界

### 2.1 独立实现

- 建立全新的代码库，独立设计核心数据模型、执行流程和实验协议。
- 不继承或复刻 baseline Harness 的架构。
- 不把 baseline 仓库的模块划分、控制流、提示词、模型选择或评测惯例视为设计要求。
- 现有 baseline 仅作为外部对比方法；未来如需公平对比，可通过薄适配层接入。
- 不把修改 baseline 仓库作为本项目工作的一部分。

### 2.2 V1 范围

V1 应当：

1. 定义稳定的任务、模块、干预、实验、运行记录和归因结果数据模型；
2. 在确定性或受控随机模拟环境中运行小规模模块化系统；
3. 记录完整实验来源信息（provenance）及模块级执行轨迹；
4. 支持基线运行、单模块消融、模块组合干预和替换干预；
5. 估计主效应与两两交互效应；
6. 对模块数量较少的系统实现精确 Shapley 贡献分配；
7. 基于重复、配对运行估计不确定性；
8. 使用隔离的模拟真值验证估计结果；
9. 生成机器可读和人类可读报告。

V1 不应当：

- 接入生产级 LLM Agent 或真实检索 benchmark；
- 实现 Critic/Proposer Agent；
- 建立先验算子库；
- 自动修改或演化 Harness 代码；
- 在演化期间使用 OOD 测试数据作为验证门槛；
- 自动优化候选 Harness。

## 3. 架构原则

1. **归因优先：** 架构选择必须服务于有效归因，而非仅仅方便开发模块。
2. **职责分离：** 执行、干预分配、记录、估计和验证彼此分离。
3. **真值隔离：** 归因估计器不得访问模拟器的真实机制参数。
4. **证据不可变：** 原始运行记录只追加、不覆盖；分析产生派生结果，不改写观测。
5. **干预语义明确：** 关闭模块、替换模块和改变参数属于不同干预。
6. **可复现：** 每个结果均可追溯到任务实例、随机种子、配置、软件版本和干预分配。
7. **渐进复杂度：** 先做小规模全因子实验；只有在枚举成本过高时才引入抽样与近似。
8. **不受 baseline 驱动：** 为兼容 baseline 而改变核心架构，不是合理理由。

## 4. 逻辑架构

### 4.1 Core（核心层）

负责共享领域模型、数据校验、标识符、配置加载和模块注册契约。

主要概念：`TaskCase`、`ModuleSpec`、`SystemConfig`、`Intervention`、`ExperimentSpec`、`RunRecord`、`AttributionResult`、`ValidationResult`。

### 4.2 Simulation（模拟层）

提供具有已知因果结构的合成任务和机制。

职责：生成任务实例；定义模块行为；采样受控噪声；定义数据生成机制；在估计流程之外保存真实系数和机制参数。模拟器真值只能由验证器在估计结果冻结后读取。

### 4.3 Execution（执行层）

根据系统配置、任务实例和干预分配执行模块化系统。记录模块输入输出、执行状态、耗时、异常及依赖关系。执行层不负责计算归因结论。

### 4.4 Intervention（干预层）

把实验方案转换为明确的模块状态或替换策略。每次干预必须记录目标模块、干预类型、干预参数、适用背景配置及执行是否成功。

### 4.5 Experiment（实验编排层）

生成实验矩阵，安排基线、消融、组合干预和重复运行；管理任务实例、随机种子与配对关系；保证实验条件可追溯。

### 4.6 Recorder / Storage（记录与存储层）

原始记录采用追加式 JSONL；初期可使用 SQLite 建立索引和查询视图。原始记录与分析产物分开存放。不得通过覆盖原始记录来“修正”实验结果。

### 4.7 Estimator（归因估计层）

只读取观测结果、干预分配和允许公开的实验元数据。负责估计条件干预效应、平均主效应、交互效应及贡献分配。不得导入或读取模拟器真值。

### 4.8 Validator（验证层）

在估计结果冻结后，读取隐藏真值并计算误差、偏差、符号识别准确率、不确定性区间覆盖率等指标。验证器不得把真值反馈给同一轮最终估计过程。

### 4.9 Reporting（报告层）

生成实验配置摘要、干预说明、样本量、估计值、不确定性、验证指标和限制说明。报告必须区分观测结果、估计结果与模拟真值。

## 5. 依赖方向与隔离要求

建议依赖方向：

`Core → Simulation / Execution / Intervention → Experiment → Recorder → Estimator → Validator → Reporting`

这是逻辑依赖示意，不表示所有模块必须形成单向调用链。关键约束是：

- Estimator 不得依赖 Simulation 的真值接口或内部机制对象；
- Validator 可以读取隐藏真值，但只在估计结果冻结后执行；
- Execution 不直接生成归因结论；
- Reporting 不重新计算或篡改原始实验数据。

## 6. 建议目录结构

```text
precise-attribution-framework/
├── README.md
├── pyproject.toml
├── configs/
│   ├── experiments/
│   └── scenarios/
├── src/precise_attribution/
│   ├── core/
│   ├── simulation/
│   ├── execution/
│   ├── interventions/
│   ├── experiments/
│   ├── recording/
│   ├── estimators/
│   ├── validation/
│   └── reporting/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── validation/
├── data/
│   ├── raw/
│   └── derived/
├── docs/
│   ├── architecture.md
│   ├── attribution_theory.md
│   ├── experiment_protocol.md
│   └── research_decisions.md
└── scripts/
```

目录是初始建议，可根据实现证据调整；不得为了贴合 baseline 而照搬其结构。

## 7. 实施里程碑

- **M0：设计基线。** 确认数据模型、归因定义、实验协议和真值隔离原则。
- **M1：最小端到端骨架。** 建立一个可运行的合成任务，从配置到运行记录再到报告贯通，并有测试。
- **M2：消融估计。** 实现单模块干预、配对运行和条件效应估计。
- **M3：组合干预。** 实现两两组合实验和交互效应估计。
- **M4：贡献分配。** 对小规模系统计算精确 Shapley 值。
- **M5：验证与稳健性。** 加入噪声、依赖关系、未见机制参数和不确定性质量测试。

先建立最小可运行端到端骨架，再逐层深化实现，避免只完成孤立模块而无法集成。

## 8. 初始技术选择（暂定）

- Python 3.12；
- pytest；
- Pydantic；
- PyYAML；
- NumPy / SciPy；
- pandas；
- JSONL 原始日志，SQLite 索引。

这些是工程起点，不是不可变约束。若更改，应在 `research_decisions.md` 记录原因和影响。

## 9. 验收原则

V1 的成功标准不是“能运行一个复杂 Agent”，而是：在已知机制的合成系统中，能通过明确干预估计单模块效应和交互效应，能报告不确定性，并能在不向估计器泄漏真值的条件下，经独立验证证明估计质量。

---

## 10. 代码实现状态（2026-11-09）

首个本地原型已在 `src/precise_attribution/` 建立：

| 设计职责 | 实现位置 | 当前状态 |
|---|---|---|
| 数据模型、配置校验 | `core/models.py`、`core/config.py` | 已实现 |
| 合成任务与异质性 | `simulation/scenario.py` | 已实现 |
| 隐藏真值 | `simulation/truth.py` | 已实现，供验证使用 |
| 绕过干预 | `interventions/policies.py` | 已实现 |
| 执行与配对全因子编排 | `execution/runner.py`、`experiments/` | 已实现 |
| JSONL 追加记录 | `recording/jsonl.py` | 已实现 |
| 主效应、两两交互与 Shapley | `estimators/factorial.py` | 已实现 |
| 真值验证 | `validation/validator.py` | 已实现 |
| 报告 | `reporting/report.py` | 已实现 |

当前明确限制：仅支持 `bypass`；背景分布 q 固定为均匀分布；区间使用可配置置信水平的配对区块正态近似，未做多重比较校正；真值隔离是软件接口边界而非操作系统级安全沙箱；完整联盟枚举最多支持 10 个模块。具体限制见项目 `README.md`。


## 11. 本地 LLM Harness 实现增补（2026-11-09）

新增实现位于 `harness/`、`llm/` 与 `evolution/`：

- `OpenAICompatibleClient` 通过本地 vLLM OpenAI-compatible endpoint 调用同一个模型服务。
- Solver 由七模块候选 Harness 组成；当前 QueryProcessor、ReasoningController、AnswerGenerator、Verifier 调用 LLM，Retriever、Ranker、ContextBuilder 使用确定性基线。
- Proposer 提出 AnswerGenerator 系统提示词候选；Critic 仅审查候选是否可进入测试。程序化开发集指标决定是否接受。
- OOD 样例仅用于最终报告，不参与接受决策。

该增补不改变七模块架构的 Proposed 状态，也不改变“归因与演化分离”的已确认决策。当前 Self-Evolve 是单轮提示词级 MVP，不是代码级或多轮自主进化。细节和限制见 `docs/llm_self_evolve_mvp.md`。
