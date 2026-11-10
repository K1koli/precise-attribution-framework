# 精确归因框架（Precise Attribution Framework）

**项目：** 进化归因  
**代码版本：** 0.2.0（合成归因原型 + 本地 LLM Self-Evolve MVP）  
**文档语言：** 中文  
**当前状态：** 本地实现与测试阶段；尚未创建 GitHub 仓库。

## 项目目标

本项目通过受控干预，估计模块化系统的单模块效应、模块间交互效应，并在明确的联盟价值函数下进行贡献分配。核心目标是**精确归因**，而不是自动演化 Harness。

本项目独立开发。`bright-aegis` / baseline 仅作为未来外部效果对比对象，不作为架构、模块划分、提示词或实验流程模板。

## 当前实现

- 可配置的合成模块系统，支持模块主效应、两两交互、任务异质性和随机噪声。
- OpenAI-compatible LLM 接口，支持连接本地 vLLM 服务。
- 七模块知识问答 Harness 候选实现，以及同一模型承担 Solver / Proposer / Critic 的单轮 Self-Evolve MVP。
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

LLM Self-Evolve（需要先启动 OpenAI-compatible vLLM 服务）：

```bash
paf llm-check --config configs/self_evolve/qwen38_27b_h200.yaml
paf evolve --config configs/self_evolve/qwen38_27b_h200.yaml --output data/derived/h200-self-evolve
```

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

- LLM Harness 当前是七模块候选原型；Retriever/Ranker 是确定性词项基线，不是完整 BM25/Dense/RRF/Cross-Encoder 实现。
- Self-Evolve 当前仅进化 AnswerGenerator 系统提示词一轮；Critic 只作准入审查，程序化指标决定接受与否。
- 当前评测使用轻量 token-F1 代理指标，不能作为正式知识任务结论。

- 合成归因实验仍不接入生产级 LLM Agent；另有独立的本地 LLM Harness MVP，但尚未接入正式真实检索基准。
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

设计变更遵循日期化历史原则。未获用户确认的研究方向只能标记为“待讨论”。
