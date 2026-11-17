# 本地 LLM Harness 与 Self-Evolve MVP

> **状态说明（2026-11-17）：** 本文描述的是旧 `paf evolve` 提示词级 MVP，作为历史实现保留。当前主要研究流程是 `paf attribute-set` 的四模块源码级变异，详见 [`code_level_mutation.md`](code_level_mutation.md)。不要把本文中的单轮提示词提案流程与当前 Critic 诊断、定向消融、先验引导的代码变异流程混为一谈。

- 日期：2026-11-09
- 状态：已实现的工程原型；当前实验 Harness 为四模块候选架构
- 模型部署目标：课题组 H200，Qwen3.8-27B BF16，单卡 vLLM

## 1. 当前实现范围

本版本增加 OpenAI-compatible API 适配层，默认连接 `http://127.0.0.1:8000/v1`。同一个模型服务由三个逻辑角色共享：

- **Solver**：通过四模块候选 Harness 执行知识问答。
- **Proposer**：根据开发集失败摘要提出答案生成提示词候选。
- **Critic**：审查候选是否存在评测泄漏、任务硬编码或违反证据约束；只决定是否允许进入实验。

Critic 的批准不是效果证据。候选是否接受由程序化开发集指标决定。

## 2. 四模块候选 Harness

| 模块 | 当前实现 | LLM 调用 |
|---|---|---|
| QueryUnderstanding | 问题理解与查询改写 | 是 |
| EvidenceRetrieval | 词项召回、融合与排序 | 否 |
| EvidencePreparation | 证据筛选与上下文构造 | 否 |
| EvidenceReasoning | 充分性判断、答案生成、验证与一次修订 | 是 |

这是用于首轮代码变异验证的粗粒度候选，不代表已证明是最优模块划分。EvidenceRetrieval 目前不是 BM25/Dense/Hybrid 的完整实现。四个模块均有独立接口和消融点；算法实现候选可按契约整体替换。

## 3. 一轮 Self-Evolve 流程

1. 用当前 Harness 在 `tasks.development` 上运行基线。
2. 根据预先配置的阈值整理失败摘要。
3. Proposer 生成一个答案生成系统提示词候选。
4. Critic 审查候选，只决定是否允许进入测试。
5. 若获准，用相同开发任务评估候选版本。
6. 只有候选平均 token-F1 至少超过基线 `minimum_gain`，才接受候选。
7. 可选 `tasks.ood` 只在最终选定版本上生成报告，`used_for_acceptance=false`，不参与选择或迭代。

本节所述旧 `evolve` 命令仍只改答案提示词。多题 `attribute-set` 另有受约束的源码级单模块变异流程，说明见 `docs/code_level_mutation.md`；它不自动修改评测器、适配器或模块边界。

## 4. 运行配置

配置文件：`configs/self_evolve/qwen38_27b_h200.yaml`

- 默认服务：`http://127.0.0.1:8000/v1`
- 模型名：`Qwen3.8-27B`
- 初始上下文任务预算：小样本、短上下文
- 同一个 vLLM 服务串行承担 Solver、Proposer、Critic；不启动三个模型实例

命令：

```bash
paf llm-check --config configs/self_evolve/qwen38_27b_h200.yaml
paf evolve --config configs/self_evolve/qwen38_27b_h200.yaml --output data/derived/h200-self-evolve
```

输出：`self_evolve_report.json`、`self_evolve_report.md`、`selected_prompt.txt`。

## 5. H200 PBS 模板

`scripts/run_self_evolve_h200.pbs` 是独立模板，不覆盖用户现有 PBS 文件。模板申请 1 张 GPU，`walltime=48:00:00`，启动 BF16 vLLM，初始 `max-model-len=32768`、`max-num-seqs=1`、`gpu-memory-utilization=0.90`，完成后自动终止服务。

提交前必须：

1. 按集群实际环境修改模块加载和虚拟环境路径；
2. 设置 `MODEL_PATH` 为共享模型目录中的实际模型路径；
3. 确认 PBS 队列名、资源字段和作业策略符合课题组规则；
4. 确认计算节点允许本地绑定 `127.0.0.1:8000`；
5. 检查 vLLM 版本支持该模型的多模态架构。当前只传文本；若模型要求显式语言模型路径或额外参数，应按集群已验证的启动命令调整模板。

不要将服务暴露到公网，也不要在作业结束后遗留 vLLM 进程。

## 6. 旧 MVP 的研究限制

- token-F1 仅是首轮轻量代理指标；正式评测需补充任务特定 EM/F1、证据支持度、完整性、人工抽检和跨领域 OOD。
- 同一模型承担三个角色，角色提示词隔离不等于认知或统计独立；Critic 可能有系统性自我偏好。
- 当前检索是词项重叠基线，中文分词和长文档检索能力有限；后续可接入 BM25、Embedding（bge-m3/Qwen Embedding）和 Dense/Hybrid。
- 当前未实现迭代检索、真实 RRF、Cross-Encoder、Chunk 扩展/压缩、空输出/中性替代/替代实现干预。
- 归因估计器和实验统计仍与 LLM 自然语言诊断分离；不得用 Critic 的理由替代受控干预证据。
- 目前只进行一轮提示词进化；多轮进化、模块实现替换和自动模块拆分均未实现。
