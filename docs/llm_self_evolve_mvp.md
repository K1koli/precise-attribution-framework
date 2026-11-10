# 本地 LLM Harness 与 Self-Evolve MVP

- 日期：2026-11-09
- 状态：已实现的工程原型；七模块架构仍为 Proposed
- 模型部署目标：课题组 H200，Qwen3.8-27B BF16，单卡 vLLM

## 1. 当前实现范围

本版本增加 OpenAI-compatible API 适配层，默认连接 `http://127.0.0.1:8000/v1`。同一个模型服务由三个逻辑角色共享：

- **Solver**：通过七模块候选 Harness 执行知识问答。
- **Proposer**：根据开发集失败摘要提出答案生成提示词候选。
- **Critic**：审查候选是否存在评测泄漏、任务硬编码或违反证据约束；只决定是否允许进入实验。

Critic 的批准不是效果证据。候选是否接受由程序化开发集指标决定。

## 2. 七模块候选 Harness

| 模块 | 当前实现 | LLM 调用 |
|---|---|---|
| QueryProcessor | 查询改写 | 是 |
| Retriever | 词项重叠检索基线 | 否 |
| Ranker | 可观测的确定性融合/排序基线 | 否 |
| ContextBuilder | 去重前的基础上下文拼接与长度限制 | 否 |
| ReasoningController | 证据充分性判断 | 是 |
| AnswerGenerator | 基于证据生成答案 | 是 |
| Verifier | 证据支持度与完整性判断 | 是 |

这是一套可运行的中粒度候选，不代表已证明是最优模块划分。Retriever 目前不是 BM25/Dense/Hybrid 的完整实现；Ranker 也尚未实现 RRF 或 Cross-Encoder。模块内部操作的替换和独立干预仍待后续实现。

## 3. 一轮 Self-Evolve 流程

1. 用当前 Harness 在 `tasks.development` 上运行基线。
2. 根据预先配置的阈值整理失败摘要。
3. Proposer 生成一个答案生成系统提示词候选。
4. Critic 审查候选，只决定是否允许进入测试。
5. 若获准，用相同开发任务评估候选版本。
6. 只有候选平均 token-F1 至少超过基线 `minimum_gain`，才接受候选。
7. 可选 `tasks.ood` 只在最终选定版本上生成报告，`used_for_acceptance=false`，不参与选择或迭代。

当前进化对象仅为 `AnswerGenerator` 的系统提示词；不自动改 Python 代码、检索算法、模块边界或模型权重。该实现是单轮 MVP，不是完整自主进化系统。

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

## 6. 研究限制与后续工作

- token-F1 仅是首轮轻量代理指标；正式评测需补充任务特定 EM/F1、证据支持度、完整性、人工抽检和跨领域 OOD。
- 同一模型承担三个角色，角色提示词隔离不等于认知或统计独立；Critic 可能有系统性自我偏好。
- 当前检索是词项重叠基线，中文分词和长文档检索能力有限；后续可接入 BM25、Embedding（bge-m3/Qwen Embedding）和 Dense/Hybrid。
- 当前未实现迭代检索、真实 RRF、Cross-Encoder、Chunk 扩展/压缩、空输出/中性替代/替代实现干预。
- 归因估计器和实验统计仍与 LLM 自然语言诊断分离；不得用 Critic 的理由替代受控干预证据。
- 目前只进行一轮提示词进化；多轮进化、模块实现替换和自动模块拆分均未实现。
