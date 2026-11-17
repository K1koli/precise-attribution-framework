# 人类先验算子库（首版）

更新日期：2026-11-16

## 目的与边界

先验库把论文中经实验研究过的 RAG 方法转成可检验的修改方向。它不把论文结果当作本项目上的保证：论文的数据集、模型、检索器、训练方式和评价指标与本项目不完全相同。每条先验都记录了来源、适用信号、证伪检查、实现建议、不变量和风险。

同一算子同时提供两个视角：

- **Critic 视角**：哪些轨迹现象使该方法值得怀疑，如何提出可证伪的模块假设。
- **Proposer 视角**：在当前接口内如何实现，以及需要保留哪些契约和信息。

Critic 先用 dev 集的正确/错误轨迹和可用先验提出假设，再只对被提名模块消融；Proposer 收到对应实现建议和消融结果。先验不能代替轨迹证据、消融或 held-out 验收。

## 首版纳入的方法

| 算子 | 主要模块 | 论文启发 | 当前实现状态 |
|---|---|---|---|
| 查询拆解并保留实体 | QueryUnderstanding、EvidenceRetrieval | 多跳问题拆分后分别检索、合并与重排；后续子问题需显式携带桥接实体 | 可在现有 `queries` 契约内尝试 |
| 推理条件下的迭代检索 | EvidenceReasoning、EvidenceRetrieval | IRCoT/FLARE 将检索与中间推理或不确定信息结合 | 需要 Harness 控制流支持，暂不提供给 Agent |
| 检索质量检查与纠正 | EvidenceRetrieval | CRAG 先判断候选证据质量，再选择纠正动作 | 可在单模块内试验，但当前没有外部搜索能力 |
| 选择性证据压缩 | EvidencePreparation | RECOMP 研究抽取/摘要压缩及选择性上下文增强 | 可尝试；现有模块契约尚未显式保留句级来源映射 |
| 区分相关性、支持度与使用情况 | EvidenceRetrieval、EvidenceReasoning | Self-RAG 将检索决策、段落相关性和答案支持性分开；Provenance 做上下文到输出的事实核验 | 可在现有验证模块内近似实现；原论文中的训练方案不可由提示词等同复现 |
| 按问题复杂度分配检索预算 | QueryUnderstanding、EvidenceRetrieval、EvidenceReasoning | Adaptive-RAG 在不检索、单轮和多轮策略间路由 | 需要 Harness 控制流支持，暂不提供给 Agent |

完整机器可读条目见 [`configs/priors/rag_operators.yaml`](../configs/priors/rag_operators.yaml)。

## 论文来源

1. Trivedi et al. (2023), [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://aclanthology.org/2023.acl-long.557/), ACL 2023.
2. Jiang et al. (2023), [Active Retrieval Augmented Generation](https://aclanthology.org/2023.emnlp-main.495/), EMNLP 2023.
3. Asai et al. (2024), [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://openreview.net/forum?id=hSyW5go0v8), ICLR 2024.
4. Xu et al. (2024), [RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation](https://proceedings.iclr.cc/paper_files/paper/2024/hash/bda88ed2892f5e61c9a9bf215c566913-Abstract-Conference.html), ICLR 2024.
5. Jeong et al. (2024), [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://aclanthology.org/2024.naacl-long.389/), NAACL 2024.
6. Yan et al. (2024), [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884), arXiv preprint; expanded publication: [Findings of AACL-IJCNLP 2026](https://aclanthology.org/2026.findings-aacl.151/).
7. Sankararaman et al. (2024), [Provenance: A Light-weight Fact-checker for Retrieval Augmented LLM Generation Output](https://aclanthology.org/2024.emnlp-industry.97/), EMNLP Industry Track 2024.
8. Ammann et al. (2025), [Question Decomposition for Retrieval-Augmented Generation](https://aclanthology.org/2025.acl-srw.32/), ACL Student Research Workshop 2025.
9. Zhu et al. (2025), [Mitigating Lost-in-Retrieval Problems in Retrieval Augmented Multi-Hop Question Answering](https://aclanthology.org/2025.acl-long.1089/), ACL 2025.

首版选择方法原语而非照搬完整论文系统。尤其是需要训练专用模型、隐藏状态或跨模块循环的方案，只作为设计参考，不宣称当前代码已能实现原论文方法。

## 运行与对照

`attribute-set` 默认从实验配置指定的先验文件加载条目。设置 `priors.enabled: false` 可关闭先验，用于之后做有先验/无先验对照。报告会记录库 ID、启用算子 ID、Critic 诊断引用的算子及候选接收的实施建议。

建议后续在固定模型、数据切分、提案数和评测预算下比较：

1. 无先验；
2. Critic 可见诊断视角，Proposer 不见实施视角；
3. Critic 与 Proposer 按当前设计分别获得同一算子的角色视角。

只有第三种相对前两种稳定改善候选质量或 held-out 表现时，才支持“角色分层先验库有帮助”的结论。
