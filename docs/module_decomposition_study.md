# 模块划分研究：文献依据与首轮实验方案

- **日期：** 2026-11-09
- **状态：** Proposed（研究提案，尚未确认正式架构）
- **目的：** 为知识密集型任务的 Modular Task Harness 确定可解释、可干预、可归因的模块边界。

## 1. 当前结论

不应直接把既有六模块当成最终答案。文献可以提供候选分解方式，但模块划分是否适合本项目，必须通过同一任务集上的受控干预验证。

首轮建议采用 **7 个功能模块作为中粒度候选架构**。这不是最终定案，而是后续比较的中心候选：

1. QueryProcessor：理解任务需求并改写查询；不负责决定是否开启下一轮检索。
2. Retriever：执行稀疏、稠密或混合召回；输出候选证据及检索分数。
3. Ranker：融合多路候选并进行重排；内部保留 Fusion 与 Reranking 两段独立日志，便于必要时拆分。
4. ContextBuilder：去重、过滤、扩展、压缩并组装证据上下文。
5. ReasoningController：决定是否检索、是否继续迭代、何时停止及调用哪条处理路径；不负责最终答案内容生成。
6. AnswerGenerator：基于上下文进行推理并生成答案；不负责独立的事实验证。
7. Verifier：检查答案主张是否有证据支持、是否覆盖问题要求；不负责格式化。

格式序列化先作为固定的接口适配层，不列为首轮可进化模块。索引构建/文档解析属于离线知识库准备流程，先作为受控环境条件记录，不混入单次查询的在线 Harness 模块。

## 2. 文献与已有研究给出的证据

| 来源 | 对模块划分的启示 | 不能据此推出什么 |
|---|---|---|
| Yang et al., EVOTOOL, ACL 2026 | 将工具策略拆成 Planner、Selector、Caller、Synthesizer，并利用轨迹定位错误、对单模块定向修改。说明按功能职责和轨迹可观测性划分具有可操作性。 | 不能证明四模块适用于知识密集型 RAG，也不能证明其划分最优。 |
| Kumar et al., RAG Components Are Not Independently Rankable, 2026 | 对 chunking、retrieval、fusion、reranking 做配对全因子比较；报告了 chunking 与 retrieval 的交互、BM25 fusion 的领域依赖，以及 reranking 在多数上游配置下的稳定收益。直接支持必须测交互，而不是只给组件独立排名。 | 不能直接确定本项目应该采用几模块；结论受其数据集、模型和实现范围约束。 |
| Kim et al., Investigating Component Contributions in Multi-Agent ML Systems, ICML 2026 | 通过 4,000 多项受控消融研究组件贡献；摘要指出迭代反馈的贡献可能高于架构复杂度，多智能体协调也可能带来负收益。提醒我们不要把“更多模块”当作更好。 | 不能将其 ML engineering agent 的结论直接外推到 RAG/知识任务。 |
| Retrieval-Augmented Generation: A Methodological Survey, 2026 | 对 191 篇研究按主要创新位置归类，区分 Retrieval-Optimized、Orchestrated、Structured-Memory 等方向；说明 RAG 改进分布在检索机制、控制策略、知识结构等不同层次。 | 这是研究方法分类，不是一个可以直接照搬的运行时模块图。 |
| 用户科研笔记中的 SWE-agent、EvoTool 记录 | SWE-agent 强调动作简单、反馈可观测、上下文紧凑；EvoTool 按工具决策链分模块。可作为设计原则和候选方案来源。 | 这些笔记中的总结不是本项目的实验证据；需回到论文和自己的任务验证。 |

## 3. 为什么原六模块需要调整

原方案：QueryProcessor、Retriever、Reranker、ContextBuilder、ReasoningController、OutputProcessor。

主要问题：

- **ReasoningController 职责过宽：** 控制“何时继续检索/何时停止”和“如何基于证据推理”是不同功能，应该拆为 ReasoningController 与 AnswerGenerator。
- **OutputProcessor 职责混合：** 格式规范是输出接口问题，事实支持度/覆盖度检查是验证问题。首轮把格式化固定在接口层，把 Verifier 单独作为模块。
- **RRF 与 Cross-Encoder 不完全同类：** 前者是候选列表融合，后者是基于 query-document 对的相关性重评分。首轮暂放在 Ranker 模块内，但分别记录中间结果；若实验发现独立贡献或交互显著，再拆成 Fusion 与 Reranker。
- **Retriever 的实现不等于模块：** BM25、Dense、Hybrid 是 Retriever 内部可替换的实现/算子；首轮不把每个参数或每种实现都定义成顶层模块。

## 4. 首轮比较的三种粒度

| 方案 | 模块粒度 | 目的 |
|---|---|---|
| A：粗粒度（4 模块） | Query & Control；Retrieval & Ranking；Context；Reason & Verify | 检查粗分是否掩盖关键贡献。 |
| B：中粒度（7 模块） | 第 1 节提出的七模块 | 首选候选，兼顾职责边界与实验成本。 |
| C：细粒度（逐边界拆分） | 仅对 B 中有证据显示内部异质性的模块拆分，例如 Retriever→Sparse/Dense、Ranker→Fusion/Reranker、ContextBuilder→Filter/Compression | 检查细分能否增加稳定、可复现的归因信息。不是一次性把所有子操作都拆开。 |

C 方案采用逐边界拆分，而不是把所有子模块一次性展开，避免模块数上升后联盟数指数增长，也避免同时改变多个边界而无法解释结果。

## 5. 如何判断哪种划分更合理

对 A/B/C 使用相同任务样本、知识库、基础模型、提示词预算和评测规则。所有运行记录模块输入、输出、耗时、token、调用次数、证据来源和异常。

评价维度：

1. **干预有效性：** 能否单独替换/关闭该模块，而不破坏其他模块接口？
2. **归因稳定性：** 不同随机种子、任务子集和重复运行下，效应方向及区间是否稳定？
3. **解释增益：** 细分后是否揭示了粗粒度方案无法定位的真实失败来源？
4. **交互可辨识性：** 重要交互能否被识别，是否因模块过细导致估计方差过大？
5. **任务泛化：** 贡献是否只在单一领域成立，还是能跨任务类型复现？
6. **实验成本：** 增加模块后，干预组合、运行成本和分析复杂度增加多少？

不以单一总分决定模块结构。至少同时报告答案正确性/完整性、证据召回与支持度、延迟、token/API 成本，以及模块效应和关键交互效应。

## 6. 实验顺序

**阶段 0：文献与接口定义（当前阶段）**
- 精读 EVOTOOL、RAG Components Are Not Independently Rankable、Investigating Component Contributions。
- 为七个模块写清输入、输出、职责边界、可观测日志和干预接口。

**阶段 1：接入一个固定 LLM 的最小 RAG Harness**
- 不训练模型，不做自动进化。
- 固定模型版本、系统提示、语料、检索库和预算。
- 选取具有可验证答案/证据的知识密集型任务，按任务类型分层。

**阶段 2：先做单模块干预**
- 每个模块至少比较原实现与一个明确替代实现/基线。
- 对关闭模块，优先采用接口兼容的中性替代；若只能 bypass，必须单独标记，不能把它解释成纯功能缺失效应。

**阶段 3：再测关键交互**
- 优先测有机制依据的交互：Retriever×Ranker、Ranker×ContextBuilder、ContextBuilder×AnswerGenerator、ReasoningController×Retriever、AnswerGenerator×Verifier。
- 先做筛查，再对重要交互进行配对确认；不必一开始对真实 LLM 的全部 2^7 联盟做昂贵全因子实验。

**阶段 4：逐边界细分**
- 只有当一个模块内部存在可重复的行为差异，且日志/干预能将其区分时，才拆分该模块。
- 比较拆分前后的归因稳定性、解释增益和成本。

## 7. 当前建议与未决事项

- **建议：** 以 7 模块作为第一套真实 Harness 的候选实现，不把它写成最终架构定案。
- **暂不建议：** 立即接入复杂多 Agent、Critic 自主归因或自动进化；先让固定模型、固定模块、可观测轨迹和受控干预跑通。
- **待确认：** 首个 LLM 选型、知识任务基准、语料与评测集、Verifier 的评测来源、可接受的 API/算力预算。
- **防泄漏：** 模块划分和参数选择只使用开发集/验证集；OOD 测试集保留到方案冻结后做最终泛化验证。

## 8. 参考文献

1. Yang, S. et al. (2026). *EVOTOOL: Self-Evolving Tool-Use Policy Optimization in LLM Agents via Blame-Aware Mutation and Diversity-Aware Selection*. ACL 2026. https://aclanthology.org/2026.acl-long.2016/
2. Kumar, V. P. V. et al. (2026). *RAG Components Are Not Independently Rankable*. Workshop on Document Intelligence and Understanding. https://aclanthology.org/2026.docinsights-1.22/
3. Kim, J. et al. (2026). *Investigating Component Contributions in Multi-Agent ML Systems*. ICML 2026. https://proceedings.mlr.press/v306/kim26ba.html
4. *Retrieval Augmented Generation for Large Language Models: A Methodological Survey* (2026). https://www.mdpi.com/2227-7080/14/11/715
5. 用户提供的科研日志与 Paper List（项目内原始材料），用于梳理研究动机和既有阅读线索。
