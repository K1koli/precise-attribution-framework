# 四模块 Harness 与源码级变异

当前用于验证代码级进化的 Harness 采用四个候选模块。边界是跨 benchmark 的知识处理能力；数据格式转换和指标计算由 Harness 外的 adapter 负责。

| 模块 | 输入 | 输出 | 默认实现概要 |
|---|---|---|---|
| `QueryUnderstanding` | `{question}` | `{queries}` | LLM 提取实体、关系和限定条件；主查询在前，可附加子查询 |
| `EvidenceRetrieval` | `{queries, documents}` | `{documents}` | 对所有 query 召回、合并去重后排序；初始实现仍是轻量确定性基线 |
| `EvidencePreparation` | `{documents}` | `{context}` | 组织和截断证据上下文；当前含文档 ID/标题前缀 |
| `EvidenceReasoning` | `{question, context}` | `{answer, verification, control}` | 判断充分性、基于证据回答、验证并最多修订一次 |

这是首轮实验分解，不是最终模块结构。未来可以根据跨题归因证据把 EvidenceRetrieval 拆成召回/重排，或把 EvidenceReasoning 拆成推理/验证。

## 源码候选流程

`paf attribute-set` 先在 train 集基线轨迹中抽取高分与低分样例，交由 Critic 做全局对照诊断；Critic 提名 1–3 个可疑模块后，系统只对这些模块执行 bypass 消融。Proposer 的目标模块从被提名且有足够配对覆盖率的模块中，按消融效应选择；每次只输出一个 `CandidateModule.run(inputs)` 的完整实现。静态检查或 smoke 失败时会收到一次带错误反馈的重试。可用 `--target-module` 限定 Critic 可诊断的模块范围，配合 `--proposals` 控制候选数。QA30 配置还包含手写的 EvidencePreparation 对照实现，可用 `--proposals 0` 单独评估。

Critic 的输入仅来自 train 集，包含训练题目的标准答案或 gold 文档 ID；它输出的模块与机制判断是待验证假设。定向 bypass 结果会随诊断一起交给 Proposer，最终候选仍由程序化 dev 门控决定。若 Critic 调用失败或没有提出有效模块，本轮不退回全模块消融，也不生成 Agent 候选；报告会记录原因。首版论文来源先验库及其局限见 `docs/human_prior_library.md`；设 `priors.enabled: false` 可关闭先验做对照。

代码级 Proposer 的失败样例只包含问题、系统答案、系统自检结果和分数；不直接附带标准答案或 gold 文档 ID；Critic 的训练集诊断可能引用这些信息。开发集和测试集内容不提供给 Proposer。它仍会看到题目和文档作为运行输入，以便改进模块行为；候选源码另由检查器扫描，拒绝包含训练集答案或文档原文片段的硬编码。

候选依次经过：

1. 源码结构、允许语法、名称/属性白名单及大小限制检查；
2. 单模块接口 smoke 检查；
3. dev 配对评测，默认要求平均增益至少 `0.01`、95% bootstrap CI 下界大于 0、配对覆盖率至少 90%，且候选无运行错误；
4. 按 dev 增益选出最佳合格候选，保存为 `selected_module.py` 并标记接受。test 不参与选择或接受。可在实验 YAML 的 `gate` 中调整 dev 阈值。

源码、硬编码检查或 smoke 校验失败后，Proposer 会收到校验器错误并重写一次；报告记录尝试次数，两次都失败则记录拒绝原因。单任务运行异常记录为 `score: null` 和错误详情，不会伪装成零分。配对统计报告有效配对数与总任务覆盖率；覆盖率低于门槛时，不用该效应自动选变异目标。候选任务运行有异常时不具备进入验收门的资格。

输出目录包含 `multi_task_attribution.json`、Markdown 报告、`candidate_modules/` 源码，以及（通过门槛时）`selected_module.py` 和带源码摘要/评测结果的 `selected_harness.json`。正式结论通过独立 `evaluate-test` 命令产生，不反馈给候选生成或版本门。

候选源码的 loader 在独立模块命名空间中执行，只暴露有限 builtins，并拒绝 import、直接文件访问、任意网络接口、反射、继承、装饰器、while 循环和 dunder 属性；模型只能经限次 `self.llm.chat` 调用。源码允许的名称必须能在受限运行环境中解析。常见字符串/容器方法列入属性白名单；`range` 单次限制 10000 次迭代。候选每题最多调用 LLM 8 次。它不是操作系统级安全沙箱；候选代码仍在当前 Python 进程中执行。Proposer 不能修改 Harness 外的 adapter、scorer 或版本门。

## Benchmark adapter

配置中的 `benchmark` 选择内置 adapter：

- `qa_token_f1`：答案式 QA，任务需有 `question`、`documents`、`expected_answer`；四个模块均纳入归因。
- `bright_ndcg10`：BRIGHT 风格检索评测，适配后的任务需有 `question`、候选 `documents`、`gold_doc_ids`；按二元相关性计算 nDCG@10，只对 QueryUnderstanding 和 EvidenceRetrieval 做归因。当前 adapter 不下载/解析 BRIGHT 原始文件，语料切分和字段转换由外部预处理完成，因此它是对接接口示例，不等同于已完成官方 BRIGHT 复现。

示例配置：

```yaml
benchmark: bright_ndcg10
harness:
  top_k_retrieval: 20
  top_k_ranked: 10
tasks_file: configs/tasks/bright_prepared.yaml
```

同一个通用 Harness 可以被不同 adapter 调用，但单个 benchmark 只为其实际参与评分的模块提供归因证据。要验证知识密集型能力泛化，应在多个领域/任务集的 held-out 上验证，不能把某个检索 benchmark 的 nDCG 变化等同于端到端问答能力提升。

## 2026-11-17：训练 / 开发 / 最终测试分离

此前 dev 同时用于诊断和选择，held-out 用于接受候选。现在默认按固定 seed 划分
60% train / 20% dev / 20% test；比例可由 CLI 调整，每份至少一道题，ID 必须唯一。
30 题示例对应 18 / 6 / 6，仅用于流程验证，正式研究需要更大数据集。

```bash
paf attribute-set --config configs/experiments/qa30_attribution.yaml \
  --output data/derived/qa30-three-way --train-ratio 0.6 --dev-ratio 0.2

# 完成开发、确定最终版本后才执行，不要逐轮查看 test。
paf evaluate-test --config configs/experiments/qa30_attribution.yaml \
  --experiment data/derived/qa30-three-way
```

报告固定三份题目 ID 和完整题库摘要。最终测试验证题库与选定源码摘要，
拒绝覆盖已有 `test_report.json`；测试结果不会修改接受状态。没有候选通过时，测试原始基线。
摘要检查只保证数据一致性，不是访问控制；不同实验目录仍可重复访问同一测试集，
研究流程必须保证测试结果不用于继续调参。旧配置项 `minimum_heldout_gain` 和
`evaluate_all_on_heldout` 已移除；CLI 会明确拒绝带这些旧项的优化配置。
本次变更只适用于 `attribute-set`，旧 `evolve` / `attribute-task` 保持原有实验语义。
多轮累积更新尚未实现；以后每轮应复用相同数据划分。
