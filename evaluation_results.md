
## Biology test 评估

- 固定划分：seed=42，train/dev/test = 41/31/31；本次仅评估 test 31 题。

### Test 结果

**BRIGHT / Biology（test，31 题）**

| 方法 | Biology | |
|---|---:|---:|
| | **nDCG@10（0–1）↑** | **RAG acc（0–1）↑** |
| Harness 0 | 0.10647923 | 未评估 |

- [本次逐题结果与检索轨迹](data/derived/biology-bm25-seed42-114001.gaas/test_report.json)，PBS 作业：114001.gaas；计算节点执行，无运行错误。

- nDCG@10 为逐题检索分数的平均值，使用 0–1 尺度，不是回答准确率。
- RAG acc 指以检索结果作为证据回答原始 query 后，回答正确的题数占评估题数的比例。当前未运行答案生成与正确性判定，因此标为“未评估”；不能从 nDCG@10 推算。运行前需固定答案生成配置、参考答案及正确性判定规则。
### Harness 0：原始 Query + 全库 BM25

| 模块 | 初始功能 |
|---|---|
| QueryUnderstanding | 保留模块，原样返回问题，不调用 LLM 改写。 |
| EvidenceRetrieval | 用原始问题在 Biology 全部 57,359 篇文档的完整正文中执行 BM25，返回 top-10；遵守 excluded_ids，不注入 gold。 |
| EvidencePreparation | 旁路默认模块；仅将检索结果拼接为截断的轨迹上下文，不影响检索排名或评分。 |
| EvidenceReasoning | 旁路，不调用 LLM 生成答案或验证答案。 |

BM25 参数：k1=1.2、b=0.75；小写英文字母/数字分词，无停用词过滤或词干化，查询词去重，同分按文档 ID 升序排列。gold_doc_ids 仅用于评分。
