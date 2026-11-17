# 单问题模块归因实验

当前实验用于检查一个具体案例中，四个 Harness 模块是否可能成为性能瓶颈，以及单模块修改是否改变答案。它是机制调试工具；单题结果不支持总体效果或稳定因果结论。

## 运行

启动 OpenAI-compatible vLLM 服务后：

```bash
paf attribute-task --config configs/experiments/single_task_attribution.yaml --output data/derived/single-task
```

配置须包含 `task.question`、`task.documents`，建议提供 `task.expected_answer`。`candidates` 中每项必须只包含一个目标模块：

```yaml
candidates:
  - id: concise-answer
    modules:
      EvidenceReasoning:
        answer_prompt: "仅依据证据直接回答；证据不足时说明。"
  - id: broader-retrieval
    modules:
      EvidenceRetrieval:
        top_k_retrieval: 12
```

各模块可修改参数：

- `QueryUnderstanding.system_prompt`
- `EvidenceRetrieval.top_k_retrieval` / `top_k_ranked`
- `EvidencePreparation.max_chars`
- `EvidenceReasoning.reasoning_prompt` / `answer_prompt` / `verifier_prompt`

程序接口还允许通过 `RAGHarness(..., modules={模块名: 实例})` 替换任意模块实现；替换实例需提供相同的 `run` 调用契约。单问题 YAML 目前用于参数/提示词修改。

## 对照与输出

每次执行包含基线、四个逐模块 `bypass` 干预和显式候选。`single_task_attribution.json` 保存结果及模块旁路效应（基线分数减消融分数）；`single_task_runs.jsonl` 按条件保存完整输入、输出、耗时和轨迹。已有 JSON 报告时命令拒绝覆盖，需指定新输出目录。

四模块分别是查询理解、证据召回与排序、证据准备、证据推理与修订。EvidenceReasoning 的充分性判断会控制答案生成是否收到检索上下文，Verifier 未通过时触发一次修订。旁路时前两个模块用输入透传，EvidencePreparation 拼接纯文本，EvidenceReasoning 返回空答案。

## 解读边界

旁路效应测量的是当前实现下移除该步骤的结果变化，不等于模块可改进空间。候选修改与基线在单题上的 token-F1 差异只作为案例证据；LLM 输出存在随机性，且 token-F1 不能衡量证据忠实度、完整性或事实风险。要主张模块贡献或稳定提升，需在多个同域 held-out 任务上重复、配对运行，并预先定义任务指标与选择门槛。
