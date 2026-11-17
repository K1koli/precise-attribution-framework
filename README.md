# 精确归因框架（Precise Attribution Framework）

**项目：** 归因驱动的知识密集型 Harness 进化

**代码版本：** 0.3.0（BRIGHT 检索 Harness 的归因与代码级进化）
**文档语言：** 中文  
**当前状态：** 研究原型；代码托管于 [GitHub](https://github.com/K1koli/precise-attribution-framework)。当前聚焦单一问题：**归因引导的代码变异能否提升 BRIGHT 检索指标**。管线可复现，但自动提案多数为负向改动，尚不能稳定提分。

## 项目目标

本项目研究如何通过模块化归因和受控干预定位 Harness 的薄弱环节，再结合 RAG 人类先验对目标模块做定向变异，并用 dev 与 held-out 评测筛选版本。当前主线是 BRIGHT Biology 候选池上的检索排序（nDCG@10）；跨领域泛化（economics）与全库检索尚未系统验证。

本项目独立开发。`bright-aegis` / baseline 仅作为未来外部效果对比对象，不作为架构、模块划分、提示词或实验流程模板。

## 当前实现

- 四模块知识检索 Harness（QueryUnderstanding / EvidenceRetrieval / EvidencePreparation / EvidenceReasoning），支持 `bypass` 干预与结构化 trace。
- Critic 对照高分/低分轨迹做诊断，只对被提名模块执行定向消融。
- 先验引导的单模块代码变异：固定外壳（只填 `run` 方法体）、AST 校验、属性/导入白名单、每题 LLM 调用预算。
- 行为差异门：候选在真实题目上与既有实现输出完全一致时判为 no-op，不进入评测。
- dev / held-out 版本门：配对 bootstrap 置信区间 + 覆盖率与错误率检查。
- BRIGHT 预处理候选池上的 nDCG@10 适配器，配套 biology / economics 各 103 题任务集。
- H200 单卡 vLLM + PBS 作业模板（48 小时上限）。

## 快速开始

建议使用 Python 3.12 或 3.13：

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -e '.[dev]'
pytest
paf llm-check --config configs/experiments/bright_biology.yaml
```

多题代码级归因与变异（BRIGHT Biology 示例）：

```bash
paf attribute-set --config configs/experiments/bright_biology.yaml --output data/derived/bright-bio --proposals 2
```

默认先让 Critic 对照 dev 集高分/低分执行轨迹，提名可疑模块；系统只对被提名模块做 bypass 消融，再把诊断与消融结果交给 Proposer 生成单模块 Python 候选。候选经过源码约束检查、契约 smoke 和 dev 对比，再用 held-out 验证。可重复传入 `--target-module` 限定 Critic 的诊断范围。候选源码及被接受的版本保存在输出目录；模块接口和限制见 `docs/code_level_mutation.md`。

实验报告会逐题记录异常和有效配对覆盖率；候选默认需在 dev 与 held-out 上分别达到至少 0.01 的平均增益且 bootstrap 区间下界大于 0。候选模块每题最多调用 LLM 8 次。

仓库包含 `configs/tasks/bright_biology_50.yaml` 和 `configs/tasks/bright_economics_50.yaml` 两个预处理任务集。Biology 文件含 103 道题，每题 50 篇候选文档；候选池由全域语料 BM25 初筛，并强制并入该题全部 gold 文档。因此它用于候选池上的重排实验，**不是全库在线检索结果**，且 gold 入池设置会使评测不同于端到端全库召回。BRIGHT 原始全量语料不包含在仓库中。

`bright_ndcg10` adapter 只评测任务文件所给候选池上的检索排序，不负责下载/索引原始数据。要运行当前 Biology 候选池实验，可使用 `configs/experiments/bright_biology.yaml`；要评估真正的全库检索，需要另行接入全量语料和共享索引，当前流程尚未实现。

## 复现 BRIGHT Biology 候选池实验

仓库已包含预处理任务集 `configs/tasks/bright_biology_50.yaml`（103 题 × 50 篇候选）和对应配置 `configs/experiments/bright_biology.yaml`，**无需下载原始语料**即可复现当前结论。

1. 安装（Python 3.12 / 3.13）：

   ```bash
   python -m venv .venv && source .venv/bin/activate
   python -m pip install -e '.[dev]'
   pytest            # 20 项测试
   ```

2. 启动 OpenAI-compatible 服务（Qwen3.8-27B + vLLM），或修改配置中的 `llm.base_url` 与 `llm.model` 指向你自己的服务。

3. 运行：

   ```bash
   paf attribute-set --config configs/experiments/bright_biology.yaml \
       --output data/derived/bright-bio --proposals 2
   ```

4. 预期结果（单卡 H200，加载约 5 分钟后评测约 5 分钟；dev 69 题 / held-out 34 题）：

   | 项 | 参考值 |
   |---|---|
   | dev 基线 nDCG@10 | ≈ 0.52 |
   | QueryUnderstanding 效应 | ≈ +0.37 |
   | EvidenceRetrieval 效应 | ≈ +0.52 |
   | EvidencePreparation / EvidenceReasoning | 不参与（`bypass_inactive_modules: true`） |
   | 版本门 | 当前提案均为负向或格式非法，`accepted = false` |

5. 产物：`multi_task_attribution.json`（逐题轨迹与得分）、`multi_task_attribution.md`（汇总表）、`candidate_modules/*.py`（提案源码）、通过版本门时另有 `selected_module.py`。

> 复现时不要期待出现被接受的版本：截至当前代码，自动生成的候选在这套任务上均为负向改动，仓库只保证流程与评测可复现，不保证自动进化能提分。

迄今为止唯一一次正向观测（供对照，不是稳定结论）：一个重写 `EvidenceRetrieval` 词面打分的候选在 dev 上 +0.0602（95% CI [−0.014, +0.135]，32 升 / 23 降 / 14 平），在 held-out 上 +0.1106（95% CI [+0.015, +0.213]）。因 dev 区间下界小于 0，版本门未接受该候选。

### 从原始 parquet 重新生成任务集（可选）

生成口径：每题用全库 BM25 取 top-`pool_size`，再**强制并入该题全部 gold chunk**（保证召回上限为 1.0）；`text` 截断 1200 字符；控制字符与孤立代理字符会被清洗。

```bash
python -m pip install pyarrow
python scripts/bright_to_tasks.py --raw-dir data/bright/raw --domain biology \
    --out configs/tasks/bright_biology_50.yaml --pool-size 50
```

原始目录需包含 `documents/<domain>-00000-of-00001.parquet`（字段 `id`,`content`）与 `examples/<domain>-00000-of-00001.parquet`（字段 `query`,`id`,`gold_ids`）。转换完成后检查输出 `stats`：`pool_size_min/max` 应等于（或略大于）`--pool-size`，金标覆盖应为 100%。

若要四个模块各生成一个候选，可传 `--proposals 4` 并重复指定四个 `--target-module`。BRIGHT 风格预处理数据可将配置 `benchmark` 设为 `bright_ndcg10`；该适配器只归因查询理解和证据检索模块。

集群 PBS 模板见 `scripts/run_self_evolve_h200.pbs`；提交前必须设置 `MODEL_PATH` 并按集群环境检查模块加载、虚拟环境和队列资源字段。

不安装命令行入口时，也可运行：

```bash
PYTHONPATH=src python -m precise_attribution.cli attribute-set \
  --config configs/experiments/bright_biology.yaml \
  --output data/derived/bright-bio --proposals 2
```

输出目录包含：

- `multi_task_attribution.json`：逐题轨迹、模块效应、候选与 held-out 对比；
- `multi_task_attribution.md`：人类可读报告；
- `candidate_modules/*.py`：每个候选的源码；
- `selected_module.py`：仅在通过版本门时生成。

> 候选源码在受限命名空间内执行：仅允许 `math` / `re` / `collections` / `heapq` 的子集导入，禁止文件、网络与反射。这是降低误用的防护，不是对抗级安全沙箱。

## 研究边界与限制

- 四模块 Harness 当前仍是候选分解；EvidenceRetrieval 内部采用确定性词项基线，不是完整 BM25/Dense/RRF/Cross-Encoder 实现。
- 候选池只有 50 篇且**强制并入 gold**，因此指标衡量的是池内重排，不能等同于全库检索能力。
- 归因只做单模块 `bypass` 消融，不是全因子分解，也没有交互效应与 Shapley 分配。
- 区间为配对 bootstrap 的近似区间，未做多重比较校正。
- 自动提案的合法率已可通过外壳与白名单保证，但**有效性仍未解决**：多数候选为负向改动。

## 目录

```text
configs/                 实验配置、题库、先验库
src/precise_attribution/ 核心实现
  cli.py                 命令行入口（llm-check / attribute-set）
  harness/modules.py     四模块 Harness、bypass、trace
  evolution/multi_task.py 归因、定向消融、变异与版本门
  evolution/code_mutation.py 候选源码沙箱与外观检查
  evolution/roles.py     Critic / Proposer 角色
  evolution/prior_library.py 人类先验算子库
  evolution/metrics.py   token-F1 指标
  evaluation/benchmarks.py qa_token_f1 / bright_ndcg10 适配器
  llm/                   OpenAI 兼容客户端
scripts/                 BRIGHT 转换器、PBS 模板
tests/                   单元测试
docs/                    设计文档
```

## 设计文档

- `docs/code_level_mutation.md`（四模块接口、源码级变异与 benchmark adapter）
- `docs/human_prior_library.md`（论文来源与先验算子库设计）

设计变更遵循日期化历史原则。未获用户确认的研究方向只能标记为“待讨论”。
