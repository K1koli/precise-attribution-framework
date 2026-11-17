from __future__ import annotations

import math
from typing import Any, Protocol

from precise_attribution.evolution.metrics import token_f1


class BenchmarkAdapter(Protocol):
    name: str
    metric: str

    def prepare(self, task: dict[str, Any]) -> dict[str, Any]: ...
    def score(self, task: dict[str, Any], result: dict[str, Any]) -> float: ...


class QATokenF1Adapter:
    name = "qa_token_f1"
    metric = "token_f1"
    active_modules = ("QueryUnderstanding", "EvidenceRetrieval", "EvidencePreparation", "EvidenceReasoning")

    def prepare(self, task: dict[str, Any]) -> dict[str, Any]:
        if not task.get("expected_answer"):
            raise ValueError(f"{task.get('id')}: QA task 缺少 expected_answer")
        return dict(task)

    def score(self, task: dict[str, Any], result: dict[str, Any]) -> float:
        return token_f1(str(result.get("answer", "")), str(task["expected_answer"]))


class BrightNDCG10Adapter:
    """Thin adapter for BRIGHT-style ranked evidence tasks, not a full data loader."""
    name = "bright_ndcg10"
    metric = "nDCG@10"
    active_modules = ("QueryUnderstanding", "EvidenceRetrieval")

    def prepare(self, task: dict[str, Any]) -> dict[str, Any]:
        if not task.get("gold_doc_ids"):
            raise ValueError(f"{task.get('id')}: BRIGHT task 缺少 gold_doc_ids")
        if not task.get("documents"):
            raise ValueError(f"{task.get('id')}: BRIGHT task 缺少已适配的 documents 候选列表")
        return dict(task)

    def score(self, task: dict[str, Any], result: dict[str, Any]) -> float:
        gold = set(map(str, task["gold_doc_ids"]))
        ranked = result.get("trace", {}).get("retrieved_documents", [])[:10]
        dcg = sum((1.0 / math.log2(rank + 2)) for rank, doc in enumerate(ranked) if str(doc.get("id")) in gold)
        ideal_n = min(len(gold), 10)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_n))
        return dcg / idcg if idcg else 0.0


class BrightFullCorpusAdapter(BrightNDCG10Adapter):
    name = "bright_full_ndcg10"

    def prepare(self, task):
        if not task.get("gold_doc_ids"):
            raise ValueError("BRIGHT task missing gold_doc_ids")
        return dict(task)


_ADAPTERS: dict[str, BenchmarkAdapter] = {
    "qa_token_f1": QATokenF1Adapter(),
    "bright_ndcg10": BrightNDCG10Adapter(),
    "bright_full_ndcg10": BrightFullCorpusAdapter(),
}


def benchmark_adapter(name: str) -> BenchmarkAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(f"未知 benchmark adapter {name!r}; 可选：{sorted(_ADAPTERS)}") from exc
