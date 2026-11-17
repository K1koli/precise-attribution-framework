from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


def _json_from_text(text: str) -> dict[str, Any]:
    """从模型输出里取出 JSON 对象。

    模型不总是守规矩：可能裹在 ``` 里、可能用 Python 单引号、可能在 JSON 前面
    先写一段中文散文（里面还带花括号）。这里按“整段 → 平衡花括号片段 → 单引号
    修复 → literal_eval”逐级降级；全都失败时把原文带进异常，方便定位。
    """
    import ast
    import json

    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)

    candidates: list[str] = [raw]
    depth = 0
    start = None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(raw[start : i + 1])
                start = None

    def _quote_bare_keys(t: str) -> str:
        # 把 {foo: 1} / , bar: 2 这类裸键名补上引号
        return re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', t)

    last_err: Exception | None = None
    for cand in candidates:
        variants = (cand, cand.replace("'", '"'), _quote_bare_keys(cand.replace("'", '"')))
        for attempt in variants:
            try:
                value = json.loads(attempt)
                if isinstance(value, dict):
                    return value
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            try:
                value = ast.literal_eval(attempt)
                if isinstance(value, dict):
                    return value
            except Exception:  # noqa: BLE001
                pass

    raise ValueError(f"模型未返回有效 JSON 对象；原文前 400 字符：{raw[:400]!r}") from last_err


@dataclass
class QueryProcessor:
    llm: ChatClient
    system_prompt: str = "你是查询处理模块。将用户问题改写为适合知识库检索的简洁查询。保留实体、时间、限定条件。只输出查询文本，不回答问题。"

    def run(self, question: str) -> str:
        result = self.llm.chat([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ], temperature=0.0, max_tokens=128)
        return result.strip() or question


@dataclass
class Retriever:
    """Dependency-light lexical retrieval baseline; dense retrieval is a later swap-in."""

    def run(self, query: str, documents: list[dict[str, str]], top_k: int = 8) -> list[dict[str, Any]]:
        q = _tokens(query)
        ranked: list[dict[str, Any]] = []
        for doc in documents:
            body = f"{doc.get('title', '')} {doc.get('text', '')}"
            terms = _tokens(body)
            overlap = len(q & terms)
            score = overlap / max(1, len(q))
            ranked.append({**doc, "retrieval_score": score})
        return sorted(ranked, key=lambda x: (x["retrieval_score"], x.get("id", "")), reverse=True)[:top_k]


@dataclass
class Ranker:
    """Initial deterministic fusion/ranking placeholder with observable scores."""

    def run(self, query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        q = _tokens(query)
        scored = []
        for doc in candidates:
            terms = _tokens(f"{doc.get('title', '')} {doc.get('text', '')}")
            coverage = len(q & terms) / max(1, len(q))
            score = 0.7 * float(doc.get("retrieval_score", 0.0)) + 0.3 * coverage
            scored.append({**doc, "rank_score": score})
        return sorted(scored, key=lambda x: (x["rank_score"], x.get("id", "")), reverse=True)[:top_k]


@dataclass
class ContextBuilder:
    max_chars: int = 9000

    def run(self, ranked: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        used = 0
        for doc in ranked:
            block = f"[文档 {doc.get('id', 'unknown')}] {doc.get('title', '')}\n{doc.get('text', '')}".strip()
            remaining = self.max_chars - used
            if remaining <= 0:
                break
            block = block[:remaining]
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)


@dataclass
class ReasoningController:
    llm: ChatClient
    system_prompt: str = '你是推理流程控制模块。判断给定证据是否足以回答问题。只输出 JSON：{"sufficient": true/false, "reason": "简短理由"}。不要编造证据。'

    def run(self, question: str, context: str) -> dict[str, Any]:
        raw = self.llm.chat([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"问题：{question}\n证据：{context[:7000]}"},
        ], temperature=0.0, max_tokens=160)
        try:
            data = _json_from_text(raw)
            return {"sufficient": bool(data.get("sufficient", bool(context.strip()))), "reason": str(data.get("reason", ""))}
        except Exception:
            return {"sufficient": bool(context.strip()), "reason": "控制器输出无法解析，采用有证据则继续的保守策略"}


@dataclass
class AnswerGenerator:
    llm: ChatClient
    system_prompt: str = "你是严谨的知识问答助手。仅依据提供的证据回答；证据不足时明确说明不确定；不要编造引用。"

    def run(self, question: str, context: str) -> str:
        return self.llm.chat([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"问题：{question}\n\n可用证据：\n{context}\n\n请给出简洁、可核查的答案。"},
        ], temperature=0.0, max_tokens=512).strip()


@dataclass
class Verifier:
    llm: ChatClient
    system_prompt: str = '你是答案验证模块。仅评估答案是否被给定证据支持、是否回应问题。只输出 JSON：{"supported": true/false, "complete": true/false, "issues": ["..."]}。'

    def run(self, question: str, context: str, answer: str) -> dict[str, Any]:
        raw = self.llm.chat([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"问题：{question}\n证据：{context[:7000]}\n答案：{answer}"},
        ], temperature=0.0, max_tokens=220)
        try:
            data = _json_from_text(raw)
            return {"supported": bool(data.get("supported", False)), "complete": bool(data.get("complete", False)), "issues": data.get("issues", [])}
        except Exception:
            return {"supported": False, "complete": False, "issues": ["Verifier 输出无法解析"]}


@dataclass
class QueryUnderstanding:
    """Canonical module contract: {question} -> {queries}; main query comes first."""
    llm: ChatClient
    system_prompt: str = "你是查询理解模块。提取问题中的实体、关系与限定条件，生成适合知识检索的查询。只输出查询文本。"

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        query = QueryProcessor(self.llm, self.system_prompt).run(str(inputs["question"]))
        return {"queries": [query]}


@dataclass
class EvidenceRetrieval:
    """Canonical module contract: {queries, documents} -> {documents}."""
    top_k_retrieval: int = 8
    top_k_ranked: int = 5

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        queries = inputs.get("queries") or [inputs.get("query", "")]
        queries = list(dict.fromkeys(str(query).strip() for query in queries if str(query).strip())) or [""]
        documents = list(inputs.get("documents", []))
        merged: dict[str, dict[str, Any]] = {}
        for query in queries:
            retrieved = Retriever().run(query, documents, self.top_k_retrieval)
            for doc in retrieved:
                doc_id = str(doc.get("id", ""))
                if doc_id not in merged:
                    merged[doc_id] = {**doc, "query_hits": 1}
                else:
                    merged[doc_id]["retrieval_score"] = max(
                        float(merged[doc_id].get("retrieval_score", 0.0)),
                        float(doc.get("retrieval_score", 0.0)),
                    )
                    merged[doc_id]["query_hits"] += 1
        for doc in merged.values():
            doc["retrieval_score"] = float(doc.get("retrieval_score", 0.0)) + 0.05 * (doc["query_hits"] - 1)
        joined_query = " ".join(queries)
        ranked = Ranker().run(joined_query, list(merged.values()), self.top_k_ranked)
        return {"documents": ranked}


@dataclass
class EvidencePreparation:
    """Canonical module contract: {documents} -> {context}."""
    max_chars: int = 9000

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"context": ContextBuilder(self.max_chars).run(list(inputs.get("documents", [])))}


@dataclass
class EvidenceReasoning:
    """Canonical module contract: {question, context} -> {answer, verification}."""
    llm: ChatClient
    reasoning_prompt: str
    answer_prompt: str
    verifier_prompt: str

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        question, context = str(inputs["question"]), str(inputs["context"])
        control = ReasoningController(self.llm, self.reasoning_prompt).run(question, context)
        answer_context = context if control.get("sufficient", False) else ""
        answer = AnswerGenerator(self.llm, self.answer_prompt).run(question, answer_context)
        verifier = Verifier(self.llm, self.verifier_prompt)
        verification = verifier.run(question, context, answer)
        if verification.get("supported") is False or verification.get("complete") is False:
            repair_context = f"{answer_context}\n\n初稿：{answer}\n验证问题：{verification.get('issues', [])}\n请仅修正这些问题；若证据不足则明确说明。"
            answer = AnswerGenerator(self.llm, self.answer_prompt).run(question, repair_context)
            verification = verifier.run(question, context, answer)
        return {"answer": answer, "verification": verification, "control": control}


@dataclass
class HarnessConfig:
    query_system_prompt: str = "你是查询理解模块。提取问题中的实体、关系与限定条件，生成适合知识检索的查询。只输出查询文本。"
    answer_system_prompt: str = "你是严谨的知识问答助手。仅依据提供的证据回答；证据不足时明确说明不确定；不要编造引用。"
    reasoning_system_prompt: str = '你是推理流程控制模块。判断给定证据是否足以回答问题。只输出 JSON：{"sufficient": true/false, "reason": "简短理由"}。不要编造证据。'
    verifier_system_prompt: str = '你是答案验证模块。仅评估答案是否被给定证据支持、是否回应问题。只输出 JSON：{"supported": true/false, "complete": true/false, "issues": ["..."]}。'
    top_k_retrieval: int = 8
    top_k_ranked: int = 5
    context_max_chars: int = 9000
    version: str = "baseline-v1"


@dataclass
class RAGHarness:
    llm: ChatClient
    config: HarnessConfig = field(default_factory=HarnessConfig)
    modules: dict[str, Any] = field(default_factory=dict)

    MODULES = ("QueryUnderstanding", "EvidenceRetrieval", "EvidencePreparation", "EvidenceReasoning")

    def _modules(self) -> dict[str, Any]:
        defaults = {
            "QueryUnderstanding": QueryUnderstanding(self.llm, self.config.query_system_prompt),
            "EvidenceRetrieval": EvidenceRetrieval(self.config.top_k_retrieval, self.config.top_k_ranked),
            "EvidencePreparation": EvidencePreparation(self.config.context_max_chars),
            "EvidenceReasoning": EvidenceReasoning(self.llm, self.config.reasoning_system_prompt, self.config.answer_system_prompt, self.config.verifier_system_prompt),
        }
        unknown = set(self.modules) - set(self.MODULES)
        if unknown:
            raise ValueError(f"未知 Harness 模块：{sorted(unknown)}")
        defaults.update(self.modules)
        return defaults

    def run(self, task: dict[str, Any], *, interventions: dict[str, Any] | None = None) -> dict[str, Any]:
        """运行 Harness；interventions 支持 bypass 或将模块实例直接替换为可调用实现。"""
        interventions = interventions or {}
        modules = self._modules()
        # Generated candidates may carry a per-task LLM budget wrapper.
        for module in modules.values():
            reset = getattr(getattr(module, "llm", None), "reset", None)
            if callable(reset):
                reset()
        question = str(task["question"])
        documents = list(task.get("documents", []))
        trace: dict[str, Any] = {"module_order": list(self.MODULES), "modules": {}}

        def step(name: str, inputs: Any, call: Any, bypass: Any) -> Any:
            started = time.perf_counter()
            entry: dict[str, Any] = {"input": inputs}
            intervention = interventions.get(name)
            try:
                if intervention == "bypass":
                    value = bypass()
                    entry["intervention"] = "bypass"
                else:
                    module = intervention if intervention is not None else modules[name]
                    entry["intervention"] = "replacement" if intervention is not None else "active"
                    value = call(module)
                entry["output"] = value
                return value
            except Exception as exc:
                entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
                trace["modules"][name] = entry
                try:
                    setattr(exc, "harness_trace", trace)
                except Exception:  # noqa: BLE001
                    pass
                raise
            finally:
                entry["duration_ms"] = (time.perf_counter() - started) * 1000
                trace["modules"][name] = entry

        query_result = step("QueryUnderstanding", {"question": question}, lambda m: m.run({"question": question}), lambda: {"queries": [question]})
        queries = query_result.get("queries") or [question]
        # 把 k 一并传给模块：否则候选只能猜 k（上一轮有提案硬编码 top-5，直接丢掉一半名额）
        retrieval_inputs = {
            "queries": queries,
            "documents": documents,
            "k": self.config.top_k_ranked,
            "retrieval_k": self.config.top_k_retrieval,
        }
        retrieval_result = step("EvidenceRetrieval", retrieval_inputs, lambda m: m.run(retrieval_inputs), lambda: {"documents": documents[:self.config.top_k_ranked]})
        prepared = step("EvidencePreparation", {"documents": retrieval_result["documents"]}, lambda m: m.run({"documents": retrieval_result["documents"]}), lambda: {"context": "\n\n".join(str(d.get("text", "")) for d in retrieval_result["documents"])[:self.config.context_max_chars]})
        outcome = step("EvidenceReasoning", {"question": question, "context": prepared["context"]}, lambda m: m.run({"question": question, "context": prepared["context"]}), lambda: {"answer": "", "verification": {"supported": False, "complete": False, "issues": ["bypassed"]}, "control": {"sufficient": False, "reason": "bypassed"}})
        trace["query"] = query_result
        trace["retrieved_documents"] = retrieval_result["documents"]
        trace["context"] = prepared["context"]
        return {"task_id": task.get("id", "task"), "question": question, "answer": outcome["answer"], "verification": outcome["verification"], "trace": trace}
