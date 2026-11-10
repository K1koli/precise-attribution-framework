from __future__ import annotations

import re
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

    def run(self, question: str) -> str:
        result = self.llm.chat([
            {"role": "system", "content": "你是查询处理模块。将用户问题改写为适合知识库检索的简洁查询。保留实体、时间、限定条件。只输出查询文本，不回答问题。"},
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

    def run(self, question: str, context: str) -> dict[str, Any]:
        raw = self.llm.chat([
            {"role": "system", "content": '你是推理流程控制模块。判断给定证据是否足以回答问题。只输出 JSON：{"sufficient": true/false, "reason": "简短理由"}。不要编造证据。'},
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

    def run(self, question: str, context: str, answer: str) -> dict[str, Any]:
        raw = self.llm.chat([
            {"role": "system", "content": '你是答案验证模块。仅评估答案是否被给定证据支持、是否回应问题。只输出 JSON：{"supported": true/false, "complete": true/false, "issues": ["..."]}。'},
            {"role": "user", "content": f"问题：{question}\n证据：{context[:7000]}\n答案：{answer}"},
        ], temperature=0.0, max_tokens=220)
        try:
            data = _json_from_text(raw)
            return {"supported": bool(data.get("supported", False)), "complete": bool(data.get("complete", False)), "issues": data.get("issues", [])}
        except Exception:
            return {"supported": False, "complete": False, "issues": ["Verifier 输出无法解析"]}


@dataclass
class HarnessConfig:
    answer_system_prompt: str = "你是严谨的知识问答助手。仅依据提供的证据回答；证据不足时明确说明不确定；不要编造引用。"
    top_k_retrieval: int = 8
    top_k_ranked: int = 5
    context_max_chars: int = 9000
    version: str = "baseline-v1"


@dataclass
class RAGHarness:
    llm: ChatClient
    config: HarnessConfig = field(default_factory=HarnessConfig)

    def run(self, task: dict[str, Any]) -> dict[str, Any]:
        question = str(task["question"])
        documents = list(task.get("documents", []))
        trace: dict[str, Any] = {"module_order": ["QueryProcessor", "Retriever", "Ranker", "ContextBuilder", "ReasoningController", "AnswerGenerator", "Verifier"]}
        query = QueryProcessor(self.llm).run(question)
        trace["QueryProcessor"] = {"query": query}
        candidates = Retriever().run(query, documents, self.config.top_k_retrieval)
        trace["Retriever"] = {"candidate_ids": [d.get("id") for d in candidates], "count": len(candidates)}
        ranked = Ranker().run(query, candidates, self.config.top_k_ranked)
        trace["Ranker"] = {"ranked_ids": [d.get("id") for d in ranked], "scores": [d["rank_score"] for d in ranked]}
        context = ContextBuilder(self.config.context_max_chars).run(ranked)
        trace["ContextBuilder"] = {"context_chars": len(context)}
        control = ReasoningController(self.llm).run(question, context)
        trace["ReasoningController"] = control
        answer = AnswerGenerator(self.llm, self.config.answer_system_prompt).run(question, context)
        trace["AnswerGenerator"] = {"answer_chars": len(answer)}
        verification = Verifier(self.llm).run(question, context, answer)
        trace["Verifier"] = verification
        return {"task_id": task.get("id", "task"), "question": question, "answer": answer, "verification": verification, "trace": trace}
