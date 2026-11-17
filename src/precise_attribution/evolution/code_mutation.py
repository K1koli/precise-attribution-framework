"""Constrained single-module source mutation for the knowledge QA Harness."""
from __future__ import annotations

import ast
import json
import textwrap
from typing import Any

from precise_attribution.harness.modules import _json_from_text, _tokens


CONTRACTS = {
    "QueryUnderstanding": "run(inputs: dict) -> dict with queries: list[str]. Input: {question: str}. Put the main query first; additional entries may be focused subqueries.",
    "EvidenceRetrieval": "run(inputs: dict) -> dict with documents: list[dict]. Input: {queries: list[str], documents: list[dict], k: int, retrieval_k: int}. k is how many documents the caller will use — return exactly min(k, len(documents)) of them. Search with every query and merge duplicate documents. Documents should retain id/title/text; optional score fields are allowed.",
    "EvidencePreparation": "run(inputs: dict) -> dict with context: str. Input: {documents: list[dict]}.",
    "EvidenceReasoning": "run(inputs: dict) -> dict with answer: str and verification: dict. Input: {question: str, context: str}. verification should contain supported and complete booleans plus issues list.",
}

_SAFE_BUILTINS = {
    "abs": abs, "bool": bool, "dict": dict, "enumerate": enumerate,
    "float": float, "format": format, "int": int, "isinstance": isinstance,
    "len": len, "list": list, "max": max, "min": min, "range": range,
    "reversed": reversed, "round": round, "set": set, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "all": all, "any": any, "__build_class__": __build_class__,
}
_SAFE_ATTRIBUTES = {
    # 字符串
    "capitalize", "casefold", "center", "count", "endswith", "expandtabs", "find",
    "format", "format_map", "index", "isalnum", "isalpha", "isdigit", "islower",
    "isnumeric", "isspace", "istitle", "isupper", "join", "ljust", "lower",
    "lstrip", "partition", "replace", "rfind", "rjust", "rpartition", "rsplit",
    "rstrip", "split", "splitlines", "startswith", "strip", "swapcase", "title",
    "upper", "zfill",
    # 列表 / 集合 / 字典
    "add", "append", "clear", "copy", "difference", "difference_update", "discard",
    "extend", "get", "insert", "intersection", "intersection_update", "isdisjoint",
    "issubset", "issuperset", "items", "keys", "pop", "remove", "reverse",
    "setdefault", "sort", "symmetric_difference", "union", "update", "values",
    # 框架允许的唯一对象属性
    "chat", "llm",
}


def _bounded_range(*args):
    value = range(*args)
    try:
        length = len(value)
    except OverflowError as exc:
        raise ValueError("候选 range 范围过大") from exc
    if length > 10000:
        raise ValueError("候选单次 range 超过 10000 次迭代预算")
    return value


_SAFE_BUILTINS["range"] = _bounded_range


class BudgetedChatClient:
    """Per-task call limiter used only by generated candidate modules."""

    def __init__(self, client: Any, max_calls: int = 8):
        self.client = client
        self.max_calls = max_calls
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def chat(self, messages, **kwargs):
        if self.calls >= self.max_calls:
            raise RuntimeError(f"候选模块单题 LLM 调用超过预算 {self.max_calls}")
        self.calls += 1
        return self.client.chat(messages, **kwargs)


def validate_source(source: str) -> ast.Module:
    """Accept exactly one plain CandidateModule class with a run method."""
    if len(source) > 12000:
        raise ValueError("候选源码超过 12000 字符复杂度上限")
    tree = ast.parse(source, mode="exec")
    if sum(1 for _ in ast.walk(tree)) > 1500:
        raise ValueError("候选源码超过 AST 节点复杂度上限")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef):
        raise ValueError("候选源码必须只定义一个 CandidateModule 类")
    cls = tree.body[0]
    if cls.name != "CandidateModule" or cls.decorator_list or cls.bases or cls.keywords:
        raise ValueError("候选类必须是无装饰器、无继承的 CandidateModule")
    class_body = list(cls.body)
    if class_body and isinstance(class_body[0], ast.Expr) and isinstance(class_body[0].value, ast.Constant) and isinstance(class_body[0].value.value, str):
        class_body = class_body[1:]
    methods = [node for node in class_body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(methods) != len(class_body) or {method.name for method in methods} != {"run"}:
        raise ValueError("CandidateModule 只能定义一个同步 run 方法")
    run_method = methods[0]
    if isinstance(run_method, ast.AsyncFunctionDef) or run_method.decorator_list or len(run_method.args.args) != 2:
        raise ValueError("run 必须是 def run(self, inputs) 且不能带装饰器")
    nodes = list(ast.walk(tree))
    declared_names = {"_tokens", "_json_from_text", "CandidateModule"}
    declared_names.update(node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))
    declared_names.update(node.arg for node in nodes if isinstance(node, ast.arg))
    declared_names.update(node.name for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    allowed_names = set(_SAFE_BUILTINS) | declared_names
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.Try, ast.While)):
            raise ValueError(f"候选源码包含不支持的语法：{type(node).__name__}")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr not in _SAFE_ATTRIBUTES):
            raise ValueError(f"候选源码使用了不允许的属性：{node.attr}")
        if isinstance(node, ast.Name) and node.id in {"open", "eval", "exec", "compile", "__import__", "__build_class__", "globals", "locals", "vars", "getattr", "setattr", "delattr"}:
            raise ValueError(f"候选源码使用了不允许的名称：{node.id}")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in allowed_names:
            raise ValueError(f"候选源码使用了未开放的名称：{node.id}")
    return tree


_SHELL = """class CandidateModule:
    def run(self, inputs):
{body}
"""


def wrap_body(body: str) -> str:
    """把模型输出的方法体包进固定外壳（措施 1：只填方法体，结构由我们保证）。"""
    lines = textwrap.dedent(body).strip("\n").splitlines()
    indented = "\n".join(("        " + line) if line.strip() else "" for line in lines)
    return _SHELL.format(body=indented)


def load_candidate_module(source: str, llm: Any, *, max_llm_calls: int = 8):
    """Compile a validated candidate with restricted builtins and return its instance.

    This is a constrained loader, not an operating-system security sandbox.
    """
    tree = validate_source(source)
    namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "__name__": "candidate_module",
        "_tokens": _tokens,
        "_json_from_text": _json_from_text,
    }
    exec(compile(tree, "<candidate-module>", "exec"), namespace, namespace)
    instance = namespace["CandidateModule"]()
    instance.llm = BudgetedChatClient(llm, max_calls=max_llm_calls)
    return instance


def propose_code_candidate(llm: Any, *, module: str, current_source: str, failures: list[dict[str, Any]], diagnosis: dict[str, Any] | None = None, training_tasks: list[dict[str, Any]] | None = None, return_attempts: bool = False):
    contract = CONTRACTS[module]
    prompt = f"""你在改进知识密集型 QA Harness 的单个模块：{module}。
接口契约：{contract}
只修改这个模块的算法/实现方法。不能改变输入输出契约，不能把失败样例里的题目、参考答案或文档内容硬编码进候选。
Critic 的轨迹诊断和针对性消融结果是待检验假设，不是已证实因果。请围绕其中的机制与修改目标做最小代码改动；不要扩展到无关模块。如果消融不支持诊断，也应谨慎处理，避免强行制造收益。
候选类必须名为 CandidateModule，且仅实现 def run(self, inputs)。如需调用模型，可使用 self.llm.chat(messages, temperature=0.0, max_tokens=512)。
允许的内建函数：{', '.join(sorted(_SAFE_BUILTINS))}
允许调用的方法名（属性白名单，超出即被拒）：{', '.join(sorted(_SAFE_ATTRIBUTES - {'chat', 'llm'}))}
禁止：import、文件/网络访问、反射、继承、装饰器、异步、异常处理（try）、lambda、while、dunder 属性。
当前实现：
{current_source}
Critic 诊断纲领及定向消融结果：
{diagnosis or {}}
失败样例（只用于开发诊断，不得逐题记忆或硬编码）：
{failures[:5]}
只输出 ``run`` 方法的**方法体**（不要写 class、不要写 def run 这一行），放在一个 python 代码块中。
方法体缩进 4 个空格即可；类外壳由框架负责，会自动包成 ``class CandidateModule``。
不要定义其它函数、类或模块级语句。"""
    feedback = ""
    last_error: Exception | None = None
    for attempt in range(2):
        attempt_prompt = prompt
        if feedback:
            attempt_prompt += (
                "\n\n你上一次提交的候选未通过检查。校验错误如下：\n"
                f"{feedback}\n请针对错误修复，仍只输出一个 CandidateModule 类的完整源码。"
            )
        try:
            raw = llm.chat([
                {"role": "system", "content": "你是单模块代码变异器。严格遵守输入输出契约，只输出完整 CandidateModule 源码。"},
                {"role": "user", "content": attempt_prompt},
            ], temperature=0.2, max_tokens=1600)
            if "```" in raw:
                raw = raw.split("```", 1)[1]
                if raw.lstrip().startswith("python"):
                    raw = raw.lstrip()[6:]
                raw = raw.rsplit("```", 1)[0]
            candidate_text = textwrap.dedent(raw).strip()
            # 兼容旧写法：模型直接给了完整类就用它；否则把方法体包进固定外壳
            source = candidate_text if "class CandidateModule" in candidate_text else wrap_body(candidate_text)
            validate_source(source)
            if training_tasks is not None:
                reject_task_literals(source, training_tasks)
            smoke_candidate(module, load_candidate_module(source, llm))
            return (source, attempt + 1) if return_attempts else source
        except Exception as exc:  # one targeted repair attempt for parse, validation, load, and smoke failures
            last_error = exc
            feedback = f"{type(exc).__name__}: {exc}"
    raise ValueError(f"候选源码在两次生成后仍未通过校验：{feedback}") from last_error


def smoke_candidate(module: str, candidate: Any) -> None:
    samples = {
        "QueryUnderstanding": {"question": "线粒体如何产生能量？"},
        "EvidenceRetrieval": {"queries": ["线粒体 ATP"], "documents": [{"id": "d1", "title": "线粒体", "text": "产生 ATP"}], "k": 5, "retrieval_k": 8},
        "EvidencePreparation": {"documents": [
            {"id": "d1", "title": "线粒体", "text": "线粒体通过氧化磷酸化产生 ATP，为细胞供能。"},
            {"id": "d2", "title": "细胞核", "text": "细胞核储存遗传物质并调控基因表达。"},
            {"id": "d3", "title": "核糖体", "text": "核糖体负责蛋白质合成。"},
        ]},
        "EvidenceReasoning": {"question": "线粒体作用？", "context": "线粒体产生 ATP。"},
    }
    result = candidate.run(samples[module])
    if not isinstance(result, dict):
        raise ValueError("候选 run 必须返回 dict")
    key, expected_type = {
        "QueryUnderstanding": ("queries", list),
        "EvidenceRetrieval": ("documents", list),
        "EvidencePreparation": ("context", str),
        "EvidenceReasoning": ("answer", str),
    }[module]
    if not isinstance(result.get(key), expected_type):
        raise ValueError(f"候选输出字段 {key!r} 必须是 {expected_type.__name__}")
    if module == "QueryUnderstanding" and (not result["queries"] or any(not isinstance(query, str) for query in result["queries"])):
        raise ValueError("QueryUnderstanding.queries 必须是非空字符串列表")
    if module == "EvidenceRetrieval" and any(not isinstance(doc, dict) or "id" not in doc for doc in result["documents"]):
        raise ValueError("EvidenceRetrieval 返回的每篇文档都必须保留 id")
    if module == "EvidenceReasoning" and not isinstance(result.get("verification"), dict):
        raise ValueError("EvidenceReasoning 候选必须返回 verification 对象")
    if module == "EvidenceReasoning":
        verification = result["verification"]
        if not isinstance(verification.get("supported"), bool) or not isinstance(verification.get("complete"), bool) or not isinstance(verification.get("issues"), list):
            raise ValueError("verification 必须包含布尔 supported/complete 和 issues 列表")


def probe_candidate(module: str, candidate: Any, default: Any) -> dict[str, Any]:
    """客观探针：候选是否真的改变了模块输出、是否去掉了文档标记、是否保留标题。"""
    sample = {
        "QueryUnderstanding": {"question": "线粒体如何产生能量？"},
        "EvidenceRetrieval": {"queries": ["线粒体 ATP"], "documents": [{"id": "d1", "title": "线粒体", "text": "产生 ATP"}], "k": 5, "retrieval_k": 8},
        "EvidencePreparation": {"documents": [
            {"id": "d1", "title": "线粒体", "text": "线粒体通过氧化磷酸化产生 ATP，为细胞供能。"},
            {"id": "d2", "title": "细胞核", "text": "细胞核储存遗传物质并调控基因表达。"},
            {"id": "d3", "title": "核糖体", "text": "核糖体负责蛋白质合成。"},
        ]},
        "EvidenceReasoning": {"question": "线粒体作用？", "context": "线粒体产生 ATP。"},
    }[module]
    key = {"QueryUnderstanding": "queries", "EvidenceRetrieval": "documents",
           "EvidencePreparation": "context", "EvidenceReasoning": "answer"}[module]

    def output_of(mod: Any) -> Any:
        try:
            return mod.run(dict(sample)).get(key)
        except Exception:  # noqa: BLE001
            return None

    cand_out, base_out = output_of(candidate), output_of(default)
    cand_text = json.dumps(cand_out, ensure_ascii=False) if not isinstance(cand_out, str) else cand_out
    base_text = json.dumps(base_out, ensure_ascii=False) if not isinstance(base_out, str) else base_out
    titles = [str(d.get("title", "")) for d in sample.get("documents", []) if d.get("title")]
    return {
        "changed_output": cand_text != base_text,
        "strips_marker": ("[文档" in (base_text or "")) and ("[文档" not in (cand_text or "")),
        "keeps_title": any(t in (cand_text or "") for t in titles),
        "output_chars": len(cand_text or ""),
        "llm_calls": getattr(getattr(candidate, "llm", None), "calls", 0),
    }


def probe_with_tasks(module: str, candidate: Any, baseline: Any, tasks: list[dict[str, Any]], *, limit: int = 3) -> dict[str, Any]:
    """措施 3：在真实任务上比较候选与既有实现的输出，完全一致视为 no-op。"""
    samples = []
    for task in tasks[:limit]:
        documents = list(task.get("documents", []))
        if module == "QueryUnderstanding":
            inputs = {"question": str(task.get("question", ""))}
        elif module == "EvidenceRetrieval":
            inputs = {"queries": [str(task.get("question", ""))], "documents": documents[:10], "k": 10, "retrieval_k": 50}
        elif module == "EvidencePreparation":
            inputs = {"documents": documents[:5]}
        else:
            inputs = {"question": str(task.get("question", "")), "context": "placeholder context"}
        def output_of(mod):
            try:
                return mod.run(dict(inputs))
            except Exception as exc:  # noqa: BLE001
                return {"__error__": f"{type(exc).__name__}: {exc}"}
        cand_out, base_out = output_of(candidate), output_of(baseline)
        samples.append({
            "task_id": task.get("id"),
            "changed": json.dumps(cand_out, ensure_ascii=False, sort_keys=True) != json.dumps(base_out, ensure_ascii=False, sort_keys=True),
        })
    changed = sum(1 for item in samples if item["changed"])
    return {"changed": changed > 0, "changed_count": changed, "probes": samples}


def reject_task_literals(source: str, tasks: list[dict[str, Any]]) -> None:
    """Reject obvious question/answer/document memorization in candidate literals."""
    literals = []
    for task in tasks:
        for key in ("question", "expected_answer"):
            value = task.get(key)
            if isinstance(value, str) and len(value.strip()) >= (10 if key == "question" else 3):
                literals.append(value.strip())
        for doc in task.get("documents", []):
            for key in ("id", "title", "text"):
                value = doc.get(key)
                if isinstance(value, str) and len(value.strip()) >= (3 if key == "id" else 5 if key == "title" else 18):
                    literals.append(value.strip())
        literals.extend(str(value) for value in task.get("gold_doc_ids", []) if len(str(value)) >= 3)
    for literal in literals:
        if literal and literal in source:
            raise ValueError("候选源码包含 训练集问题、参考答案或文档的原文片段，疑似任务硬编码")


def default_source(module: str) -> str:
    """Compact source exemplars used as the mutation prompt's baseline."""
    return _DEFAULT_SOURCES[module]


_DEFAULT_SOURCES = {
    "QueryUnderstanding": '''class CandidateModule:
    def run(self, inputs):
        question = str(inputs["question"])
        query = self.llm.chat([
            {"role": "system", "content": "提取实体、关系和限定条件，改写为简洁检索查询。只输出查询。"},
            {"role": "user", "content": question},
        ], temperature=0.0, max_tokens=128).strip()
        query = query or question
        return {"queries": [query]}
''',
    "EvidenceRetrieval": '''class CandidateModule:
    def run(self, inputs):
        queries = inputs.get("queries") or [inputs.get("query", "")]
        unique_queries = []
        for query in queries:
            query = str(query).strip()
            if query and query not in unique_queries:
                unique_queries.append(query)
        if not unique_queries:
            unique_queries = [""]
        merged = {}
        for query in unique_queries:
            q = _tokens(query)
            for doc in inputs.get("documents", []):
                terms = _tokens(str(doc.get("title", "")) + " " + str(doc.get("text", "")))
                score = len(q & terms) / max(1, len(q))
                doc_id = str(doc.get("id", ""))
                if doc_id not in merged:
                    merged[doc_id] = {**doc, "retrieval_score": score, "query_hits": 1}
                else:
                    merged[doc_id]["retrieval_score"] = max(merged[doc_id]["retrieval_score"], score)
                    merged[doc_id]["query_hits"] += 1
        for doc in merged.values():
            doc["retrieval_score"] += 0.05 * (doc["query_hits"] - 1)
        ranked = []
        joined_query = " ".join(unique_queries)
        for doc in merged.values():
            terms = _tokens(str(doc.get("title", "")) + " " + str(doc.get("text", "")))
            q = _tokens(joined_query)
            coverage = len(q & terms) / max(1, len(q))
            rank_score = 0.7 * float(doc.get("retrieval_score", 0.0)) + 0.3 * coverage
            ranked.append((rank_score, str(doc.get("id", "")), {**doc, "rank_score": rank_score}))
        ranked.sort(reverse=True)
        k = int(inputs.get("k", 5))
        return {"documents": [row[2] for row in ranked[:k]]}
''',
    "EvidencePreparation": '''class CandidateModule:
    def run(self, inputs):
        blocks = []
        for doc in inputs.get("documents", []):
            text = ("[文档 " + str(doc.get("id", "unknown")) + "] " + str(doc.get("title", "")) + "\\n" + str(doc.get("text", ""))).strip()
            if text:
                blocks.append(text)
        return {"context": "\\n\\n".join(blocks)[:9000]}
''',
    "EvidenceReasoning": '''class CandidateModule:
    def run(self, inputs):
        question, context = str(inputs["question"]), str(inputs["context"])
        control_raw = self.llm.chat([
            {"role": "system", "content": "判断证据是否足以回答。只输出 JSON：{\\\"sufficient\\\":true/false,\\\"reason\\\":\\\"\\\"}"},
            {"role": "user", "content": "问题：" + question + "\\n证据：" + context},
        ], temperature=0.0, max_tokens=160)
        control = _json_from_text(control_raw)
        answer_context = context if control.get("sufficient", bool(context.strip())) else ""
        answer = self.llm.chat([
            {"role": "system", "content": "仅依据证据回答；证据不足时说明不确定。"},
            {"role": "user", "content": "问题：" + question + "\\n证据：" + answer_context},
        ], temperature=0.0, max_tokens=512).strip()
        verification = _json_from_text(self.llm.chat([
            {"role": "system", "content": "只判断答案是否有证据支持且完整，输出 JSON。"},
            {"role": "user", "content": "问题：" + question + "\\n证据：" + context + "\\n答案：" + answer},
        ], temperature=0.0, max_tokens=220))
        if verification.get("supported") is False or verification.get("complete") is False:
            answer = self.llm.chat([
                {"role": "system", "content": "根据验证问题修正答案；证据不足时说明。"},
                {"role": "user", "content": "证据：" + answer_context + "\\n初稿：" + answer + "\\n问题：" + str(verification.get("issues", []))},
            ], temperature=0.0, max_tokens=512).strip()
            verification = _json_from_text(self.llm.chat([
                {"role": "system", "content": "复核修订答案是否有证据支持且完整，输出 JSON。"},
                {"role": "user", "content": "问题：" + question + "\\n证据：" + context + "\\n答案：" + answer},
            ], temperature=0.0, max_tokens=220))
        return {"answer": answer, "verification": verification}
''',
}
