from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from precise_attribution.harness.modules import _json_from_text


def _chat_json(llm: "ChatClient", messages: list[dict[str, str]], *, retries: int = 1, **kwargs: Any) -> dict[str, Any]:
    """让模型输出 JSON；解析失败就追加一条纠正指令重试一次。

    同一个本地模型既当 Solver 又当 Critic，偶尔会输出 Python 字典或夹带散文，
    直接崩掉整轮演化代价太高（一次冷启动要重载 52G 权重），所以这里做一次重试。
    """
    last_raw = ""
    last_exc: Exception | None = None
    attempt_messages = list(messages)
    for _ in range(retries + 1):
        last_raw = llm.chat(attempt_messages, **kwargs)
        try:
            return _json_from_text(last_raw)
        except ValueError as exc:  # noqa: PERF203
            last_exc = exc
            attempt_messages = attempt_messages + [
                {"role": "user", "content": "上一条输出不是合法 JSON。请只输出一个 JSON 对象：键名和字符串都用双引号，不要代码块、不要解释、不要前后缀文字。"}
            ]
    raise ValueError(f"连续 {retries + 1} 次未得到合法 JSON；最后一次原文前 300 字符：{last_raw[:300]!r}") from last_exc


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


@dataclass
class Proposer:
    llm: ChatClient

    def propose(self, current_prompt: str, failure_summary: str, examples: list[dict[str, Any]]) -> dict[str, Any]:
        data = _chat_json(self.llm, [
            {"role": "system", "content": '你是 Harness Evolve 的 Proposer。根据失败证据提出一个最小、可检验的答案生成系统提示词改动。不得要求访问隐藏测试集、修改评测器或编造事实。只输出 JSON：{"rationale":"...","candidate_prompt":"...","expected_effect":"..."}。'},
            {"role": "user", "content": f"当前提示词：\n{current_prompt}\n\n开发集诊断：\n{failure_summary}\n\n失败样例：\n{examples[:5]}"},
        ], temperature=0.2, max_tokens=700)
        if not isinstance(data.get("candidate_prompt"), str) or not data["candidate_prompt"].strip():
            raise ValueError("Proposer 未提供 candidate_prompt")
        return data


@dataclass
class Critic:
    llm: ChatClient

    def review(self, current_prompt: str, proposal: dict[str, Any]) -> dict[str, Any]:
        data = _chat_json(self.llm, [
            {"role": "system", "content": '你是独立的 Harness Evolve Critic。检查候选提示词是否越权、泄漏评测答案、依赖任务特定硬编码或破坏证据约束。只输出 JSON：{"approve_for_testing":true/false,"risks":["..."],"reason":"..."}。批准只表示允许进入实验，不代表改进有效。'},
            {"role": "user", "content": f"当前提示词：\n{current_prompt}\n\n候选方案：\n{proposal}"},
        ], temperature=0.0, max_tokens=350)
        return {"approve_for_testing": bool(data.get("approve_for_testing", False)), "risks": data.get("risks", []), "reason": str(data.get("reason", ""))}
