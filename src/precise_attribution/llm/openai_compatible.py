from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class LLMError(RuntimeError):
    """Raised when an OpenAI-compatible inference endpoint fails."""


@dataclass
class OpenAICompatibleClient:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen3.8-27B"
    api_key: str = field(default="EMPTY", repr=False)
    timeout: float = 180.0
    max_retries: int = 2
    retry_backoff: float = 1.5
    provider: str = "vllm"
    disable_thinking: bool | None = None

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.provider not in {"vllm", "deepseek", "openai_compatible"}:
            raise ValueError("不支持的 LLM provider")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Provider-specific fields must not leak to other compatible APIs.
        disabled = self.disable_thinking
        if self.provider == "vllm":
            if disabled is None:
                disabled = os.environ.get("PAF_DISABLE_THINKING", "1") == "1"
            if disabled:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif self.provider == "deepseek":
            # Small per-role output budgets require non-thinking mode by default.
            payload["thinking"] = {"type": "disabled" if disabled is not False else "enabled"}
        if response_format:
            payload["response_format"] = response_format
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return str(body["choices"][0]["message"]["content"] or "")
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * (attempt + 1))
        raise LLMError(f"LLM 请求失败（{self.base_url}，模型 {self.model}）：{last_error}") from last_error

    def health(self) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout, 15)) as response:
                data = json.loads(response.read().decode("utf-8"))
            return {"ok": True, "models": [x.get("id") for x in data.get("data", [])]}
        except Exception as exc:  # surface endpoint errors in the CLI report
            return {"ok": False, "error": str(exc)}
