from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class LLMError(RuntimeError):
    """Raised when an OpenAI-compatible inference endpoint fails."""


@dataclass
class OpenAICompatibleClient:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen3.8-27B"
    api_key: str = "EMPTY"
    timeout: float = 180.0
    max_retries: int = 2
    retry_backoff: float = 1.5

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

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
        # Qwen3.8 的 chat template 默认开思考：思考内容会吃满 max_tokens，导致
        # content 变成空串（JSON 角色直接失败）。本地 vLLM 支持请求级
        # chat_template_kwargs，用它把思考关掉；PAF_DISABLE_THINKING=0 可复原。
        if os.environ.get("PAF_DISABLE_THINKING", "1") == "1":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
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
