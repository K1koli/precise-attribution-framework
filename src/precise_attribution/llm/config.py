from __future__ import annotations

from typing import Any

from .openai_compatible import OpenAICompatibleClient


def client_from_config(config: dict[str, Any]) -> OpenAICompatibleClient:
    llm = config.get("llm", {})
    return OpenAICompatibleClient(
        base_url=str(llm.get("base_url", "http://127.0.0.1:8000/v1")),
        model=str(llm.get("model", "Qwen3.8-27B")),
        api_key=str(llm.get("api_key", "EMPTY")),
        timeout=float(llm.get("timeout", 180)),
        max_retries=int(llm.get("max_retries", 2)),
    )
