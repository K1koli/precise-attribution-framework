from __future__ import annotations

import os
from typing import Any

from .openai_compatible import OpenAICompatibleClient


def client_from_config(config: dict[str, Any]) -> OpenAICompatibleClient:
    llm = config.get("llm", {})
    # Existing local configurations keep their vLLM behavior.
    provider = str(llm.get("provider", "vllm"))
    if provider not in {"vllm", "deepseek", "openai_compatible"}:
        raise ValueError("llm.provider 必须为 vllm、deepseek 或 openai_compatible")
    is_deepseek = provider == "deepseek"
    key_env = llm.get("api_key_env", "DEEPSEEK_API_KEY" if is_deepseek else None)
    if key_env:
        api_key = os.environ.get(str(key_env), "").strip()
        if not api_key:
            raise ValueError(f"请先设置环境变量 {key_env}，不要将 API Key 写入配置文件")
    elif is_deepseek:
        raise ValueError("DeepSeek 必须通过 api_key_env 从环境变量读取 Key")
    else:
        api_key = str(llm.get("api_key", "EMPTY"))
    disable_thinking = llm.get("disable_thinking")
    if disable_thinking is not None and not isinstance(disable_thinking, bool):
        raise ValueError("llm.disable_thinking 必须是布尔值")
    return OpenAICompatibleClient(
        base_url=str(llm.get("base_url", "https://api.deepseek.com" if is_deepseek else "http://127.0.0.1:8000/v1")),
        model=str(llm.get("model", "deepseek-flash" if is_deepseek else "Qwen3.8-27B")),
        api_key=api_key,
        timeout=float(llm.get("timeout", 180)),
        max_retries=int(llm.get("max_retries", 2)),
        provider=provider,
        disable_thinking=disable_thinking,
    )
