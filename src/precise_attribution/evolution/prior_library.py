"""Load and prepare research-derived human prior operators for RAG evolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def default_prior_library_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "priors" / "rag_operators.yaml"


def load_prior_library(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else default_prior_library_path()
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("operators"), list):
        raise ValueError(f"先验库格式错误：{source}")
    ids = [operator.get("id") for operator in data["operators"] if isinstance(operator, dict)]
    if len(ids) != len(data["operators"]) or any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("先验库每个算子必须是对象，且 id 非空、不重复")
    for operator in data["operators"]:
        if not isinstance(operator.get("critic"), dict) or not isinstance(operator.get("proposer"), dict):
            raise ValueError(f"算子 {operator['id']} 缺少 Critic/Proposer 视角")
        if not isinstance(operator.get("sources"), list) or not operator["sources"]:
            raise ValueError(f"算子 {operator['id']} 缺少可追溯论文来源")
    return data


def available_prior_operators(library: dict[str, Any], modules: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    allowed = set(modules)
    return [
        operator for operator in library["operators"]
        if operator.get("implementation_status") in {"available", "available_within_module"}
        and allowed.intersection(operator.get("modules", []))
    ]


def critic_prior_cards(operators: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards = []
    for operator in operators:
        cards.append({
            "id": operator["id"],
            "modules": operator.get("modules", []),
            "title": operator.get("title", ""),
            "evidence_summary": operator.get("evidence_summary", ""),
            "diagnostic_signals": operator["critic"].get("diagnostic_signals", []),
            "falsification_checks": operator["critic"].get("falsification_checks", []),
            "sources": operator.get("sources", []),
        })
    return cards


def proposer_priors(operators: list[dict[str, Any]], selected_ids: list[str]) -> list[dict[str, Any]]:
    wanted = set(selected_ids)
    return [
        {
            "id": operator["id"],
            "title": operator.get("title", ""),
            "implementation_guidance": operator["proposer"].get("implementation_guidance", []),
            "invariants": operator["proposer"].get("invariants", []),
            "risks": operator["proposer"].get("risks", []),
            "sources": operator.get("sources", []),
        }
        for operator in operators if operator.get("id") in wanted
    ]
