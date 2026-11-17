from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from precise_attribution.harness.modules import HarnessConfig, RAGHarness
from precise_attribution.evaluation.benchmarks import benchmark_adapter


def _config_for_candidate(base: HarnessConfig, changes: dict[str, Any]) -> HarnessConfig:
    """Apply a module-scoped prompt/parameter change while retaining other settings."""
    values = asdict(base)
    module_fields = {
        "QueryUnderstanding": {"system_prompt": "query_system_prompt"},
        "EvidenceRetrieval": {"top_k_retrieval": "top_k_retrieval", "top_k_ranked": "top_k_ranked"},
        "EvidencePreparation": {"max_chars": "context_max_chars"},
        "EvidenceReasoning": {
            "reasoning_prompt": "reasoning_system_prompt",
            "answer_prompt": "answer_system_prompt",
            "verifier_prompt": "verifier_system_prompt",
        },
    }
    unknown = set(changes) - set(module_fields)
    if unknown:
        raise ValueError(f"未知候选模块：{sorted(unknown)}")
    for module_name, params in changes.items():
        if not isinstance(params, dict):
            raise ValueError(f"{module_name} 候选必须是参数对象")
        for key, value in params.items():
            field = module_fields[module_name].get(key)
            if field is None:
                raise ValueError(f"{module_name} 不支持候选参数 {key!r}")
            values[field] = value
    values["version"] = "candidate"
    return HarnessConfig(**values)


def run_single_task_attribution(
    task: dict[str, Any], llm: Any, output_dir: str | Path, *,
    harness_config: dict[str, Any] | None = None,
    candidates: list[dict[str, Any]] | None = None,
    benchmark: str = "qa_token_f1",
) -> dict[str, Any]:
    """Baseline + four one-at-a-time ablations + supplied per-module changes."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "single_task_attribution.json"
    if report_path.exists():
        raise FileExistsError(f"拒绝覆盖已有单问题实验：{report_path}")
    base = HarnessConfig(**(harness_config or {}))
    adapter = benchmark_adapter(benchmark)
    task = adapter.prepare(task)

    def evaluate(config: HarnessConfig, interventions: dict[str, Any] | None = None) -> dict[str, Any]:
        result = RAGHarness(llm, config).run(task, interventions=interventions)
        result["score"] = adapter.score(task, result)
        return result

    baseline = evaluate(base)
    ablations = {}
    for module in adapter.active_modules:
        ablations[module] = evaluate(base, {module: "bypass"})
    for module in set(RAGHarness.MODULES) - set(adapter.active_modules):
        ablations[module] = None
    candidate_results = []
    for candidate in candidates or []:
        module_changes = candidate.get("modules", {})
        if len(module_changes) != 1:
            raise ValueError("每个候选必须只修改一个模块，以便归因比较")
        module = next(iter(module_changes))
        if module not in adapter.active_modules:
            raise ValueError(f"{module} 不参与 {adapter.name} 评分，不能在该 benchmark 上归因候选")
        candidate_result = evaluate(_config_for_candidate(base, module_changes))
        candidate_results.append({
            "id": candidate.get("id", f"candidate-{len(candidate_results) + 1}"),
            "target_module": module,
            "changes": module_changes,
            "score_delta_vs_baseline": (None if baseline["score"] is None else candidate_result["score"] - baseline["score"]),
            "result": candidate_result,
        })
    report = {
        "task_id": task.get("id", "task"), "question": task.get("question"),
        "benchmark": adapter.name,
        "metric": adapter.metric,
        "baseline": baseline,
        "ablations": ablations,
        "module_effects": {
            name: (None if run is None or baseline["score"] is None else baseline["score"] - run["score"])
            for name, run in ablations.items()
        },
        "candidates": candidate_results,
        "interpretation": "单题分数是案例证据，不构成稳定提升或因果贡献的总体估计；消融只测当前模块实例的旁路效应。",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "single_task_runs.jsonl").open("w", encoding="utf-8") as handle:
        recorded = [(f"ablation:{k}", v) for k, v in ablations.items() if v is not None]
        for condition, result in [("baseline", baseline), *recorded, *[(f"candidate:{x['id']}", x["result"]) for x in candidate_results]]:
            handle.write(json.dumps({"condition": condition, **result}, ensure_ascii=False) + "\n")
    return report
