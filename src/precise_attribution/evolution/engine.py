from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from precise_attribution.evolution.metrics import token_f1
from precise_attribution.evolution.roles import Critic, Proposer
from precise_attribution.harness.modules import HarnessConfig, RAGHarness


def _evaluate(llm: Any, tasks: list[dict[str, Any]], prompt: str, version: str) -> tuple[float, list[dict[str, Any]]]:
    harness = RAGHarness(llm, HarnessConfig(answer_system_prompt=prompt, version=version))
    results = []
    for task in tasks:
        result = harness.run(task)
        score = token_f1(result["answer"], str(task["expected_answer"]))
        results.append({"task_id": task.get("id"), "score": score, "answer": result["answer"], "verification": result["verification"], "trace": result["trace"]})
    return (sum(x["score"] for x in results) / max(1, len(results))), results


def run_self_evolve(config: dict[str, Any], llm: Any, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    current_prompt = str(config["harness"]["answer_system_prompt"])
    tasks = list(config["tasks"]["development"])
    if not tasks:
        raise ValueError("tasks.development 不能为空")
    baseline_score, baseline_runs = _evaluate(llm, tasks, current_prompt, "baseline-v1")
    failures = [x for x in baseline_runs if x["score"] < float(config.get("evolution", {}).get("failure_threshold", 0.8))]
    failure_summary = json.dumps({"mean_token_f1": baseline_score, "failed_count": len(failures), "task_count": len(tasks)}, ensure_ascii=False)
    examples = [{"task_id": x["task_id"], "answer": x["answer"], "score": x["score"]} for x in failures[:5]]

    proposer = Proposer(llm)
    critic = Critic(llm)
    proposal = proposer.propose(current_prompt, failure_summary, examples)
    review = critic.review(current_prompt, proposal)

    candidate_score = None
    candidate_runs: list[dict[str, Any]] = []
    accepted = False
    minimum_gain = float(config.get("evolution", {}).get("minimum_gain", 0.01))
    if review["approve_for_testing"]:
        candidate_score, candidate_runs = _evaluate(llm, tasks, proposal["candidate_prompt"], "candidate-v1")
        accepted = candidate_score >= baseline_score + minimum_gain
    selected_prompt = proposal["candidate_prompt"] if accepted else current_prompt

    # Optional OOD evaluation is reporting-only; its result never affects acceptance.
    ood_results = None
    if config.get("tasks", {}).get("ood"):
        ood_score, ood_runs = _evaluate(llm, list(config["tasks"]["ood"]), selected_prompt, "ood-report-only")
        ood_results = {"mean_token_f1": ood_score, "runs": ood_runs, "used_for_acceptance": False}

    report = {
        "model": config.get("llm", {}).get("model"),
        "roles": {"Solver": "RAGHarness", "Proposer": "prompt proposal", "Critic": "proposal review"},
        "baseline": {"mean_token_f1": baseline_score, "runs": baseline_runs},
        "proposal": proposal,
        "critic": review,
        "candidate": {"mean_token_f1": candidate_score, "runs": candidate_runs},
        "acceptance": {"accepted": accepted, "minimum_gain": minimum_gain, "reason": "候选开发集分数达到预设提升阈值" if accepted else "未通过 Critic 或未达到预设开发集提升阈值"},
        "selected_prompt": selected_prompt,
        "ood_report_only": ood_results,
        "caveats": [
            "当前是单轮提示词级 Self-Evolve MVP，不会修改 Python 模块代码或检索算法。",
            "Critic 只决定候选是否进入测试；是否接受由预先指定的程序化指标决定。",
            "token-F1 是轻量代理指标，正式研究需替换/补充任务适配的 EM、证据支持度和人工抽检。",
            "同一模型承担三个角色，角色隔离不等于统计独立。",
        ],
    }
    (output / "self_evolve_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "selected_prompt.txt").write_text(selected_prompt, encoding="utf-8")
    (output / "self_evolve_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict[str, Any]) -> str:
    cand = report["candidate"]["mean_token_f1"]
    cand_text = "未运行" if cand is None else f"{cand:.4f}"
    lines = [
        "# Self-Evolve 单轮实验报告", "",
        f"- 模型：`{report['model']}`",
        f"- Solver 开发集 token-F1：{report['baseline']['mean_token_f1']:.4f}",
        f"- Critic 允许测试：{report['critic']['approve_for_testing']}",
        f"- Candidate 开发集 token-F1：{cand_text}",
        f"- 是否接受：**{report['acceptance']['accepted']}**",
        f"- 判定原因：{report['acceptance']['reason']}", "",
        "## Proposer 理由", str(report["proposal"].get("rationale", "")), "",
        "## Critic 意见", str(report["critic"].get("reason", "")), "",
        "## 限制", *[f"- {x}" for x in report["caveats"]], "",
    ]
    if report.get("ood_report_only"):
        lines += [f"OOD 报告分数（不参与接受决策）：{report['ood_report_only']['mean_token_f1']:.4f}", ""]
    return "\n".join(lines)
