"""训练集诊断与生成、开发集选择、独立测试集最终评测。

候选只修改一个模块；测试结果不参与版本接受。
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from precise_attribution.harness.modules import HarnessConfig, RAGHarness
from precise_attribution.evolution.code_mutation import (
    default_source, load_candidate_module, probe_candidate, probe_with_tasks, propose_code_candidate,
    reject_task_literals, smoke_candidate,
)
from precise_attribution.evolution.roles import Critic
from precise_attribution.evolution.prior_library import (
    available_prior_operators, critic_prior_cards, load_prior_library, proposer_priors,
)
from precise_attribution.evaluation.benchmarks import benchmark_adapter


def split_tasks(tasks: list[dict[str, Any]], *, train_ratio: float = 0.6, dev_ratio: float = 0.2, seed: int = 42):
    """Deterministic, disjoint train/dev/test split with nonempty partitions."""
    if len(tasks) < 3:
        raise ValueError("至少需要 3 道题才能划分 train/dev/test")
    ids = [task.get("id") for task in tasks]
    if any(not isinstance(task_id, str) or not task_id for task_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("每道题必须有非空且唯一的字符串 id")
    if not (0 < train_ratio < 1 and 0 < dev_ratio < 1 and train_ratio + dev_ratio < 1):
        raise ValueError("train_ratio 和 dev_ratio 必须大于 0，且两者之和小于 1")
    order = list(range(len(tasks)))
    random.Random(seed).shuffle(order)
    n_train = max(1, min(len(tasks) - 2, round(len(tasks) * train_ratio)))
    n_dev = max(1, min(len(tasks) - n_train - 1, round(len(tasks) * dev_ratio)))
    return tuple(
        [tasks[i] for i in sorted(indices)]
        for indices in (order[:n_train], order[n_train:n_train + n_dev], order[n_train + n_dev:])
    )


def prepared_groups(tasks, split):
    """Validate and resolve a prepared partition without reshuffling."""
    if not isinstance(split, dict) or type(split.get("seed")) is not int:
        raise ValueError("tasks.json 缺少有效的 split/seed；请重新 prepare")
    ids = [t.get("id") for t in tasks]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("题目 ID 必须唯一且非空")
    groups = [split.get(k) for k in ("train", "dev", "test")]
    if any(not isinstance(g, list) or not g for g in groups):
        raise ValueError("train/dev/test 划分必须为非空列表")
    flat = [i for g in groups for i in g]
    if any(not isinstance(i, str) for i in flat) or len(set(flat)) != len(flat) or set(flat) != set(ids):
        raise ValueError("划分必须互斥且完整覆盖题目 ID")
    by_id = {t["id"]: t for t in tasks}
    return tuple([by_id[i] for i in g] for g in groups)


def task_fingerprint(tasks: list[dict[str, Any]]) -> str:
    payload = json.dumps(tasks, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate_set(tasks, llm, config: HarnessConfig, adapter, interventions: dict[str, Any] | None = None, modules: dict[str, Any] | None = None, bypass_modules: Iterable[str] = ()) -> list[dict[str, Any]]:
    """评测一组任务；``bypass_modules`` 里的模块默认走 bypass（可被 interventions 覆盖）。"""
    default_interventions = {module: "bypass" for module in bypass_modules}
    rows = []
    for task in tasks:
        try:
            merged = {**default_interventions, **(interventions or {})}
            result = RAGHarness(llm, config, modules or {}).run(task, interventions=merged or None)
            score = adapter.score(task, result)
            rows.append({
                "task_id": task.get("id"), "question": task.get("question"), "score": score,
                "answer": result.get("answer"), "verification": result.get("verification"),
                "context_head": str(result.get("trace", {}).get("context", ""))[:600],
                "trace": result.get("trace"),
                "error": None,
            })
        except Exception as exc:  # isolate one failed task; never convert infra failure into a zero score
            rows.append({
                "task_id": task.get("id"), "question": task.get("question"), "score": None, "answer": None,
                "verification": None, "context_head": None,
                "trace": getattr(exc, "harness_trace", None),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            })
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def paired_bootstrap(a: dict[str, float], b: dict[str, float], *, n_boot: int = 5000, seed: int = 42, minimum_coverage: float = 0.0, total_tasks: int | None = None):
    """对共享 task_id 求 mean(b - a) 及其 95% 百分位区间。"""
    keys = sorted(set(a) & set(b))
    if not keys:
        n_possible = total_tasks if total_tasks is not None else max(len(a), len(b))
        return {"n": 0, "n_possible": n_possible, "coverage": 0.0, "mean": None, "raw_mean": None, "ci95": [None, None]}
    diffs = [b[k] - a[k] for k in keys]
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    raw_mean = _mean(diffs)
    n_possible = total_tasks if total_tasks is not None else max(len(a), len(b), 1)
    coverage = n / max(n_possible, 1)
    return {
        "n": n,
        "n_possible": n_possible,
        "coverage": coverage,
        "mean": raw_mean if coverage >= minimum_coverage else None,
        "raw_mean": raw_mean,
        "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]],
    }


def _as_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {r["task_id"]: r["score"] for r in rows if r["score"] is not None}


def _diagnostic_examples(rows: list[dict[str, Any]], tasks: list[dict[str, Any]], *, limit_per_class: int = 4) -> list[dict[str, Any]]:
    """Expose training labels only to the diagnostic role."""
    tasks_by_id = {task["id"]: task for task in tasks}
    successful = sorted(
        (row for row in rows if row.get("score") == 1.0),
        key=lambda row: str(row.get("task_id", "")),
    )[:limit_per_class]
    failed = sorted(
        (row for row in rows if row.get("score") is not None and row["score"] < 1.0),
        key=lambda row: (row["score"], str(row.get("task_id", ""))),
    )[:limit_per_class]
    failed.extend(row for row in rows if row.get("error") is not None)
    examples = []
    for label, group in (("success", successful), ("failure", failed[:limit_per_class])):
        for row in group:
            trace = row.get("trace") or {}
            modules = {}
            for name, entry in trace.get("modules", {}).items():
                modules[name] = {
                    "input": str(entry.get("input", ""))[:700],
                    "output": str(entry.get("output", ""))[:1000],
                    "duration_ms": entry.get("duration_ms"),
                    "error": entry.get("error"),
                    "intervention": entry.get("intervention"),
                }
            examples.append({
                "outcome_group": label,
                "reference": {key: tasks_by_id[row["task_id"]][key] for key in ("expected_answer", "gold_doc_ids") if key in tasks_by_id[row["task_id"]]},
                "task_id": row.get("task_id"),
                "question": row.get("question"),
                "score": row.get("score"),
                "answer": row.get("answer"),
                "verification": row.get("verification"),
                "error": row.get("error"),
                "modules": modules,
            })
    return examples


def run_multi_task_attribution(
    tasks: list[dict[str, Any]],
    llm,
    output_dir: str | Path,
    *,
    harness_config: dict[str, Any] | None = None,
    candidates: list[dict[str, Any]] | None = None,
    train_ratio: float = 0.6,
    dev_ratio: float = 0.2,
    seed: int = 42,
    max_proposals: int = 2,
    target_modules: list[str] | None = None,
    benchmark: str = "qa_token_f1",
    minimum_dev_gain: float = 0.01,
    minimum_pair_coverage: float = 0.90,
    require_positive_ci: bool = True,
    prior_library: dict[str, Any] | None = None,
    bypass_inactive_modules: bool = False,
    history_reports: list[str] | None = None,
    prepared_split: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(tasks) < 3:
        raise ValueError("多题归因至少需要 3 道题")
    if max_proposals < 0:
        raise ValueError("max_proposals 不能小于 0")
    if minimum_pair_coverage < 0 or minimum_pair_coverage > 1:
        raise ValueError("minimum_pair_coverage 必须在 [0, 1] 范围内")
    if minimum_dev_gain < 0:
        raise ValueError("提升门槛不能小于 0")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "multi_task_attribution.json"
    if report_path.exists():
        raise FileExistsError(f"拒绝覆盖已有实验：{report_path}")

    base = HarnessConfig(**(harness_config or {}))
    if benchmark == "bright_full_ndcg10":
        from precise_attribution.harness.bm25 import corpus_index
        if base.retrieval_backend != "bm25" or base.query_mode != "identity" or base.top_k_ranked != 10:
            raise ValueError("Full BRIGHT requires identity query, BM25 and top-10 baseline")
        corpus_index(base.corpus_file, base.corpus_sha256)
    adapter = benchmark_adapter(benchmark)
    dataset_sha256 = task_fingerprint(tasks)
    if prepared_split is not None:
        train, dev, test = prepared_groups(tasks, prepared_split)
        seed = prepared_split["seed"]
    elif benchmark == "bright_full_ndcg10":
        raise ValueError("Full BRIGHT requires prepared split in tasks.json")
    else:
        train, dev, test = split_tasks(tasks, train_ratio=train_ratio, dev_ratio=dev_ratio, seed=seed)
    train = [adapter.prepare(task) for task in train]
    dev = [adapter.prepare(task) for task in dev]
    targets = target_modules or list(adapter.active_modules)
    invalid_targets = set(targets) - set(adapter.active_modules)
    if invalid_targets:
        raise ValueError(f"当前 benchmark 不支持归因/变异这些模块：{sorted(invalid_targets)}")
    # 检索型 benchmark 只给检索打分；把不参与计分的模块旁路掉，省掉无意义的 LLM 调用
    bypassed = (
        [module for module in RAGHarness.MODULES if module not in adapter.active_modules]
        if bypass_inactive_modules else []
    )

    from .history import load_history
    history = load_history(history_reports or [], dataset_sha256=dataset_sha256,
                           split={"train": [t["id"] for t in train], "dev": [t["id"] for t in dev],
                                  "test": [t["id"] for t in test], "seed": seed},
                           benchmark=benchmark, harness_config=asdict(base))
    baseline_train = evaluate_set(train, llm, base, adapter, bypass_modules=bypassed)
    train_scores = _as_scores(baseline_train)

    # Critic first compares successful and failed baseline traces from train only.
    # Explicit --target-module values constrain the diagnostic scope; they do not
    # bypass the diagnosis or force an ablation.
    diagnostic_scope = list(target_modules or adapter.active_modules)
    prior_library = prior_library if prior_library is not None else load_prior_library()
    active_priors = available_prior_operators(prior_library, diagnostic_scope)
    diagnostic_prior_cards = critic_prior_cards(active_priors)
    critic_diagnosis: dict[str, Any]
    try:
        critic_diagnosis = {
            "status": "completed",
            **Critic(llm).diagnose_modules(
                _diagnostic_examples(baseline_train, train), diagnostic_scope, diagnostic_prior_cards,
                **({"history": history} if history else {}),
            ),
        }
    except Exception as exc:  # keep the experiment reportable if the diagnostic call fails
        critic_diagnosis = {
            "status": "failed",
            "summary": "Critic 诊断失败；为避免退回全模块盲目消融，本轮不执行模块消融或 Agent 代码提案。",
            "hypotheses": [],
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    suspected_modules = [
        item["module"] for item in critic_diagnosis.get("hypotheses", [])
        if item.get("module") in adapter.active_modules
    ]
    ablations: dict[str, list[dict[str, Any]]] = {}
    effects: dict[str, Any] = {}
    for module in suspected_modules:
        rows = evaluate_set(train, llm, base, adapter, {module: "bypass"}, bypass_modules=bypassed)
        ablations[module] = rows
        effects[module] = paired_bootstrap(_as_scores(rows), train_scores, seed=seed, minimum_coverage=minimum_pair_coverage, total_tasks=len(train))

    for module in set(RAGHarness.MODULES) - set(adapter.active_modules):
        effects[module] = {"n": 0, "mean": None, "ci95": [None, None], "status": "not_applicable_for_benchmark"}
    for module in set(adapter.active_modules) - set(suspected_modules):
        effects[module] = {"n": 0, "mean": None, "ci95": [None, None], "status": "not_selected_by_critic"}
    estimable_modules = [m for m in suspected_modules if effects[m]["mean"] is not None]
    weakest = min(estimable_modules, key=lambda m: effects[m]["mean"]) if estimable_modules else None
    # Direct proposer examples contain training inputs and behavior only.
    # Critic diagnoses may refer to training labels; no dev/test feedback is included.
    failures = [
        {
            "task_id": r["task_id"],
            "question": r.get("question", ""),
            "answer": r["answer"],
            "score": r["score"],
            "verification": r.get("verification"),
        }
        for r in baseline_train if r["score"] is not None and r["score"] < 1.0
    ]

    hypothesis_by_module = {
        item["module"]: item for item in critic_diagnosis.get("hypotheses", [])
        if item.get("module") in suspected_modules
    }
    ranked_targets = sorted(
        (module for module in suspected_modules if effects[module].get("mean") is not None),
        key=lambda module: effects[module]["mean"],
    )
    proposed: list[dict[str, Any]] = []
    for i in range(max_proposals):
        target = ranked_targets[i % len(ranked_targets)] if ranked_targets else None
        if target is None:
            proposed.append({"id": f"proposal-{i + 1}", "target_module": None, "status": "skipped", "reason": "Critic 未提出可检验模块假设，或目标模块的消融配对覆盖率不足"})
            continue
        mutation_brief = {
            "critic_summary": critic_diagnosis.get("summary", ""),
            "hypothesis": hypothesis_by_module[target],
            "ablation_result": effects[target],
            "human_priors": proposer_priors(
                active_priors, hypothesis_by_module[target].get("prior_ids", []),
            ),
        }
        try:
            source, attempts = propose_code_candidate(
                llm, module=target, current_source=default_source(target, base),
                failures=failures, diagnosis=mutation_brief,
                training_tasks=train, return_attempts=True,
            )
            candidate_module = load_candidate_module(source, llm)
            baseline_module = load_candidate_module(default_source(target, base), llm)
            probe = probe_candidate(target, candidate_module, baseline_module)
            diff = probe_with_tasks(target, candidate_module, baseline_module, train, config=base)
            if not diff["changed"]:
                proposed.append({"id": f"proposal-{i + 1}", "target_module": target, "diagnosis": mutation_brief,
                                 "source": source, "status": "rejected_no_behavioral_change",
                                 "attempts": attempts, "probe": probe, "behavior_diff": diff})
                continue
            proposed.append({"id": f"proposal-{i + 1}", "target_module": target, "diagnosis": mutation_brief,
                             "source": source, "status": "smoke_passed", "smoke_passed": True,
                             "attempts": attempts, "probe": probe, "behavior_diff": diff})
        except Exception as exc:  # noqa: BLE001
            proposed.append({"id": f"proposal-{i + 1}", "target_module": target, "diagnosis": mutation_brief, "status": "rejected", "attempts": 2, "error": str(exc)})

    baseline_dev = evaluate_set(dev, llm, base, adapter, bypass_modules=bypassed)
    base_scores = _as_scores(baseline_dev)
    candidate_results = []
    valid_candidates = [p for p in proposed if p.get("status") == "smoke_passed"]
    valid_candidates.extend(candidates or [])
    for cand in valid_candidates:
        candidate_id = cand.get("id", f"candidate-{len(candidate_results) + 1}")
        module, source = cand.get("target_module"), cand.get("source", "")
        try:
            if module not in adapter.active_modules:
                raise ValueError(f"{module!r} 不适用于 {adapter.name}")
            reject_task_literals(source, train)
            candidate_module = load_candidate_module(source, llm)
            if not cand.get("smoke_passed"):
                smoke_candidate(module, candidate_module)
            dev_rows = evaluate_set(dev, llm, base, adapter, modules={module: candidate_module}, bypass_modules=bypassed)
            n_errors = sum(row["error"] is not None for row in dev_rows)
            _b, _c = base_scores, _as_scores(dev_rows)
            _keys = sorted(set(_b) & set(_c))
            dev_up = sum(1 for k in _keys if _c[k] > _b[k] + 1e-9)
            dev_down = sum(1 for k in _keys if _c[k] < _b[k] - 1e-9)
            dev_tie = len(_keys) - dev_up - dev_down
            candidate_results.append({
                "id": candidate_id, "target_module": module, "source": source,
                "rationale": cand.get("rationale"), "dev": dev_rows,
                "dev_mean": _mean(list(_as_scores(dev_rows).values())),
                "n_errors": n_errors,
                "dev_improved": dev_up, "dev_worsened": dev_down, "dev_tied": dev_tie,
                "status": "evaluated" if n_errors == 0 else "rejected_runtime_errors",
                "dev_delta": paired_bootstrap(base_scores, _as_scores(dev_rows), seed=seed, minimum_coverage=minimum_pair_coverage, total_tasks=len(dev)),
            })
        except Exception as exc:  # candidate-level failure is reported, not allowed to kill the experiment
            candidate_results.append({
                "id": candidate_id, "target_module": module, "source": source,
                "n_errors": 1, "status": "rejected_preflight", "error": {"type": type(exc).__name__, "message": str(exc)},
                "dev": [], "dev_delta": {"n": 0, "n_possible": len(dev), "coverage": 0.0, "mean": None, "raw_mean": None, "ci95": [None, None]},
            })

    # 只用 dev 选择和接受；最终 test 通过独立入口运行。
    def eligible_on_dev(candidate):
        delta = candidate["dev_delta"]
        ci_ok = not require_positive_ci or (delta["ci95"][0] is not None and delta["ci95"][0] > 0)
        return candidate["n_errors"] == 0 and delta["mean"] is not None and delta["mean"] >= minimum_dev_gain and ci_ok

    best = max(
        (c for c in candidate_results if eligible_on_dev(c)),
        key=lambda c: c["dev_delta"]["mean"],
        default=None,
    )
    selected = best
    artifact_dir = out / "candidate_modules"
    artifact_dir.mkdir(exist_ok=True)
    for candidate in candidate_results:
        safe_id = "".join(ch for ch in candidate["id"] if ch.isalnum() or ch in "-_") or "candidate"
        (artifact_dir / f"{safe_id}.py").write_text(candidate["source"], encoding="utf-8")
    if selected:
        (out / "selected_module.py").write_text(selected["source"], encoding="utf-8")
        (out / "selected_harness.json").write_text(json.dumps({
            "version": "candidate-v1",
            "module": selected["target_module"],
            "candidate_id": selected["id"],
            "source_sha256": hashlib.sha256(selected["source"].encode("utf-8")).hexdigest(),
            "benchmark": adapter.name,
            "metric": adapter.metric,
            "harness_config": asdict(base),
            "dev_delta": selected["dev_delta"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "benchmark": adapter.name,
        "metric": adapter.metric,
        "bypassed_modules": bypassed,
        "n_tasks": len(tasks),
        "dataset_sha256": dataset_sha256,
        "harness_config": asdict(base),
        "split": {"train": [t["id"] for t in train], "dev": [t["id"] for t in dev], "test": [t["id"] for t in test], "seed": seed},
        "history_summary": history,
        "baseline_train": baseline_train,
        "baseline_dev": baseline_dev,
        "run_errors": {
            "baseline_train": sum(row["error"] is not None for row in baseline_train),
            "baseline_dev": sum(row["error"] is not None for row in baseline_dev),
            "ablations_train": {module: sum(row["error"] is not None for row in rows) for module, rows in ablations.items()},
            "candidate_dev": {candidate["id"]: candidate["n_errors"] for candidate in candidate_results},
        },
        "baseline_dev_mean": _mean([s for s in base_scores.values()]),
        "ablations": ablations,
        "module_effects": effects,
        "critic_diagnosis": critic_diagnosis,
        "prior_library": {
            "library_id": prior_library.get("library_id"),
            "schema_version": prior_library.get("schema_version"),
            "enabled_operator_ids": [operator["id"] for operator in active_priors],
            "diagnostic_prior_ids": [card["id"] for card in diagnostic_prior_cards],
        },
        "suspected_modules": suspected_modules,
        "weakest_module": weakest,
        "mutation_targets": ranked_targets,
        "gate": {
            "minimum_dev_gain": minimum_dev_gain,
            "minimum_pair_coverage": minimum_pair_coverage,
            "require_positive_ci_lower_bound": require_positive_ci,
        },
        "proposals": proposed,
        "candidates": candidate_results,
        "test_evaluation": {"status": "not_run", "used_for_acceptance": False},
        "selection": {
            "accepted": bool(selected),
            "candidate_id": selected["id"] if selected else None,
            "reason": "候选通过 dev 的增益、CI、覆盖率与错误率门槛" if selected else "未通过 dev 增益、CI、覆盖率或错误率门槛",
            "artifact": "selected_module.py" if selected else None,
        },
        "interpretation": (
            "module_effects 为训练集 baseline − bypass 的配对均值：>0 表示该模块当前有正贡献，<0 表示去掉它更好。"
            f"诊断与生成只使用 train；接受只依据 dev；test 仅通过独立命令最终评测。结果仅对 {adapter.metric} 指标成立。"
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace")
    return report


def evaluate_final_test(tasks, llm, experiment_dir: str | Path) -> dict[str, Any]:
    """Evaluate the frozen, dev-selected version; never change acceptance."""
    out = Path(experiment_dir)
    report = json.loads((out / "multi_task_attribution.json").read_text(encoding="utf-8"))
    result_path = out / "test_report.json"
    if result_path.exists():
        raise FileExistsError(f"最终测试已运行，拒绝覆盖：{result_path}")
    if task_fingerprint(tasks) != report["dataset_sha256"]:
        raise ValueError("题库与生成候选时不同；拒绝改变最终测试集")
    test_ids = set(report["split"]["test"])
    adapter = benchmark_adapter(report["benchmark"])
    test = [adapter.prepare(task) for task in tasks if task["id"] in test_ids]
    base = HarnessConfig(**report["harness_config"])
    modules = {}
    if report["selection"]["accepted"]:
        selected = json.loads((out / "selected_harness.json").read_text(encoding="utf-8"))
        source = (out / "selected_module.py").read_text(encoding="utf-8")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != selected["source_sha256"]:
            raise ValueError("已选模块源码发生变化；拒绝测试非冻结版本")
        modules[selected["module"]] = load_candidate_module(source, llm)
    baseline = evaluate_set(test, llm, base, adapter, bypass_modules=report["bypassed_modules"])
    final = evaluate_set(test, llm, base, adapter, modules=modules, bypass_modules=report["bypassed_modules"]) if modules else baseline
    result = {
        "used_for_acceptance": False,
        "candidate_id": report["selection"]["candidate_id"],
        "test_ids": report["split"]["test"],
        "baseline": baseline,
        "final": final,
        "delta": paired_bootstrap(_as_scores(baseline), _as_scores(final), seed=report["split"]["seed"],
                                  minimum_coverage=report["gate"]["minimum_pair_coverage"], total_tasks=len(test)),
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def render_markdown(report: dict[str, Any]) -> str:
    split = report["split"]
    lines = ["# 多题归因报告", "",
             f"- train / dev / test：{len(split['train'])} / {len(split['dev'])} / {len(split['test'])}",
             f"- dev baseline {report['metric']}：{report['baseline_dev_mean']:.4f}",
             "- train 用于诊断、消融和生成；dev 用于选择；test 尚未运行。",
             "", "## 训练集诊断", "", report['critic_diagnosis'].get('summary', ''),
             "", "## 训练集消融（baseline − bypass）", "",
             "| 模块 | 平均效应 | 95% CI |", "|---|---|---|"]
    for module, effect in report['module_effects'].items():
        lines.append(f"| {module} | {effect['mean']} | {effect['ci95']} |")
    lines += ["", "## 开发集候选", "", "| 候选 | 模块 | 状态 | dev Δ | 95% CI |", "|---|---|---|---|---|"]
    for candidate in report['candidates']:
        delta = candidate['dev_delta']
        lines.append(f"| {candidate['id']} | {candidate['target_module']} | {candidate['status']} | {delta['mean']} | {delta['ci95']} |")
    lines += ["", "## 版本选择", "", f"- 接受：{report['selection']['accepted']}",
              f"- 原因：{report['selection']['reason']}", "", report['interpretation']]
    return "\n".join(lines) + "\n"
