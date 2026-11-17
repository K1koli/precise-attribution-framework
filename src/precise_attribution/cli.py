from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from precise_attribution.llm.config import client_from_config
from precise_attribution.evolution.multi_task import run_multi_task_attribution, render_markdown
from precise_attribution.evolution.prior_library import load_prior_library


def _resolve_candidate_sources(config: dict, config_path: str) -> list[dict]:
    candidates = [dict(candidate) for candidate in config.get("candidates", [])]
    config_dir = Path(config_path).resolve().parent
    for candidate in candidates:
        source_file = candidate.pop("source_file", None)
        if source_file:
            path = Path(source_file)
            if not path.is_absolute():
                path = config_dir / path
            candidate["source"] = path.read_text(encoding="utf-8")
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(prog="paf", description="精确归因框架命令行工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("llm-check", help="检查 OpenAI 兼容推理服务")
    check_parser.add_argument("--config", required=True, help="包含 llm 配置的 YAML 文件")
    set_parser = subparsers.add_parser("attribute-set", help="对一组题目运行模块消融、候选生成与 held-out 验证")
    set_parser.add_argument("--config", required=True, help="包含 llm/tasks/harness（可选 candidates）的 YAML 文件")
    set_parser.add_argument("--output", required=True, help="实验输出目录")
    set_parser.add_argument("--dev-ratio", type=float, default=0.67, help="dev 占比（其余为 held-out）")
    set_parser.add_argument("--seed", type=int, default=0, help="切分与 bootstrap 的随机种子")
    set_parser.add_argument("--proposals", type=int, default=2, help="生成多少个单模块源码候选")
    set_parser.add_argument("--target-module", action="append", choices=["QueryUnderstanding", "EvidenceRetrieval", "EvidencePreparation", "EvidenceReasoning"], help="限定 Critic 可诊断的模块范围；可重复传入，不指定时检查所有适用模块")
    args = parser.parse_args()
    if args.command == "llm-check":
        with Path(args.config).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        health = client_from_config(config).health()
        print(json.dumps(health, ensure_ascii=False, indent=2))
        if not health.get("ok"):
            raise SystemExit(1)
    elif args.command == "attribute-set":
        with Path(args.config).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError("配置文件必须是 YAML 对象")
        tasks = config.get("tasks")
        if not tasks and config.get("tasks_file"):
            with Path(config["tasks_file"]).open("r", encoding="utf-8") as handle:
                tasks = (yaml.safe_load(handle) or {}).get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("配置必须包含 tasks 列表，或用 tasks_file 指向题库")
        if len(tasks) < 3:
            raise ValueError("至少需要 3 道题")
        prior_config = config.get("priors")
        if isinstance(prior_config, dict) and prior_config.get("enabled") is False:
            prior_library = {"library_id": "disabled", "schema_version": 1, "operators": []}
        elif isinstance(prior_config, dict) and prior_config.get("file"):
            prior_path = Path(prior_config["file"])
            if not prior_path.is_absolute():
                prior_path = Path(args.config).resolve().parent / prior_path
            prior_library = load_prior_library(prior_path)
        else:
            prior_library = load_prior_library()
        client = client_from_config(config)
        health = client.health()
        if not health.get("ok"):
            raise RuntimeError(f"LLM 服务不可用：{health}")
        report = run_multi_task_attribution(
            tasks, client, args.output,
            harness_config=config.get("harness"), candidates=_resolve_candidate_sources(config, args.config),
            dev_ratio=args.dev_ratio, seed=args.seed, max_proposals=args.proposals,
            target_modules=args.target_module or config.get("mutation_targets"),
            benchmark=config.get("benchmark", "qa_token_f1"),
            minimum_dev_gain=float(config.get("gate", {}).get("minimum_dev_gain", 0.01)),
            minimum_heldout_gain=float(config.get("gate", {}).get("minimum_heldout_gain", 0.01)),
            minimum_pair_coverage=float(config.get("gate", {}).get("minimum_pair_coverage", 0.90)),
            require_positive_ci=bool(config.get("gate", {}).get("require_positive_ci_lower_bound", True)),
            evaluate_all_on_heldout=bool(config.get("gate", {}).get("evaluate_all_on_heldout", False)),
            bypass_inactive_modules=bool(config.get("bypass_inactive_modules", False)),
            prior_library=prior_library,
        )
        out = Path(args.output)
        (out / "multi_task_attribution.md").write_text(render_markdown(report), encoding="utf-8", errors="replace")
        print(f"多题归因完成；dev baseline {report['metric']}={report['baseline_dev_mean']:.4f}；最弱模块={report['weakest_module']}")
        hv = report.get("heldout_validation") or {}
        if hv.get("delta", {}).get("mean") is not None:
            print(f"held-out Δ={hv['delta']['mean']:+.4f}（{hv['candidate_id']}）")
        print(f"版本接受：{report['selection']['accepted']}；{report['selection']['reason']}")
        print(f"报告：{(out / 'multi_task_attribution.md').resolve()}")


if __name__ == "__main__":
    main()
