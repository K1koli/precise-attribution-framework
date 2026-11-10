from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import yaml

from precise_attribution.core.config import load_config
from precise_attribution.estimators.factorial import estimate_factorial
from precise_attribution.experiments.orchestrator import run_experiment
from precise_attribution.recording.jsonl import JsonlRecorder
from precise_attribution.reporting.report import write_json, write_report
from precise_attribution.simulation.truth import truth_from_config
from precise_attribution.validation.validator import validate_estimate
from precise_attribution.llm.config import client_from_config
from precise_attribution.evolution.engine import run_self_evolve


def run_command(config_path: str, output_path: str) -> None:
    config = load_config(config_path)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    records, scenario = run_experiment(config, output)

    # 估计器仅接收记录和模块名称，不接收 scenario 或 truth。
    observed_records = JsonlRecorder(output / "runs.jsonl").read_all()
    estimate = estimate_factorial(
        observed_records, scenario.modules,
        confidence_level=float(config["experiment"].get("confidence_level", 0.95)),
    )
    estimate_payload = estimate.to_dict()
    estimate_bytes = json.dumps(estimate_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    estimate_digest = hashlib.sha256(estimate_bytes).hexdigest()
    write_json(output / "estimate.json", {**estimate_payload, "frozen_sha256": estimate_digest})

    # 估计结果写盘并计算摘要后，才创建真值对象并调用验证器。
    truth = truth_from_config(config["scenario"])
    validation = validate_estimate(estimate, truth)
    write_json(output / "validation.json", validation.to_dict())
    write_report(output / "report.md", estimate, validation)

    manifest = {
        "experiment_id": config["experiment"]["id"],
        "scenario_id": scenario.scenario_id,
        "config_path": str(Path(config_path)),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "run_count": len(records),
        "coalition_count": 2 ** len(scenario.modules),
        "paired_block_count": int(config["sampling"]["blocks"]),
        "estimate_sha256": estimate_digest,
        "intervention_semantics": "bypass",
        "python_runtime": __import__("sys").version,
    }
    write_json(output / "manifest.json", manifest)
    print(f"实验完成：{output.resolve()}")
    print(f"运行记录：{len(records)} 条；完整联盟：{2 ** len(scenario.modules)}；配对区块：{estimate.n_blocks}")
    print(f"报告：{(output / 'report.md').resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="paf", description="精确归因框架命令行工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="运行合成归因实验")
    run_parser.add_argument("--config", required=True, help="实验 YAML 配置")
    run_parser.add_argument("--output", required=True, help="实验输出目录")
    evolve_parser = subparsers.add_parser("evolve", help="运行单轮 LLM Harness Self-Evolve")
    evolve_parser.add_argument("--config", required=True, help="Self-Evolve YAML 配置")
    evolve_parser.add_argument("--output", required=True, help="实验输出目录")
    check_parser = subparsers.add_parser("llm-check", help="检查 OpenAI 兼容推理服务")
    check_parser.add_argument("--config", required=True, help="包含 llm 配置的 YAML 文件")
    args = parser.parse_args()
    if args.command == "run":
        run_command(args.config, args.output)
    elif args.command == "evolve":
        with Path(args.config).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError("配置文件必须是 YAML 对象")
        client = client_from_config(config)
        health = client.health()
        if not health.get("ok"):
            raise RuntimeError(f"LLM 服务不可用：{health}")
        print(f"LLM 服务正常：{health.get('models')}")
        report = run_self_evolve(config, client, args.output)
        print(f"Self-Evolve 完成；accepted={report['acceptance']['accepted']}")
        print(f"报告：{(Path(args.output) / 'self_evolve_report.md').resolve()}")
    elif args.command == "llm-check":
        with Path(args.config).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        health = client_from_config(config).health()
        print(json.dumps(health, ensure_ascii=False, indent=2))
        if not health.get("ok"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
