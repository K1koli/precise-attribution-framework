from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from precise_attribution.core.models import RunRecord
from precise_attribution.execution.runner import execute_coalition
from precise_attribution.experiments.design import all_coalitions
from precise_attribution.recording.jsonl import JsonlRecorder
from precise_attribution.simulation.scenario import SyntheticScenario


def stable_seed(master_seed: int, index: int) -> int:
    raw = f"{master_seed}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def run_experiment(config: dict[str, Any], output_dir: str | Path) -> tuple[list[RunRecord], SyntheticScenario]:
    output = Path(output_dir)
    if (output / "runs.jsonl").exists() and (output / "runs.jsonl").stat().st_size > 0:
        raise FileExistsError(f"输出目录已有原始运行记录，拒绝覆盖或混入新实验：{output / 'runs.jsonl'}")
    output.mkdir(parents=True, exist_ok=True)
    scenario = SyntheticScenario(config["scenario"])
    experiment_id = config["experiment"]["id"]
    blocks = int(config["sampling"]["blocks"])
    master_seed = int(config["sampling"]["seed"])
    coalitions = all_coalitions(scenario.modules)
    records: list[RunRecord] = []
    for index in range(blocks):
        task = scenario.make_task(index, stable_seed(master_seed, index))
        # 同一任务/随机种子区块下所有联盟共享任务潜变量和随机噪声，形成配对比较。
        for coalition in coalitions:
            records.append(execute_coalition(scenario, task, experiment_id, set(coalition)))
    recorder = JsonlRecorder(output / "runs.jsonl")
    recorder.append_many(records)
    return records, scenario
