from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SyntheticTask:
    task_id: str
    block_id: str
    seed: int
    baseline_offset: float
    main_effects: dict[str, float]
    interactions: dict[tuple[str, str], float]
    shared_noise: float


class SyntheticScenario:
    """模拟器持有机制参数；执行器只调用 outcome，不把参数交给估计器。"""

    def __init__(self, config: dict[str, Any]):
        self.scenario_id = config["id"]
        self.modules = tuple(config["modules"])
        self.baseline = float(config.get("baseline", 0.0))
        self.main_effects = {k: float(v) for k, v in config["main_effects"].items()}
        self.interactions = {
            tuple(pair.split("|")): float(value)
            for pair, value in config.get("pairwise_interactions", {}).items()
        }
        self.task_heterogeneity_sd = float(config.get("task_heterogeneity_sd", 0.0))
        self.interaction_heterogeneity_sd = float(config.get("interaction_heterogeneity_sd", 0.0))
        self.noise_sd = float(config.get("noise_sd", 0.0))

    def make_task(self, index: int, seed: int) -> SyntheticTask:
        rng = random.Random(seed)
        task_main = {
            module: effect + rng.gauss(0.0, self.task_heterogeneity_sd)
            for module, effect in self.main_effects.items()
        }
        task_interactions = {
            pair: effect + rng.gauss(0.0, self.interaction_heterogeneity_sd)
            for pair, effect in self.interactions.items()
        }
        baseline_offset = rng.gauss(0.0, self.task_heterogeneity_sd)
        shared_noise = rng.gauss(0.0, self.noise_sd)
        return SyntheticTask(
            task_id=f"task-{index:05d}",
            block_id=f"block-{index:05d}",
            seed=seed,
            baseline_offset=baseline_offset,
            main_effects=task_main,
            interactions=task_interactions,
            shared_noise=shared_noise,
        )

    def outcome(self, task: SyntheticTask, active: set[str]) -> float:
        value = self.baseline + task.baseline_offset + task.shared_noise
        value += sum(task.main_effects[m] for m in active)
        value += sum(effect for pair, effect in task.interactions.items() if set(pair).issubset(active))
        return value
