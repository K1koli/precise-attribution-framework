"""隐藏机制真值。仅验证层可导入；估计器包不得依赖此模块。"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial
from typing import Any


@dataclass(frozen=True)
class TruthBundle:
    modules: tuple[str, ...]
    baseline: float
    main_effects: dict[str, float]
    interactions: dict[tuple[str, str], float]

    def coalition_value(self, coalition: set[str]) -> float:
        value = self.baseline + sum(self.main_effects[m] for m in coalition)
        value += sum(v for pair, v in self.interactions.items() if set(pair).issubset(coalition))
        return value


def truth_from_config(config: dict[str, Any]) -> TruthBundle:
    return TruthBundle(
        modules=tuple(config["modules"]),
        baseline=float(config.get("baseline", 0.0)),
        main_effects={k: float(v) for k, v in config["main_effects"].items()},
        interactions={
            tuple(pair.split("|")): float(value)
            for pair, value in config.get("pairwise_interactions", {}).items()
        },
    )


def shapley_from_values(modules: tuple[str, ...], values: dict[frozenset[str], float]) -> dict[str, float]:
    n = len(modules)
    result: dict[str, float] = {}
    for module in modules:
        others = [m for m in modules if m != module]
        total = 0.0
        for size in range(n):
            for subset_tuple in combinations(others, size):
                subset = frozenset(subset_tuple)
                weight = factorial(size) * factorial(n - size - 1) / factorial(n)
                total += weight * (values[subset | {module}] - values[subset])
        result[module] = total
    return result
