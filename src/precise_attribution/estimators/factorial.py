from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import factorial, sqrt
from statistics import NormalDist

from precise_attribution.core.models import Estimate, RunRecord


def _interval(values: list[float], confidence_level: float) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return {"estimate": mean, "standard_error": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_blocks": n}
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    se = sqrt(variance / n)
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return {"estimate": mean, "standard_error": se, "ci_low": mean - z * se, "ci_high": mean + z * se, "n_blocks": n}


def _shapley(modules: tuple[str, ...], values: dict[frozenset[str], float]) -> dict[str, float]:
    n = len(modules)
    result = {}
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


def estimate_factorial(records: list[RunRecord], modules: tuple[str, ...], confidence_level: float = 0.95) -> Estimate:
    """从完整 2^n 观测记录估计均匀背景分布下的主效应、两两交互和 Shapley。

    此模块不接收场景对象或真值参数；输入只允许是观测运行记录。
    """
    if not records:
        raise ValueError("没有可用于估计的运行记录")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level 必须在 0 和 1 之间")
    experiment_ids = {r.experiment_id for r in records}
    scenario_ids = {r.scenario_id for r in records}
    if len(experiment_ids) != 1 or len(scenario_ids) != 1:
        raise ValueError("一次估计只能包含一个实验和一个场景")
    grouped: dict[str, dict[frozenset[str], float]] = defaultdict(dict)
    block_identity: dict[str, tuple[str, int]] = {}
    for record in records:
        if record.status != "success":
            continue
        coalition = frozenset(record.coalition)
        if record.block_id in block_identity and block_identity[record.block_id] != (record.task_id, record.seed):
            raise ValueError(f"配对区块 {record.block_id} 的 task_id/seed 不一致")
        block_identity[record.block_id] = (record.task_id, record.seed)
        expected_states = {m: int(m in coalition) for m in modules}
        if record.module_states != expected_states:
            raise ValueError(f"运行记录 {record.run_id} 的模块状态与联盟不一致")
        if not coalition.issubset(set(modules)):
            raise ValueError("运行记录包含未知模块")
        if coalition in grouped[record.block_id]:
            raise ValueError(f"配对区块 {record.block_id} 中存在重复联盟 {sorted(coalition)}")
        grouped[record.block_id][coalition] = record.outcome

    expected = {frozenset(c) for c in _all_subsets(modules)}
    valid_blocks = {bid: values for bid, values in grouped.items() if set(values) == expected}
    if len(valid_blocks) < 2:
        raise ValueError("至少需要两个包含完整 2^n 联盟的成功配对区块")

    block_main: dict[str, list[float]] = {m: [] for m in modules}
    pair_names = list(combinations(modules, 2))
    block_interactions: dict[tuple[str, str], list[float]] = {p: [] for p in pair_names}
    for values in valid_blocks.values():
        for module in modules:
            contrasts = []
            others = [m for m in modules if m != module]
            for background in _all_subsets(tuple(others)):
                bg = frozenset(background)
                contrasts.append(values[bg | {module}] - values[bg])
            block_main[module].append(sum(contrasts) / len(contrasts))
        for left, right in pair_names:
            others = [m for m in modules if m not in (left, right)]
            contrasts = []
            for background in _all_subsets(tuple(others)):
                bg = frozenset(background)
                contrasts.append(
                    values[bg | {left, right}]
                    - values[bg | {left}]
                    - values[bg | {right}]
                    + values[bg]
                )
            block_interactions[(left, right)].append(sum(contrasts) / len(contrasts))

    valid_coalition_values: dict[frozenset[str], list[float]] = defaultdict(list)
    for block_values in valid_blocks.values():
        for coalition, outcome in block_values.items():
            valid_coalition_values[coalition].append(outcome)
    coalition_means = {
        coalition: sum(values) / len(values)
        for coalition, values in valid_coalition_values.items()
        if values
    }
    if set(coalition_means) != expected:
        raise ValueError("联盟观测不完整，无法计算精确 Shapley 值")
    coalition_labels = {"|".join(m for m in modules if m in coalition) or "∅": value for coalition, value in coalition_means.items()}
    main = {m: _interval(values, confidence_level) for m, values in block_main.items()}
    interactions = {f"{a}|{b}": _interval(values, confidence_level) for (a, b), values in block_interactions.items()}
    return Estimate(
        experiment_id=next(iter(experiment_ids)),
        scenario_id=next(iter(scenario_ids)),
        modules=modules,
        n_blocks=len(valid_blocks),
        confidence_level=confidence_level,
        coalition_values=coalition_labels,
        main_effects=main,
        pairwise_interactions=interactions,
        shapley_values=_shapley(modules, coalition_means),
        notes=(
            "主效应对其余模块状态采用均匀背景分布平均。",
            "两两交互对其余模块状态采用均匀背景分布平均。",
            "区间为配对区块正态近似区间，置信水平由实验配置指定；未做多重比较校正。",
            "Shapley 是指定联盟价值函数下的分配，不等同于机制性因果真值。",
        ),
    )


def _all_subsets(items: tuple[str, ...]) -> list[tuple[str, ...]]:
    result = []
    for size in range(len(items) + 1):
        result.extend(combinations(items, size))
    return result
