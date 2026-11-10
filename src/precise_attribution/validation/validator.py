from __future__ import annotations

from itertools import combinations

from precise_attribution.core.models import Estimate, ValidationResult
from precise_attribution.simulation.truth import TruthBundle, shapley_from_values


def validate_estimate(estimate: Estimate, truth: TruthBundle) -> ValidationResult:
    """只在估计完成后调用；真值对象不传入估计器。"""
    # 当前估计量是对其他模块均匀平均后的主效应；有交互时不等于裸结构系数 beta_i。
    expected_main = {
        module: truth.main_effects[module]
        + 0.5 * sum(value for pair, value in truth.interactions.items() if module in pair)
        for module in truth.modules
    }
    main_errors = {
        module: estimate.main_effects[module]["estimate"] - expected_main[module]
        for module in truth.modules
    }
    interaction_errors = {}
    for left, right in combinations(truth.modules, 2):
        key = f"{left}|{right}"
        estimated = estimate.pairwise_interactions[key]["estimate"]
        actual = truth.interactions.get((left, right), 0.0)
        interaction_errors[key] = estimated - actual
    truth_values = {}
    for size in range(len(truth.modules) + 1):
        for subset in combinations(truth.modules, size):
            key = frozenset(subset)
            truth_values[key] = truth.coalition_value(set(subset))
    truth_shapley = shapley_from_values(truth.modules, truth_values)
    shapley_errors = {m: estimate.shapley_values[m] - truth_shapley[m] for m in truth.modules}
    main_sign = [
        _sign(estimate.main_effects[m]["estimate"]) == _sign(expected_main[m])
        for m in truth.modules if expected_main[m] != 0
    ]
    interaction_sign = [
        _sign(estimate.pairwise_interactions[f"{k[0]}|{k[1]}"]["estimate"]) == _sign(v)
        for k, v in truth.interactions.items() if v != 0
    ]
    full_key = "|".join(truth.modules)
    empty_key = "∅"
    estimated_total = estimate.coalition_values[full_key] - estimate.coalition_values[empty_key]
    shapley_total = sum(estimate.shapley_values.values())
    return ValidationResult(
        experiment_id=estimate.experiment_id,
        main_effect_errors=main_errors,
        interaction_errors=interaction_errors,
        shapley_errors=shapley_errors,
        main_effect_sign_accuracy=sum(main_sign) / len(main_sign) if main_sign else 1.0,
        interaction_sign_accuracy=sum(interaction_sign) / len(interaction_sign) if interaction_sign else 1.0,
        shapley_efficiency_error=shapley_total - estimated_total,
    )


def _sign(value: float, tolerance: float = 1e-9) -> int:
    return 1 if value > tolerance else -1 if value < -tolerance else 0
