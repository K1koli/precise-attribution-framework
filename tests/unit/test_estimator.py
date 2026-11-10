from precise_attribution.core.models import RunRecord
from precise_attribution.estimators.factorial import estimate_factorial
from precise_attribution.experiments.design import all_coalitions


def make_records():
    modules = ("A", "B")
    records = []
    for block in range(4):
        for coalition in all_coalitions(modules):
            a, b = "A" in coalition, "B" in coalition
            y = 1.0 + 2.0 * a - 1.0 * b + 0.5 * a * b + block * 0.01
            records.append(RunRecord(
                run_id=f"{block}-{a}-{b}", experiment_id="e", scenario_id="s", task_id=str(block),
                block_id=str(block), seed=block, coalition=tuple(m for m in modules if m in coalition),
                module_states={m: int(m in coalition) for m in modules}, interventions=(), outcome=y
            ))
    return modules, records


def test_factorial_estimates_average_main_effects_and_interaction():
    modules, records = make_records()
    estimate = estimate_factorial(records, modules)
    assert abs(estimate.main_effects["A"]["estimate"] - 2.25) < 1e-9
    assert abs(estimate.main_effects["B"]["estimate"] - (-0.75)) < 1e-9
    assert abs(estimate.pairwise_interactions["A|B"]["estimate"] - 0.5) < 1e-9


def test_shapley_efficiency():
    modules, records = make_records()
    estimate = estimate_factorial(records, modules)
    total = sum(estimate.shapley_values.values())
    assert abs(total - (estimate.coalition_values["A|B"] - estimate.coalition_values["∅"])) < 1e-9


def test_estimator_has_no_simulation_import():
    from pathlib import Path
    source = Path(__file__).resolve().parents[2] / "src/precise_attribution/estimators/factorial.py"
    assert "precise_attribution.simulation" not in source.read_text(encoding="utf-8")


def test_incomplete_factorial_block_is_rejected():
    import pytest
    modules, records = make_records()
    incomplete = [r for r in records if r.coalition]
    with pytest.raises(ValueError, match=r"完整 2\^n"):
        estimate_factorial(incomplete, modules)


def test_confidence_level_is_validated():
    import pytest
    modules, records = make_records()
    with pytest.raises(ValueError, match="confidence_level"):
        estimate_factorial(records, modules, confidence_level=1.0)


def test_incomplete_block_is_excluded_from_all_estimands():
    modules, records = make_records()
    # 删除最后一个区块中的空联盟；其余三个完整区块仍可估计。
    filtered = [r for r in records if not (r.block_id == "3" and not r.coalition)]
    estimate = estimate_factorial(filtered, modules)
    assert estimate.n_blocks == 3
    assert abs(estimate.main_effects["A"]["estimate"] - 2.25) < 1e-9
    assert abs(sum(estimate.shapley_values.values()) - (estimate.coalition_values["A|B"] - estimate.coalition_values["∅"])) < 1e-9
