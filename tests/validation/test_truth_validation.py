from pathlib import Path
from precise_attribution.core.config import load_config
from precise_attribution.estimators.factorial import estimate_factorial
from precise_attribution.experiments.orchestrator import run_experiment
from precise_attribution.simulation.truth import truth_from_config
from precise_attribution.validation.validator import validate_estimate

ROOT = Path(__file__).resolve().parents[2]


def test_estimate_validates_against_hidden_truth(tmp_path):
    config = load_config(ROOT / "configs/experiments/demo.yaml")
    config["sampling"]["blocks"] = 120
    records, _ = run_experiment(config, tmp_path)
    estimate = estimate_factorial(records, tuple(config["scenario"]["modules"]))
    validation = validate_estimate(estimate, truth_from_config(config["scenario"]))
    assert max(abs(x) for x in validation.main_effect_errors.values()) < 0.12
    assert max(abs(x) for x in validation.interaction_errors.values()) < 0.12
    assert validation.shapley_efficiency_error < 1e-8
