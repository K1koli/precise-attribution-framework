from pathlib import Path
import pytest

from precise_attribution.core.config import load_config, validate_config
from precise_attribution.experiments.design import all_coalitions

ROOT = Path(__file__).resolve().parents[2]


def test_demo_config_loads():
    config = load_config(ROOT / "configs/experiments/demo.yaml")
    assert config["scenario"]["modules"] == ["检索", "推理", "校验"]


def test_full_factorial_has_two_to_n_coalitions():
    assert len(all_coalitions(("A", "B", "C"))) == 8
    assert frozenset() in all_coalitions(("A", "B"))
    assert frozenset({"A", "B"}) in all_coalitions(("A", "B"))


def test_rejects_too_many_modules():
    config = {
        "experiment": {}, "sampling": {"blocks": 2},
        "scenario": {"modules": [f"m{i}" for i in range(11)], "main_effects": {f"m{i}": 1 for i in range(11)}}
    }
    with pytest.raises(ValueError, match="最多支持 10 个模块"):
        validate_config(config)
