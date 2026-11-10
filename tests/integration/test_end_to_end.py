import json
from pathlib import Path
import yaml

from precise_attribution.cli import run_command

ROOT = Path(__file__).resolve().parents[2]


def test_end_to_end_creates_all_artifacts(tmp_path):
    config = yaml.safe_load((ROOT / "configs/experiments/demo.yaml").read_text(encoding="utf-8"))
    config["sampling"]["blocks"] = 24
    config["experiment"]["confidence_level"] = 0.90
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    output = tmp_path / "result"
    run_command(str(config_path), str(output))
    expected = {"runs.jsonl", "estimate.json", "validation.json", "report.md", "manifest.json"}
    assert expected.issubset({p.name for p in output.iterdir()})
    estimate = json.loads((output / "estimate.json").read_text(encoding="utf-8"))
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert estimate["n_blocks"] == 24
    assert estimate["confidence_level"] == 0.90
    assert validation["shapley_efficiency_error"] < 1e-8
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "两两交互效应" in report
    assert "90% 区间" in report


def test_existing_raw_records_are_not_overwritten(tmp_path):
    from precise_attribution.experiments.orchestrator import run_experiment
    from precise_attribution.core.config import load_config
    config = load_config(ROOT / "configs/experiments/demo.yaml")
    output = tmp_path / "result"
    run_experiment(config, output)
    import pytest
    with pytest.raises(FileExistsError):
        run_experiment(config, output)
