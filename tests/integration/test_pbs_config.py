import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prepare_pbs_run', ROOT / 'scripts/prepare_pbs_run.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_runtime_config_paths_and_independent_test(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    output = tmp_path / 'experiment'
    path = module.prepare('configs/experiments/qa30_attribution.yaml', str(output), 'develop', 8123, 'local-model')
    original = path.read_bytes()
    config = yaml.safe_load(original)
    assert config['llm']['base_url'] == 'http://127.0.0.1:8123/v1'
    assert config['llm']['model'] == 'local-model'
    assert config['llm']['provider'] == 'vllm'
    assert Path(config['tasks_file']).is_file()
    assert Path(config['priors']['file']).is_file()
    assert all(Path(c['source_file']).is_file() for c in config['candidates'])
    with pytest.raises(FileExistsError):
        module.prepare('configs/experiments/qa30_attribution.yaml', str(output), 'develop', 8123, 'local-model')
    with pytest.raises(FileNotFoundError):
        module.prepare('unused', str(output), 'test', 8124, 'local-model')
    (output / 'multi_task_attribution.json').write_text('{}')
    with pytest.raises(ValueError, match='模型'):
        module.prepare('unused', str(output), 'test', 8124, 'different-model')
    test_path = module.prepare('unused', str(output), 'test', 8124, 'local-model')
    assert yaml.safe_load(test_path.read_text())['llm']['base_url'].endswith(':8124/v1')
    assert path.read_bytes() == original
    (output / 'test_report.json').write_text('{}')
    with pytest.raises(FileExistsError):
        module.prepare('unused', str(output), 'test', 8124, 'local-model')


def test_deepseek_config_is_switched_to_local(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    path = module.prepare('configs/experiments/qa30_deepseek.yaml', str(tmp_path / 'run'), 'develop', 8000, 'Qwen3.8-27B')
    llm = yaml.safe_load(path.read_text())['llm']
    assert 'api_key_env' not in llm
    assert llm['provider'] == 'vllm'
    assert llm['disable_thinking'] is True
