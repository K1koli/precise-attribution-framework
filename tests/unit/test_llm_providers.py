import io
import json

import pytest

from precise_attribution.llm.config import client_from_config


def test_deepseek_requires_environment_key(monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    with pytest.raises(ValueError, match='DEEPSEEK_API_KEY'):
        client_from_config({'llm': {'provider': 'deepseek', 'api_key': 'not-used'}})


def test_environment_key_is_not_in_repr(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-secret-value')
    client = client_from_config({'llm': {'provider': 'deepseek'}})
    assert client.api_key == 'test-secret-value'
    assert 'test-secret-value' not in repr(client)
    assert client.base_url == 'https://api.deepseek.com'


@pytest.mark.parametrize('provider,options,expected', [
    ('deepseek', {}, {'thinking': {'type': 'disabled'}}),
    ('deepseek', {'disable_thinking': False}, {'thinking': {'type': 'enabled'}}),
    ('vllm', {}, {'chat_template_kwargs': {'enable_thinking': False}}),
    ('vllm', {'disable_thinking': False}, {}),
    ('openai_compatible', {}, {}),
])
def test_request_parameters_are_provider_specific(monkeypatch, provider, options, expected):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-only')
    monkeypatch.setenv('PAF_DISABLE_THINKING', '1')
    captured = []

    def urlopen(request, **kwargs):
        captured.append(json.loads(request.data))
        return io.BytesIO(b'{"choices":[{"message":{"content":"pong"}}]}')

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    client = client_from_config({'llm': {'provider': provider, **options}})
    assert client.chat([{'role': 'user', 'content': 'ping'}]) == 'pong'
    extras = {k: v for k, v in captured[0].items() if k in {'thinking', 'chat_template_kwargs'}}
    assert extras == expected


def test_legacy_local_config_and_environment_override(monkeypatch):
    monkeypatch.setenv('LOCAL_LLM_KEY', 'local-test-only')
    client = client_from_config({'llm': {'api_key_env': 'LOCAL_LLM_KEY'}})
    assert client.provider == 'vllm'
    assert client.model == 'Qwen3.8-27B'
    assert client.api_key == 'local-test-only'


def test_unknown_provider_fails():
    with pytest.raises(ValueError, match='provider'):
        client_from_config({'llm': {'provider': 'typo'}})
