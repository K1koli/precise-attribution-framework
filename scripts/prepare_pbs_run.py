#!/usr/bin/env python3
"""Prepare a local-vLLM configuration without changing tracked experiment files."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def prepare(config_path: str, experiment: str, mode: str, port: int, model: str) -> Path:
    if mode not in {'develop', 'test'}:
        raise ValueError('JOB_MODE 必须为 develop 或 test')
    if not 1 <= port <= 65535:
        raise ValueError('PORT 必须在 1–65535 范围内')
    out = Path(experiment).resolve()
    frozen = out / 'runtime_config.yaml'
    if mode == 'test':
        if (out / 'test_report.json').exists():
            raise FileExistsError('最终测试已运行，拒绝重复提交')
        if not (out / 'multi_task_attribution.json').exists():
            raise FileNotFoundError('test 模式需要已完成的实验目录')
        config_file = frozen
        destination = out / 'test_runtime_config.yaml'
    else:
        config_file = Path(config_path).resolve()
        destination = frozen
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('实验目录非空；请使用新的 EXPERIMENT_DIR')
    config = yaml.safe_load(config_file.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('实验配置必须是 YAML 对象')
    if mode == 'test' and config.get('llm', {}).get('model') != model:
        raise ValueError('最终测试的模型名称必须与开发实验一致')
    llm = config.setdefault('llm', {})
    llm.update(provider='vllm', base_url=f'http://127.0.0.1:{port}/v1', model=model,
               api_key='EMPTY', disable_thinking=True)
    llm.pop('api_key_env', None)
    # Candidate/prior paths are relative to config; tasks_file follows the CLI's cwd rule.
    if config.get('tasks_dir'):
        task_dir = Path(config['tasks_dir']).resolve()
        if not all((task_dir / f'{part}.json').is_file() for part in ('train', 'dev', 'test')):
            raise FileNotFoundError(f'Prepared partitions missing: {task_dir}')
        config['tasks_dir'] = str(task_dir)
    if config.get('tasks_file'):
        task_path = Path(config['tasks_file']).resolve()
        if not task_path.is_file():
            raise FileNotFoundError(task_path)
        config['tasks_file'] = str(task_path)
    if config.get('proposal_seeds'):
        raise ValueError('proposal_seeds 已停用，请使用 history_reports')
    if config.get('history_reports'):
        paths = [Path(p).resolve() for p in config['history_reports']]
        if not all(p.is_file() for p in paths):
            raise FileNotFoundError('历史实验报告不存在')
        config['history_reports'] = [str(p) for p in paths]
    harness = config.get('harness', {})
    if harness.get('corpus_file'):
        import hashlib
        corpus = Path(harness['corpus_file']).resolve()
        digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
        if digest != harness.get('corpus_sha256'):
            raise ValueError('Corpus checksum mismatch')
        harness['corpus_file'] = str(corpus)
    for candidate in config.get('candidates', []):
        if candidate.get('source_file'):
            source = (config_file.parent / candidate['source_file']).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            candidate['source_file'] = str(source)
    priors = config.get('priors', {})
    if priors.get('file'):
        prior_path = (config_file.parent / priors['file']).resolve()
        if not prior_path.is_file():
            raise FileNotFoundError(prior_path)
        priors['file'] = str(prior_path)
    out.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--mode', choices=['develop', 'test'], required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    print(prepare(args.config, args.experiment, args.mode, args.port, args.model))


if __name__ == '__main__':
    main()
