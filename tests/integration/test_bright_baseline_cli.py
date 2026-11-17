import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('baseline_cli', ROOT / 'scripts/evaluate_bright_baseline.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.mark.parametrize('subset', ['biology', 'economics'])
def test_subset_selects_own_corpus_and_tasks(tmp_path, monkeypatch, subset):
    data = tmp_path / 'processed'
    folder = data / subset
    folder.mkdir(parents=True)
    docs = [{'id': f'{subset}-{i}', 'title': '', 'text': 'cell'} for i in range(12)]
    tasks = [{'id': f'q-{i}', 'question': 'cell', 'gold_doc_ids': [docs[0]['id']], 'excluded_ids': []} for i in range(10)]
    (folder/'corpus.json').write_text(json.dumps(docs))
    for name, indices in [('train',list(range(4))),('dev',list(range(4,7))),('test',list(range(7,10)))]:
        (folder/f'{name}.json').write_text(json.dumps({'partition':name,'seed':42,'split_ratio':[4,3,3],
            'task_indices':indices,'tasks':[tasks[i] for i in indices]}))
    out = tmp_path/'result'
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, 'argv', ['baseline', '--subset', subset, '--data-dir', str(data), '--output', str(out)])
    cli.main()
    report = json.loads((out/'test_report.json').read_text())
    assert report['subset'] == subset and report['seed'] == 42
    assert len(report['baseline']) == 3 and report['errors'] == 0
    assert report['mean_ndcg10'] == 1
    assert all(d['id'].startswith(subset) for row in report['baseline'] for d in row['trace']['retrieved_documents'])


def test_missing_subset_does_not_create_output(tmp_path, monkeypatch):
    out = tmp_path/'result'
    monkeypatch.setattr(sys, 'argv', ['baseline', '--subset', 'robotics', '--data-dir', str(tmp_path), '--output', str(out)])
    with pytest.raises(SystemExit):
        cli.main()
    assert not out.exists()
