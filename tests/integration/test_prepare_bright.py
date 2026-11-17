import importlib.util
from pathlib import Path

from precise_attribution.evolution.prepared_data import load_prepared_data
from precise_attribution.evolution.multi_task import split_tasks

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prepare_bright',ROOT/'scripts/prepare_bright_full.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_prepare_writes_three_partitions(monkeypatch, tmp_path):
    class Table:
        def __init__(self, rows): self.rows = rows
        def to_pylist(self): return self.rows
    docs = [{'id':'doc','content':'science'}]
    examples = [{'id':str(i),'query':f'question {i}','gold_ids':['doc'],'excluded_ids':[]} for i in range(10)]
    monkeypatch.setattr(module.pq, 'read_table', lambda p: Table(docs if p.parent.name == 'documents' else examples))
    out=tmp_path/'processed'
    module.prepare(tmp_path/'raw',out,'biology',seed=42,split_ratio=[4,3,3])
    assert {p.name for p in out.iterdir()} == {'train.json','dev.json','test.json','corpus.json'}
    payload=load_prepared_data(out)
    groups=split_tasks(payload['tasks'],train_ratio=.4,dev_ratio=.3,seed=42)
    for name, group in zip(['train','dev','test'],groups):
        assert payload['split'][name] == [t['id'] for t in group]
    assert [len(g) for g in groups] == [4,3,3]
    assert [t['id'] for t in payload['tasks']] == [f'bright-biology-{i}' for i in range(10)]
