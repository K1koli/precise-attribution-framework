#!/usr/bin/env python3
"""Export full BRIGHT corpus and queries without gold-based candidate construction."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from precise_attribution.evolution.multi_task import split_tasks


def prepare(raw, out, domain, *, seed=42, split_ratio=(4, 3, 3)):
    if len(split_ratio) != 3 or any(type(v) is not int or v <= 0 for v in split_ratio):
        raise ValueError('split_ratio must contain three positive integers')
    total = sum(split_ratio)
    train_ratio, dev_ratio = split_ratio[0] / total, split_ratio[1] / total
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing nonempty directory: {out}')
    docs = pq.read_table(raw / 'documents' / f'{domain}-00000-of-00001.parquet').to_pylist()
    examples = pq.read_table(raw / 'examples' / f'{domain}-00000-of-00001.parquet').to_pylist()
    corpus = [{'id': str(d['id']), 'title': '', 'text': str(d['content'])} for d in docs]
    ids = {d['id'] for d in corpus}
    assert len(ids) == len(corpus), 'Duplicate document IDs'
    tasks = []
    for row in examples:
        gold = list(map(str, row['gold_ids']))
        excluded = [str(x) for x in row.get('excluded_ids', []) if str(x) in ids]
        assert set(gold) <= ids and not set(gold) & set(excluded)
        tasks.append({'id': f"bright-{domain}-{row['id']}", 'question': row['query'],
                      'gold_doc_ids': gold, 'excluded_ids': excluded})
    train, dev, test = split_tasks(tasks, train_ratio=train_ratio, dev_ratio=dev_ratio, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    raw_corpus = json.dumps(corpus, ensure_ascii=False).encode('utf-8')
    (out / 'corpus.json').write_bytes(raw_corpus)
    original_indices = {t['id']: i for i, t in enumerate(tasks)}
    for name, group in [('train', train), ('dev', dev), ('test', test)]:
        payload = {'partition': name, 'seed': seed, 'split_ratio': list(split_ratio),
                   'task_indices': [original_indices[t['id']] for t in group], 'tasks': group}
        (out / f'{name}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'corpus_sha256': hashlib.sha256(raw_corpus).hexdigest(),
                'documents': len(corpus), 'queries': len(tasks)}
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--domain', default='biology')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split-ratio', type=int, nargs=3, default=[4, 3, 3], metavar=('TRAIN', 'DEV', 'TEST'))
    a = p.parse_args()
    prepare(a.raw_dir, a.out, a.domain, seed=a.seed, split_ratio=a.split_ratio)
