"""Evaluate only the seeded BRIGHT subset test partition with the fixed BM25 baseline."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dataclasses import asdict
import re
from precise_attribution.evolution.multi_task import prepared_groups, evaluate_set, task_fingerprint
from precise_attribution.evaluation.benchmarks import benchmark_adapter
from precise_attribution.harness.modules import HarnessConfig
from precise_attribution.evolution.prepared_data import load_prepared_data


class NoLLM:
    def chat(self, *args, **kwargs):
        raise RuntimeError('Baseline evaluation must not call an LLM')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset', default='biology', help='BRIGHT subset, e.g. biology, economics, robotics')
    parser.add_argument('--data-dir', type=Path, default=Path('data/bright/processed'))
    parser.add_argument('--seed', type=int, help='Optional check against the prepared seed; does not resplit')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]*', args.subset):
        parser.error('--subset must be a subset name, not a path')
    subset_dir = args.data_dir / args.subset
    corpus = (subset_dir / 'corpus.json').resolve()
    if not corpus.is_file() or not all((subset_dir / f'{part}.json').is_file() for part in ('train', 'dev', 'test')):
        parser.error(f'Missing processed data in {subset_dir}; run prepare_bright_full.py --domain {args.subset} first')
    cfg = HarnessConfig(query_mode='identity', retrieval_backend='bm25',
                        corpus_file=str(corpus), corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
                        top_k_retrieval=10, top_k_ranked=10)
    c = {'harness': asdict(cfg), 'tasks_dir': str(subset_dir.resolve()), 'benchmark': 'bright_full_ndcg10'}
    payload = load_prepared_data(subset_dir)
    tasks = payload['tasks']
    train, dev, test = prepared_groups(tasks, payload.get('split'))
    seed = payload['split']['seed']
    if args.seed is not None and args.seed != seed:
        parser.error('--seed differs from prepared data; regenerate data to change the split')
    args.seed = seed
    args.output.mkdir(parents=True, exist_ok=False)
    rows = evaluate_set(test, NoLLM(), cfg, benchmark_adapter(c['benchmark']),
                        bypass_modules=['EvidencePreparation', 'EvidenceReasoning'])
    errors = sum(r['error'] is not None for r in rows)
    result = {'subset': args.subset, 'config': c, 'seed': args.seed, 'evaluated_split': 'test',
              'split': {k: [t['id'] for t in group] for k, group in [('train',train),('dev',dev),('test',test)]},
              'timestamp': datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
              'dataset_sha256': task_fingerprint(tasks), 'errors':errors,
              'source_sha256': {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                               ['src/precise_attribution/harness/bm25.py','src/precise_attribution/harness/modules.py','src/precise_attribution/evaluation/benchmarks.py']},
              'baseline': rows, 'used_for_acceptance': False}
    if errors == 0:
        for task, row in zip(test, rows):
            assert row['trace']['query']['queries'] == [task['question']]
            docs = row['trace']['retrieved_documents']
            assert len(docs) == len({d['id'] for d in docs}) == 10
            assert not {d['id'] for d in docs} & set(task.get('excluded_ids', []))
        result['mean_ndcg10'] = sum(r['score'] for r in rows) / len(rows)
    (args.output/'test_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    if errors:
        raise RuntimeError(f'{errors} evaluation errors; inspect test_report.json')
    print(json.dumps({k:v for k,v in result.items() if k in ['subset','seed','evaluated_split','timestamp','errors','mean_ndcg10']},indent=2))


if __name__ == '__main__':
    main()
