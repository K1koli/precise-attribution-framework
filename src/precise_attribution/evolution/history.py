"""Allowlisted historical feedback. Never forward raw dev/test rows or errors."""
import json
import math
from pathlib import Path


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def summarize(report):
    baseline = number(report.get('baseline_dev_mean'))
    selection = report.get('selection', {})
    gate = report.get('gate', {})
    attempts = []
    candidates = report.get('candidates', [])
    seen = {c.get('id') for c in candidates}
    for c in candidates + [p for p in report.get('proposals', []) if p.get('id') not in seen]:
        delta = c.get('dev_delta', {})
        gain = number(delta.get('mean'))
        ci = delta.get('ci95', [None, None])
        ci = [number(x) for x in ci] if isinstance(ci, list) and len(ci) == 2 else [None, None]
        accepted = bool(selection.get('accepted') and selection.get('candidate_id') == c.get('id'))
        reasons = []
        if not accepted:
            if c.get('status') != 'evaluated': reasons.append('未通过源码/运行/行为检查或未评估')
            if gain is None: reasons.append('无有效配对增益')
            elif gain < gate.get('minimum_dev_gain', .01): reasons.append('dev平均增益未达门槛')
            if gate.get('require_positive_ci_lower_bound', True) and (ci[0] is None or ci[0] <= 0):
                reasons.append('增益区间下界未大于0')
            if number(delta.get('coverage')) is not None and delta['coverage'] < gate.get('minimum_pair_coverage', .9):
                reasons.append('有效配对覆盖率不足')
            if c.get('n_errors', 0): reasons.append('候选运行存在错误')
            if not reasons: reasons.append('未被选择为最佳合格候选')
        attempts.append({'candidate_id': c.get('id'), 'module': c.get('target_module'),
                         'source': c.get('source', ''), 'accepted': accepted,
                         'baseline_dev_mean': baseline,
                         'candidate_dev_mean': number(c.get('dev_mean')),
                         'dev_mean_delta': gain, 'dev_delta_ci95': ci,
                         'coverage': number(delta.get('coverage')),
                         'reason': '已接受' if accepted else '；'.join(reasons)})
        # Historical reports predate dev_mean: derive only a scalar, never forward rows.
        if attempts[-1]['candidate_dev_mean'] is None:
            scores = [number(r.get('score')) for r in c.get('dev', [])]
            scores = [s for s in scores if s is not None]
            attempts[-1]['candidate_dev_mean'] = sum(scores) / len(scores) if scores else None
    return {'attempts': attempts}


def load_history(paths, *, dataset_sha256, split, benchmark, harness_config):
    history = []
    for path in paths:
        p = Path(path)
        report = json.loads(p.read_text(encoding='utf-8'))
        if report.get('dataset_sha256') != dataset_sha256 or report.get('split') != split or report.get('benchmark') != benchmark:
            raise ValueError('历史实验的题库、划分或指标不一致')
        if report.get('harness_config', {}).get('corpus_sha256') != harness_config.get('corpus_sha256'):
            raise ValueError('历史实验语料不一致')
        history.append({'run': p.parent.name, **summarize(report)})
    return history
