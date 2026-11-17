import hashlib
import json

import pytest

from precise_attribution.harness.bm25 import BM25Index, corpus_index
from precise_attribution.harness.modules import HarnessConfig, RAGHarness
from precise_attribution.evolution.code_mutation import default_source, load_candidate_module


class NoLLM:
    def chat(self, *args, **kwargs):
        raise AssertionError('Baseline must not call LLM')


def test_full_corpus_no_gold_injection_or_llm(tmp_path):
    docs = [{'id': str(i), 'title': '', 'text': 'cell biology' if i < 12 else 'unrelated'} for i in range(15)]
    raw = json.dumps(docs).encode()
    path = tmp_path / 'corpus.json'
    path.write_bytes(raw)
    cfg = HarnessConfig(query_mode='identity', retrieval_backend='bm25', top_k_ranked=10,
                        corpus_file=str(path), corpus_sha256=hashlib.sha256(raw).hexdigest())
    task = {'id': 'q', 'question': 'cell', 'gold_doc_ids': ['14'], 'excluded_ids': ['0']}
    bypass = {'EvidencePreparation': 'bypass', 'EvidenceReasoning': 'bypass'}
    result = RAGHarness(NoLLM(), cfg).run(task, interventions=bypass)
    ranked = result['trace']['retrieved_documents']
    assert len(ranked) == 10
    assert '14' not in {d['id'] for d in ranked}
    assert '0' not in {d['id'] for d in ranked}
    task['gold_doc_ids'] = ['1']
    assert RAGHarness(NoLLM(), cfg).run(task, interventions=bypass)['trace']['retrieved_documents'] == ranked
    for name in ['QueryUnderstanding', 'EvidenceRetrieval']:
        module = load_candidate_module(default_source(name, cfg), NoLLM())
        replaced = RAGHarness(NoLLM(), cfg, {name: module}).run(task, interventions=bypass)
        assert replaced['trace']['retrieved_documents'] == ranked
    with pytest.raises(ValueError, match='checksum'):
        corpus_index(str(path), 'bad')


def test_bm25_prefers_rare_term():
    docs = [{'id': 'a', 'text': 'common rare'}, {'id': 'b', 'text': 'common common'}, {'id': 'c', 'text': 'common'}]
    assert BM25Index(docs).search(['rare'], 1)[0]['id'] == 'a'
