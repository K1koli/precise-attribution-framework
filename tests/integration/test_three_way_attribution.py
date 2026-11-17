import copy
import json

import pytest

from precise_attribution.evolution import multi_task as mt


SOURCE = '''class CandidateModule:
    def run(self, inputs):
        return {"context": "updated evidence"}
'''


def task_set():
    return [
        {"id": f"task-{i}", "question": f"training-question-{i}",
         "expected_answer": f"reference-{i}",
         "documents": [{"id": f"doc-{i}", "text": f"evidence-{i}"}]}
        for i in range(15)
    ]


def test_split_is_reproducible_disjoint_and_complete():
    tasks = task_set()
    groups = mt.split_tasks(tasks)
    assert groups == mt.split_tasks(tasks)
    assert [len(group) for group in groups] == [9, 3, 3]
    ids = [task['id'] for group in groups for task in group]
    assert len(ids) == len(set(ids)) == len(tasks)
    assert all(mt.split_tasks(tasks[:3]))


@pytest.mark.parametrize('train,dev', [(0, .2), (.6, 0), (.8, .2), (-.1, .2), (float('nan'), .2)])
def test_invalid_ratios_are_rejected(train, dev):
    with pytest.raises(ValueError):
        mt.split_tasks(task_set(), train_ratio=train, dev_ratio=dev)


def test_duplicate_ids_are_rejected():
    tasks = task_set()
    tasks[1]['id'] = tasks[0]['id']
    with pytest.raises(ValueError):
        mt.split_tasks(tasks)


def install_spies(monkeypatch, tasks):
    train, dev, test = mt.split_tasks(tasks)
    train_ids = {t['id'] for t in train}
    test_ids = {t['id'] for t in test}
    state = {'evaluated': [], 'proposed': False, 'test_open': False}

    def evaluate(batch, llm, config, adapter, interventions=None, modules=None, bypass_modules=()):
        ids = {t['id'] for t in batch}
        if ids & test_ids:
            assert state['test_open'], 'test evaluated during optimization'
        state['evaluated'].append(ids)
        # Candidate improves dev but worsens test: final test must not undo acceptance.
        score = (0.0 if ids & test_ids else 1.0) if modules else .5
        return [{'task_id': t['id'], 'question': t['question'], 'score': score,
                 'answer': 'baseline answer', 'verification': {}, 'trace': {}, 'error': None}
                for t in batch]

    def diagnose(self, examples, available, priors):
        assert all(e['task_id'] in train_ids for e in examples)
        by_id = {t['id']: t for t in train}
        assert examples
        for e in examples:
            assert e['reference']['expected_answer'] == by_id[e['task_id']]['expected_answer']
        return {'summary': 'training diagnosis', 'hypotheses': [{'module': 'EvidencePreparation', 'prior_ids': []}]}

    def propose(llm, **kwargs):
        assert {t['id'] for t in kwargs['training_tasks']} == train_ids
        assert all(f['task_id'] in train_ids for f in kwargs['failures'])
        # Dev evaluation must happen only after all proposals have been produced.
        assert all(ids <= train_ids for ids in state['evaluated'])
        state['proposed'] = True
        return SOURCE, 1

    monkeypatch.setattr(mt, 'evaluate_set', evaluate)
    monkeypatch.setattr(mt.Critic, 'diagnose_modules', diagnose)
    monkeypatch.setattr(mt, 'propose_code_candidate', propose)
    return state


def run_experiment(monkeypatch, tmp_path):
    tasks = task_set()
    state = install_spies(monkeypatch, tasks)
    report = mt.run_multi_task_attribution(
        tasks, None, tmp_path, max_proposals=1,
        prior_library={'library_id': 'empty', 'operators': []},
    )
    return tasks, state, report


def test_train_only_generation_dev_selection_and_independent_test(monkeypatch, tmp_path):
    tasks, state, report = run_experiment(monkeypatch, tmp_path)
    assert state['proposed']
    assert report['selection']['accepted']
    assert report['test_evaluation']['status'] == 'not_run'
    assert (tmp_path / 'selected_module.py').exists()
    assert 'train / dev / test' in mt.render_markdown(report)
    frozen_report = (tmp_path / 'multi_task_attribution.json').read_bytes()
    state['test_open'] = True
    result = mt.evaluate_final_test(tasks, None, tmp_path)
    assert result['delta']['mean'] < 0
    assert result['used_for_acceptance'] is False
    assert (tmp_path / 'multi_task_attribution.json').read_bytes() == frozen_report
    with pytest.raises(FileExistsError):
        mt.evaluate_final_test(tasks, None, tmp_path)


def test_final_test_rejects_changed_data_and_source(monkeypatch, tmp_path):
    tasks, state, report = run_experiment(monkeypatch, tmp_path)
    changed = copy.deepcopy(tasks)
    changed[-1]['expected_answer'] = 'changed'
    with pytest.raises(ValueError, match='题库'):
        mt.evaluate_final_test(changed, None, tmp_path)
    (tmp_path / 'selected_module.py').write_text(SOURCE + '\n# modified\n')
    with pytest.raises(ValueError, match='源码'):
        mt.evaluate_final_test(tasks, None, tmp_path)


def test_no_accepted_candidate_tests_baseline(monkeypatch, tmp_path):
    tasks = task_set()
    state = install_spies(monkeypatch, tasks)
    report = mt.run_multi_task_attribution(tasks, None, tmp_path, max_proposals=0,
                                         prior_library={'library_id': 'empty', 'operators': []})
    assert not report['selection']['accepted']
    state['test_open'] = True
    result = mt.evaluate_final_test(tasks, None, tmp_path)
    assert result['candidate_id'] is None
    assert result['final'] == result['baseline']



def test_history_summary_reaches_critic_but_not_candidate_start(monkeypatch, tmp_path):
    tasks, _, old = run_experiment(monkeypatch, tmp_path / 'old')
    old['selection']['accepted'] = False
    old['candidates'][0]['dev'][0]['question'] = 'SECRET_DEV_QUESTION'
    old['test_report'] = 'SECRET_TEST_RESULT'
    old['selection']['reason'] = 'SECRET_DEV_REASON_DETAIL'
    path = tmp_path / 'old' / 'multi_task_attribution.json'
    path.write_text(json.dumps(old))
    state = install_spies(monkeypatch, tasks)
    diagnose = mt.Critic.diagnose_modules
    propose = mt.propose_code_candidate
    seen = []

    def inspect_history(self, examples, available, priors, history=None):
        assert history
        text = json.dumps(history)
        assert 'SECRET_' not in text
        assert history[0]['attempts'][0]['candidate_dev_mean'] == 1.0
        assert history[0]['attempts'][0]['accepted'] is False
        seen.append(history)
        return diagnose(self, examples, available, priors)

    def inspect_source(llm, **kwargs):
        assert kwargs['current_source'] == mt.default_source('EvidencePreparation')
        return propose(llm, **kwargs)

    monkeypatch.setattr(mt.Critic, 'diagnose_modules', inspect_history)
    monkeypatch.setattr(mt, 'propose_code_candidate', inspect_source)
    report = mt.run_multi_task_attribution(tasks, None, tmp_path / 'new', max_proposals=1,
        history_reports=[str(path)], prior_library={'library_id': 'empty', 'operators': []})
    assert seen and state['proposed']
    assert report['history_summary'] == seen[0]
    assert report['test_evaluation']['status'] == 'not_run'


def test_prepared_split_is_used_without_reshuffling(monkeypatch, tmp_path):
    tasks = task_set()
    groups = mt.split_tasks(tasks)
    split = {k: [t['id'] for t in g] for k, g in zip(['train','dev','test'], groups)}
    split['seed'] = 42
    state = install_spies(monkeypatch, tasks)
    def no_runtime_split(*args, **kwargs):
        raise AssertionError('Runtime reshuffled prepared data')
    monkeypatch.setattr(mt, 'split_tasks', no_runtime_split)
    report = mt.run_multi_task_attribution(tasks, None, tmp_path, max_proposals=0, prepared_split=split,
                                         prior_library={'library_id':'empty','operators':[]})
    assert report['split'] == split
    assert report['test_evaluation']['status'] == 'not_run'


@pytest.mark.parametrize('kind', ['duplicate', 'missing', 'unknown'])
def test_invalid_prepared_split_rejected(kind):
    tasks = task_set()
    split = {'train':[t['id'] for t in tasks[:9]], 'dev':[t['id'] for t in tasks[9:12]],
             'test':[t['id'] for t in tasks[12:]], 'seed':42}
    if kind == 'duplicate': split['dev'][0] = split['train'][0]
    elif kind == 'missing': split['test'].pop()
    else: split['test'][0] = 'unknown'
    with pytest.raises(ValueError): mt.prepared_groups(tasks, split)
