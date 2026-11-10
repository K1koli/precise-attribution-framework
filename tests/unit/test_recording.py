from precise_attribution.core.models import RunRecord, Intervention
from precise_attribution.recording.jsonl import JsonlRecorder


def test_jsonl_round_trip_and_append(tmp_path):
    record = RunRecord(
        run_id="r1", experiment_id="e1", scenario_id="s1", task_id="t1", block_id="b1",
        seed=7, coalition=("A",), module_states={"A": 1}, interventions=(Intervention("B"),), outcome=1.25
    )
    recorder = JsonlRecorder(tmp_path / "runs.jsonl")
    recorder.append_many([record])
    recorder.append_many([record])
    loaded = recorder.read_all()
    assert len(loaded) == 2
    assert loaded[0] == record
