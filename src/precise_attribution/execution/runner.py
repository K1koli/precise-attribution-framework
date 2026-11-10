from __future__ import annotations

import uuid
from precise_attribution.core.models import RunRecord
from precise_attribution.interventions.policies import apply_interventions, interventions_for_coalition
from precise_attribution.simulation.scenario import SyntheticScenario, SyntheticTask


def execute_coalition(
    scenario: SyntheticScenario,
    task: SyntheticTask,
    experiment_id: str,
    active: set[str],
) -> RunRecord:
    interventions = interventions_for_coalition(scenario.modules, active)
    effective = apply_interventions(scenario.modules, active, interventions)
    outcome = scenario.outcome(task, effective)
    return RunRecord(
        run_id=str(uuid.uuid4()),
        experiment_id=experiment_id,
        scenario_id=scenario.scenario_id,
        task_id=task.task_id,
        block_id=task.block_id,
        seed=task.seed,
        coalition=tuple(m for m in scenario.modules if m in effective),
        module_states={m: int(m in effective) for m in scenario.modules},
        interventions=interventions,
        outcome=outcome,
        metadata={"intervention_semantics": "bypass", "metric": "score"},
    )
