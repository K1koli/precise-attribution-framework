from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Intervention:
    module: str
    kind: str = "bypass"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    experiment_id: str
    scenario_id: str
    task_id: str
    block_id: str
    seed: int
    coalition: tuple[str, ...]
    module_states: dict[str, int]
    interventions: tuple[Intervention, ...]
    outcome: float
    status: str = "success"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["coalition"] = list(self.coalition)
        data["interventions"] = [asdict(item) for item in self.interventions]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        return cls(
            **{
                **data,
                "coalition": tuple(data["coalition"]),
                "interventions": tuple(Intervention(**x) for x in data["interventions"]),
            }
        )


@dataclass(frozen=True)
class Estimate:
    experiment_id: str
    scenario_id: str
    modules: tuple[str, ...]
    n_blocks: int
    confidence_level: float
    coalition_values: dict[str, float]
    main_effects: dict[str, dict[str, float]]
    pairwise_interactions: dict[str, dict[str, float]]
    shapley_values: dict[str, float]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationResult:
    experiment_id: str
    main_effect_errors: dict[str, float]
    interaction_errors: dict[str, float]
    shapley_errors: dict[str, float]
    main_effect_sign_accuracy: float
    interaction_sign_accuracy: float
    shapley_efficiency_error: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
