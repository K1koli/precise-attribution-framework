from __future__ import annotations

from precise_attribution.core.models import Intervention

SUPPORTED_KINDS = {"bypass"}


def interventions_for_coalition(modules: tuple[str, ...], active: set[str]) -> tuple[Intervention, ...]:
    return tuple(Intervention(module=m, kind="bypass") for m in modules if m not in active)


def apply_interventions(modules: tuple[str, ...], requested_active: set[str], interventions: tuple[Intervention, ...]) -> set[str]:
    """当前合成执行器支持 bypass；其他语义明确报错，不静默等价处理。"""
    effective = set(requested_active)
    for intervention in interventions:
        if intervention.kind not in SUPPORTED_KINDS:
            raise NotImplementedError(f"当前执行器尚不支持干预语义：{intervention.kind}")
        if intervention.module not in modules:
            raise ValueError(f"干预目标模块不存在：{intervention.module}")
        effective.discard(intervention.module)
    return effective
