from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("实验配置必须是 YAML 对象")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("experiment", "scenario", "sampling"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"配置缺少对象字段：{section}")
    scenario = config["scenario"]
    modules = scenario.get("modules")
    if not isinstance(modules, list) or not modules or len(set(modules)) != len(modules):
        raise ValueError("scenario.modules 必须是非空且不重复的列表")
    if len(modules) > 10:
        raise ValueError("当前原型最多支持 10 个模块，以限制完整联盟枚举成本")
    effects = scenario.get("main_effects", {})
    if set(effects) != set(modules):
        raise ValueError("main_effects 必须为每个模块提供且仅提供一个系数")
    confidence = float(config["experiment"].get("confidence_level", 0.95))
    if not 0.0 < confidence < 1.0:
        raise ValueError("experiment.confidence_level 必须在 0 和 1 之间")
    if int(config["sampling"].get("blocks", 0)) < 2:
        raise ValueError("sampling.blocks 至少为 2，才能估计区间")
    for key in ("noise_sd", "task_heterogeneity_sd", "interaction_heterogeneity_sd"):
        if float(scenario.get(key, 0.0)) < 0:
            raise ValueError(f"{key} 不得为负数")
    for pair in scenario.get("pairwise_interactions", {}):
        parts = pair.split("|")
        if len(parts) != 2 or any(p not in modules for p in parts) or parts[0] == parts[1]:
            raise ValueError(f"无效的交互模块对：{pair}")
        if modules.index(parts[0]) >= modules.index(parts[1]):
            raise ValueError(f"交互模块对必须按 scenario.modules 顺序书写：{pair}")
