from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from precise_attribution.core.models import Estimate, ValidationResult


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_report(path: Path, estimate: Estimate, validation: ValidationResult) -> None:
    confidence_label = f"{estimate.confidence_level * 100:g}%"
    lines = [
        f"# 实验报告：{estimate.experiment_id}",
        "",
        f"- 场景：`{estimate.scenario_id}`",
        f"- 完整配对区块数：{estimate.n_blocks}",
        f"- 模块：{', '.join(estimate.modules)}",
        "- 主效应与交互效应均按其余模块状态的均匀分布平均。",
        "",
        "## 单模块平均主效应",
        "",
        "| 模块 | 估计值 | 标准误 | 95% 区间 |",
        "|---|---:|---:|---:|",
    ]
    for module, item in estimate.main_effects.items():
        lines.append(f"| {module} | {item['estimate']:.4f} | {item['standard_error']:.4f} | [{item['ci_low']:.4f}, {item['ci_high']:.4f}] |")
    lines = [line.replace("95% 区间", f"{confidence_label} 区间") for line in lines]
    lines += ["", "## 两两交互效应", "", f"| 模块对 | 估计值 | 标准误 | {confidence_label} 区间 |", "|---|---:|---:|---:|"]
    for pair, item in estimate.pairwise_interactions.items():
        lines.append(f"| {pair} | {item['estimate']:.4f} | {item['standard_error']:.4f} | [{item['ci_low']:.4f}, {item['ci_high']:.4f}] |")
    lines += ["", "## Shapley 贡献分配", "", "| 模块 | Shapley 值 |", "|---|---:|"]
    for module, value in estimate.shapley_values.items():
        lines.append(f"| {module} | {value:.4f} |")
    lines += [
        "", "## 独立真值验证", "",
        f"- 主效应符号识别准确率：{validation.main_effect_sign_accuracy:.3f}",
        f"- 非零交互符号识别准确率：{validation.interaction_sign_accuracy:.3f}",
        f"- Shapley 效率误差：{validation.shapley_efficiency_error:.8f}",
        "", "### 主效应误差（估计值 - 模拟真值）", "",
    ]
    for key, value in validation.main_effect_errors.items():
        lines.append(f"- {key}：{value:+.4f}")
    lines += ["", "### 交互效应误差", ""]
    for key, value in validation.interaction_errors.items():
        lines.append(f"- {key}：{value:+.4f}")
    lines += ["", "## 解释限制", ""]
    lines.extend(f"- {note}" for note in estimate.notes)
    lines += [
        "- 当前仅支持 `bypass` 干预；其他干预语义尚未实现。",
        "- 当前真值隔离是软件接口边界，不是操作系统级安全隔离。",
        "- 本报告用于研究原型验证，不代表真实知识密集型系统的外部有效性。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
