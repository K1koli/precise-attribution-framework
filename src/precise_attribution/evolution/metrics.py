from __future__ import annotations

import re


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize(prediction), normalize(reference)
    if not pred or not ref:
        return float(pred == ref)
    from collections import Counter
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)
