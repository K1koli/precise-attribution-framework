from __future__ import annotations

from itertools import combinations


def all_coalitions(modules: tuple[str, ...]) -> list[frozenset[str]]:
    result: list[frozenset[str]] = []
    for size in range(len(modules) + 1):
        result.extend(frozenset(x) for x in combinations(modules, size))
    return result
