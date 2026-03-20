from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


RISKY_LABEL = 1
SAFE_LABEL = 0
ACTIVE_LABEL = 1
INACTIVE_LABEL = 0


def hard_gap_percent(*, best_baseline_gap: float, best_baseline_score: float) -> float:
    denominator = float(best_baseline_score)
    if not np.isfinite(denominator) or denominator <= 1e-9:
        return float("nan")
    return float(100.0 * float(best_baseline_gap) / denominator)


def best_baseline_method(row: Mapping[str, object], *, baseline_methods: Sequence[str]) -> str | None:
    best_method: str | None = None
    best_score = float("-inf")
    for method in baseline_methods:
        score = float(row.get(f"gap_score_{method}", float("nan")))
        if not np.isfinite(score):
            continue
        if best_method is None or score > best_score:
            best_method = str(method)
            best_score = float(score)
    return best_method


def weak_direction_labels(
    *,
    risky: bool,
    sage_previous_action: float,
    best_baseline_previous_action: float,
    action_margin: float,
) -> tuple[int, int]:
    if not bool(risky):
        return INACTIVE_LABEL, INACTIVE_LABEL
    if not np.isfinite(float(best_baseline_previous_action)):
        return INACTIVE_LABEL, INACTIVE_LABEL
    delta = float(sage_previous_action) - float(best_baseline_previous_action)
    if delta > float(action_margin):
        return ACTIVE_LABEL, INACTIVE_LABEL
    if delta < -float(action_margin):
        return INACTIVE_LABEL, ACTIVE_LABEL
    return INACTIVE_LABEL, INACTIVE_LABEL
