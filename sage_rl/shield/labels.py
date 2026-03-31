from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


RISKY_LABEL = 1
SAFE_LABEL = 0
ACTIVE_LABEL = 1
INACTIVE_LABEL = 0
NOOP_LABEL = 0
BACKOFF_LABEL = 1
PUSH_HARDER_LABEL = 2

ACTION_LABEL_NAMES: dict[int, str] = {
    int(NOOP_LABEL): "noop",
    int(BACKOFF_LABEL): "back_off",
    int(PUSH_HARDER_LABEL): "push_harder",
}


def hard_gap_percent(*, best_baseline_gap: float, best_baseline_score: float) -> float:
    denominator = float(best_baseline_score)
    if not np.isfinite(denominator) or denominator <= 1e-9:
        return float("nan")
    return float(100.0 * float(best_baseline_gap) / denominator)


def action_label_name(label: int) -> str:
    return str(ACTION_LABEL_NAMES.get(int(label), "noop"))


def is_risky_state(
    *,
    hard_gap_percent: float,
    hard_baseline_score: float,
    risk_gap_pct: float,
    baseline_score_floor: float,
) -> bool:
    gap_pct = float(hard_gap_percent)
    baseline_score = float(hard_baseline_score)
    valid = np.isfinite(gap_pct) and np.isfinite(baseline_score) and baseline_score >= float(baseline_score_floor)
    return bool(valid and gap_pct >= float(risk_gap_pct))


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


def unified_action_label(
    *,
    hard_gap_percent_value: float,
    hard_baseline_score: float,
    risk_gap_pct: float,
    baseline_score_floor: float,
    sage_previous_action: float,
    best_baseline_previous_action: float,
    action_margin: float,
) -> tuple[int, str, str, bool]:
    risky = is_risky_state(
        hard_gap_percent=float(hard_gap_percent_value),
        hard_baseline_score=float(hard_baseline_score),
        risk_gap_pct=float(risk_gap_pct),
        baseline_score_floor=float(baseline_score_floor),
    )
    if not bool(risky):
        return int(NOOP_LABEL), action_label_name(NOOP_LABEL), "not_risky", True
    if not np.isfinite(float(sage_previous_action)):
        return int(NOOP_LABEL), action_label_name(NOOP_LABEL), "missing_sage_previous_action", False
    if not np.isfinite(float(best_baseline_previous_action)):
        return int(NOOP_LABEL), action_label_name(NOOP_LABEL), "missing_best_baseline_previous_action", False

    delta = float(sage_previous_action) - float(best_baseline_previous_action)
    if delta > float(action_margin):
        return int(BACKOFF_LABEL), action_label_name(BACKOFF_LABEL), "sage_more_aggressive_than_reference", True
    if delta < -float(action_margin):
        return int(PUSH_HARDER_LABEL), action_label_name(PUSH_HARDER_LABEL), "sage_less_aggressive_than_reference", True
    return int(NOOP_LABEL), action_label_name(NOOP_LABEL), "within_action_margin", True
