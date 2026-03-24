from __future__ import annotations

from typing import Any

import numpy as np


DIFFERENCE_LABEL_ADV = 1
DIFFERENCE_LABEL_CLEAN = 0
CHALLENGE_LABEL_POSITIVE = 1
CHALLENGE_LABEL_NEGATIVE = 0

MECHANISM_THROUGHPUT = "throughput_harm"
MECHANISM_RTT = "rtt_harm"
MECHANISM_LOSS = "loss_harm"
BASELINE_WINNER_RENO = "reno_wins"
BASELINE_WINNER_BBR = "bbr_wins"
BASELINE_WINNER_CUBIC = "cubic_wins"
ATTRIBUTION_ENV_DRIVEN = "env_driven_gap"
ATTRIBUTION_ACTION_AMPLIFIED = "action_amplified_gap"
ATTRIBUTION_POLICY_INDUCED = "policy_induced_gap"

BASELINE_WINNER_LABELS: tuple[str, ...] = (
    BASELINE_WINNER_RENO,
    BASELINE_WINNER_BBR,
    BASELINE_WINNER_CUBIC,
)
MECHANISM_LABELS: tuple[str, ...] = (
    MECHANISM_THROUGHPUT,
    MECHANISM_RTT,
    MECHANISM_LOSS,
)
ATTRIBUTION_LABELS: tuple[str, ...] = (
    ATTRIBUTION_ENV_DRIVEN,
    ATTRIBUTION_ACTION_AMPLIFIED,
    ATTRIBUTION_POLICY_INDUCED,
)

RATE_DEFICIT_COL = "best_minus_sage_rate_contrib_mean"
RTT_DEFICIT_COL = "best_minus_sage_rtt_contrib_mean"
LOSS_EXCESS_COL = "sage_minus_best_loss_penalty_mean"
HARD_GAP_PCT_MEAN_COL = "hard_gap_percent_mean"
HARD_BASELINE_SCORE_MEAN_COL = "hard_baseline_score_mean"
DOMINANT_BEST_BASELINE_METHOD_COL = "dominant_best_baseline_method"
ACTION_REFERENCE_AVAILABILITY_COL = "action_reference_available_fraction"
ENV_STRESS_SCORE_COL = "env_stress_score"
ACTION_MISMATCH_SCORE_COL = "action_mismatch_score"
INTERACTION_AMPLIFICATION_SCORE_COL = "interaction_amplification_score"


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def difference_label(trace_type: str) -> int:
    return int(DIFFERENCE_LABEL_ADV if str(trace_type).strip().lower() == "adv" else DIFFERENCE_LABEL_CLEAN)


def challenge_label(
    row: dict[str, Any],
    *,
    gap_pct_threshold: float,
    baseline_score_floor: float,
) -> int:
    gap_pct = _to_float(row.get(HARD_GAP_PCT_MEAN_COL, 0.0))
    baseline_score = _to_float(row.get(HARD_BASELINE_SCORE_MEAN_COL, 0.0))
    if not np.isfinite(gap_pct) or not np.isfinite(baseline_score):
        return int(CHALLENGE_LABEL_NEGATIVE)
    if baseline_score < float(baseline_score_floor):
        return int(CHALLENGE_LABEL_NEGATIVE)
    return int(CHALLENGE_LABEL_POSITIVE if gap_pct >= float(gap_pct_threshold) else CHALLENGE_LABEL_NEGATIVE)


def mechanism_strengths(row: dict[str, Any]) -> dict[str, float]:
    rate_deficit = max(_to_float(row.get(RATE_DEFICIT_COL, 0.0)), 0.0)
    rtt_deficit = max(_to_float(row.get(RTT_DEFICIT_COL, 0.0)), 0.0)
    loss_excess = max(_to_float(row.get(LOSS_EXCESS_COL, 0.0)), 0.0)
    return {
        MECHANISM_THROUGHPUT: float(rate_deficit),
        MECHANISM_RTT: float(rtt_deficit),
        MECHANISM_LOSS: float(loss_excess),
    }


def mechanism_shares(row: dict[str, Any]) -> dict[str, float]:
    strengths = mechanism_strengths(row)
    total = float(sum(strengths.values()))
    if total <= 1e-12:
        return {key: 0.0 for key in strengths}
    return {key: float(value) / total for key, value in strengths.items()}


def mechanism_label_map(
    row: dict[str, Any],
    *,
    challenge_gap_pct_threshold: float,
    baseline_score_floor: float,
    share_threshold: float,
    min_strength: float,
) -> dict[str, int]:
    if challenge_label(
        row,
        gap_pct_threshold=float(challenge_gap_pct_threshold),
        baseline_score_floor=float(baseline_score_floor),
    ) != int(CHALLENGE_LABEL_POSITIVE):
        return {label: 0 for label in MECHANISM_LABELS}

    strengths = mechanism_strengths(row)
    shares = mechanism_shares(row)
    output: dict[str, int] = {}
    for label in MECHANISM_LABELS:
        output[label] = int(
            strengths[label] >= float(min_strength) and shares[label] >= float(share_threshold)
        )
    return output


def baseline_winner_label(
    row: dict[str, Any],
    *,
    method: str,
    challenge_gap_pct_threshold: float,
    baseline_score_floor: float,
    min_fraction: float,
) -> int:
    if challenge_label(
        row,
        gap_pct_threshold=float(challenge_gap_pct_threshold),
        baseline_score_floor=float(baseline_score_floor),
    ) != int(CHALLENGE_LABEL_POSITIVE):
        return 0

    target_method = str(method).strip().lower()
    dominant_method = str(row.get(DOMINANT_BEST_BASELINE_METHOD_COL, "")).strip().lower()
    winner_fraction = max(_to_float(row.get(f"best_baseline_fraction_{target_method}", 0.0)), 0.0)
    return int(dominant_method == target_method and winner_fraction >= float(min_fraction))


def attribution_scores(row: dict[str, Any]) -> dict[str, float]:
    return {
        "env": max(_to_float(row.get(ENV_STRESS_SCORE_COL, 0.0)), 0.0),
        "action": max(_to_float(row.get(ACTION_MISMATCH_SCORE_COL, 0.0)), 0.0),
        "interaction": max(_to_float(row.get(INTERACTION_AMPLIFICATION_SCORE_COL, 0.0)), 0.0),
        "availability": max(_to_float(row.get(ACTION_REFERENCE_AVAILABILITY_COL, 0.0)), 0.0),
    }


def attribution_label_map(
    row: dict[str, Any],
    *,
    challenge_gap_pct_threshold: float,
    baseline_score_floor: float,
    env_high_threshold: float,
    action_high_threshold: float,
    action_low_threshold: float,
    interaction_high_threshold: float,
    interaction_low_threshold: float,
    availability_floor: float,
) -> dict[str, int]:
    if challenge_label(
        row,
        gap_pct_threshold=float(challenge_gap_pct_threshold),
        baseline_score_floor=float(baseline_score_floor),
    ) != int(CHALLENGE_LABEL_POSITIVE):
        return {label: 0 for label in ATTRIBUTION_LABELS}

    scores = attribution_scores(row)
    if scores["availability"] < float(availability_floor):
        return {label: 0 for label in ATTRIBUTION_LABELS}

    env_score = float(scores["env"])
    action_score = float(scores["action"])
    interaction_score = float(scores["interaction"])
    output = {label: 0 for label in ATTRIBUTION_LABELS}
    if (
        env_score >= float(env_high_threshold)
        and action_score <= float(action_low_threshold)
        and interaction_score <= float(interaction_low_threshold)
    ):
        output[ATTRIBUTION_ENV_DRIVEN] = 1
    if env_score >= float(env_high_threshold) and (
        action_score >= float(action_high_threshold) or interaction_score >= float(interaction_high_threshold)
    ):
        output[ATTRIBUTION_ACTION_AMPLIFIED] = 1
    if env_score < float(env_high_threshold) and (
        action_score >= float(action_high_threshold) or interaction_score >= float(interaction_high_threshold)
    ):
        output[ATTRIBUTION_POLICY_INDUCED] = 1
    return output
