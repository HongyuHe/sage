from __future__ import annotations

from typing import Any

import numpy as np


SCORE_RATE_WEIGHT = 0.6
SCORE_RTT_WEIGHT = 0.25
SCORE_LOSS_WEIGHT = 0.15


def bounded_linear_score_terms(
    *,
    base_rtt_ms: float,
    current_rtt_ms: float,
    windowed_rate_mbps: float,
    current_loss_mbps: float,
    path_cap_mbps: float,
) -> dict[str, float]:
    current_rtt_ms = float(np.nan_to_num(current_rtt_ms, nan=0.0, posinf=0.0, neginf=0.0))
    windowed_rate_mbps = float(np.nan_to_num(windowed_rate_mbps, nan=0.0, posinf=0.0, neginf=0.0))
    current_loss_mbps = float(np.nan_to_num(current_loss_mbps, nan=0.0, posinf=0.0, neginf=0.0))
    base_rtt_ms = float(np.nan_to_num(base_rtt_ms, nan=0.0, posinf=0.0, neginf=0.0))
    path_cap_mbps = float(np.nan_to_num(path_cap_mbps, nan=0.0, posinf=0.0, neginf=0.0))
    current_rtt_ms = max(current_rtt_ms, 1e-6)
    windowed_rate_mbps = max(windowed_rate_mbps, 0.0)
    current_loss_mbps = max(current_loss_mbps, 0.0)
    base_rtt_ms = max(base_rtt_ms, 1e-6)
    path_cap_mbps = max(path_cap_mbps, 1e-6)
    rate_norm = float(np.clip(windowed_rate_mbps / path_cap_mbps, 0.0, 1.0))
    rtt_norm = float(np.clip(base_rtt_ms / current_rtt_ms, 0.0, 1.0))
    loss_norm = float(np.clip(current_loss_mbps / path_cap_mbps, 0.0, 1.0))
    rate_contrib = float(SCORE_RATE_WEIGHT * rate_norm)
    rtt_contrib = float(SCORE_RTT_WEIGHT * rtt_norm)
    loss_penalty = float(SCORE_LOSS_WEIGHT * loss_norm)
    score = float(np.clip(rate_contrib + rtt_contrib - loss_penalty, 0.0, 1.0))
    return {
        "rate_norm": rate_norm,
        "rtt_norm": rtt_norm,
        "loss_norm": loss_norm,
        "rate_contrib": rate_contrib,
        "rtt_contrib": rtt_contrib,
        "loss_penalty": loss_penalty,
        "score": score,
    }


def bounded_linear_score_terms_from_info(
    info: dict[str, Any],
    *,
    base_rtt_ms: float,
    path_cap_mbps: float,
) -> dict[str, float]:
    return bounded_linear_score_terms(
        base_rtt_ms=base_rtt_ms,
        current_rtt_ms=float(info.get("sage/current_rtt_ms", 0.0)),
        windowed_rate_mbps=float(info.get("sage/windowed_delivery_rate_mbps", 0.0)),
        current_loss_mbps=float(info.get("sage/current_loss_mbps", 0.0)),
        path_cap_mbps=path_cap_mbps,
    )
