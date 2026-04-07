"""
Generate a synthetic Sage shield dataset with controlled, closed-set semantics.

Example usage:
python attacks/shield/generate_synthetic_shield_dataset.py \
  --out-dir attacks/shield/shield-dataset/synthetic-deficiency-part1 \
  --allowed-deficiencies under_aggressiveness,over_aggressiveness,delayed_recovery \
  --num-clean-episodes 36 \
  --num-adv-episodes-per-deficiency 12 \
  --episode-steps 48 \
  --deficiencies-per-adv-episode 1

python attacks/shield/generate_synthetic_shield_dataset.py \
  --out-dir attacks/shield/shield-dataset/synthetic-deficiency-part2 \
  --allowed-deficiencies under_aggressiveness,delayed_backoff,rtt_insensitivity \
  --num-clean-episodes 36 \
  --num-adv-episodes-per-deficiency 12 \
  --episode-steps 48 \
  --deficiencies-per-adv-episode 1

python attacks/shield/generate_synthetic_shield_dataset.py \
  --out-dir attacks/shield/shield-dataset/synthetic-deficiency-part3 \
  --allowed-deficiencies loss_insensitivity,over_aggressiveness,delayed_recovery \
  --num-clean-episodes 36 \
  --num-adv-episodes-per-deficiency 12 \
  --episode-steps 48 \
  --deficiencies-per-adv-episode 1
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
import sys
from typing import Any

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from sage_rl.shield.features import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS, ShieldFeatureTracker
from sage_rl.shield.labels import (
    hard_gap_percent,
    unified_action_label,
    weak_direction_labels,
)
from scripts.gen_sage_shield_dataset import write_clean_feature_thresholds_from_dataset


BASELINE_METHODS: tuple[str, ...] = ("reno", "bbr", "cubic")
CLEAN_PROFILE_NAMES: tuple[str, ...] = (
    "benign_steady",
    # "benign_capacity_swing",
    # "benign_mild_loss",
)

SEMANTIC_TAXONOMY: dict[str, dict[str, Any]] = {
    "under_aggressiveness": {
        "intended_action": "push_harder",
        "description": "Sage keeps its action too low despite favorable RTT, low loss, and available delivery-rate headroom.",
        "salient_features": [
            "previous_action",
            "current_rtt_ms",
            "current_loss_mbps",
            "windowed_delivery_rate_mbps",
            "windowed_vs_max_rate_ratio",
        ],
    },
    "over_aggressiveness": {
        "intended_action": "back_off",
        "description": "Sage remains too aggressive under persistent congestion, inflating RTT and loss while not converting that aggression into delivery rate.",
        "salient_features": [
            "previous_action",
            "current_rtt_ms",
            "current_loss_mbps",
            "current_min_rtt_ratio",
            "windowed_vs_max_rate_ratio",
        ],
    },
    "delayed_recovery": {
        "intended_action": "push_harder",
        "description": "Network conditions improve, but Sage recovers too slowly and fails to increase its action quickly enough.",
        "salient_features": [
            "delivery_growth_ratio",
            "max_delivery_growth_ratio",
            "previous_action",
            "current_loss_mbps",
            "current_min_rtt_ratio",
        ],
    },
    "delayed_backoff": {
        "intended_action": "back_off",
        "description": "Congestion signals rise rapidly, yet Sage keeps its action elevated instead of backing off promptly.",
        "salient_features": [
            "current_rtt_ms_delta",
            "current_loss_mbps_delta",
            "previous_action",
            "previous_action_max",
            "windowed_delivery_rate_mbps_delta",
        ],
    },
    "rtt_insensitivity": {
        "intended_action": "back_off",
        "description": "Sage under-reacts to RTT inflation specifically, keeping its action high even when the path latency grows.",
        "salient_features": [
            "current_rtt_ms",
            "current_min_rtt_ratio",
            "rtt_inflation",
            "current_loss_mbps",
            "previous_action",
        ],
    },
    "loss_insensitivity": {
        "intended_action": "back_off",
        "description": "Sage under-reacts to loss spikes, continuing to probe aggressively despite clear loss signals.",
        "salient_features": [
            "current_loss_mbps",
            "current_loss_mbps_avg",
            "current_loss_mbps_max",
            "loss_to_windowed_rate_ratio",
            "previous_action",
        ],
    },
    "unstable_probing": {
        "intended_action": "back_off",
        "description": "Sage oscillates aggressively, producing unstable action deltas and RTT variation rather than steady probing.",
        "salient_features": [
            "current_rttvar_ms",
            "previous_action_delta",
            "previous_action_max",
            "delivery_growth_ratio",
            "windowed_delivery_rate_mbps_delta",
        ],
    },
}


@dataclass(frozen=True)
class SegmentAnnotation:
    segment_index: int
    segment_label: str
    segment_role: str
    start_step: int
    end_step: int
    severity: float
    intended_action: str
    best_baseline_method: str
    description: str


@dataclass
class RollingEpisodeState:
    prev_rtt_ms: float | None = None
    prev_delivery_rate_mbps: float | None = None
    rolling_max_growth_ratio: float = 1.0
    rolling_max_windowed_rate_mbps: float = 1.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)


def _resolve_repo_path(repo_root: str, path: str) -> str:
    expanded = os.path.expanduser(str(path))
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(repo_root, expanded))


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(max(float(value), float(low)), float(high)))


def _jitter(rng: np.random.Generator, *, center: float, scale: float, low: float, high: float) -> float:
    return _clamp(float(rng.normal(float(center), float(scale))), float(low), float(high))


def _segment_lengths(total_steps: int, num_segments: int) -> list[int]:
    base = max(int(total_steps) // max(int(num_segments), 1), 1)
    remainder = max(int(total_steps) - base * int(num_segments), 0)
    lengths = [int(base) for _ in range(int(num_segments))]
    for index in range(remainder):
        lengths[index % len(lengths)] += 1
    return lengths


def _clean_profile_step(
    *,
    profile_name: str,
    phase: float,
    severity: float,
    rng: np.random.Generator,
) -> dict[str, float | str]:
    wave = math.sin(2.0 * math.pi * float(phase))
    if str(profile_name) == "benign_capacity_swing":
        rate = _jitter(rng, center=58.0 + 28.0 * wave, scale=6.0, low=15.0, high=140.0)
        rtt = _jitter(rng, center=58.0 + 5.0 * math.cos(2.0 * math.pi * float(phase)), scale=1.8, low=45.0, high=75.0)
        loss = _jitter(rng, center=0.012 + 0.01 * abs(wave), scale=0.01, low=0.0, high=0.08)
        min_ratio = _jitter(rng, center=0.92 - 0.03 * abs(wave), scale=0.02, low=0.78, high=0.99)
        action = _jitter(rng, center=0.03 + 0.03 * wave, scale=0.015, low=-0.08, high=0.12)
        reference_action = _jitter(rng, center=action + 0.01, scale=0.02, low=-0.08, high=0.14)
        gap_pct = _jitter(rng, center=8.0, scale=2.0, low=1.0, high=16.0)
        best_score = _jitter(rng, center=0.77, scale=0.02, low=0.62, high=0.88)
        best_method = "bbr"
    elif str(profile_name) == "benign_mild_loss":
        rate = _jitter(rng, center=44.0 + 12.0 * wave, scale=4.5, low=10.0, high=100.0)
        rtt = _jitter(rng, center=60.0 + 4.0 * abs(wave), scale=2.0, low=48.0, high=78.0)
        loss = _jitter(rng, center=0.05 + 0.03 * float(severity), scale=0.015, low=0.0, high=0.14)
        min_ratio = _jitter(rng, center=0.88, scale=0.02, low=0.76, high=0.98)
        action = _jitter(rng, center=0.02 + 0.02 * wave, scale=0.012, low=-0.06, high=0.10)
        reference_action = _jitter(rng, center=action + 0.005, scale=0.015, low=-0.06, high=0.12)
        gap_pct = _jitter(rng, center=9.0, scale=2.5, low=2.0, high=18.0)
        best_score = _jitter(rng, center=0.74, scale=0.025, low=0.60, high=0.86)
        best_method = "reno"
    else:
        rate = _jitter(rng, center=38.0 + 10.0 * wave, scale=3.0, low=10.0, high=90.0)
        rtt = _jitter(rng, center=53.0 + 3.5 * math.cos(2.0 * math.pi * float(phase)), scale=1.5, low=42.0, high=70.0)
        loss = _jitter(rng, center=0.004 + 0.004 * abs(wave), scale=0.004, low=0.0, high=0.03)
        min_ratio = _jitter(rng, center=0.95, scale=0.015, low=0.82, high=0.995)
        action = _jitter(rng, center=0.02 + 0.015 * wave, scale=0.01, low=-0.05, high=0.08)
        reference_action = _jitter(rng, center=action, scale=0.012, low=-0.05, high=0.10)
        gap_pct = _jitter(rng, center=6.0, scale=1.5, low=0.5, high=12.0)
        best_score = _jitter(rng, center=0.79, scale=0.02, low=0.64, high=0.89)
        best_method = "cubic"

    return {
        "current_rtt_ms": float(rtt),
        "current_delivery_rate_mbps": float(rate),
        "current_loss_mbps": float(loss),
        "current_min_rtt_ratio": float(min_ratio),
        "previous_action": float(action),
        "reference_action": float(reference_action),
        "gap_percent_target": float(gap_pct),
        "best_baseline_score": float(best_score),
        "windowed_scale": float(_jitter(rng, center=0.97, scale=0.025, low=0.88, high=1.05)),
        "time_delta_ms": float(_jitter(rng, center=0.34, scale=0.025, low=0.26, high=0.44)),
        "best_baseline_method": str(best_method),
        "rttvar_override": float("nan"),
    }


def _deficiency_profile_step(
    *,
    deficiency: str,
    phase: float,
    severity: float,
    rng: np.random.Generator,
) -> dict[str, float | str]:
    wave = math.sin(2.0 * math.pi * float(phase))
    if str(deficiency) == "under_aggressiveness":
        rate = _jitter(rng, center=86.0 + 12.0 * wave, scale=5.0, low=40.0, high=165.0)
        rtt = _jitter(rng, center=51.0 + 2.0 * math.cos(2.0 * math.pi * float(phase)), scale=1.2, low=45.0, high=64.0)
        loss = _jitter(rng, center=0.006, scale=0.006, low=0.0, high=0.04)
        min_ratio = _jitter(rng, center=0.965, scale=0.012, low=0.88, high=0.995)
        action = _jitter(rng, center=-0.045 + 0.012 * wave, scale=0.015, low=-0.16, high=0.04)
        reference_action = _jitter(rng, center=0.17 + 0.05 * severity, scale=0.015, low=0.10, high=0.30)
        gap_pct = _jitter(rng, center=28.0 + 12.0 * severity, scale=3.0, low=22.0, high=58.0)
        best_score = _jitter(rng, center=0.80, scale=0.02, low=0.65, high=0.90)
        best_method = "bbr"
        windowed_scale = _jitter(rng, center=0.985, scale=0.015, low=0.94, high=1.02)
        time_delta_ms = _jitter(rng, center=0.34, scale=0.02, low=0.26, high=0.42)
    elif str(deficiency) == "delayed_recovery":
        rate = _jitter(rng, center=18.0 + 72.0 * float(phase), scale=5.0, low=8.0, high=150.0)
        rtt = _jitter(rng, center=74.0 - 20.0 * float(phase), scale=2.2, low=45.0, high=95.0)
        loss = _jitter(rng, center=max(0.02, 0.18 - 0.15 * float(phase)), scale=0.02, low=0.0, high=0.25)
        min_ratio = _jitter(rng, center=0.70 + 0.22 * float(phase), scale=0.02, low=0.55, high=0.96)
        action = _jitter(rng, center=-0.02 + 0.03 * float(phase), scale=0.012, low=-0.10, high=0.08)
        reference_action = _jitter(rng, center=0.15 + 0.03 * float(phase) + 0.04 * severity, scale=0.012, low=0.10, high=0.32)
        gap_pct = _jitter(rng, center=32.0 + 10.0 * severity, scale=3.5, low=22.0, high=60.0)
        best_score = _jitter(rng, center=0.78, scale=0.025, low=0.62, high=0.89)
        best_method = "bbr"
        windowed_scale = _jitter(rng, center=0.76 + 0.08 * float(phase), scale=0.025, low=0.62, high=0.92)
        time_delta_ms = _jitter(rng, center=0.35, scale=0.025, low=0.26, high=0.44)
    elif str(deficiency) == "over_aggressiveness":
        rate = _jitter(rng, center=26.0 + 7.0 * wave, scale=3.5, low=8.0, high=70.0)
        rtt = _jitter(rng, center=80.0 + 14.0 * severity + 5.0 * abs(wave), scale=3.0, low=58.0, high=130.0)
        loss = _jitter(rng, center=0.18 + 0.22 * severity, scale=0.05, low=0.02, high=1.00)
        min_ratio = _jitter(rng, center=0.64 - 0.06 * severity, scale=0.03, low=0.35, high=0.82)
        action = _jitter(rng, center=0.18 + 0.06 * severity, scale=0.015, low=0.08, high=0.34)
        reference_action = _jitter(rng, center=0.015 + 0.01 * wave, scale=0.012, low=-0.04, high=0.08)
        gap_pct = _jitter(rng, center=30.0 + 10.0 * severity, scale=3.0, low=22.0, high=55.0)
        best_score = _jitter(rng, center=0.75, scale=0.02, low=0.60, high=0.87)
        best_method = "reno"
        windowed_scale = _jitter(rng, center=0.84, scale=0.03, low=0.65, high=0.95)
        time_delta_ms = _jitter(rng, center=0.34, scale=0.03, low=0.26, high=0.46)
    elif str(deficiency) == "delayed_backoff":
        rate = _jitter(rng, center=46.0 - 18.0 * float(phase), scale=4.0, low=8.0, high=85.0)
        rtt = _jitter(rng, center=60.0 + 36.0 * float(phase), scale=3.0, low=48.0, high=126.0)
        loss = _jitter(rng, center=0.03 + 0.55 * float(phase), scale=0.04, low=0.0, high=1.10)
        min_ratio = _jitter(rng, center=0.88 - 0.28 * float(phase), scale=0.025, low=0.42, high=0.95)
        action = _jitter(rng, center=0.17 + 0.05 * severity, scale=0.014, low=0.08, high=0.32)
        reference_action = _jitter(rng, center=0.01 + 0.015 * (1.0 - float(phase)), scale=0.01, low=-0.05, high=0.08)
        gap_pct = _jitter(rng, center=34.0 + 10.0 * severity, scale=3.2, low=24.0, high=60.0)
        best_score = _jitter(rng, center=0.74, scale=0.02, low=0.60, high=0.86)
        best_method = "reno"
        windowed_scale = _jitter(rng, center=0.92 - 0.15 * float(phase), scale=0.03, low=0.58, high=0.95)
        time_delta_ms = _jitter(rng, center=0.36, scale=0.03, low=0.26, high=0.48)
    elif str(deficiency) == "rtt_insensitivity":
        rate = _jitter(rng, center=34.0 + 4.0 * wave, scale=3.0, low=10.0, high=70.0)
        rtt = _jitter(rng, center=96.0 + 22.0 * severity + 4.0 * abs(wave), scale=3.5, low=72.0, high=150.0)
        loss = _jitter(rng, center=0.018, scale=0.015, low=0.0, high=0.08)
        min_ratio = _jitter(rng, center=0.48 + 0.05 * (1.0 - severity), scale=0.025, low=0.32, high=0.70)
        action = _jitter(rng, center=0.16 + 0.05 * severity, scale=0.013, low=0.08, high=0.30)
        reference_action = _jitter(rng, center=0.015 + 0.01 * wave, scale=0.01, low=-0.04, high=0.08)
        gap_pct = _jitter(rng, center=30.0 + 8.0 * severity, scale=2.8, low=22.0, high=52.0)
        best_score = _jitter(rng, center=0.73, scale=0.02, low=0.58, high=0.84)
        best_method = "cubic"
        windowed_scale = _jitter(rng, center=0.87, scale=0.025, low=0.66, high=0.95)
        time_delta_ms = _jitter(rng, center=0.35, scale=0.025, low=0.26, high=0.45)
    elif str(deficiency) == "loss_insensitivity":
        rate = _jitter(rng, center=31.0 + 5.0 * wave, scale=3.0, low=10.0, high=70.0)
        rtt = _jitter(rng, center=64.0 + 8.0 * severity, scale=2.5, low=52.0, high=92.0)
        loss = _jitter(rng, center=0.55 + 0.55 * severity, scale=0.07, low=0.18, high=1.60)
        min_ratio = _jitter(rng, center=0.84, scale=0.02, low=0.68, high=0.94)
        action = _jitter(rng, center=0.17 + 0.04 * severity, scale=0.014, low=0.08, high=0.30)
        reference_action = _jitter(rng, center=0.01, scale=0.012, low=-0.05, high=0.08)
        gap_pct = _jitter(rng, center=31.0 + 9.0 * severity, scale=3.0, low=22.0, high=54.0)
        best_score = _jitter(rng, center=0.72, scale=0.02, low=0.56, high=0.83)
        best_method = "reno"
        windowed_scale = _jitter(rng, center=0.83, scale=0.03, low=0.60, high=0.94)
        time_delta_ms = _jitter(rng, center=0.34, scale=0.03, low=0.26, high=0.46)
    elif str(deficiency) == "unstable_probing":
        fast_wave = math.sin(6.0 * math.pi * float(phase))
        rate = _jitter(rng, center=44.0 + 18.0 * wave, scale=5.0, low=8.0, high=110.0)
        rtt = _jitter(rng, center=70.0 + 9.0 * abs(wave), scale=4.0, low=52.0, high=120.0)
        loss = _jitter(rng, center=0.08 + 0.10 * abs(fast_wave), scale=0.03, low=0.0, high=0.40)
        min_ratio = _jitter(rng, center=0.72 - 0.08 * abs(wave), scale=0.03, low=0.45, high=0.88)
        action = _jitter(rng, center=0.21 + 0.09 * fast_wave, scale=0.02, low=0.05, high=0.36)
        reference_action = _jitter(rng, center=0.03, scale=0.01, low=-0.04, high=0.08)
        gap_pct = _jitter(rng, center=29.0 + 8.0 * severity, scale=3.2, low=22.0, high=52.0)
        best_score = _jitter(rng, center=0.74, scale=0.02, low=0.58, high=0.86)
        best_method = "cubic"
        windowed_scale = _jitter(rng, center=0.76 + 0.10 * wave, scale=0.04, low=0.50, high=0.95)
        time_delta_ms = _jitter(rng, center=0.35, scale=0.035, low=0.24, high=0.50)
        rttvar_override = _jitter(rng, center=2.8 + 2.4 * abs(fast_wave), scale=0.35, low=1.2, high=7.0)
        return {
            "current_rtt_ms": float(rtt),
            "current_delivery_rate_mbps": float(rate),
            "current_loss_mbps": float(loss),
            "current_min_rtt_ratio": float(min_ratio),
            "previous_action": float(action),
            "reference_action": float(reference_action),
            "gap_percent_target": float(gap_pct),
            "best_baseline_score": float(best_score),
            "windowed_scale": float(windowed_scale),
            "time_delta_ms": float(time_delta_ms),
            "best_baseline_method": str(best_method),
            "rttvar_override": float(rttvar_override),
        }
    else:
        raise RuntimeError(f"unknown synthetic deficiency label: {deficiency}")

    return {
        "current_rtt_ms": float(rtt),
        "current_delivery_rate_mbps": float(rate),
        "current_loss_mbps": float(loss),
        "current_min_rtt_ratio": float(min_ratio),
        "previous_action": float(action),
        "reference_action": float(reference_action),
        "gap_percent_target": float(gap_pct),
        "best_baseline_score": float(best_score),
        "windowed_scale": float(windowed_scale),
        "time_delta_ms": float(time_delta_ms),
        "best_baseline_method": str(best_method),
        "rttvar_override": float("nan"),
    }


def _segment_profile_step(
    *,
    segment_label: str,
    phase: float,
    severity: float,
    rng: np.random.Generator,
) -> dict[str, float | str]:
    if str(segment_label) in CLEAN_PROFILE_NAMES:
        return _clean_profile_step(
            profile_name=str(segment_label),
            phase=float(phase),
            severity=float(severity),
            rng=rng,
        )
    return _deficiency_profile_step(
        deficiency=str(segment_label),
        phase=float(phase),
        severity=float(severity),
        rng=rng,
    )


def _baseline_method_payload(
    *,
    rng: np.random.Generator,
    best_method: str,
    best_score: float,
    reference_action: float,
    current_delivery_rate_mbps: float,
    current_rtt_ms: float,
    intended_action: str,
) -> dict[str, float]:
    score_penalties = {
        "reno": 0.05,
        "bbr": 0.04,
        "cubic": 0.06,
    }
    payload: dict[str, float] = {}
    for method in BASELINE_METHODS:
        method_score = float(best_score)
        if str(method) != str(best_method):
            method_score = float(best_score - score_penalties[str(method)] - float(rng.uniform(0.0, 0.02)))
        if str(intended_action) == "push_harder":
            rate_multiplier = 1.14 if str(method) == str(best_method) else 1.03
            rtt_multiplier = 0.97 if str(method) == str(best_method) else 1.00
        elif str(intended_action) == "back_off":
            rate_multiplier = 1.02 if str(method) == str(best_method) else 0.97
            rtt_multiplier = 0.82 if str(method) == str(best_method) else 0.96
        else:
            rate_multiplier = 1.01 if str(method) == str(best_method) else 0.99
            rtt_multiplier = 0.98 if str(method) == str(best_method) else 1.00
        action_offset = {
            "reno": -0.012,
            "bbr": 0.015,
            "cubic": 0.0,
        }[str(method)]
        payload[f"gap_score_{method}"] = float(method_score)
        payload[f"baseline_rate_{method}_mbps"] = float(max(current_delivery_rate_mbps * rate_multiplier, 0.1))
        payload[f"baseline_rtt_{method}_ms"] = float(max(current_rtt_ms * rtt_multiplier, 1.0))
        payload[f"baseline_previous_action_{method}"] = float(
            _clamp(
                reference_action + action_offset + float(rng.normal(0.0, 0.01)),
                -0.25,
                0.40,
            )
        )
    payload[f"gap_score_{best_method}"] = float(best_score)
    payload[f"baseline_previous_action_{best_method}"] = float(reference_action)
    return payload


def _enforce_intended_label(
    *,
    row: dict[str, Any],
    best_method: str,
    intended_action: str,
    risk_gap_pct: float,
    baseline_score_floor: float,
    action_margin: float,
) -> None:
    baseline_score = max(float(row["hard_baseline_score"]), float(baseline_score_floor) + 0.12)
    sage_action = float(row["sage_previous_action"])
    reference_action = float(row["best_baseline_previous_action"])
    gap_pct_value = float(row["hard_gap_percent"])

    if str(intended_action) == "push_harder":
        gap_pct_value = max(gap_pct_value, float(risk_gap_pct) + 6.0)
        reference_action = max(reference_action, sage_action + float(action_margin) + 0.08)
    elif str(intended_action) == "back_off":
        gap_pct_value = max(gap_pct_value, float(risk_gap_pct) + 6.0)
        reference_action = min(reference_action, sage_action - float(action_margin) - 0.08)
    else:
        gap_pct_value = min(gap_pct_value, max(float(risk_gap_pct) - 6.0, 2.0))
        reference_action = _clamp(reference_action, sage_action - float(action_margin) * 0.45, sage_action + float(action_margin) * 0.45)

    row["hard_baseline_score"] = float(baseline_score)
    row["hard_gap_percent"] = float(gap_pct_value)
    row["hard_gap_value"] = float(baseline_score * gap_pct_value / 100.0)
    row["sage_score"] = float(max(baseline_score - float(row["hard_gap_value"]), 0.0))
    row["reward"] = float(row["hard_gap_value"] * baseline_score)
    row["attacker_reward"] = float(row["reward"])
    row["sage_reward"] = float(row["sage_score"])
    row["best_baseline_previous_action"] = float(reference_action)
    row[f"baseline_previous_action_{best_method}"] = float(reference_action)
    row["smoothed_baseline_score"] = float(baseline_score * 0.98)
    row["smoothed_gap_value"] = float(max(float(row["hard_gap_value"]) * 0.85, 0.0))
    row["smoothed_gap_percent"] = float(
        hard_gap_percent(
            best_baseline_gap=float(row["smoothed_gap_value"]),
            best_baseline_score=float(row["smoothed_baseline_score"]),
        )
    )


def _current_values_from_segment_step(
    *,
    runtime: RollingEpisodeState,
    step_payload: dict[str, float | str],
    rng: np.random.Generator,
) -> dict[str, float]:
    current_rate = max(float(step_payload["current_delivery_rate_mbps"]), 0.1)
    current_rtt = max(float(step_payload["current_rtt_ms"]), 1.0)
    windowed_scale = float(step_payload["windowed_scale"])
    windowed_rate = max(current_rate * windowed_scale + float(rng.normal(0.0, 0.7)), 0.1)
    runtime.rolling_max_windowed_rate_mbps = max(float(runtime.rolling_max_windowed_rate_mbps) * 0.985, windowed_rate)
    if runtime.prev_delivery_rate_mbps is None or float(runtime.prev_delivery_rate_mbps) <= 1e-6:
        delivery_growth_ratio = 1.0
    else:
        delivery_growth_ratio = current_rate / float(runtime.prev_delivery_rate_mbps)
    runtime.rolling_max_growth_ratio = max(float(runtime.rolling_max_growth_ratio) * 0.985, float(delivery_growth_ratio), 1.0)
    rttvar_override = float(step_payload["rttvar_override"])
    if math.isfinite(rttvar_override):
        current_rttvar_ms = float(rttvar_override)
    elif runtime.prev_rtt_ms is None:
        current_rttvar_ms = float(_jitter(rng, center=0.35, scale=0.08, low=0.08, high=1.4))
    else:
        current_rttvar_ms = float(
            _clamp(
                0.18 + 0.22 * abs(current_rtt - float(runtime.prev_rtt_ms)) + float(abs(rng.normal(0.0, 0.12))),
                0.08,
                6.0,
            )
        )
    runtime.prev_rtt_ms = float(current_rtt)
    runtime.prev_delivery_rate_mbps = float(current_rate)
    return {
        "current_rtt_ms": float(current_rtt),
        "current_rttvar_ms": float(current_rttvar_ms),
        "current_delivery_rate_mbps": float(current_rate),
        "windowed_delivery_rate_mbps": float(windowed_rate),
        "max_windowed_delivery_rate_mbps": float(max(windowed_rate, runtime.rolling_max_windowed_rate_mbps)),
        "current_loss_mbps": float(max(float(step_payload["current_loss_mbps"]), 0.0)),
        "current_min_rtt_ratio": float(_clamp(float(step_payload["current_min_rtt_ratio"]), 1e-3, 1.0)),
        "previous_action": float(step_payload["previous_action"]),
        "time_delta_ms": float(step_payload["time_delta_ms"]),
        "delivery_growth_ratio": float(_clamp(delivery_growth_ratio, 0.35, 1.8)),
        "max_delivery_growth_ratio": float(_clamp(runtime.rolling_max_growth_ratio, 1.0, 1.8)),
    }


def _fieldnames(baseline_methods: tuple[str, ...]) -> list[str]:
    return [
        "setup",
        "trace_type",
        "episode_id",
        "episode_step",
        "reward",
        "attacker_reward",
        "sage_reward",
        "sage_score",
        "sage_previous_action",
        "hard_gap_value",
        "hard_baseline_score",
        "hard_gap_percent",
        "smoothed_gap_value",
        "smoothed_baseline_score",
        "smoothed_gap_percent",
        "best_baseline_method",
        "best_baseline_previous_action",
        "shield_risk_label",
        "shield_backoff_label",
        "shield_push_label",
        "shield_action_label_id",
        "shield_action_label",
        "shield_action_delta_to_best_baseline",
        "shield_label_valid",
        "shield_label_reason",
        "has_env_error",
        "env_bootstrap_placeholder",
        "env_nonfinite_sage_values",
        *FEATURE_COLUMNS,
        *[
            metric
            for method in baseline_methods
            for metric in (
                f"gap_score_{method}",
                f"baseline_rate_{method}_mbps",
                f"baseline_rtt_{method}_ms",
                f"baseline_previous_action_{method}",
            )
        ],
    ]


def _row_from_step(
    *,
    tracker: ShieldFeatureTracker,
    setup_name: str,
    trace_type: str,
    episode_id: str,
    episode_step: int,
    intended_action: str,
    current_values: dict[str, float],
    step_payload: dict[str, float | str],
    risk_gap_pct: float,
    baseline_score_floor: float,
    action_margin: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    features = tracker.update_from_current_values(current_values)
    best_method = str(step_payload["best_baseline_method"])

    baseline_score = float(step_payload["best_baseline_score"])
    hard_gap_percent_target = float(step_payload["gap_percent_target"])
    hard_gap_value_target = float(baseline_score * hard_gap_percent_target / 100.0)
    sage_score = float(max(baseline_score - hard_gap_value_target, 0.0))
    smoothed_baseline_score = float(max(baseline_score * 0.98 + float(rng.normal(0.0, 0.005)), 0.05))
    smoothed_gap_value = float(max(hard_gap_value_target * (0.82 + 0.08 * float(rng.uniform())), 0.0))

    row: dict[str, Any] = {
        "setup": str(setup_name),
        "trace_type": str(trace_type),
        "episode_id": str(episode_id),
        "episode_step": int(episode_step),
        "reward": float(hard_gap_value_target * baseline_score),
        "attacker_reward": float(hard_gap_value_target * baseline_score),
        "sage_reward": float(sage_score),
        "sage_score": float(sage_score),
        "sage_previous_action": float(current_values["previous_action"]),
        "hard_gap_value": float(hard_gap_value_target),
        "hard_baseline_score": float(baseline_score),
        "hard_gap_percent": float(
            hard_gap_percent(
                best_baseline_gap=float(hard_gap_value_target),
                best_baseline_score=float(baseline_score),
            )
        ),
        "smoothed_gap_value": float(smoothed_gap_value),
        "smoothed_baseline_score": float(smoothed_baseline_score),
        "smoothed_gap_percent": float(
            hard_gap_percent(
                best_baseline_gap=float(smoothed_gap_value),
                best_baseline_score=float(smoothed_baseline_score),
            )
        ),
        "best_baseline_method": str(best_method),
        "best_baseline_previous_action": float(step_payload["reference_action"]),
        "has_env_error": 0,
        "env_bootstrap_placeholder": 0,
        "env_nonfinite_sage_values": 0.0,
    }
    row.update(features)
    row.update(
        _baseline_method_payload(
            rng=rng,
            best_method=str(best_method),
            best_score=float(baseline_score),
            reference_action=float(step_payload["reference_action"]),
            current_delivery_rate_mbps=float(current_values["current_delivery_rate_mbps"]),
            current_rtt_ms=float(current_values["current_rtt_ms"]),
            intended_action=str(intended_action),
        )
    )
    _enforce_intended_label(
        row=row,
        best_method=str(best_method),
        intended_action=str(intended_action),
        risk_gap_pct=float(risk_gap_pct),
        baseline_score_floor=float(baseline_score_floor),
        action_margin=float(action_margin),
    )
    shield_risky = bool(float(row["hard_gap_percent"]) >= float(risk_gap_pct) and float(row["hard_baseline_score"]) >= float(baseline_score_floor))
    backoff_label, push_label = weak_direction_labels(
        risky=bool(shield_risky),
        sage_previous_action=float(row["sage_previous_action"]),
        best_baseline_previous_action=float(row["best_baseline_previous_action"]),
        action_margin=float(action_margin),
    )
    row["shield_risk_label"] = int(1 if shield_risky else 0)
    row["shield_backoff_label"] = int(backoff_label)
    row["shield_push_label"] = int(push_label)
    row["shield_action_delta_to_best_baseline"] = float(
        float(row["sage_previous_action"]) - float(row["best_baseline_previous_action"])
    )
    action_label_id, action_label, action_label_reason, action_label_valid = unified_action_label(
        hard_gap_percent_value=float(row["hard_gap_percent"]),
        hard_baseline_score=float(row["hard_baseline_score"]),
        risk_gap_pct=float(risk_gap_pct),
        baseline_score_floor=float(baseline_score_floor),
        sage_previous_action=float(row["sage_previous_action"]),
        best_baseline_previous_action=float(row["best_baseline_previous_action"]),
        action_margin=float(action_margin),
    )
    row["shield_action_label_id"] = int(action_label_id)
    row["shield_action_label"] = str(action_label)
    row["shield_label_valid"] = int(1 if bool(action_label_valid) else 0)
    row["shield_label_reason"] = str(action_label_reason)
    return row


def _build_clean_episode_segments(
    *,
    rng: np.random.Generator,
    episode_steps: int,
    episode_index: int,
) -> list[SegmentAnnotation]:
    profile_name = str(CLEAN_PROFILE_NAMES[int(episode_index) % len(CLEAN_PROFILE_NAMES)])
    return [
        SegmentAnnotation(
            segment_index=0,
            segment_label=profile_name,
            segment_role="clean_profile",
            start_step=0,
            end_step=max(int(episode_steps) - 1, 0),
            severity=float(rng.uniform(0.25, 0.75)),
            intended_action="noop",
            best_baseline_method=str(BASELINE_METHODS[int(episode_index) % len(BASELINE_METHODS)]),
            description=f"Benign synthetic clean profile: {profile_name}.",
        )
    ]


def _build_adv_episode_segments(
    *,
    rng: np.random.Generator,
    episode_steps: int,
    feature_history_len: int,
    deficiency_labels: list[str],
) -> list[SegmentAnnotation]:
    num_deficiencies = max(len(deficiency_labels), 1)
    warmup_len = max(int(feature_history_len) + 2, min(8, max(int(episode_steps) // 6, 4)))
    cooldown_len = warmup_len
    bridge_len = max(4, int(feature_history_len))
    num_segments = 2 + int(num_deficiencies) + max(int(num_deficiencies) - 1, 0)
    lengths = _segment_lengths(
        max(int(episode_steps) - warmup_len - cooldown_len - max(int(num_deficiencies) - 1, 0) * bridge_len, int(num_deficiencies)),
        int(num_deficiencies),
    )
    segments: list[SegmentAnnotation] = []
    cursor = 0
    warmup_profile = str(CLEAN_PROFILE_NAMES[int(rng.integers(0, len(CLEAN_PROFILE_NAMES)))])
    segments.append(
        SegmentAnnotation(
            segment_index=len(segments),
            segment_label=warmup_profile,
            segment_role="warmup_context",
            start_step=int(cursor),
            end_step=int(cursor + warmup_len - 1),
            severity=float(rng.uniform(0.25, 0.65)),
            intended_action="noop",
            best_baseline_method="bbr",
            description=f"Benign warm-up context using {warmup_profile}.",
        )
    )
    cursor += warmup_len
    for deficiency_index, deficiency_label in enumerate(deficiency_labels):
        severity = float(rng.uniform(0.45, 1.0))
        metadata = dict(SEMANTIC_TAXONOMY[str(deficiency_label)])
        length = int(lengths[min(deficiency_index, len(lengths) - 1)])
        segments.append(
            SegmentAnnotation(
                segment_index=len(segments),
                segment_label=str(deficiency_label),
                segment_role="deficiency",
                start_step=int(cursor),
                end_step=int(cursor + length - 1),
                severity=float(severity),
                intended_action=str(metadata["intended_action"]),
                best_baseline_method=(
                    "bbr"
                    if str(metadata["intended_action"]) == "push_harder"
                    else ("cubic" if str(deficiency_label) in {"rtt_insensitivity", "unstable_probing"} else "reno")
                ),
                description=str(metadata["description"]),
            )
        )
        cursor += length
        if deficiency_index < int(num_deficiencies) - 1:
            bridge_profile = str(CLEAN_PROFILE_NAMES[int(rng.integers(0, len(CLEAN_PROFILE_NAMES)))])
            segments.append(
                SegmentAnnotation(
                    segment_index=len(segments),
                    segment_label=bridge_profile,
                    segment_role="bridge_context",
                    start_step=int(cursor),
                    end_step=int(cursor + bridge_len - 1),
                    severity=float(rng.uniform(0.2, 0.6)),
                    intended_action="noop",
                    best_baseline_method="reno",
                    description=f"Benign bridge context using {bridge_profile}.",
                )
            )
            cursor += bridge_len
    cooldown_profile = str(CLEAN_PROFILE_NAMES[int(rng.integers(0, len(CLEAN_PROFILE_NAMES)))])
    segments.append(
        SegmentAnnotation(
            segment_index=len(segments),
            segment_label=cooldown_profile,
            segment_role="cooldown_context",
            start_step=int(cursor),
            end_step=int(episode_steps - 1),
            severity=float(rng.uniform(0.25, 0.65)),
            intended_action="noop",
            best_baseline_method="cubic",
            description=f"Benign cool-down context using {cooldown_profile}.",
        )
    )
    return segments


def _episode_rows_and_ground_truth(
    *,
    rng: np.random.Generator,
    setup_name: str,
    trace_type: str,
    episode_id: str,
    segments: list[SegmentAnnotation],
    risk_gap_pct: float,
    baseline_score_floor: float,
    action_margin: float,
    feature_history_len: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tracker = ShieldFeatureTracker(history_len=int(feature_history_len))
    runtime = RollingEpisodeState()
    rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    for segment in segments:
        segment_length = int(segment.end_step) - int(segment.start_step) + 1
        for local_index, step_index in enumerate(range(int(segment.start_step), int(segment.end_step) + 1)):
            phase = float(local_index / max(segment_length - 1, 1))
            step_payload = _segment_profile_step(
                segment_label=str(segment.segment_label),
                phase=float(phase),
                severity=float(segment.severity),
                rng=rng,
            )
            step_payload["best_baseline_method"] = str(segment.best_baseline_method)
            current_values = _current_values_from_segment_step(
                runtime=runtime,
                step_payload=step_payload,
                rng=rng,
            )
            row = _row_from_step(
                tracker=tracker,
                setup_name=str(setup_name),
                trace_type=str(trace_type),
                episode_id=str(episode_id),
                episode_step=int(step_index),
                intended_action=str(segment.intended_action),
                current_values=current_values,
                step_payload=step_payload,
                risk_gap_pct=float(risk_gap_pct),
                baseline_score_floor=float(baseline_score_floor),
                action_margin=float(action_margin),
                rng=rng,
            )
            rows.append(row)
            gt_rows.append(
                {
                    "setup": str(setup_name),
                    "trace_type": str(trace_type),
                    "episode_id": str(episode_id),
                    "episode_step": int(step_index),
                    "segment_index": int(segment.segment_index),
                    "segment_role": str(segment.segment_role),
                    "segment_label": str(segment.segment_label),
                    "severity": float(segment.severity),
                    "intended_action": str(segment.intended_action),
                    "best_baseline_method": str(segment.best_baseline_method),
                    "description": str(segment.description),
                }
            )
    return rows, gt_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic Sage shield dataset with controlled semantic deficiencies.")
    parser.add_argument("--repo-root", type=str, default=REPO_ROOT)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--trace-set-name", type=str, default=None)
    parser.add_argument("--num-clean-episodes", type=int, default=36)
    parser.add_argument("--num-adv-episodes-per-deficiency", type=int, default=10)
    parser.add_argument("--episode-steps", type=int, default=48)
    parser.add_argument("--deficiencies-per-adv-episode", type=int, default=1)
    parser.add_argument("--allowed-deficiencies", type=str, default=None)
    parser.add_argument("--feature-history-len", type=int, default=4)
    parser.add_argument("--risk-gap-pct", type=float, default=20.0)
    parser.add_argument("--baseline-score-floor", type=float, default=0.3)
    parser.add_argument("--action-margin", type=float, default=0.15)
    parser.add_argument("--threshold-percentiles", type=str, default="10,25,90,95")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = os.path.abspath(str(args.repo_root))
    out_dir = _resolve_repo_path(repo_root, str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    trace_set_name = str(args.trace_set_name or os.path.basename(os.path.normpath(out_dir)) or "synthetic_shield_dataset")
    rng = np.random.default_rng(int(args.seed))

    fieldnames = _fieldnames(BASELINE_METHODS)
    csv_path = os.path.join(out_dir, "sage_shield_dataset.csv")
    meta_path = os.path.join(out_dir, "sage_shield_dataset_meta.json")
    threshold_path = os.path.join(out_dir, "clean_feature_thresholds.csv")
    ground_truth_json_path = os.path.join(out_dir, "synthetic_semantic_ground_truth.json")
    ground_truth_csv_path = os.path.join(out_dir, "synthetic_semantic_ground_truth.csv")

    all_deficiency_names = list(SEMANTIC_TAXONOMY.keys())
    allowed_raw = [item.strip() for item in str(args.allowed_deficiencies or "").split(",") if item.strip()]
    if allowed_raw:
        invalid_deficiencies = [name for name in allowed_raw if name not in SEMANTIC_TAXONOMY]
        if invalid_deficiencies:
            raise RuntimeError(
                "unknown --allowed-deficiencies values: "
                + ", ".join(sorted(str(name) for name in invalid_deficiencies))
            )
        deficiency_names = list(dict.fromkeys(allowed_raw))
    else:
        deficiency_names = list(all_deficiency_names)
    if not deficiency_names:
        raise RuntimeError("no semantic deficiencies selected for synthetic generation")
    all_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    episode_ground_truth: list[dict[str, Any]] = []
    episode_counts: dict[str, int] = {name: 0 for name in deficiency_names}

    for clean_index in range(int(args.num_clean_episodes)):
        episode_id = f"clean-{clean_index:03d}"
        segments = _build_clean_episode_segments(
            rng=rng,
            episode_steps=int(args.episode_steps),
            episode_index=int(clean_index),
        )
        rows, per_step_gt = _episode_rows_and_ground_truth(
            rng=rng,
            setup_name="clean",
            trace_type="clean",
            episode_id=episode_id,
            segments=segments,
            risk_gap_pct=float(args.risk_gap_pct),
            baseline_score_floor=float(args.baseline_score_floor),
            action_margin=float(args.action_margin),
            feature_history_len=int(args.feature_history_len),
        )
        all_rows.extend(rows)
        gt_rows.extend(per_step_gt)
        episode_ground_truth.append(
            {
                "setup": "clean",
                "trace_type": "clean",
                "episode_id": episode_id,
                "segments": [segment.__dict__ for segment in segments],
            }
        )

    total_adv_episodes = int(args.num_adv_episodes_per_deficiency) * len(deficiency_names)
    primary_labels = [
        str(label)
        for label in deficiency_names
        for _ in range(max(int(args.num_adv_episodes_per_deficiency), 0))
    ]
    rng.shuffle(primary_labels)
    deficiencies_per_adv_episode = max(int(args.deficiencies_per_adv_episode), 1)
    for adv_index, primary_label in enumerate(primary_labels):
        selected_labels = [str(primary_label)]
        if deficiencies_per_adv_episode > 1 and len(deficiency_names) > 1:
            remaining_labels = [label for label in deficiency_names if str(label) != str(primary_label)]
            rng.shuffle(remaining_labels)
            selected_labels.extend(remaining_labels[: max(deficiencies_per_adv_episode - 1, 0)])
        episode_id = f"adv-{'-'.join(selected_labels)}-{adv_index:03d}"
        for label in selected_labels:
            episode_counts[str(label)] += 1
        segments = _build_adv_episode_segments(
            rng=rng,
            episode_steps=int(args.episode_steps),
            feature_history_len=int(args.feature_history_len),
            deficiency_labels=selected_labels,
        )
        rows, per_step_gt = _episode_rows_and_ground_truth(
            rng=rng,
            setup_name=trace_set_name,
            trace_type="adv",
            episode_id=episode_id,
            segments=segments,
            risk_gap_pct=float(args.risk_gap_pct),
            baseline_score_floor=float(args.baseline_score_floor),
            action_margin=float(args.action_margin),
            feature_history_len=int(args.feature_history_len),
        )
        all_rows.extend(rows)
        gt_rows.extend(per_step_gt)
        episode_ground_truth.append(
            {
                "setup": trace_set_name,
                "trace_type": "adv",
                "episode_id": episode_id,
                "segments": [segment.__dict__ for segment in segments],
                "deficiency_subset": list(selected_labels),
            }
        )

    with open(csv_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    with open(ground_truth_csv_path, "w", encoding="utf-8", newline="") as file_obj:
        gt_fieldnames = [
            "setup",
            "trace_type",
            "episode_id",
            "episode_step",
            "segment_index",
            "segment_role",
            "segment_label",
            "severity",
            "intended_action",
            "best_baseline_method",
            "description",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=gt_fieldnames)
        writer.writeheader()
        for row in gt_rows:
            writer.writerow(row)

    threshold_percentiles = [int(item.strip()) for item in str(args.threshold_percentiles).split(",") if item.strip()]
    threshold_metadata = write_clean_feature_thresholds_from_dataset(
        dataset_path=csv_path,
        out_path=threshold_path,
        percentiles=threshold_percentiles,
    )

    ground_truth_payload = {
        "created_at_utc": _utc_now_iso(),
        "trace_set_name": trace_set_name,
        "seed": int(args.seed),
        "taxonomy": {
            label: {
                **dict(metadata),
                "label": str(label),
            }
            for label, metadata in SEMANTIC_TAXONOMY.items()
            if label in deficiency_names
        },
        "allowed_deficiencies": list(deficiency_names),
        "full_semantic_taxonomy": list(all_deficiency_names),
        "clean_profiles": list(CLEAN_PROFILE_NAMES),
        "episodes": episode_ground_truth,
        "ground_truth_csv_path": os.path.relpath(ground_truth_csv_path, repo_root),
    }
    _save_json(ground_truth_json_path, ground_truth_payload)

    summary_payload = {
        "created_at_utc": _utc_now_iso(),
        "repo_root": repo_root,
        "generated_manifest_path": None,
        "training_config_path": None,
        "clean_manifest_path": None,
        "runtime_dir_resolved": None,
        "trace_set_name": trace_set_name,
        "baseline_methods": list(BASELINE_METHODS),
        "pipeline_mode": "synthetic_ground_truth",
        "feature_history_len": int(args.feature_history_len),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "labeling": {
            "risk_gap_pct": float(args.risk_gap_pct),
            "baseline_score_floor": float(args.baseline_score_floor),
            "action_margin": float(args.action_margin),
        },
        "csv_path": os.path.relpath(csv_path, repo_root),
        "thresholds_path": os.path.relpath(threshold_path, repo_root),
        "threshold_percentiles": [int(item) for item in threshold_percentiles],
        "threshold_metadata": threshold_metadata,
        "ground_truth_json_path": os.path.relpath(ground_truth_json_path, repo_root),
        "ground_truth_csv_path": os.path.relpath(ground_truth_csv_path, repo_root),
        "synthetic_generator": "closed_taxonomy_v1",
        "synthetic_taxonomy": list(deficiency_names),
        "full_semantic_taxonomy": list(all_deficiency_names),
        "allowed_deficiencies": list(deficiency_names),
        "synthetic_episode_counts": {
            "num_clean_episodes": int(args.num_clean_episodes),
            "num_adv_episodes": int(total_adv_episodes),
            "episodes_per_deficiency": {str(label): int(count) for label, count in episode_counts.items()},
        },
        "num_rows": int(len(all_rows)),
        "num_clean_episodes": int(args.num_clean_episodes),
        "num_adv_episodes": int(total_adv_episodes),
    }
    _save_json(meta_path, summary_payload)

    print(csv_path)
    print(threshold_path)
    print(ground_truth_json_path)


if __name__ == "__main__":
    main()
