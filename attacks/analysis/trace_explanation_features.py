from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


CATEGORY_BANDWIDTH_COUPLING_KIND = "bandwidth_coupling_kind"
CATEGORY_BANDWIDTH_PROFILE_KIND = "bandwidth_profile_kind"
CATEGORY_LOSS_PROFILE_KIND = "loss_profile_kind"
CATEGORY_DELAY_PROFILE_KIND = "delay_profile_kind"
CATEGORY_BASELINE_METHODS_KEY = "baseline_methods_key"
CATEGORY_ATTACK_MODE = "attack_mode"

CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    CATEGORY_BANDWIDTH_COUPLING_KIND,
    CATEGORY_BANDWIDTH_PROFILE_KIND,
    CATEGORY_LOSS_PROFILE_KIND,
    CATEGORY_DELAY_PROFILE_KIND,
    CATEGORY_BASELINE_METHODS_KEY,
    CATEGORY_ATTACK_MODE,
)

_SHARED_BW_PREFIX = "shared_bw"
_DIRECTIONAL_BW_PREFIXES = ("uplink_bw", "downlink_bw")
_LOSS_DELAY_PREFIXES = ("uplink_loss", "downlink_loss", "uplink_delay", "downlink_delay")
_SHARED_WINDOW_STEPS = (5, 10, 20)


def _bandwidth_feature_columns(prefix: str) -> list[str]:
    base = [
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_cv",
        f"{prefix}_min",
        f"{prefix}_max",
        f"{prefix}_p10",
        f"{prefix}_p90",
        f"{prefix}_span",
        f"{prefix}_slope",
        f"{prefix}_early_late_delta",
        f"{prefix}_abs_diff_mean",
        f"{prefix}_abs_diff_p90",
        f"{prefix}_abs_second_diff_mean",
        f"{prefix}_sign_change_rate",
        f"{prefix}_plateau_fraction",
        f"{prefix}_high_fraction",
        f"{prefix}_low_fraction",
        f"{prefix}_high_run_count",
        f"{prefix}_low_run_count",
        f"{prefix}_longest_high_run",
        f"{prefix}_longest_low_run",
        f"{prefix}_peak_to_mean",
        f"{prefix}_autocorr_lag1",
    ]
    return base


def _directional_bandwidth_feature_columns(prefix: str) -> list[str]:
    return [
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_min",
        f"{prefix}_max",
        f"{prefix}_p10",
        f"{prefix}_p90",
        f"{prefix}_slope",
        f"{prefix}_abs_diff_mean",
        f"{prefix}_abs_second_diff_mean",
        f"{prefix}_sign_change_rate",
        f"{prefix}_plateau_fraction",
        f"{prefix}_peak_to_mean",
        f"{prefix}_autocorr_lag1",
    ]


def _loss_delay_feature_columns(prefix: str) -> list[str]:
    return [
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_max",
        f"{prefix}_span",
        f"{prefix}_nonzero_fraction",
        f"{prefix}_abs_diff_mean",
    ]


def _window_feature_columns(window_steps: int) -> list[str]:
    return [
        f"shared_bw_window{window_steps}_min_mean",
        f"shared_bw_window{window_steps}_max_mean",
        f"shared_bw_window{window_steps}_max_cv",
        f"shared_bw_window{window_steps}_max_curvature",
        f"shared_bw_window{window_steps}_negative_slope_fraction",
    ]


CROSS_FEATURE_COLUMNS: tuple[str, ...] = (
    "num_steps",
    "duration_seconds",
    "bandwidth_symmetry_fraction",
    "bandwidth_corr",
    "bandwidth_abs_diff_mean",
    "bandwidth_abs_diff_p90",
    "bandwidth_ratio_mean",
    "loss_abs_diff_mean",
    "delay_abs_diff_mean",
)

NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    [*_bandwidth_feature_columns(_SHARED_BW_PREFIX)]
    + [feature for prefix in _DIRECTIONAL_BW_PREFIXES for feature in _directional_bandwidth_feature_columns(prefix)]
    + [feature for prefix in _LOSS_DELAY_PREFIXES for feature in _loss_delay_feature_columns(prefix)]
    + [feature for window_steps in _SHARED_WINDOW_STEPS for feature in _window_feature_columns(window_steps)]
    + list(CROSS_FEATURE_COLUMNS)
)

FEATURE_COLUMNS: tuple[str, ...] = tuple([*CATEGORICAL_FEATURE_COLUMNS, *NUMERIC_FEATURE_COLUMNS])


def _description_map() -> dict[str, str]:
    descriptions: dict[str, str] = {
        CATEGORY_BANDWIDTH_COUPLING_KIND: "Categorical summary of whether uplink and downlink bandwidth move together like a shared bottleneck or behave independently.",
        CATEGORY_BANDWIDTH_PROFILE_KIND: "Categorical summary of whether the shared bottleneck bandwidth is nearly flat or meaningfully time-varying.",
        CATEGORY_LOSS_PROFILE_KIND: "Categorical summary of whether the loss process is zero, fixed, or time-varying across the trace.",
        CATEGORY_DELAY_PROFILE_KIND: "Categorical summary of whether the delay process is fixed or time-varying across the trace.",
        CATEGORY_BASELINE_METHODS_KEY: "Enabled baseline methods rendered as a categorical key so tree models can learn setup-specific structure when multiple baseline families appear in one dataset.",
        CATEGORY_ATTACK_MODE: "Attack-environment mode from the originating attacker config, treated as a categorical setup descriptor.",
        "num_steps": "Number of replay steps in the trace schedule.",
        "duration_seconds": "Trace duration in seconds, computed as num_steps times attack interval.",
        "bandwidth_symmetry_fraction": "Fraction of steps where uplink and downlink bandwidth are effectively equal, capturing how often the trace behaves like a shared bottleneck.",
        "bandwidth_corr": "Pearson correlation between uplink and downlink bandwidth series, capturing whether both directions trend together.",
        "bandwidth_abs_diff_mean": "Mean absolute uplink/downlink bandwidth gap, capturing persistent directional asymmetry.",
        "bandwidth_abs_diff_p90": "90th percentile of absolute uplink/downlink bandwidth gap, capturing extreme asymmetric bursts.",
        "bandwidth_ratio_mean": "Mean ratio between the smaller and larger directional bandwidths, where lower values imply stronger asymmetry.",
        "loss_abs_diff_mean": "Mean absolute difference between uplink and downlink loss, capturing asymmetric loss pressure.",
        "delay_abs_diff_mean": "Mean absolute difference between uplink and downlink delay, capturing asymmetric queuing or propagation effects.",
    }

    for prefix in (_SHARED_BW_PREFIX,):
        descriptions.update(
            {
                f"{prefix}_mean": "Average shared bottleneck bandwidth, using the minimum of uplink and downlink bandwidth at each step.",
                f"{prefix}_std": "Standard deviation of shared bottleneck bandwidth, measuring variability around the mean.",
                f"{prefix}_cv": "Coefficient of variation of shared bottleneck bandwidth, i.e. std divided by mean.",
                f"{prefix}_min": "Minimum shared bottleneck bandwidth across the trace.",
                f"{prefix}_max": "Maximum shared bottleneck bandwidth across the trace.",
                f"{prefix}_p10": "10th percentile shared bottleneck bandwidth, capturing the low-bandwidth tail.",
                f"{prefix}_p90": "90th percentile shared bottleneck bandwidth, capturing the high-bandwidth tail.",
                f"{prefix}_span": "Range of shared bottleneck bandwidth, i.e. max minus min.",
                f"{prefix}_slope": "Least-squares linear slope of shared bottleneck bandwidth over time; negative values indicate an overall downward trend.",
                f"{prefix}_early_late_delta": "Difference between late-trace and early-trace mean shared bottleneck bandwidth, capturing coarse trend direction.",
                f"{prefix}_abs_diff_mean": "Mean absolute first difference of shared bottleneck bandwidth, measuring average jump size between consecutive steps.",
                f"{prefix}_abs_diff_p90": "90th percentile of absolute first differences in shared bottleneck bandwidth, capturing large bursts or cliffs.",
                f"{prefix}_abs_second_diff_mean": "Mean absolute second difference of shared bottleneck bandwidth, capturing curvature and sawtooth-like motion.",
                f"{prefix}_sign_change_rate": "Fraction of first-difference steps where the sign flips, capturing oscillation frequency.",
                f"{prefix}_plateau_fraction": "Fraction of steps whose step-to-step change is tiny, capturing plateau-like behavior.",
                f"{prefix}_high_fraction": "Fraction of steps above mean plus one standard deviation, capturing burst prevalence.",
                f"{prefix}_low_fraction": "Fraction of steps below mean minus one standard deviation, capturing trough prevalence.",
                f"{prefix}_high_run_count": "Number of contiguous high-bandwidth burst runs in the shared bottleneck series.",
                f"{prefix}_low_run_count": "Number of contiguous low-bandwidth trough runs in the shared bottleneck series.",
                f"{prefix}_longest_high_run": "Length of the longest contiguous high-bandwidth burst run in the shared bottleneck series.",
                f"{prefix}_longest_low_run": "Length of the longest contiguous low-bandwidth trough run in the shared bottleneck series.",
                f"{prefix}_peak_to_mean": "Peak-to-mean ratio of shared bottleneck bandwidth, highlighting traces with rare large spikes.",
                f"{prefix}_autocorr_lag1": "Lag-1 autocorrelation of shared bottleneck bandwidth, capturing persistence from one step to the next.",
            }
        )

    for prefix in _DIRECTIONAL_BW_PREFIXES:
        direction = "uplink" if prefix.startswith("uplink") else "downlink"
        descriptions.update(
            {
                f"{prefix}_mean": f"Average {direction} bandwidth across the trace.",
                f"{prefix}_std": f"Standard deviation of {direction} bandwidth.",
                f"{prefix}_min": f"Minimum {direction} bandwidth.",
                f"{prefix}_max": f"Maximum {direction} bandwidth.",
                f"{prefix}_p10": f"10th percentile {direction} bandwidth.",
                f"{prefix}_p90": f"90th percentile {direction} bandwidth.",
                f"{prefix}_slope": f"Least-squares linear slope of {direction} bandwidth over time.",
                f"{prefix}_abs_diff_mean": f"Mean absolute first difference of {direction} bandwidth, measuring average per-step jump size.",
                f"{prefix}_abs_second_diff_mean": f"Mean absolute second difference of {direction} bandwidth, measuring curvature.",
                f"{prefix}_sign_change_rate": f"Rate at which the {direction} bandwidth derivative changes sign, measuring oscillation frequency.",
                f"{prefix}_plateau_fraction": f"Fraction of tiny step-to-step changes in {direction} bandwidth, capturing plateaus.",
                f"{prefix}_peak_to_mean": f"Peak-to-mean ratio of {direction} bandwidth.",
                f"{prefix}_autocorr_lag1": f"Lag-1 autocorrelation of {direction} bandwidth.",
            }
        )

    for prefix in _LOSS_DELAY_PREFIXES:
        metric = "loss" if "loss" in prefix else "delay"
        direction = "uplink" if prefix.startswith("uplink") else "downlink"
        units = "" if metric == "loss" else " in milliseconds"
        descriptions.update(
            {
                f"{prefix}_mean": f"Average {direction} {metric}{units} across the trace.",
                f"{prefix}_std": f"Standard deviation of {direction} {metric}{units}.",
                f"{prefix}_max": f"Maximum {direction} {metric}{units}.",
                f"{prefix}_span": f"Range of {direction} {metric}{units}, i.e. max minus min.",
                f"{prefix}_nonzero_fraction": f"Fraction of steps where {direction} {metric} is nonzero, capturing how persistent this impairment is.",
                f"{prefix}_abs_diff_mean": f"Mean absolute first difference of {direction} {metric}{units}, capturing how abruptly it changes.",
            }
        )

    for window_steps in _SHARED_WINDOW_STEPS:
        descriptions.update(
            {
                f"shared_bw_window{window_steps}_min_mean": f"Minimum mean shared bottleneck bandwidth over any sliding window of {window_steps} steps, capturing sustained low-bandwidth segments.",
                f"shared_bw_window{window_steps}_max_mean": f"Maximum mean shared bottleneck bandwidth over any sliding window of {window_steps} steps, capturing sustained high-bandwidth segments.",
                f"shared_bw_window{window_steps}_max_cv": f"Maximum coefficient of variation of shared bottleneck bandwidth over any {window_steps}-step window, capturing localized burstiness.",
                f"shared_bw_window{window_steps}_max_curvature": f"Maximum mean absolute second difference of shared bottleneck bandwidth over any {window_steps}-step window, capturing localized curvature.",
                f"shared_bw_window{window_steps}_negative_slope_fraction": f"Fraction of {window_steps}-step windows whose linear slope is meaningfully negative, capturing repeated downward ramps.",
            }
        )

    return descriptions


FEATURE_DESCRIPTIONS: dict[str, str] = _description_map()


def _safe_series(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return np.zeros(1, dtype=np.float64)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, float(q)))


def _linear_slope(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    x_centered = x - float(np.mean(x))
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 1e-12:
        return 0.0
    y_centered = values - float(np.mean(values))
    return float(np.dot(x_centered, y_centered) / denom)


def _early_late_delta(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    split = max(int(values.size // 3), 1)
    early = values[:split]
    late = values[-split:]
    return float(np.mean(late) - np.mean(early))


def _autocorr_lag1(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    left = values[:-1]
    right = values[1:]
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return 0.0
    corr = np.corrcoef(left, right)[0, 1]
    return float(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))


def _count_runs(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    count = 0
    in_run = False
    for item in mask.tolist():
        if bool(item):
            if not in_run:
                count += 1
                in_run = True
        else:
            in_run = False
    return int(count)


def _longest_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for item in mask.tolist():
        if bool(item):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _series_plateau_fraction(values: np.ndarray) -> float:
    diffs = np.abs(np.diff(values))
    if diffs.size == 0:
        return 1.0
    tolerance = max(float(np.mean(np.abs(values))) * 0.01, 1e-6)
    return float(np.mean(diffs <= tolerance))


def _series_sign_change_rate(values: np.ndarray) -> float:
    diffs = np.diff(values)
    if diffs.size <= 1:
        return 0.0
    signs = np.sign(diffs)
    sign_changes = (signs[1:] * signs[:-1]) < 0.0
    return float(np.mean(sign_changes))


def _shared_profile_kind(shared_bw: np.ndarray) -> str:
    cv = float(np.std(shared_bw) / max(np.mean(shared_bw), 1e-6))
    abs_diff_mean = float(np.mean(np.abs(np.diff(shared_bw)))) if shared_bw.size > 1 else 0.0
    if cv <= 0.03 and abs_diff_mean <= max(float(np.mean(shared_bw)) * 0.01, 0.1):
        return "flat"
    return "variable"


def _bandwidth_coupling_kind(uplink_bw: np.ndarray, downlink_bw: np.ndarray) -> str:
    diff = np.abs(uplink_bw - downlink_bw)
    shared_tol = max(float(np.mean(np.minimum(uplink_bw, downlink_bw))) * 0.02, 1e-3)
    symmetry_fraction = float(np.mean(diff <= shared_tol)) if diff.size > 0 else 1.0
    if symmetry_fraction >= 0.98:
        return "shared"
    if uplink_bw.size > 1 and float(np.std(uplink_bw)) > 1e-12 and float(np.std(downlink_bw)) > 1e-12:
        corr = float(np.corrcoef(uplink_bw, downlink_bw)[0, 1])
        corr = float(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
    else:
        corr = 0.0
    if corr >= 0.5:
        return "loosely_coupled"
    return "independent"


def _profile_kind(values: np.ndarray, *, zero_tolerance: float, const_tolerance: float) -> str:
    if bool(np.all(np.abs(values) <= float(zero_tolerance))):
        return "zero"
    if float(np.max(values) - np.min(values)) <= float(const_tolerance):
        return "fixed"
    return "variable"


def _bandwidth_feature_values(prefix: str, values: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    diffs = np.abs(np.diff(values))
    second = np.abs(np.diff(values, n=2))
    high_threshold = mean + std
    low_threshold = mean - std
    high_mask = values >= high_threshold
    low_mask = values <= low_threshold
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_cv": float(std / max(mean, 1e-6)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_p10": _percentile(values, 10.0),
        f"{prefix}_p90": _percentile(values, 90.0),
        f"{prefix}_span": float(np.max(values) - np.min(values)),
        f"{prefix}_slope": _linear_slope(values),
        f"{prefix}_early_late_delta": _early_late_delta(values),
        f"{prefix}_abs_diff_mean": float(np.mean(diffs)) if diffs.size > 0 else 0.0,
        f"{prefix}_abs_diff_p90": _percentile(diffs, 90.0) if diffs.size > 0 else 0.0,
        f"{prefix}_abs_second_diff_mean": float(np.mean(second)) if second.size > 0 else 0.0,
        f"{prefix}_sign_change_rate": _series_sign_change_rate(values),
        f"{prefix}_plateau_fraction": _series_plateau_fraction(values),
        f"{prefix}_high_fraction": float(np.mean(high_mask)),
        f"{prefix}_low_fraction": float(np.mean(low_mask)),
        f"{prefix}_high_run_count": float(_count_runs(high_mask)),
        f"{prefix}_low_run_count": float(_count_runs(low_mask)),
        f"{prefix}_longest_high_run": float(_longest_run(high_mask)),
        f"{prefix}_longest_low_run": float(_longest_run(low_mask)),
        f"{prefix}_peak_to_mean": float(np.max(values) / max(mean, 1e-6)),
        f"{prefix}_autocorr_lag1": _autocorr_lag1(values),
    }


def _directional_bandwidth_feature_values(prefix: str, values: np.ndarray) -> dict[str, float]:
    diffs = np.abs(np.diff(values))
    second = np.abs(np.diff(values, n=2))
    mean = float(np.mean(values))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_p10": _percentile(values, 10.0),
        f"{prefix}_p90": _percentile(values, 90.0),
        f"{prefix}_slope": _linear_slope(values),
        f"{prefix}_abs_diff_mean": float(np.mean(diffs)) if diffs.size > 0 else 0.0,
        f"{prefix}_abs_second_diff_mean": float(np.mean(second)) if second.size > 0 else 0.0,
        f"{prefix}_sign_change_rate": _series_sign_change_rate(values),
        f"{prefix}_plateau_fraction": _series_plateau_fraction(values),
        f"{prefix}_peak_to_mean": float(np.max(values) / max(mean, 1e-6)),
        f"{prefix}_autocorr_lag1": _autocorr_lag1(values),
    }


def _loss_delay_feature_values(prefix: str, values: np.ndarray) -> dict[str, float]:
    diffs = np.abs(np.diff(values))
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_span": float(np.max(values) - np.min(values)),
        f"{prefix}_nonzero_fraction": float(np.mean(np.abs(values) > 1e-9)),
        f"{prefix}_abs_diff_mean": float(np.mean(diffs)) if diffs.size > 0 else 0.0,
    }


def _window_feature_values(shared_bw: np.ndarray, *, window_steps: int) -> dict[str, float]:
    if shared_bw.size <= 1:
        return {
            f"shared_bw_window{window_steps}_min_mean": float(np.mean(shared_bw)),
            f"shared_bw_window{window_steps}_max_mean": float(np.mean(shared_bw)),
            f"shared_bw_window{window_steps}_max_cv": 0.0,
            f"shared_bw_window{window_steps}_max_curvature": 0.0,
            f"shared_bw_window{window_steps}_negative_slope_fraction": 0.0,
        }
    effective = min(int(window_steps), int(shared_bw.size))
    means: list[float] = []
    cvs: list[float] = []
    curvatures: list[float] = []
    negative_slope_count = 0
    slope_threshold = -max(float(np.std(shared_bw)) / max(float(effective - 1), 1.0), 1e-6)
    for start in range(0, int(shared_bw.size) - effective + 1):
        window = shared_bw[start : start + effective]
        window_mean = float(np.mean(window))
        window_std = float(np.std(window))
        means.append(window_mean)
        cvs.append(float(window_std / max(window_mean, 1e-6)))
        second = np.abs(np.diff(window, n=2))
        curvatures.append(float(np.mean(second)) if second.size > 0 else 0.0)
        if _linear_slope(window) <= slope_threshold:
            negative_slope_count += 1
    total_windows = max(len(means), 1)
    return {
        f"shared_bw_window{window_steps}_min_mean": float(np.min(means)),
        f"shared_bw_window{window_steps}_max_mean": float(np.max(means)),
        f"shared_bw_window{window_steps}_max_cv": float(np.max(cvs)),
        f"shared_bw_window{window_steps}_max_curvature": float(np.max(curvatures)),
        f"shared_bw_window{window_steps}_negative_slope_fraction": float(negative_slope_count) / float(total_windows),
    }


def _categorical_feature_values(
    *,
    uplink_bw: np.ndarray,
    downlink_bw: np.ndarray,
    uplink_loss: np.ndarray,
    downlink_loss: np.ndarray,
    uplink_delay: np.ndarray,
    downlink_delay: np.ndarray,
    baseline_methods_key: str,
    attack_mode: str,
) -> dict[str, Any]:
    shared_bw = np.minimum(uplink_bw, downlink_bw)
    delay_series = np.concatenate([uplink_delay, downlink_delay], axis=0)
    loss_series = np.concatenate([uplink_loss, downlink_loss], axis=0)
    return {
        CATEGORY_BANDWIDTH_COUPLING_KIND: _bandwidth_coupling_kind(uplink_bw, downlink_bw),
        CATEGORY_BANDWIDTH_PROFILE_KIND: _shared_profile_kind(shared_bw),
        CATEGORY_LOSS_PROFILE_KIND: _profile_kind(loss_series, zero_tolerance=1e-9, const_tolerance=1e-6),
        CATEGORY_DELAY_PROFILE_KIND: _profile_kind(delay_series, zero_tolerance=1e-9, const_tolerance=1e-6),
        CATEGORY_BASELINE_METHODS_KEY: str(baseline_methods_key),
        CATEGORY_ATTACK_MODE: str(attack_mode),
    }


def extract_trace_explanation_features(
    action_schedule: Sequence[Sequence[float]] | np.ndarray,
    *,
    attack_interval_ms: float,
    baseline_methods_key: str,
    attack_mode: str,
) -> dict[str, Any]:
    actions = np.asarray(action_schedule, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError(f"expected action schedule with shape [num_steps, >=6], received {actions.shape}")

    uplink_bw = _safe_series(actions[:, 0])
    downlink_bw = _safe_series(actions[:, 1])
    uplink_loss = _safe_series(actions[:, 2])
    downlink_loss = _safe_series(actions[:, 3])
    uplink_delay = _safe_series(actions[:, 4])
    downlink_delay = _safe_series(actions[:, 5])
    shared_bw = np.minimum(uplink_bw, downlink_bw)

    feature_values: dict[str, Any] = _categorical_feature_values(
        uplink_bw=uplink_bw,
        downlink_bw=downlink_bw,
        uplink_loss=uplink_loss,
        downlink_loss=downlink_loss,
        uplink_delay=uplink_delay,
        downlink_delay=downlink_delay,
        baseline_methods_key=str(baseline_methods_key),
        attack_mode=str(attack_mode),
    )
    feature_values.update(_bandwidth_feature_values(_SHARED_BW_PREFIX, shared_bw))
    feature_values.update(_directional_bandwidth_feature_values("uplink_bw", uplink_bw))
    feature_values.update(_directional_bandwidth_feature_values("downlink_bw", downlink_bw))
    feature_values.update(_loss_delay_feature_values("uplink_loss", uplink_loss))
    feature_values.update(_loss_delay_feature_values("downlink_loss", downlink_loss))
    feature_values.update(_loss_delay_feature_values("uplink_delay", uplink_delay))
    feature_values.update(_loss_delay_feature_values("downlink_delay", downlink_delay))
    for window_steps in _SHARED_WINDOW_STEPS:
        feature_values.update(_window_feature_values(shared_bw, window_steps=int(window_steps)))

    diff_bw = np.abs(uplink_bw - downlink_bw)
    min_bw = np.minimum(uplink_bw, downlink_bw)
    max_bw = np.maximum(uplink_bw, downlink_bw)
    shared_tolerance = max(float(np.mean(min_bw)) * 0.02, 1e-3)
    feature_values.update(
        {
            "num_steps": float(actions.shape[0]),
            "duration_seconds": float(actions.shape[0]) * float(attack_interval_ms) / 1000.0,
            "bandwidth_symmetry_fraction": float(np.mean(diff_bw <= shared_tolerance)),
            "bandwidth_corr": float(np.nan_to_num(np.corrcoef(uplink_bw, downlink_bw)[0, 1], nan=0.0, posinf=0.0, neginf=0.0))
            if actions.shape[0] > 1 and float(np.std(uplink_bw)) > 1e-12 and float(np.std(downlink_bw)) > 1e-12
            else 0.0,
            "bandwidth_abs_diff_mean": float(np.mean(diff_bw)),
            "bandwidth_abs_diff_p90": _percentile(diff_bw, 90.0),
            "bandwidth_ratio_mean": float(np.mean(min_bw / np.maximum(max_bw, 1e-6))),
            "loss_abs_diff_mean": float(np.mean(np.abs(uplink_loss - downlink_loss))),
            "delay_abs_diff_mean": float(np.mean(np.abs(uplink_delay - downlink_delay))),
        }
    )

    missing = [feature for feature in FEATURE_COLUMNS if feature not in feature_values]
    if missing:
        raise RuntimeError(f"missing trace explanation features: {missing}")
    return feature_values
