from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
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
_LOSS_DELAY_PREFIXES = ("uplink_loss", "downlink_loss", "uplink_delay", "downlink_delay")
_ACTION_PREFIXES = ("sage_action", "best_baseline_action")
DEFAULT_SHARED_WINDOW_STEPS: tuple[int, ...] = (5, 10, 20, 40, 100, 200)
_WINDOW_FEATURE_SUFFIXES: tuple[str, ...] = (
    "min_mean",
    "max_mean",
    "max_cv",
    "max_curvature",
    "negative_slope_fraction",
)
_WINDOW_FEATURE_PATTERN = re.compile(
    r"^shared_bw_window(?P<steps>\d+)_(?P<suffix>min_mean|max_mean|max_cv|max_curvature|negative_slope_fraction)$"
)


def _bandwidth_feature_columns(prefix: str) -> list[str]:
    return [
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


def _loss_delay_feature_columns(prefix: str) -> list[str]:
    return [
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_max",
        f"{prefix}_span",
        f"{prefix}_nonzero_fraction",
        f"{prefix}_abs_diff_mean",
    ]


def _action_feature_columns(prefix: str) -> list[str]:
    return [
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_p10",
        f"{prefix}_p90",
        f"{prefix}_span",
        f"{prefix}_slope",
        f"{prefix}_abs_diff_mean",
        f"{prefix}_abs_second_diff_mean",
        f"{prefix}_increase_fraction",
        f"{prefix}_decrease_fraction",
        f"{prefix}_hold_fraction",
        f"{prefix}_longest_increase_run",
        f"{prefix}_longest_decrease_run",
        f"{prefix}_autocorr_lag1",
    ]


ACTION_RELATION_FEATURE_COLUMNS: tuple[str, ...] = (
    "action_reference_available_fraction",
    "action_gap_mean",
    "action_gap_abs_mean",
    "action_gap_p90",
    "action_gap_positive_fraction",
    "action_gap_negative_fraction",
    "action_gap_positive_mean",
    "action_gap_negative_mean_abs",
    "action_gap_longest_positive_run",
    "action_gap_longest_negative_run",
    "action_gap_positive_mean_scaled",
    "action_gap_negative_mean_scaled",
)


INTERACTION_FEATURE_COLUMNS: tuple[str, ...] = (
    "interaction_low_bw_fraction",
    "interaction_high_bw_fraction",
    "interaction_bw_drop_fraction",
    "interaction_bw_rebound_fraction",
    "interaction_delay_stress_fraction",
    "interaction_high_gap_fraction",
    "interaction_sage_increase_during_low_bw_fraction",
    "interaction_sage_increase_during_bw_drop_fraction",
    "interaction_sage_no_backoff_during_delay_stress_fraction",
    "interaction_sage_no_backoff_during_high_gap_fraction",
    "interaction_sage_no_increase_during_high_bw_fraction",
    "interaction_sage_no_increase_during_bw_rebound_fraction",
    "interaction_sage_no_increase_during_high_gap_fraction",
    "interaction_positive_action_gap_during_low_bw_fraction",
    "interaction_positive_action_gap_during_bw_drop_fraction",
    "interaction_positive_action_gap_during_high_gap_fraction",
    "interaction_negative_action_gap_during_high_bw_fraction",
    "interaction_negative_action_gap_during_bw_rebound_fraction",
    "interaction_negative_action_gap_during_high_gap_fraction",
    "interaction_aggressive_stress_fraction",
    "interaction_aggressive_stress_longest_run",
    "interaction_conservative_recovery_fraction",
    "interaction_conservative_recovery_longest_run",
    "interaction_backoff_delay_after_low_bw_mean",
    "interaction_backoff_delay_after_bw_drop_mean",
    "interaction_increase_delay_after_high_bw_mean",
    "interaction_increase_delay_after_bw_rebound_mean",
    "interaction_hard_gap_growth_after_increase_lag1_mean",
    "interaction_hard_gap_growth_after_increase_lag1_positive_fraction",
    "interaction_hard_gap_growth_after_increase_lag2_mean",
    "interaction_hard_gap_growth_after_positive_action_gap_lag1_mean",
    "interaction_hard_gap_growth_after_positive_action_gap_lag1_positive_fraction",
    "interaction_hard_gap_growth_after_negative_action_gap_lag1_mean",
    "interaction_hard_gap_growth_after_negative_action_gap_lag1_positive_fraction",
    "interaction_rate_deficit_after_negative_action_gap_lag1_mean",
    "interaction_rtt_deficit_after_increase_lag1_mean",
    "interaction_loss_excess_after_increase_lag1_mean",
)


SCORE_FEATURE_COLUMNS: tuple[str, ...] = (
    "env_stress_score",
    "action_aggressive_score",
    "action_conservative_score",
    "action_mismatch_score",
    "interaction_aggressive_score",
    "interaction_conservative_score",
    "interaction_amplification_score",
)


def _window_feature_columns(window_steps: int) -> list[str]:
    return [
        f"shared_bw_window{window_steps}_min_mean",
        f"shared_bw_window{window_steps}_max_mean",
        f"shared_bw_window{window_steps}_max_cv",
        f"shared_bw_window{window_steps}_max_curvature",
        f"shared_bw_window{window_steps}_negative_slope_fraction",
    ]


def _action_window_feature_columns(window_steps: int) -> list[str]:
    return [
        f"sage_action_window{window_steps}_max_increase_fraction",
        f"sage_action_window{window_steps}_max_decrease_fraction",
        f"action_gap_window{window_steps}_max_positive_fraction",
        f"action_gap_window{window_steps}_max_negative_fraction",
        f"interaction_window{window_steps}_max_aggressive_stress_fraction",
        f"interaction_window{window_steps}_max_conservative_recovery_fraction",
        f"interaction_window{window_steps}_max_no_backoff_high_gap_fraction",
        f"interaction_window{window_steps}_max_no_increase_high_gap_fraction",
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


_BASE_NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    [*_bandwidth_feature_columns(_SHARED_BW_PREFIX)]
    + [feature for prefix in _LOSS_DELAY_PREFIXES for feature in _loss_delay_feature_columns(prefix)]
    + [feature for prefix in _ACTION_PREFIXES for feature in _action_feature_columns(prefix)]
    + list(ACTION_RELATION_FEATURE_COLUMNS)
    + list(INTERACTION_FEATURE_COLUMNS)
    + list(SCORE_FEATURE_COLUMNS)
    + list(CROSS_FEATURE_COLUMNS)
)


def normalize_trace_explanation_window_steps(window_steps: Sequence[int] | None = None) -> tuple[int, ...]:
    source = DEFAULT_SHARED_WINDOW_STEPS if window_steps is None else tuple(window_steps)
    normalized: list[int] = []
    seen: set[int] = set()
    for value in source:
        step_count = int(value)
        if step_count <= 0:
            raise ValueError(f"window step counts must be positive integers, received {value}")
        if step_count in seen:
            continue
        seen.add(step_count)
        normalized.append(step_count)
    return tuple(normalized)


def trace_explanation_numeric_feature_columns(window_steps: Sequence[int] | None = None) -> tuple[str, ...]:
    resolved_window_steps = normalize_trace_explanation_window_steps(window_steps)
    return tuple(
        list(_BASE_NUMERIC_FEATURE_COLUMNS)
        + [feature for step_count in resolved_window_steps for feature in _window_feature_columns(int(step_count))]
        + [feature for step_count in resolved_window_steps for feature in _action_window_feature_columns(int(step_count))]
    )


def trace_explanation_feature_columns(window_steps: Sequence[int] | None = None) -> tuple[str, ...]:
    return tuple([*CATEGORICAL_FEATURE_COLUMNS, *trace_explanation_numeric_feature_columns(window_steps)])


def _description_map(window_steps: Sequence[int] | None = None) -> dict[str, str]:
    resolved_window_steps = normalize_trace_explanation_window_steps(window_steps)
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
        "action_reference_available_fraction": "Fraction of replay steps for which a best-baseline action proxy was available, used to guard action-aware diagnostics.",
        "action_gap_mean": "Mean difference between Sage's previous action and the best baseline's previous action at each step.",
        "action_gap_abs_mean": "Mean absolute Sage-versus-best-baseline action gap.",
        "action_gap_p90": "90th percentile of the action-gap magnitude, highlighting large action mismatches.",
        "action_gap_positive_fraction": "Fraction of steps where Sage is more aggressive than the best baseline according to the action proxy.",
        "action_gap_negative_fraction": "Fraction of steps where Sage is more conservative than the best baseline according to the action proxy.",
        "action_gap_positive_mean": "Mean positive action gap on steps where Sage is more aggressive than the best baseline.",
        "action_gap_negative_mean_abs": "Mean absolute negative action gap on steps where Sage is more conservative than the best baseline.",
        "action_gap_longest_positive_run": "Longest run of consecutive steps where Sage remains more aggressive than the best baseline.",
        "action_gap_longest_negative_run": "Longest run of consecutive steps where Sage remains more conservative than the best baseline.",
        "action_gap_positive_mean_scaled": "Mean positive action gap scaled by the larger action span of Sage and the best baseline, making action mismatch more comparable across traces.",
        "action_gap_negative_mean_scaled": "Mean absolute negative action gap scaled by the larger action span of Sage and the best baseline, capturing how strongly Sage underreacts relative to the best baseline.",
        "interaction_low_bw_fraction": "Fraction of replay steps identified as low-bandwidth stress steps within the trace.",
        "interaction_high_bw_fraction": "Fraction of replay steps identified as high-bandwidth opportunity steps within the trace.",
        "interaction_bw_drop_fraction": "Fraction of replay steps identified as sharp shared-bandwidth drops.",
        "interaction_bw_rebound_fraction": "Fraction of replay steps identified as sharp shared-bandwidth rebounds or recoveries.",
        "interaction_delay_stress_fraction": "Fraction of replay steps identified as high-RTT stress steps for Sage.",
        "interaction_high_gap_fraction": "Fraction of replay steps where the hard Sage-versus-best-baseline gap is high relative to the trace.",
        "interaction_sage_increase_during_low_bw_fraction": "Among low-bandwidth stress steps, fraction where Sage still increases its action proxy.",
        "interaction_sage_increase_during_bw_drop_fraction": "Among sharp bandwidth-drop steps, fraction where Sage still increases its action proxy.",
        "interaction_sage_no_backoff_during_delay_stress_fraction": "Among RTT-stress steps, fraction where Sage does not decrease its action proxy.",
        "interaction_sage_no_backoff_during_high_gap_fraction": "Among high-gap steps, fraction where Sage does not decrease its action proxy.",
        "interaction_sage_no_increase_during_high_bw_fraction": "Among high-bandwidth opportunity steps, fraction where Sage does not increase its action proxy.",
        "interaction_sage_no_increase_during_bw_rebound_fraction": "Among sharp bandwidth rebounds, fraction where Sage does not increase its action proxy.",
        "interaction_sage_no_increase_during_high_gap_fraction": "Among high-gap steps, fraction where Sage does not increase its action proxy.",
        "interaction_positive_action_gap_during_low_bw_fraction": "Among low-bandwidth stress steps, fraction where Sage is more aggressive than the best baseline.",
        "interaction_positive_action_gap_during_bw_drop_fraction": "Among sharp bandwidth-drop steps, fraction where Sage is more aggressive than the best baseline.",
        "interaction_positive_action_gap_during_high_gap_fraction": "Among high-gap steps, fraction where Sage is more aggressive than the best baseline.",
        "interaction_negative_action_gap_during_high_bw_fraction": "Among high-bandwidth opportunity steps, fraction where Sage is more conservative than the best baseline.",
        "interaction_negative_action_gap_during_bw_rebound_fraction": "Among sharp bandwidth rebounds, fraction where Sage is more conservative than the best baseline.",
        "interaction_negative_action_gap_during_high_gap_fraction": "Among high-gap steps, fraction where Sage is more conservative than the best baseline.",
        "interaction_aggressive_stress_fraction": "Fraction of all replay steps where Sage is aggressive or more aggressive than the best baseline while the environment is stressed.",
        "interaction_aggressive_stress_longest_run": "Longest run of consecutive steps where stress and aggressive Sage behavior co-occur.",
        "interaction_conservative_recovery_fraction": "Fraction of all replay steps where the environment offers recovery opportunity but Sage remains conservative or below the best baseline.",
        "interaction_conservative_recovery_longest_run": "Longest run of consecutive steps where recovery opportunity and conservative Sage behavior co-occur.",
        "interaction_backoff_delay_after_low_bw_mean": "Mean number of steps Sage takes to produce a backoff after entering a low-bandwidth regime.",
        "interaction_backoff_delay_after_bw_drop_mean": "Mean number of steps Sage takes to back off after a sharp bandwidth drop.",
        "interaction_increase_delay_after_high_bw_mean": "Mean number of steps Sage takes to increase after entering a high-bandwidth opportunity regime.",
        "interaction_increase_delay_after_bw_rebound_mean": "Mean number of steps Sage takes to increase after a sharp bandwidth rebound.",
        "interaction_hard_gap_growth_after_increase_lag1_mean": "Mean next-step change in hard gap after Sage increases its action.",
        "interaction_hard_gap_growth_after_increase_lag1_positive_fraction": "Fraction of Sage action increases that are followed by a larger hard gap one step later.",
        "interaction_hard_gap_growth_after_increase_lag2_mean": "Mean two-step change in hard gap after Sage increases its action.",
        "interaction_hard_gap_growth_after_positive_action_gap_lag1_mean": "Mean next-step change in hard gap after Sage is more aggressive than the best baseline.",
        "interaction_hard_gap_growth_after_positive_action_gap_lag1_positive_fraction": "Fraction of positive action-gap steps that are followed by a larger hard gap one step later.",
        "interaction_hard_gap_growth_after_negative_action_gap_lag1_mean": "Mean next-step change in hard gap after Sage is more conservative than the best baseline.",
        "interaction_hard_gap_growth_after_negative_action_gap_lag1_positive_fraction": "Fraction of negative action-gap steps that are followed by a larger hard gap one step later.",
        "interaction_rate_deficit_after_negative_action_gap_lag1_mean": "Mean rate-related score deficit one step after Sage is more conservative than the best baseline.",
        "interaction_rtt_deficit_after_increase_lag1_mean": "Mean RTT-related score deficit one step after Sage increases its action.",
        "interaction_loss_excess_after_increase_lag1_mean": "Mean loss-penalty excess one step after Sage increases its action.",
        "env_stress_score": "Composite score summarizing environmental severity from bandwidth scarcity, trough persistence, downward ramps, and localized low windows.",
        "action_aggressive_score": "Composite score summarizing how persistently Sage diverges toward greater aggressiveness relative to the best baseline.",
        "action_conservative_score": "Composite score summarizing how persistently Sage diverges toward greater conservatism or underreaction relative to the best baseline.",
        "action_mismatch_score": "Symmetric composite score summarizing either persistent over-aggressiveness or persistent underreaction relative to the best baseline.",
        "interaction_aggressive_score": "Composite score summarizing whether aggressive Sage behavior coincides with stress and is followed by worsening gaps.",
        "interaction_conservative_score": "Composite score summarizing whether conservative Sage behavior coincides with recovery opportunity and is followed by worsening throughput gaps.",
        "interaction_amplification_score": "Symmetric composite score summarizing whether either aggressive or conservative Sage behavior amplifies the observed gap under the current conditions.",
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

    for prefix, actor_name in (("sage_action", "Sage"), ("best_baseline_action", "the best baseline")):
        descriptions.update(
            {
                f"{prefix}_mean": f"Average previous-action proxy for {actor_name} across the replayed trace.",
                f"{prefix}_std": f"Standard deviation of the previous-action proxy for {actor_name}.",
                f"{prefix}_p10": f"10th percentile of the previous-action proxy for {actor_name}.",
                f"{prefix}_p90": f"90th percentile of the previous-action proxy for {actor_name}.",
                f"{prefix}_span": f"Range of the previous-action proxy for {actor_name}.",
                f"{prefix}_slope": f"Least-squares linear slope of the previous-action proxy for {actor_name} over time.",
                f"{prefix}_abs_diff_mean": f"Mean absolute step-to-step action change for {actor_name}.",
                f"{prefix}_abs_second_diff_mean": f"Mean absolute second difference of the action proxy for {actor_name}, capturing action curvature or oscillatory acceleration.",
                f"{prefix}_increase_fraction": f"Fraction of steps where {actor_name} increases its action proxy.",
                f"{prefix}_decrease_fraction": f"Fraction of steps where {actor_name} decreases its action proxy.",
                f"{prefix}_hold_fraction": f"Fraction of steps where {actor_name} approximately holds its action proxy steady.",
                f"{prefix}_longest_increase_run": f"Longest run of consecutive action increases for {actor_name}.",
                f"{prefix}_longest_decrease_run": f"Longest run of consecutive action decreases for {actor_name}.",
                f"{prefix}_autocorr_lag1": f"Lag-1 autocorrelation of the action proxy for {actor_name}, capturing action persistence.",
            }
        )

    for window_steps in resolved_window_steps:
        descriptions.update(
            {
                f"shared_bw_window{window_steps}_min_mean": f"Minimum mean shared bottleneck bandwidth over any sliding window of {window_steps} steps, capturing sustained low-bandwidth segments.",
                f"shared_bw_window{window_steps}_max_mean": f"Maximum mean shared bottleneck bandwidth over any sliding window of {window_steps} steps, capturing sustained high-bandwidth segments.",
                f"shared_bw_window{window_steps}_max_cv": f"Maximum coefficient of variation of shared bottleneck bandwidth over any {window_steps}-step window, capturing localized burstiness.",
                f"shared_bw_window{window_steps}_max_curvature": f"Maximum mean absolute second difference of shared bottleneck bandwidth over any {window_steps}-step window, capturing localized curvature.",
                f"shared_bw_window{window_steps}_negative_slope_fraction": f"Fraction of {window_steps}-step windows whose linear slope is meaningfully negative, capturing repeated downward ramps.",
                f"sage_action_window{window_steps}_max_increase_fraction": f"Maximum fraction of action increases for Sage over any {window_steps}-step window, capturing localized aggressive ramps.",
                f"sage_action_window{window_steps}_max_decrease_fraction": f"Maximum fraction of action decreases for Sage over any {window_steps}-step window, capturing localized conservative pullbacks.",
                f"action_gap_window{window_steps}_max_positive_fraction": f"Maximum fraction of steps where Sage is more aggressive than the best baseline over any {window_steps}-step window.",
                f"action_gap_window{window_steps}_max_negative_fraction": f"Maximum fraction of steps where Sage is more conservative than the best baseline over any {window_steps}-step window.",
                f"interaction_window{window_steps}_max_aggressive_stress_fraction": f"Maximum fraction of steps where Sage behaves aggressively under stress over any {window_steps}-step window.",
                f"interaction_window{window_steps}_max_conservative_recovery_fraction": f"Maximum fraction of steps where Sage behaves conservatively despite recovery opportunity over any {window_steps}-step window.",
                f"interaction_window{window_steps}_max_no_backoff_high_gap_fraction": f"Maximum conditional fraction of high-gap steps without a Sage backoff over any {window_steps}-step window.",
                f"interaction_window{window_steps}_max_no_increase_high_gap_fraction": f"Maximum conditional fraction of high-gap steps without a Sage increase over any {window_steps}-step window.",
            }
        )

    return descriptions


def trace_explanation_feature_descriptions(window_steps: Sequence[int] | None = None) -> dict[str, str]:
    return _description_map(window_steps)


def window_steps_from_feature_columns(columns: Sequence[str]) -> tuple[int, ...]:
    resolved: list[int] = []
    seen: set[int] = set()
    for column in columns:
        match = _WINDOW_FEATURE_PATTERN.match(str(column))
        if match is None:
            continue
        if str(match.group("suffix")) not in _WINDOW_FEATURE_SUFFIXES:
            continue
        step_count = int(match.group("steps"))
        if step_count in seen:
            continue
        seen.add(step_count)
        resolved.append(step_count)
    return tuple(sorted(resolved))


def infer_trace_explanation_feature_schema(
    columns: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    available = {str(column) for column in columns}
    categorical = tuple(column for column in CATEGORICAL_FEATURE_COLUMNS if column in available)
    window_steps = window_steps_from_feature_columns(tuple(available))
    numeric_candidates = trace_explanation_numeric_feature_columns(window_steps)
    numeric = tuple(column for column in numeric_candidates if column in available)
    return categorical, numeric, tuple([*categorical, *numeric])


def feature_descriptions_for_columns(columns: Sequence[str]) -> dict[str, str]:
    _, _, feature_columns = infer_trace_explanation_feature_schema(columns)
    descriptions = trace_explanation_feature_descriptions(window_steps_from_feature_columns(feature_columns))
    return {column: descriptions[column] for column in feature_columns if column in descriptions}


NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = trace_explanation_numeric_feature_columns()
FEATURE_COLUMNS: tuple[str, ...] = trace_explanation_feature_columns()
FEATURE_DESCRIPTIONS: dict[str, str] = trace_explanation_feature_descriptions()


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


def _fraction(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.mean(np.asarray(mask, dtype=np.float64)))


def _conditional_fraction(base_mask: np.ndarray, condition_mask: np.ndarray) -> float:
    base = np.asarray(base_mask, dtype=bool)
    condition = np.asarray(condition_mask, dtype=bool)
    count = int(np.sum(base))
    if count <= 0:
        return 0.0
    return float(np.mean(condition[base]))


def _positive_mean(values: np.ndarray) -> float:
    positive = values[values > 0.0]
    if positive.size == 0:
        return 0.0
    return float(np.mean(positive))


def _negative_mean_abs(values: np.ndarray) -> float:
    negative = values[values < 0.0]
    if negative.size == 0:
        return 0.0
    return float(np.mean(np.abs(negative)))


def _clip01(value: float) -> float:
    return float(min(max(float(value), 0.0), 1.0))


def _action_tolerance(*series: np.ndarray) -> float:
    finite_parts = [np.asarray(values, dtype=np.float64).reshape(-1) for values in series if values is not None]
    finite_parts = [values[np.isfinite(values)] for values in finite_parts if values.size > 0]
    if not finite_parts:
        return 1e-3
    merged = np.concatenate(finite_parts, axis=0)
    if merged.size == 0:
        return 1e-3
    span = float(np.percentile(merged, 90.0) - np.percentile(merged, 10.0))
    return max(0.05 * span, 1e-3)


def _stress_mask_low(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    threshold = _percentile(values, 25.0)
    return np.asarray(values <= threshold + 1e-9, dtype=bool)


def _stress_mask_high(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    if float(np.std(values)) <= 1e-9:
        return np.zeros(values.shape[0], dtype=bool)
    threshold = _percentile(values, 75.0)
    return np.asarray(values >= threshold, dtype=bool)


def _drop_mask(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.zeros(values.shape[0], dtype=bool)
    diffs = np.diff(values, prepend=values[0])
    negative_mag = np.maximum(-diffs, 0.0)
    active = negative_mag[negative_mag > 0.0]
    if active.size == 0:
        return np.zeros(values.shape[0], dtype=bool)
    threshold = max(float(np.percentile(active, 75.0)), float(np.mean(active)))
    return np.asarray(negative_mag >= threshold, dtype=bool)


def _rebound_mask(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.zeros(values.shape[0], dtype=bool)
    diffs = np.diff(values, prepend=values[0])
    positive_mag = np.maximum(diffs, 0.0)
    active = positive_mag[positive_mag > 0.0]
    if active.size == 0:
        return np.zeros(values.shape[0], dtype=bool)
    threshold = max(float(np.percentile(active, 75.0)), float(np.mean(active)))
    return np.asarray(positive_mag >= threshold, dtype=bool)


def _onset_mask(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.zeros(0, dtype=bool)
    previous = np.concatenate([np.asarray([False], dtype=bool), np.asarray(mask[:-1], dtype=bool)], axis=0)
    return np.asarray(mask, dtype=bool) & (~previous)


def _mean_delay_to_condition(onset_mask: np.ndarray, target_mask: np.ndarray, *, horizon: int) -> float:
    onset_indices = np.flatnonzero(np.asarray(onset_mask, dtype=bool))
    if onset_indices.size == 0:
        return 0.0
    effective_horizon = max(int(horizon), 1)
    delays: list[float] = []
    target = np.asarray(target_mask, dtype=bool)
    for onset_index in onset_indices.tolist():
        max_index = min(int(onset_index) + effective_horizon, target.shape[0] - 1)
        delay = float(effective_horizon)
        for idx in range(int(onset_index), int(max_index) + 1):
            if bool(target[idx]):
                delay = float(idx - int(onset_index))
                break
        delays.append(delay)
    return float(np.mean(delays)) if delays else 0.0


def _triggered_delta_stats(trigger_mask: np.ndarray, values: np.ndarray, *, lag: int) -> tuple[float, float]:
    effective_lag = max(int(lag), 1)
    if values.size <= effective_lag or trigger_mask.size <= effective_lag:
        return 0.0, 0.0
    trigger_indices = np.flatnonzero(np.asarray(trigger_mask[:-effective_lag], dtype=bool))
    if trigger_indices.size == 0:
        return 0.0, 0.0
    deltas = values[trigger_indices + effective_lag] - values[trigger_indices]
    return float(np.mean(deltas)), float(np.mean(deltas > 0.0))


def _triggered_future_mean(trigger_mask: np.ndarray, values: np.ndarray, *, lag: int) -> float:
    effective_lag = max(int(lag), 1)
    if values.size <= effective_lag or trigger_mask.size <= effective_lag:
        return 0.0
    trigger_indices = np.flatnonzero(np.asarray(trigger_mask[:-effective_lag], dtype=bool))
    if trigger_indices.size == 0:
        return 0.0
    return float(np.mean(values[trigger_indices + effective_lag]))


def _series_from_mapping(
    mapping: Mapping[str, Sequence[float]] | None,
    key: str,
    *,
    length: int,
) -> tuple[np.ndarray, float]:
    if mapping is None or key not in mapping:
        return np.zeros(length, dtype=np.float64), 0.0
    raw = np.asarray(mapping.get(key, []), dtype=np.float64).reshape(-1)
    if raw.size < length:
        padded = np.zeros(length, dtype=np.float64)
        if raw.size > 0:
            padded[: raw.size] = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        availability = float(np.mean(np.isfinite(raw))) if raw.size > 0 else 0.0
        return padded, availability * float(raw.size) / float(max(length, 1))
    raw = raw[:length]
    availability = float(np.mean(np.isfinite(raw))) if raw.size > 0 else 0.0
    return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), availability


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


def _action_feature_values(prefix: str, values: np.ndarray, *, tolerance: float) -> dict[str, float]:
    diffs = np.diff(values)
    second = np.abs(np.diff(values, n=2))
    increase_mask = diffs > float(tolerance)
    decrease_mask = diffs < -float(tolerance)
    hold_mask = np.abs(diffs) <= float(tolerance)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p10": _percentile(values, 10.0),
        f"{prefix}_p90": _percentile(values, 90.0),
        f"{prefix}_span": float(np.max(values) - np.min(values)),
        f"{prefix}_slope": _linear_slope(values),
        f"{prefix}_abs_diff_mean": float(np.mean(np.abs(diffs))) if diffs.size > 0 else 0.0,
        f"{prefix}_abs_second_diff_mean": float(np.mean(second)) if second.size > 0 else 0.0,
        f"{prefix}_increase_fraction": _fraction(increase_mask),
        f"{prefix}_decrease_fraction": _fraction(decrease_mask),
        f"{prefix}_hold_fraction": _fraction(hold_mask),
        f"{prefix}_longest_increase_run": float(_longest_run(increase_mask)),
        f"{prefix}_longest_decrease_run": float(_longest_run(decrease_mask)),
        f"{prefix}_autocorr_lag1": _autocorr_lag1(values),
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


def _action_window_feature_values(
    *,
    window_steps: int,
    increase_mask: np.ndarray,
    decrease_mask: np.ndarray,
    positive_action_gap_mask: np.ndarray,
    negative_action_gap_mask: np.ndarray,
    aggressive_stress_mask: np.ndarray,
    conservative_recovery_mask: np.ndarray,
    high_gap_mask: np.ndarray,
    no_backoff_mask: np.ndarray,
    no_increase_mask: np.ndarray,
) -> dict[str, float]:
    base_length = int(
        max(
            increase_mask.shape[0],
            decrease_mask.shape[0],
            positive_action_gap_mask.shape[0],
            negative_action_gap_mask.shape[0],
            aggressive_stress_mask.shape[0],
            conservative_recovery_mask.shape[0],
            1,
        )
    )
    effective = min(int(window_steps), base_length)
    max_increase = 0.0
    max_decrease = 0.0
    max_positive_gap = 0.0
    max_negative_gap = 0.0
    max_aggressive_stress = 0.0
    max_conservative_recovery = 0.0
    max_no_backoff_high_gap = 0.0
    max_no_increase_high_gap = 0.0
    for start in range(0, base_length - effective + 1):
        window_slice = slice(start, start + effective)
        max_increase = max(max_increase, _fraction(increase_mask[window_slice]))
        max_decrease = max(max_decrease, _fraction(decrease_mask[window_slice]))
        max_positive_gap = max(max_positive_gap, _fraction(positive_action_gap_mask[window_slice]))
        max_negative_gap = max(max_negative_gap, _fraction(negative_action_gap_mask[window_slice]))
        max_aggressive_stress = max(max_aggressive_stress, _fraction(aggressive_stress_mask[window_slice]))
        max_conservative_recovery = max(
            max_conservative_recovery,
            _fraction(conservative_recovery_mask[window_slice]),
        )
        max_no_backoff_high_gap = max(
            max_no_backoff_high_gap,
            _conditional_fraction(high_gap_mask[window_slice], no_backoff_mask[window_slice]),
        )
        max_no_increase_high_gap = max(
            max_no_increase_high_gap,
            _conditional_fraction(high_gap_mask[window_slice], no_increase_mask[window_slice]),
        )
    return {
        f"sage_action_window{window_steps}_max_increase_fraction": float(max_increase),
        f"sage_action_window{window_steps}_max_decrease_fraction": float(max_decrease),
        f"action_gap_window{window_steps}_max_positive_fraction": float(max_positive_gap),
        f"action_gap_window{window_steps}_max_negative_fraction": float(max_negative_gap),
        f"interaction_window{window_steps}_max_aggressive_stress_fraction": float(max_aggressive_stress),
        f"interaction_window{window_steps}_max_conservative_recovery_fraction": float(max_conservative_recovery),
        f"interaction_window{window_steps}_max_no_backoff_high_gap_fraction": float(max_no_backoff_high_gap),
        f"interaction_window{window_steps}_max_no_increase_high_gap_fraction": float(max_no_increase_high_gap),
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
    shared_window_steps: Sequence[int] | None = None,
    replay_series: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    resolved_window_steps = normalize_trace_explanation_window_steps(shared_window_steps)
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
    num_steps = int(actions.shape[0])

    sage_action, sage_action_availability = _series_from_mapping(replay_series, "sage_action", length=num_steps)
    best_action, best_action_availability = _series_from_mapping(replay_series, "best_baseline_action", length=num_steps)
    hard_gap_percent, _ = _series_from_mapping(replay_series, "hard_gap_percent", length=num_steps)
    rate_deficit, _ = _series_from_mapping(replay_series, "best_minus_sage_rate_contrib", length=num_steps)
    rtt_deficit, _ = _series_from_mapping(replay_series, "best_minus_sage_rtt_contrib", length=num_steps)
    loss_excess, _ = _series_from_mapping(replay_series, "sage_minus_best_loss_penalty", length=num_steps)
    sage_rtt_ms, _ = _series_from_mapping(replay_series, "sage_rtt_ms", length=num_steps)

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
    feature_values.update(_loss_delay_feature_values("uplink_loss", uplink_loss))
    feature_values.update(_loss_delay_feature_values("downlink_loss", downlink_loss))
    feature_values.update(_loss_delay_feature_values("uplink_delay", uplink_delay))
    feature_values.update(_loss_delay_feature_values("downlink_delay", downlink_delay))
    for window_steps in resolved_window_steps:
        feature_values.update(_window_feature_values(shared_bw, window_steps=int(window_steps)))

    action_tolerance = _action_tolerance(sage_action, best_action)
    feature_values.update(_action_feature_values("sage_action", sage_action, tolerance=action_tolerance))
    feature_values.update(_action_feature_values("best_baseline_action", best_action, tolerance=action_tolerance))

    action_gap = sage_action - best_action
    action_gap_positive_mask = np.asarray(action_gap > action_tolerance, dtype=bool)
    action_gap_negative_mask = np.asarray(action_gap < -action_tolerance, dtype=bool)
    action_gap_scale = max(
        float(feature_values.get("sage_action_span", 0.0)),
        float(feature_values.get("best_baseline_action_span", 0.0)),
        action_tolerance,
    )
    feature_values.update(
        {
            "action_reference_available_fraction": float(min(sage_action_availability, best_action_availability)),
            "action_gap_mean": float(np.mean(action_gap)),
            "action_gap_abs_mean": float(np.mean(np.abs(action_gap))),
            "action_gap_p90": _percentile(np.abs(action_gap), 90.0),
            "action_gap_positive_fraction": _fraction(action_gap_positive_mask),
            "action_gap_negative_fraction": _fraction(action_gap_negative_mask),
            "action_gap_positive_mean": _positive_mean(action_gap),
            "action_gap_negative_mean_abs": _negative_mean_abs(action_gap),
            "action_gap_longest_positive_run": float(_longest_run(action_gap_positive_mask)),
            "action_gap_longest_negative_run": float(_longest_run(action_gap_negative_mask)),
            "action_gap_positive_mean_scaled": float(_positive_mean(action_gap) / max(action_gap_scale, 1e-6)),
            "action_gap_negative_mean_scaled": float(_negative_mean_abs(action_gap) / max(action_gap_scale, 1e-6)),
        }
    )

    sage_action_diffs = np.diff(sage_action)
    sage_increase_mask = np.asarray(sage_action_diffs > action_tolerance, dtype=bool)
    sage_decrease_mask = np.asarray(sage_action_diffs < -action_tolerance, dtype=bool)
    sage_hold_mask = np.asarray(np.abs(sage_action_diffs) <= action_tolerance, dtype=bool)
    if num_steps > 1:
        sage_increase_step_mask = np.concatenate([np.asarray([False], dtype=bool), sage_increase_mask], axis=0)
        sage_decrease_step_mask = np.concatenate([np.asarray([False], dtype=bool), sage_decrease_mask], axis=0)
        sage_hold_step_mask = np.concatenate([np.asarray([True], dtype=bool), sage_hold_mask], axis=0)
    else:
        sage_increase_step_mask = np.zeros(num_steps, dtype=bool)
        sage_decrease_step_mask = np.zeros(num_steps, dtype=bool)
        sage_hold_step_mask = np.ones(num_steps, dtype=bool)

    low_bw_mask = _stress_mask_low(shared_bw)
    high_bw_mask = _stress_mask_high(shared_bw)
    bw_drop_mask = _drop_mask(shared_bw)
    bw_rebound_mask = _rebound_mask(shared_bw)
    delay_stress_mask = _stress_mask_high(sage_rtt_ms)
    high_gap_mask = _stress_mask_high(hard_gap_percent)
    no_backoff_mask = np.asarray(~sage_decrease_step_mask, dtype=bool)
    no_increase_mask = np.asarray(~sage_increase_step_mask, dtype=bool)
    stress_mask = np.asarray(low_bw_mask | bw_drop_mask | delay_stress_mask, dtype=bool)
    recovery_mask = np.asarray(high_bw_mask | bw_rebound_mask, dtype=bool)
    aggressive_stress_mask = np.asarray((sage_increase_step_mask | action_gap_positive_mask) & stress_mask, dtype=bool)
    conservative_recovery_mask = np.asarray((no_increase_mask | action_gap_negative_mask) & recovery_mask, dtype=bool)

    feature_values.update(
        {
            "interaction_low_bw_fraction": _fraction(low_bw_mask),
            "interaction_high_bw_fraction": _fraction(high_bw_mask),
            "interaction_bw_drop_fraction": _fraction(bw_drop_mask),
            "interaction_bw_rebound_fraction": _fraction(bw_rebound_mask),
            "interaction_delay_stress_fraction": _fraction(delay_stress_mask),
            "interaction_high_gap_fraction": _fraction(high_gap_mask),
            "interaction_sage_increase_during_low_bw_fraction": _conditional_fraction(low_bw_mask, sage_increase_step_mask),
            "interaction_sage_increase_during_bw_drop_fraction": _conditional_fraction(bw_drop_mask, sage_increase_step_mask),
            "interaction_sage_no_backoff_during_delay_stress_fraction": _conditional_fraction(delay_stress_mask, no_backoff_mask),
            "interaction_sage_no_backoff_during_high_gap_fraction": _conditional_fraction(high_gap_mask, no_backoff_mask),
            "interaction_sage_no_increase_during_high_bw_fraction": _conditional_fraction(high_bw_mask, no_increase_mask),
            "interaction_sage_no_increase_during_bw_rebound_fraction": _conditional_fraction(bw_rebound_mask, no_increase_mask),
            "interaction_sage_no_increase_during_high_gap_fraction": _conditional_fraction(high_gap_mask, no_increase_mask),
            "interaction_positive_action_gap_during_low_bw_fraction": _conditional_fraction(low_bw_mask, action_gap_positive_mask),
            "interaction_positive_action_gap_during_bw_drop_fraction": _conditional_fraction(bw_drop_mask, action_gap_positive_mask),
            "interaction_positive_action_gap_during_high_gap_fraction": _conditional_fraction(high_gap_mask, action_gap_positive_mask),
            "interaction_negative_action_gap_during_high_bw_fraction": _conditional_fraction(high_bw_mask, action_gap_negative_mask),
            "interaction_negative_action_gap_during_bw_rebound_fraction": _conditional_fraction(bw_rebound_mask, action_gap_negative_mask),
            "interaction_negative_action_gap_during_high_gap_fraction": _conditional_fraction(high_gap_mask, action_gap_negative_mask),
            "interaction_aggressive_stress_fraction": _fraction(aggressive_stress_mask),
            "interaction_aggressive_stress_longest_run": float(_longest_run(aggressive_stress_mask)),
            "interaction_conservative_recovery_fraction": _fraction(conservative_recovery_mask),
            "interaction_conservative_recovery_longest_run": float(_longest_run(conservative_recovery_mask)),
            "interaction_backoff_delay_after_low_bw_mean": _mean_delay_to_condition(
                _onset_mask(low_bw_mask),
                sage_decrease_step_mask,
                horizon=min(max(int(num_steps // 4), 1), 5),
            ),
            "interaction_backoff_delay_after_bw_drop_mean": _mean_delay_to_condition(
                _onset_mask(bw_drop_mask),
                sage_decrease_step_mask,
                horizon=min(max(int(num_steps // 4), 1), 5),
            ),
            "interaction_increase_delay_after_high_bw_mean": _mean_delay_to_condition(
                _onset_mask(high_bw_mask),
                sage_increase_step_mask,
                horizon=min(max(int(num_steps // 4), 1), 5),
            ),
            "interaction_increase_delay_after_bw_rebound_mean": _mean_delay_to_condition(
                _onset_mask(bw_rebound_mask),
                sage_increase_step_mask,
                horizon=min(max(int(num_steps // 4), 1), 5),
            ),
        }
    )

    gap_growth_lag1_mean, gap_growth_lag1_positive_fraction = _triggered_delta_stats(
        sage_increase_step_mask,
        hard_gap_percent,
        lag=1,
    )
    gap_growth_lag2_mean, _ = _triggered_delta_stats(
        sage_increase_step_mask,
        hard_gap_percent,
        lag=2,
    )
    gap_after_positive_gap_lag1_mean, gap_after_positive_gap_lag1_positive_fraction = _triggered_delta_stats(
        action_gap_positive_mask,
        hard_gap_percent,
        lag=1,
    )
    gap_after_negative_gap_lag1_mean, gap_after_negative_gap_lag1_positive_fraction = _triggered_delta_stats(
        action_gap_negative_mask,
        hard_gap_percent,
        lag=1,
    )
    feature_values.update(
        {
            "interaction_hard_gap_growth_after_increase_lag1_mean": gap_growth_lag1_mean,
            "interaction_hard_gap_growth_after_increase_lag1_positive_fraction": gap_growth_lag1_positive_fraction,
            "interaction_hard_gap_growth_after_increase_lag2_mean": gap_growth_lag2_mean,
            "interaction_hard_gap_growth_after_positive_action_gap_lag1_mean": gap_after_positive_gap_lag1_mean,
            "interaction_hard_gap_growth_after_positive_action_gap_lag1_positive_fraction": gap_after_positive_gap_lag1_positive_fraction,
            "interaction_hard_gap_growth_after_negative_action_gap_lag1_mean": gap_after_negative_gap_lag1_mean,
            "interaction_hard_gap_growth_after_negative_action_gap_lag1_positive_fraction": gap_after_negative_gap_lag1_positive_fraction,
            "interaction_rate_deficit_after_negative_action_gap_lag1_mean": _triggered_future_mean(
                action_gap_negative_mask,
                rate_deficit,
                lag=1,
            ),
            "interaction_rtt_deficit_after_increase_lag1_mean": _triggered_future_mean(
                sage_increase_step_mask,
                rtt_deficit,
                lag=1,
            ),
            "interaction_loss_excess_after_increase_lag1_mean": _triggered_future_mean(
                sage_increase_step_mask,
                loss_excess,
                lag=1,
            ),
        }
    )

    for window_steps in resolved_window_steps:
        feature_values.update(
            _action_window_feature_values(
                window_steps=int(window_steps),
                increase_mask=sage_increase_step_mask,
                decrease_mask=sage_decrease_step_mask,
                positive_action_gap_mask=action_gap_positive_mask,
                negative_action_gap_mask=action_gap_negative_mask,
                aggressive_stress_mask=aggressive_stress_mask,
                conservative_recovery_mask=conservative_recovery_mask,
                high_gap_mask=high_gap_mask,
                no_backoff_mask=no_backoff_mask,
                no_increase_mask=no_increase_mask,
            )
        )

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

    max_negative_slope_fraction = max(
        [
            float(feature_values.get(f"shared_bw_window{int(window_steps)}_negative_slope_fraction", 0.0))
            for window_steps in resolved_window_steps
        ]
        or [0.0]
    )
    min_window_mean = min(
        [
            float(feature_values.get(f"shared_bw_window{int(window_steps)}_min_mean", float(np.mean(shared_bw))))
            for window_steps in resolved_window_steps
        ]
        or [float(np.mean(shared_bw))]
    )
    scarcity = _clip01(1.0 - float(feature_values.get("shared_bw_p10", 0.0)) / max(float(feature_values.get("shared_bw_mean", 0.0)), 1e-6))
    trough_persistence = _clip01(float(feature_values.get("shared_bw_longest_low_run", 0.0)) / max(float(actions.shape[0]), 1.0))
    local_window_scarcity = _clip01(1.0 - min_window_mean / max(float(feature_values.get("shared_bw_mean", 0.0)), 1e-6))
    env_stress_score = float(
        np.mean(
            [
                scarcity,
                trough_persistence,
                _clip01(max_negative_slope_fraction),
                local_window_scarcity,
                _clip01(float(feature_values.get("shared_bw_cv", 0.0))),
            ]
        )
    )
    action_aggressive_score = float(
        np.mean(
            [
                _clip01(float(feature_values.get("action_gap_positive_fraction", 0.0))),
                _clip01(float(feature_values.get("action_gap_positive_mean_scaled", 0.0))),
                _clip01(float(feature_values.get("action_gap_longest_positive_run", 0.0)) / max(float(actions.shape[0]), 1.0)),
                _clip01(float(feature_values.get("sage_action_increase_fraction", 0.0))),
            ]
        )
    )
    action_conservative_score = float(
        np.mean(
            [
                _clip01(float(feature_values.get("action_gap_negative_fraction", 0.0))),
                _clip01(float(feature_values.get("action_gap_negative_mean_scaled", 0.0))),
                _clip01(float(feature_values.get("action_gap_longest_negative_run", 0.0)) / max(float(actions.shape[0]), 1.0)),
                _clip01(
                    max(
                        float(feature_values.get("best_baseline_action_increase_fraction", 0.0))
                        - float(feature_values.get("sage_action_increase_fraction", 0.0)),
                        0.0,
                    )
                ),
            ]
        )
    )
    action_mismatch_score = max(action_aggressive_score, action_conservative_score)
    interaction_aggressive_score = float(
        np.mean(
            [
                _clip01(float(feature_values.get("interaction_sage_increase_during_low_bw_fraction", 0.0))),
                _clip01(float(feature_values.get("interaction_positive_action_gap_during_low_bw_fraction", 0.0))),
                _clip01(float(feature_values.get("interaction_sage_no_backoff_during_high_gap_fraction", 0.0))),
                _clip01(float(feature_values.get("interaction_aggressive_stress_fraction", 0.0))),
                _clip01(float(feature_values.get("interaction_hard_gap_growth_after_increase_lag1_positive_fraction", 0.0))),
            ]
        )
    )
    interaction_conservative_score = float(
        np.mean(
            [
                _clip01(float(feature_values.get("interaction_negative_action_gap_during_high_bw_fraction", 0.0))),
                _clip01(float(feature_values.get("interaction_negative_action_gap_during_bw_rebound_fraction", 0.0))),
                _clip01(float(feature_values.get("interaction_sage_no_increase_during_high_gap_fraction", 0.0))),
                _clip01(float(feature_values.get("interaction_conservative_recovery_fraction", 0.0))),
                _clip01(float(feature_values.get("interaction_hard_gap_growth_after_negative_action_gap_lag1_positive_fraction", 0.0))),
            ]
        )
    )
    interaction_amplification_score = max(interaction_aggressive_score, interaction_conservative_score)
    feature_values["env_stress_score"] = env_stress_score
    feature_values["action_aggressive_score"] = action_aggressive_score
    feature_values["action_conservative_score"] = action_conservative_score
    feature_values["action_mismatch_score"] = action_mismatch_score
    feature_values["interaction_aggressive_score"] = interaction_aggressive_score
    feature_values["interaction_conservative_score"] = interaction_conservative_score
    feature_values["interaction_amplification_score"] = interaction_amplification_score

    expected_feature_columns = trace_explanation_feature_columns(resolved_window_steps)
    missing = [feature for feature in expected_feature_columns if feature not in feature_values]
    if missing:
        raise RuntimeError(f"missing trace explanation features: {missing}")
    return feature_values
