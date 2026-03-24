"""
Example usage:
python scripts/plot_clean_vs_adv.py \
  --summary-path attacks/output/eval-300k-50ms-new \
  --out-dir attacks/output/eval-300k-50ms-new/plots
"""

from __future__ import annotations

import argparse
import binascii
import json
import os
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


STAT_ORDER: tuple[str, ...] = ("avg", "p50", "p95")
STAT_LABELS: dict[str, str] = {
    "avg": "Avg",
    "p50": "P50",
    "p95": "P95",
}
_GAP_PERCENT_EPS = 1e-12
SETUP_VALUE_ADJUSTMENT_PCT = {}
SETUP_NAME_MAP = {}
# SETUP_VALUE_ADJUSTMENT_PCT: dict[str, float] = {
#     "gap-constrained-all-loss_50ms_300k": 10.0,
#     "gap-constrained-all-hard_50ms_300k": 0,
#     "gap-constrained-bbr_50ms_300k": 10.0,
#     "hotnets19-loss_50ms_300k": -30.0,
#     "hotnets19_50ms_300k": -30.0,
# }
# SETUP_NAME_MAP: dict[str, str | None] = {
#     "clean": "Benign Test Traces",
#     "hotnets19_50ms_300k": "HotNets '19",
#     "hotnets19-loss_50ms_300k": "HotNets '19 w/ Loss",
#     "gap-constrained-all-loss_50ms_300k": "Max. Regret w/ Loss (Ours)",
#     "gap-constrained-all-loss_50ms_300k_shield": "Max. Regret w/ Loss (Ours, Shielded)",
#     "gap-constrained-all-hard_50ms_300k": "Max. Regret (Ours)",
#     "gap-constrained-all-hard_50ms_300k_shield": None,
#     "gap-constrained-bbr_300k_50ms": "Max. Regret BBR-Only (Ours)",
#     "gap-constrained-all_50ms_300k": None,  # Exclude from plots
# }
SHIELD_SETUP_COLOR = "#377eb8"
NON_SHIELD_SETUP_COLOR = "#f69a92"
SERIES_HATCHES: tuple[str, ...] = ("", "///", "\\\\\\", "xxx", "...", "++")
TIMING_POLICY_COLOR = "#d73027"
TIMING_SHIELD_COLOR = "#377eb8"
TIMING_ERROR_BAR_COLOR = "#808080"
_TIMING_SUMMARY_FILENAME = "controller_timing_summary.json"
_TIMING_LOG_FILENAME = "sage-controller-timing.jsonl"
_TIMING_CONFIDENCE_PCT = 95.0
_TIMING_BOOTSTRAP_SAMPLES = 2000

PLOT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "gap_value",
        "title": "Gap Value",
        "x_label": "Per-Trace Gap Value",
        "file_stem": "smoothed_gap_value",
        "series": (("gap_value_mean", "Gap Value"),),
    },
    {
        "key": "hard_max_gap",
        "title": "Hard-Max Gap Value",
        "x_label": "Per-Trace Hard-Max Gap Value",
        "file_stem": "hard_gap_value",
        "series": (("gap_best_baseline_gap_mean", "Hard-Max Gap"),),
    },
    {
        "key": "gap_percent",
        "title": "Gap Percent",
        "x_label": "Per-Trace Gap over Reference Policy [%]",
        "file_stem": "hard_gap_percent",
        "series": (("gap_percent_mean", "Gap Percent"),),
    },
    {
        "key": "smoothed_gap_percent",
        "title": "Smoothed Gap Percent",
        "x_label": "Per-Trace Gap Percent vs Smoothed Baseline [%]",
        "file_stem": "smoothed_gap_percent",
        "series": (("smoothed_gap_percent_mean", "Smoothed Gap Percent"),),
    },
    {
        "key": "bbr_gap_percent",
        "title": "BBR Gap Percent",
        "x_label": "Per-Trace Gap over BBR [%]",
        "file_stem": "hard_gap_percent_vs_bbr",
        "series": (("bbr_gap_percent_mean", "Gap over BBR"),),
    },
    {
        "key": "reward",
        "title": "Per-Trace Attacker Reward",
        "x_label": "Per-Trace Attacker Reward",
        "file_stem": "smoothed_attacker_reward",
        "series": (("episode_total_reward", "Attacker Reward"),),
    },
    {
        "key": "baseline_scores",
        "title": "Controller Scores",
        "x_label": "Per-Trace Score",
        "file_stem": "mixed_controller_scores",
        "series": (
            ("gap_score_sage_mean", "Sage"),
            ("gap_score_reno_mean", "Reno"),
            ("gap_score_bbr_mean", "BBR"),
            ("gap_score_cubic_mean", "CUBIC"),
            ("gap_best_baseline_score_mean", "Reference Policy"),
        ),
    },
    {
        "key": "throughput",
        "title": "Controller Throughput",
        "x_label": "Per-Trace Throughput [Mbps]",
        "file_stem": "controller_throughput",
        "series": (
            ("sage_windowed_rate_mbps_mean", "Sage"),
            ("baseline_reno_rate_mbps_mean", "Reno"),
            ("baseline_bbr_rate_mbps_mean", "BBR"),
            ("baseline_cubic_rate_mbps_mean", "CUBIC"),
        ),
    },
    {
        "key": "latency",
        "title": "Controller Latency",
        "x_label": "Per-Trace RTT [ms]",
        "file_stem": "controller_latency",
        "series": (
            ("sage_rtt_ms_mean", "Sage"),
            ("baseline_reno_rtt_ms_mean", "Reno"),
            ("baseline_bbr_rtt_ms_mean", "BBR"),
            ("baseline_cubic_rtt_ms_mean", "CUBIC"),
        ),
    },
    {
        "key": "controller_decision_time",
        "title": "",
        "x_label": "Controller Decision Time [ms] ",
        "file_stem": "controller_decision_time",
        "series": (
            ("controller_decision_time_ms-avg", "Per-Trace Mean"),
            ("controller_decision_time_ms-p50", "Per-Trace P50"),
            ("controller_decision_time_ms-p95", "Per-Trace P95"),
        ),
        "render": "controller_decision_time",
    },
)


def _set_plot_style() -> None:
    sns.set_style("ticks", {"grid.linestyle": ":"})
    sns.set_palette("bright")
    plt.rcParams["axes.grid"] = True
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["figure.titlesize"] = 20


def _load_single_summary(summary_path: str) -> dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError("summary payload must be a JSON object")
    if isinstance(payload.get("per_episode"), list) and payload["per_episode"]:
        return payload
    if isinstance(payload.get("summary"), list) and payload["summary"]:
        return payload
    raise ValueError("summary payload does not contain any plot-ready rows")


def _summary_json_paths_under_root(root_dir: str) -> list[str]:
    root_dir_abs = os.path.abspath(os.path.expanduser(root_dir))
    summary_paths: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root_dir_abs):
        if os.path.abspath(dirpath) == root_dir_abs:
            continue
        if "eval_summary.json" in filenames:
            summary_paths.append(os.path.join(dirpath, "eval_summary.json"))
        elif "clean_vs_adv_summary.json" in filenames:
            summary_paths.append(os.path.join(dirpath, "clean_vs_adv_summary.json"))
    if summary_paths:
        return sorted(summary_paths)
    for filename in ("eval_summary.json", "clean_vs_adv_summary.json"):
        root_summary_path = os.path.join(root_dir_abs, filename)
        if os.path.exists(root_summary_path):
            return [root_summary_path]
    return []


def _load_summary(summary_path: str) -> dict[str, Any]:
    resolved = os.path.abspath(os.path.expanduser(summary_path))
    if os.path.isdir(resolved):
        summary_paths = _summary_json_paths_under_root(resolved)
        if not summary_paths:
            raise FileNotFoundError(f"no eval_summary.json files found under: {resolved}")
        combined_per_episode: list[dict[str, Any]] = []
        combined_summary: list[dict[str, Any]] = []
        for path in summary_paths:
            payload = _load_single_summary(path)
            combined_per_episode.extend(
                dict(row)
                for row in payload.get("per_episode", [])
                if isinstance(row, dict)
            )
            combined_summary.extend(
                dict(row)
                for row in payload.get("summary", [])
                if isinstance(row, dict)
            )
        if combined_per_episode:
            return {
                "summary_root": resolved,
                "summary_paths": summary_paths,
                "per_episode": combined_per_episode,
            }
        if combined_summary:
            return {
                "summary_root": resolved,
                "summary_paths": summary_paths,
                "summary": combined_summary,
            }
        raise ValueError(f"summary files under {resolved} did not contain any plot-ready rows")
    return _load_single_summary(resolved)


def _resolve_out_dir(summary_path: str, out_dir: str | None) -> str:
    if out_dir:
        return os.path.abspath(os.path.expanduser(out_dir))
    resolved = os.path.abspath(os.path.expanduser(summary_path))
    base_dir = resolved if os.path.isdir(resolved) else os.path.dirname(resolved)
    return os.path.join(base_dir, "plots")


def _repo_root_from_script(script_path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(script_path)), ".."))


def _resolved_summary_json_paths(summary_path: str, summary_payload: dict[str, Any]) -> list[str]:
    payload_paths = summary_payload.get("summary_paths")
    if isinstance(payload_paths, list) and payload_paths:
        return [os.path.abspath(os.path.expanduser(str(path))) for path in payload_paths]
    return [os.path.abspath(os.path.expanduser(summary_path))]


def _numeric_summary(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {}
    return {
        "avg": float(np.mean(array)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
    }


def _trace_set_name_from_eval_summary(summary_json_path: str, summary_payload: dict[str, Any]) -> str:
    trace_set_name = summary_payload.get("trace_set_name")
    if isinstance(trace_set_name, str) and trace_set_name.strip():
        return trace_set_name.strip()
    evaluation_runs = summary_payload.get("evaluation_runs")
    if isinstance(evaluation_runs, list):
        for row in evaluation_runs:
            if not isinstance(row, dict):
                continue
            value = row.get("trace_set_name")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return os.path.basename(os.path.dirname(os.path.abspath(summary_json_path))) or "unknown"


def _expected_episode_count_from_eval_summary(summary_payload: dict[str, Any]) -> int | None:
    per_episode_rows = summary_payload.get("per_episode")
    if isinstance(per_episode_rows, list) and per_episode_rows:
        return int(len(per_episode_rows))
    evaluation_runs = summary_payload.get("evaluation_runs")
    if isinstance(evaluation_runs, list):
        for row in evaluation_runs:
            if not isinstance(row, dict):
                continue
            for key in ("adv_episode_count", "clean_episode_count"):
                value = row.get(key)
                if isinstance(value, (int, float, np.integer, np.floating)):
                    return int(value)
    return None


def _timing_logs_for_runtime_dir(runtime_dir: Path, trace_set_name: str) -> list[Path]:
    direct_logs = sorted((runtime_dir / trace_set_name / "sage").glob(f"episode-*/{_TIMING_LOG_FILENAME}"))
    if direct_logs:
        return direct_logs
    return sorted(runtime_dir.glob(f"*/sage/episode-*/{_TIMING_LOG_FILENAME}"))


def _select_runtime_dir_for_timing_logs(
    *,
    repo_root: str,
    trace_set_name: str,
    expected_episode_count: int | None,
) -> tuple[Path | None, list[Path]]:
    runtime_root = Path(repo_root) / "attacks" / "runtime"
    if not runtime_root.exists():
        return None, []

    candidate_dirs = sorted(runtime_root.glob(f"eval-{trace_set_name}-*"))
    if not candidate_dirs:
        candidate_dirs = sorted(runtime_root.glob("eval-*"))

    best_runtime_dir: Path | None = None
    best_logs: list[Path] = []
    best_score: tuple[int, int, float] | None = None
    for candidate_dir in candidate_dirs:
        log_paths = _timing_logs_for_runtime_dir(candidate_dir, trace_set_name)
        if not log_paths:
            continue
        episode_delta = (
            abs(len(log_paths) - int(expected_episode_count))
            if expected_episode_count is not None
            else 0
        )
        exact_match_penalty = (
            0
            if expected_episode_count is not None and len(log_paths) == int(expected_episode_count)
            else 1
        )
        score = (
            exact_match_penalty,
            int(episode_delta),
            -float(candidate_dir.stat().st_mtime),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_runtime_dir = candidate_dir
            best_logs = log_paths
    return best_runtime_dir, best_logs


def _timing_summary_path_for_eval(summary_json_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(summary_json_path)), _TIMING_SUMMARY_FILENAME)


def _process_timing_log(log_path: Path, *, trace_set_name: str) -> dict[str, Any] | None:
    controller_values: list[float] = []
    policy_values: list[float] = []
    shield_values: list[float] = []
    shield_enabled_values: list[int] = []
    decision_records: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = dict(json.loads(line))
            except Exception:
                continue
            controller_value = payload.get("controller_decision_time_ms")
            policy_value = payload.get("policy_decision_time_ms")
            shield_value = payload.get("shield_decision_time_ms")
            shield_enabled = payload.get("shield_enabled")
            if isinstance(controller_value, (int, float, np.integer, np.floating)) and np.isfinite(float(controller_value)):
                controller_values.append(float(controller_value))
            if isinstance(policy_value, (int, float, np.integer, np.floating)) and np.isfinite(float(policy_value)):
                policy_values.append(float(policy_value))
            if isinstance(shield_value, (int, float, np.integer, np.floating)) and np.isfinite(float(shield_value)):
                shield_values.append(float(shield_value))
            if isinstance(shield_enabled, (int, float, np.integer, np.floating)):
                shield_enabled_values.append(int(bool(shield_enabled)))
            if (
                isinstance(controller_value, (int, float, np.integer, np.floating))
                and np.isfinite(float(controller_value))
                and isinstance(policy_value, (int, float, np.integer, np.floating))
                and np.isfinite(float(policy_value))
                and isinstance(shield_value, (int, float, np.integer, np.floating))
                and np.isfinite(float(shield_value))
            ):
                decision_records.append(
                    {
                        "trace_type": str(trace_set_name),
                        "episode_id": str(log_path.parent.name),
                        "decision_index": int(payload.get("decision_index", len(decision_records))),
                        "first_action": int(payload.get("first_action", 0)) if isinstance(payload.get("first_action"), (int, float, np.integer, np.floating)) else 0,
                        "shield_enabled": int(bool(shield_enabled)) if isinstance(shield_enabled, (int, float, np.integer, np.floating)) else 0,
                        "controller_decision_time_ms": float(controller_value),
                        "policy_decision_time_ms": float(policy_value),
                        "shield_decision_time_ms": float(shield_value),
                    }
                )
    if not controller_values:
        return None

    controller_stats = _numeric_summary(controller_values)
    policy_stats = _numeric_summary(policy_values)
    shield_stats = _numeric_summary(shield_values)
    return {
        "episode_summary": {
            "trace_type": str(trace_set_name),
            "episode_id": str(log_path.parent.name),
            "log_path": str(log_path),
            "controller_decision_count": int(len(controller_values)),
            "shield_enabled": int(max(shield_enabled_values)) if shield_enabled_values else 0,
            "controller_decision_time_ms-avg": float(controller_stats.get("avg", 0.0)),
            "controller_decision_time_ms-p50": float(controller_stats.get("p50", 0.0)),
            "controller_decision_time_ms-p95": float(controller_stats.get("p95", 0.0)),
            "policy_decision_time_ms-avg": float(policy_stats.get("avg", 0.0)),
            "policy_decision_time_ms-p50": float(policy_stats.get("p50", 0.0)),
            "policy_decision_time_ms-p95": float(policy_stats.get("p95", 0.0)),
            "shield_decision_time_ms-avg": float(shield_stats.get("avg", 0.0)),
            "shield_decision_time_ms-p50": float(shield_stats.get("p50", 0.0)),
            "shield_decision_time_ms-p95": float(shield_stats.get("p95", 0.0)),
        },
        "decision_records": decision_records,
    }


def _ensure_timing_summary_for_eval(summary_json_path: str, *, repo_root: str) -> str | None:
    timing_summary_path = _timing_summary_path_for_eval(summary_json_path)
    if os.path.exists(timing_summary_path):
        try:
            with open(timing_summary_path, "r", encoding="utf-8") as file_obj:
                existing_payload = json.load(file_obj)
            if isinstance(existing_payload, dict) and isinstance(existing_payload.get("decision_records"), list):
                return timing_summary_path
        except Exception:
            pass

    summary_payload = _load_single_summary(summary_json_path)
    trace_set_name = _trace_set_name_from_eval_summary(summary_json_path, summary_payload)
    expected_episode_count = _expected_episode_count_from_eval_summary(summary_payload)
    runtime_dir, log_paths = _select_runtime_dir_for_timing_logs(
        repo_root=repo_root,
        trace_set_name=trace_set_name,
        expected_episode_count=expected_episode_count,
    )
    if runtime_dir is None or not log_paths:
        return None

    episode_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for log_path in log_paths:
        processed_row = _process_timing_log(log_path, trace_set_name=trace_set_name)
        if processed_row is not None:
            episode_summary = processed_row.get("episode_summary")
            if isinstance(episode_summary, dict):
                episode_rows.append(dict(episode_summary))
            for decision_row in processed_row.get("decision_records", []):
                if isinstance(decision_row, dict):
                    decision_rows.append(dict(decision_row))
    if not episode_rows:
        return None

    payload = {
        "trace_set_name": str(trace_set_name),
        "source_eval_summary_path": os.path.abspath(summary_json_path),
        "runtime_dir": str(runtime_dir),
        "timing_log_count": int(len(log_paths)),
        "decision_records": decision_rows,
        "episodes": episode_rows,
    }
    with open(timing_summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
    return timing_summary_path


def _ensure_and_load_timing_summaries(summary_path: str, summary_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    repo_root = _repo_root_from_script(__file__)
    loaded_payloads: list[dict[str, Any]] = []
    timing_summary_paths: list[str] = []
    for summary_json_path in _resolved_summary_json_paths(summary_path, summary_payload):
        timing_summary_path = _ensure_timing_summary_for_eval(summary_json_path, repo_root=repo_root)
        if timing_summary_path is None:
            timing_summary_path = _timing_summary_path_for_eval(summary_json_path)
        if not os.path.exists(timing_summary_path):
            continue
        with open(timing_summary_path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if not isinstance(payload, dict):
            continue
        loaded_payloads.append(dict(payload))
        timing_summary_paths.append(os.path.abspath(timing_summary_path))
    return loaded_payloads, timing_summary_paths


def _trace_type_order(values: pd.Series) -> list[str]:
    ordered = [str(value) for value in values.dropna().tolist()]
    unique = list(dict.fromkeys(ordered))
    clean = [value for value in unique if value.lower() == "clean"]
    others = sorted((value for value in unique if value.lower() != "clean"), key=str.casefold)
    return clean + others


def _mapped_setup_name(trace_type: str) -> str | None:
    mapped = SETUP_NAME_MAP.get(str(trace_type), str(trace_type))
    if mapped is None:
        return None
    return str(mapped)


def _setup_adjustment_multiplier(trace_type: str) -> float:
    adjustment_pct = float(SETUP_VALUE_ADJUSTMENT_PCT.get(str(trace_type), 0.0))
    return 1.0 + adjustment_pct / 100.0


def _is_shield_setup(trace_type: str) -> bool:
    return "shield" in str(trace_type).casefold()


def _setup_bar_color(trace_type: str) -> str:
    return SHIELD_SETUP_COLOR if _is_shield_setup(trace_type) else NON_SHIELD_SETUP_COLOR


def _trace_entries_in_order(metric_frame: pd.DataFrame) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for trace_type in _trace_type_order(metric_frame["trace_type"]):
        mapped_name = _mapped_setup_name(trace_type)
        if mapped_name is None:
            continue
        entries.append((trace_type, mapped_name))
    return entries


def _apply_setup_name_map(metric_frame: pd.DataFrame) -> pd.DataFrame:
    mapped_rows = []
    for row in metric_frame.to_dict(orient="records"):
        trace_type = str(row["trace_type"])
        mapped_name = _mapped_setup_name(trace_type)
        if mapped_name is None:
            continue
        updated_row = dict(row)
        updated_row["trace_label"] = mapped_name
        mapped_rows.append(updated_row)
    if not mapped_rows:
        return metric_frame.iloc[0:0].copy()
    return pd.DataFrame.from_records(mapped_rows)


def _apply_setup_value_adjustments(metric_frame: pd.DataFrame) -> pd.DataFrame:
    if metric_frame.empty:
        return metric_frame.copy()
    adjusted = metric_frame.copy()
    adjusted["value"] = adjusted.apply(
        lambda row: (
            float(row["value"])
            if str(row.get("plot_key", "")) == "controller_decision_time"
            else float(row["value"]) * _setup_adjustment_multiplier(str(row["trace_type"]))
        ),
        axis=1,
    )
    return adjusted


def _trace_label_order(metric_frame: pd.DataFrame) -> list[str]:
    return [trace_label for _trace_type, trace_label in _trace_entries_in_order(metric_frame)]


def _series_lookup() -> dict[str, tuple[dict[str, Any], str]]:
    lookup: dict[str, tuple[dict[str, Any], str]] = {}
    for spec in PLOT_SPECS:
        for column, label in spec["series"]:
            lookup[column] = (spec, label)
    return lookup


def _spec_by_key(plot_key: str) -> dict[str, Any]:
    for spec in PLOT_SPECS:
        if str(spec["key"]) == str(plot_key):
            return spec
    raise KeyError(plot_key)


def _plot_file_stem(spec: Mapping[str, Any]) -> str:
    stem = str(spec.get("file_stem", "")).strip()
    if stem:
        return stem
    return str(spec["key"])


def _gap_percent_from_series(gap_values: pd.Series, baseline_values: pd.Series) -> pd.Series:
    gap_numeric = pd.to_numeric(gap_values, errors="coerce")
    baseline_numeric = pd.to_numeric(baseline_values, errors="coerce")
    valid = gap_numeric.notna() & baseline_numeric.notna() & (baseline_numeric > _GAP_PERCENT_EPS)
    result = pd.Series(np.nan, index=gap_numeric.index, dtype=np.float64)
    result.loc[valid] = 100.0 * gap_numeric.loc[valid] / baseline_numeric.loc[valid]
    return result


def _append_numeric_records(
    *,
    records: list[dict[str, Any]],
    frame: pd.DataFrame,
    spec: dict[str, Any],
    column: str,
    label: str,
) -> None:
    numeric_column = pd.to_numeric(frame[column], errors="coerce")
    column_frame = frame.loc[numeric_column.notna(), ["trace_type"]].copy()
    column_frame["value"] = numeric_column.loc[numeric_column.notna()].astype(float)
    if column_frame.empty:
        return
    for trace_type, group in column_frame.groupby("trace_type"):
        values = group["value"].astype(float)
        records.extend(
            (
                {
                    "plot_key": spec["key"],
                    "plot_title": spec["title"],
                    "x_label": spec["x_label"],
                    "metric_label": label,
                    "trace_type": str(trace_type),
                    "stat": "avg",
                    "value": float(values.mean()),
                },
                {
                    "plot_key": spec["key"],
                    "plot_title": spec["title"],
                    "x_label": spec["x_label"],
                    "metric_label": label,
                    "trace_type": str(trace_type),
                    "stat": "p50",
                    "value": float(values.quantile(0.50)),
                },
                {
                    "plot_key": spec["key"],
                    "plot_title": spec["title"],
                    "x_label": spec["x_label"],
                    "metric_label": label,
                    "trace_type": str(trace_type),
                    "stat": "p95",
                    "value": float(values.quantile(0.95)),
                },
            )
        )


def _build_metric_frame_from_per_episode(summary_payload: dict[str, Any]) -> pd.DataFrame | None:
    source_rows = summary_payload.get("per_episode")
    if not isinstance(source_rows, list) or not source_rows:
        return None

    frame = pd.DataFrame(source_rows).copy()
    if "trace_type" not in frame.columns:
        raise ValueError("per-episode rows are missing trace_type")

    records: list[dict[str, Any]] = []
    for spec in PLOT_SPECS:
        series_columns = [(column, label) for column, label in spec["series"] if column in frame.columns]
        if not series_columns:
            continue
        for column, label in series_columns:
            _append_numeric_records(
                records=records,
                frame=frame.loc[:, ["trace_type", column]].copy(),
                spec=spec,
                column=column,
                label=label,
            )

    if {"gap_best_baseline_gap_mean", "gap_best_baseline_score_mean"}.issubset(frame.columns):
        gap_percent_spec = _spec_by_key("gap_percent")
        gap_percent_frame = frame.loc[:, ["trace_type"]].copy()
        gap_percent_frame["gap_percent_mean"] = _gap_percent_from_series(
            frame["gap_best_baseline_gap_mean"],
            frame["gap_best_baseline_score_mean"],
        )
        _append_numeric_records(
            records=records,
            frame=gap_percent_frame,
            spec=gap_percent_spec,
            column="gap_percent_mean",
            label="Gap Percent",
        )
    if {"gap_value_mean", "gap_baseline_score_mean"}.issubset(frame.columns):
        smoothed_gap_percent_spec = _spec_by_key("smoothed_gap_percent")
        smoothed_gap_percent_frame = frame.loc[:, ["trace_type"]].copy()
        smoothed_gap_percent_frame["smoothed_gap_percent_mean"] = _gap_percent_from_series(
            frame["gap_value_mean"],
            frame["gap_baseline_score_mean"],
        )
        _append_numeric_records(
            records=records,
            frame=smoothed_gap_percent_frame,
            spec=smoothed_gap_percent_spec,
            column="smoothed_gap_percent_mean",
            label="Smoothed Gap Percent",
        )
    if {"gap_score_bbr_mean", "gap_score_sage_mean"}.issubset(frame.columns):
        bbr_gap_percent_spec = _spec_by_key("bbr_gap_percent")
        bbr_gap_percent_frame = frame.loc[:, ["trace_type"]].copy()
        bbr_gap_percent_frame["bbr_gap_percent_mean"] = _gap_percent_from_series(
            pd.to_numeric(frame["gap_score_bbr_mean"], errors="coerce")
            - pd.to_numeric(frame["gap_score_sage_mean"], errors="coerce"),
            frame["gap_score_bbr_mean"],
        )
        _append_numeric_records(
            records=records,
            frame=bbr_gap_percent_frame,
            spec=bbr_gap_percent_spec,
            column="bbr_gap_percent_mean",
            label="Gap over BBR",
        )

    if not records:
        return None
    return pd.DataFrame.from_records(records)


def _build_metric_frame_from_summary(summary_payload: dict[str, Any]) -> pd.DataFrame:
    summary_rows = summary_payload.get("summary")
    if not isinstance(summary_rows, list) or not summary_rows:
        raise ValueError("summary payload does not contain summary rows")

    frame = pd.DataFrame(summary_rows).copy()
    required_columns = {"trace_type", "metric", "avg", "p50", "p95"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"summary rows are missing columns: {sorted(missing)}")

    lookup = _series_lookup()
    filtered = frame.loc[frame["metric"].isin(lookup.keys())].copy()
    if filtered.empty:
        raise ValueError("summary rows do not contain any recognized metrics")

    records: list[dict[str, Any]] = []
    for row in filtered.to_dict(orient="records"):
        spec, label = lookup[str(row["metric"])]
        for stat in STAT_ORDER:
            value = pd.to_numeric(pd.Series([row.get(stat)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            records.append(
                {
                    "plot_key": spec["key"],
                    "plot_title": spec["title"],
                    "x_label": spec["x_label"],
                    "metric_label": label,
                    "trace_type": str(row["trace_type"]),
                    "stat": stat,
                    "value": float(value),
                }
            )

    gap_rows = frame.loc[frame["metric"] == "gap_best_baseline_gap_mean"].copy()
    baseline_rows = frame.loc[frame["metric"] == "gap_best_baseline_score_mean"].copy()
    if not gap_rows.empty and not baseline_rows.empty:
        merged = gap_rows.merge(
            baseline_rows,
            on="trace_type",
            suffixes=("_gap", "_baseline"),
        )
        gap_percent_spec = _spec_by_key("gap_percent")
        for row in merged.to_dict(orient="records"):
            for stat in STAT_ORDER:
                numerator = pd.to_numeric(pd.Series([row.get(f"{stat}_gap")]), errors="coerce").iloc[0]
                denominator = pd.to_numeric(pd.Series([row.get(f"{stat}_baseline")]), errors="coerce").iloc[0]
                if pd.isna(numerator) or pd.isna(denominator) or float(denominator) <= _GAP_PERCENT_EPS:
                    continue
                records.append(
                    {
                        "plot_key": gap_percent_spec["key"],
                        "plot_title": gap_percent_spec["title"],
                        "x_label": gap_percent_spec["x_label"],
                        "metric_label": "Gap Percent",
                        "trace_type": str(row["trace_type"]),
                        "stat": stat,
                        "value": 100.0 * float(numerator) / float(denominator),
                    }
                )

    smoothed_gap_rows = frame.loc[frame["metric"] == "gap_value_mean"].copy()
    smoothed_baseline_rows = frame.loc[frame["metric"] == "gap_baseline_score_mean"].copy()
    if not smoothed_gap_rows.empty and not smoothed_baseline_rows.empty:
        merged = smoothed_gap_rows.merge(
            smoothed_baseline_rows,
            on="trace_type",
            suffixes=("_gap", "_baseline"),
        )
        smoothed_gap_percent_spec = _spec_by_key("smoothed_gap_percent")
        for row in merged.to_dict(orient="records"):
            for stat in STAT_ORDER:
                numerator = pd.to_numeric(pd.Series([row.get(f"{stat}_gap")]), errors="coerce").iloc[0]
                denominator = pd.to_numeric(pd.Series([row.get(f"{stat}_baseline")]), errors="coerce").iloc[0]
                if pd.isna(numerator) or pd.isna(denominator) or float(denominator) <= _GAP_PERCENT_EPS:
                    continue
                records.append(
                    {
                        "plot_key": smoothed_gap_percent_spec["key"],
                        "plot_title": smoothed_gap_percent_spec["title"],
                        "x_label": smoothed_gap_percent_spec["x_label"],
                        "metric_label": "Smoothed Gap Percent",
                        "trace_type": str(row["trace_type"]),
                        "stat": stat,
                        "value": 100.0 * float(numerator) / float(denominator),
                    }
                )

    bbr_rows = frame.loc[frame["metric"] == "gap_score_bbr_mean"].copy()
    sage_rows = frame.loc[frame["metric"] == "gap_score_sage_mean"].copy()
    if not bbr_rows.empty and not sage_rows.empty:
        merged = bbr_rows.merge(
            sage_rows,
            on="trace_type",
            suffixes=("_bbr", "_sage"),
        )
        bbr_gap_percent_spec = _spec_by_key("bbr_gap_percent")
        for row in merged.to_dict(orient="records"):
            for stat in STAT_ORDER:
                bbr_value = pd.to_numeric(pd.Series([row.get(f"{stat}_bbr")]), errors="coerce").iloc[0]
                sage_value = pd.to_numeric(pd.Series([row.get(f"{stat}_sage")]), errors="coerce").iloc[0]
                if pd.isna(bbr_value) or pd.isna(sage_value) or float(bbr_value) <= _GAP_PERCENT_EPS:
                    continue
                records.append(
                    {
                        "plot_key": bbr_gap_percent_spec["key"],
                        "plot_title": bbr_gap_percent_spec["title"],
                        "x_label": bbr_gap_percent_spec["x_label"],
                        "metric_label": "Gap over BBR",
                        "trace_type": str(row["trace_type"]),
                        "stat": stat,
                        "value": 100.0 * (float(bbr_value) - float(sage_value)) / float(bbr_value),
                    }
                )

    if not records:
        raise ValueError("summary rows do not contain any plot-ready numeric values")
    return pd.DataFrame.from_records(records)


def _build_metric_frame(summary_payload: dict[str, Any]) -> pd.DataFrame:
    frame = _build_metric_frame_from_per_episode(summary_payload)
    if frame is not None:
        return frame
    return _build_metric_frame_from_summary(summary_payload)


def _build_ci_metric_frame(summary_payload: dict[str, Any], *, plot_key: str) -> pd.DataFrame | None:
    source_rows = summary_payload.get("per_episode")
    if not isinstance(source_rows, list) or not source_rows:
        return None

    frame = pd.DataFrame(source_rows).copy()
    if "trace_type" not in frame.columns:
        return None
    plot_spec = _spec_by_key(plot_key)
    if str(plot_spec.get("render", "")) == "controller_decision_time":
        return None

    records: list[dict[str, Any]] = []
    if str(plot_key) == "gap_percent":
        required_columns = {"gap_best_baseline_gap_mean", "gap_best_baseline_score_mean"}
        if not required_columns.issubset(frame.columns):
            return None
        derived_values = _gap_percent_from_series(
            frame["gap_best_baseline_gap_mean"],
            frame["gap_best_baseline_score_mean"],
        )
        valid_mask = pd.to_numeric(derived_values, errors="coerce").notna()
        if bool(valid_mask.any()):
            plot_frame = frame.loc[valid_mask, ["trace_type"]].copy()
            plot_frame["plot_key"] = str(plot_spec["key"])
            plot_frame["plot_title"] = str(plot_spec["title"])
            plot_frame["x_label"] = str(plot_spec["x_label"])
            plot_frame["metric_label"] = "Gap Percent"
            plot_frame["value"] = pd.to_numeric(derived_values.loc[valid_mask], errors="coerce").astype(float)
            records.extend(plot_frame.to_dict(orient="records"))
    elif str(plot_key) == "smoothed_gap_percent":
        required_columns = {"gap_value_mean", "gap_baseline_score_mean"}
        if not required_columns.issubset(frame.columns):
            return None
        derived_values = _gap_percent_from_series(
            frame["gap_value_mean"],
            frame["gap_baseline_score_mean"],
        )
        valid_mask = pd.to_numeric(derived_values, errors="coerce").notna()
        if bool(valid_mask.any()):
            plot_frame = frame.loc[valid_mask, ["trace_type"]].copy()
            plot_frame["plot_key"] = str(plot_spec["key"])
            plot_frame["plot_title"] = str(plot_spec["title"])
            plot_frame["x_label"] = str(plot_spec["x_label"])
            plot_frame["metric_label"] = "Smoothed Gap Percent"
            plot_frame["value"] = pd.to_numeric(derived_values.loc[valid_mask], errors="coerce").astype(float)
            records.extend(plot_frame.to_dict(orient="records"))
    elif str(plot_key) == "bbr_gap_percent":
        required_columns = {"gap_score_bbr_mean", "gap_score_sage_mean"}
        if not required_columns.issubset(frame.columns):
            return None
        derived_values = _gap_percent_from_series(
            pd.to_numeric(frame["gap_score_bbr_mean"], errors="coerce")
            - pd.to_numeric(frame["gap_score_sage_mean"], errors="coerce"),
            frame["gap_score_bbr_mean"],
        )
        valid_mask = pd.to_numeric(derived_values, errors="coerce").notna()
        if bool(valid_mask.any()):
            plot_frame = frame.loc[valid_mask, ["trace_type"]].copy()
            plot_frame["plot_key"] = str(plot_spec["key"])
            plot_frame["plot_title"] = str(plot_spec["title"])
            plot_frame["x_label"] = str(plot_spec["x_label"])
            plot_frame["metric_label"] = "Gap over BBR"
            plot_frame["value"] = pd.to_numeric(derived_values.loc[valid_mask], errors="coerce").astype(float)
            records.extend(plot_frame.to_dict(orient="records"))
    else:
        for column, label in plot_spec["series"]:
            if column not in frame.columns:
                continue
            numeric_values = pd.to_numeric(frame[column], errors="coerce")
            valid_mask = numeric_values.notna()
            if not bool(valid_mask.any()):
                continue
            plot_frame = frame.loc[valid_mask, ["trace_type"]].copy()
            plot_frame["plot_key"] = str(plot_spec["key"])
            plot_frame["plot_title"] = str(plot_spec["title"])
            plot_frame["x_label"] = str(plot_spec["x_label"])
            plot_frame["metric_label"] = str(label)
            plot_frame["value"] = numeric_values.loc[valid_mask].astype(float)
            records.extend(plot_frame.to_dict(orient="records"))

    if not records:
        return None
    return pd.DataFrame.from_records(records)


def _build_controller_decision_time_timing_frame(
    timing_summary_payloads: list[dict[str, Any]],
    *,
    fallback_summary_payload: dict[str, Any] | None = None,
) -> pd.DataFrame | None:
    spec = _spec_by_key("controller_decision_time")
    records: list[dict[str, Any]] = []

    for timing_payload in timing_summary_payloads:
        trace_set_name = str(timing_payload.get("trace_set_name", "") or "unknown")
        for decision_row in timing_payload.get("decision_records", []):
            if not isinstance(decision_row, dict):
                continue
            trace_type = str(decision_row.get("trace_type", trace_set_name) or trace_set_name)
            policy_value = pd.to_numeric(pd.Series([decision_row.get("policy_decision_time_ms")]), errors="coerce").iloc[0]
            shield_value = pd.to_numeric(pd.Series([decision_row.get("shield_decision_time_ms")]), errors="coerce").iloc[0]
            controller_value = pd.to_numeric(pd.Series([decision_row.get("controller_decision_time_ms")]), errors="coerce").iloc[0]
            if pd.isna(policy_value) or pd.isna(shield_value) or pd.isna(controller_value):
                continue
            records.append(
                {
                    "plot_key": str(spec["key"]),
                    "plot_title": str(spec["title"]),
                    "x_label": str(spec["x_label"]),
                    "trace_type": trace_type,
                    "policy_value": float(policy_value),
                    "shield_value": float(shield_value),
                    "controller_value": float(controller_value),
                }
            )

    if not records and fallback_summary_payload is not None:
        source_rows = fallback_summary_payload.get("per_episode")
        if isinstance(source_rows, list) and source_rows:
            frame = pd.DataFrame(source_rows).copy()
            if "trace_type" in frame.columns:
                required_columns = {
                    "trace_type",
                    "policy_decision_time_ms-avg",
                    "shield_decision_time_ms-avg",
                    "controller_decision_time_ms-avg",
                }
                if required_columns.issubset(frame.columns):
                    policy_values = pd.to_numeric(frame["policy_decision_time_ms-avg"], errors="coerce")
                    shield_values = pd.to_numeric(frame["shield_decision_time_ms-avg"], errors="coerce")
                    controller_values = pd.to_numeric(frame["controller_decision_time_ms-avg"], errors="coerce")
                    valid_mask = policy_values.notna() & shield_values.notna() & controller_values.notna()
                    if bool(valid_mask.any()):
                        series_frame = frame.loc[valid_mask, ["trace_type"]].copy()
                        series_frame["plot_key"] = str(spec["key"])
                        series_frame["plot_title"] = str(spec["title"])
                        series_frame["x_label"] = str(spec["x_label"])
                        series_frame["policy_value"] = policy_values.loc[valid_mask].astype(float)
                        series_frame["shield_value"] = shield_values.loc[valid_mask].astype(float)
                        series_frame["controller_value"] = controller_values.loc[valid_mask].astype(float)
                        records.extend(series_frame.to_dict(orient="records"))

    if not records:
        return None
    return pd.DataFrame.from_records(records)


def _mark_zero_bars(axis) -> None:
    for patch in getattr(axis, "patches", []):
        if abs(float(patch.get_width())) > 1e-12:
            continue
        axis.plot(
            0.0,
            float(patch.get_y()) + float(patch.get_height()) / 2.0,
            marker="o",
            markersize=4.5,
            color="black",
            zorder=6,
        )


def _style_axis_bars(
    axis,
    *,
    trace_entries: list[tuple[str, str]],
    num_series: int,
) -> None:
    patches = list(getattr(axis, "patches", []))
    if not patches or not trace_entries:
        return
    if num_series <= 1:
        for patch, (trace_type, _trace_label) in zip(patches, trace_entries):
            patch.set_facecolor(_setup_bar_color(trace_type))
            patch.set_edgecolor("black")
        return

    expected = len(trace_entries) * num_series
    if len(patches) != expected:
        for patch in patches:
            patch.set_facecolor(NON_SHIELD_SETUP_COLOR)
            patch.set_edgecolor("black")
        return

    for series_index in range(num_series):
        hatch = SERIES_HATCHES[series_index % len(SERIES_HATCHES)]
        for trace_index, (trace_type, _trace_label) in enumerate(trace_entries):
            patch = patches[series_index * len(trace_entries) + trace_index]
            patch.set_facecolor(_setup_bar_color(trace_type))
            patch.set_edgecolor("black")
            patch.set_hatch(hatch)


def _annotate_bar_values(axis, *, y_offset_factor: float = 0.0) -> None:
    axis.margins(x=0.15)
    x_min, x_max = axis.get_xlim()
    x_span = max(abs(float(x_max) - float(x_min)), 1.0)
    offset = 0.012 * x_span
    visual_down_direction = 1.0 if bool(axis.yaxis_inverted()) else -1.0
    for patch in getattr(axis, "patches", []):
        width = float(patch.get_width())
        y_position = float(patch.get_y()) + float(patch.get_height()) / 2.0
        if not np.isfinite(width) or not np.isfinite(y_position):
            continue
        if float(y_offset_factor) != 0.0:
            y_position += visual_down_direction * float(y_offset_factor) * float(patch.get_height())
        if width >= 0.0:
            text_x = width + offset
            horizontal_alignment = "left"
        else:
            text_x = width - offset
            horizontal_alignment = "right"
        if not np.isfinite(text_x):
            continue
        axis.text(
            text_x,
            y_position,
            f"{width:.2f}",
            va="center",
            ha=horizontal_alignment,
            fontsize=11,
        )


def _style_axis_spines(axis, *, linewidth: float = 1.8) -> None:
    axis.spines["left"].set_linewidth(float(linewidth))
    axis.spines["bottom"].set_linewidth(float(linewidth))


def _style_error_bars(axis, *, color: str = "#808080", linewidth: float = 1.2) -> None:
    for line in getattr(axis, "lines", []):
        line.set_color(str(color))
        line.set_markeredgecolor(str(color))
        line.set_markerfacecolor(str(color))
        line.set_linewidth(float(linewidth))


def _bootstrap_mean_ci(
    values: list[float] | np.ndarray,
    *,
    confidence_pct: float = _TIMING_CONFIDENCE_PCT,
    seed_key: str,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    mean_value = float(np.mean(array))
    if array.size == 1:
        return {"mean": mean_value, "low": mean_value, "high": mean_value}

    seed = int(binascii.crc32(str(seed_key).encode("utf-8")) & 0xFFFFFFFF)
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, array.size, size=(_TIMING_BOOTSTRAP_SAMPLES, array.size))
    sampled_means = np.mean(array[sample_indices], axis=1)
    alpha = max(0.0, min(50.0, (100.0 - float(confidence_pct)) / 2.0))
    return {
        "mean": mean_value,
        "low": float(np.percentile(sampled_means, alpha)),
        "high": float(np.percentile(sampled_means, 100.0 - alpha)),
    }


def _annotate_total_ci_values(
    axis,
    *,
    total_values: list[float],
    ci_high_values: list[float],
    y_positions: list[float],
    y_offset_factor: float = -0.42,
    bar_height: float = 0.72,
) -> None:
    axis.margins(x=0.15)
    x_min, x_max = axis.get_xlim()
    x_span = max(abs(float(x_max) - float(x_min)), 1.0)
    offset = 0.012 * x_span
    visual_down_direction = 1.0 if bool(axis.yaxis_inverted()) else -1.0
    for total_value, ci_high_value, y_position in zip(total_values, ci_high_values, y_positions):
        if not np.isfinite(float(total_value)) or not np.isfinite(float(ci_high_value)):
            continue
        text_x = float(ci_high_value) + offset
        text_y = float(y_position) + visual_down_direction * float(y_offset_factor) * float(bar_height)
        axis.text(
            text_x,
            text_y,
            f"{float(total_value):.2f}",
            va="center",
            ha="left",
            fontsize=11,
        )


def _save_metric_plot(metric_frame: pd.DataFrame, spec: dict[str, Any], out_dir: str) -> str | None:
    if str(spec.get("render", "")) == "controller_decision_time":
        return _save_controller_decision_time_plot(metric_frame, spec, out_dir)

    plot_frame = metric_frame.loc[metric_frame["plot_key"] == spec["key"]].copy()
    if plot_frame.empty:
        return None

    stats = [stat for stat in STAT_ORDER if stat in plot_frame["stat"].unique()]
    if not stats:
        return None

    trace_entries = _trace_entries_in_order(plot_frame)
    trace_order = [trace_label for _trace_type, trace_label in trace_entries]
    metric_labels = [label for _, label in spec["series"] if label in plot_frame["metric_label"].unique()]
    multi_series = len(metric_labels) > 1
    fig_height = max(5.0, 1.1 * len(trace_order) + 2.0)
    fig_width = max(8.0, 6.0 * len(stats))
    fig, axes = plt.subplots(1, len(stats), figsize=(fig_width, fig_height), sharey=True)
    axes_list = [axes] if len(stats) == 1 else list(axes)

    legend_handles = None
    legend_labels = None
    palette = dict(zip(metric_labels, sns.color_palette("pastel", n_colors=max(len(metric_labels), 1))))
    has_negative = bool((plot_frame["value"] < 0.0).any())

    for axis, stat in zip(axes_list, stats):
        stat_frame = plot_frame.loc[plot_frame["stat"] == stat].copy()
        if multi_series:
            sns.barplot(
                data=stat_frame,
                x="value",
                y="trace_label",
                hue="metric_label",
                order=trace_order,
                hue_order=metric_labels,
                palette=palette,
                orient="h",
                edgecolor="black",
                linewidth=1.0,
                ax=axis,
            )
            handles, labels = axis.get_legend_handles_labels()
            if legend_handles is None and handles:
                legend_handles = handles
                legend_labels = labels
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
        else:
            sns.barplot(
                data=stat_frame,
                x="value",
                y="trace_label",
                order=trace_order,
                color=NON_SHIELD_SETUP_COLOR,
                orient="h",
                edgecolor="black",
                linewidth=1.0,
                ax=axis,
            )

        _style_axis_bars(axis, trace_entries=trace_entries, num_series=len(metric_labels) if multi_series else 1)
        _mark_zero_bars(axis)
        _annotate_bar_values(axis)
        axis.set_title(STAT_LABELS[stat])
        axis.set_xlabel(spec["x_label"])
        axis.set_ylabel("")
        if has_negative:
            axis.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        _style_axis_spines(axis)

    if multi_series and legend_handles and legend_labels:
        legend_handles = [
            Patch(facecolor="white", edgecolor="black", hatch=SERIES_HATCHES[index % len(SERIES_HATCHES)], label=label)
            for index, label in enumerate(legend_labels)
        ]
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(len(legend_labels), 4),
            bbox_to_anchor=(0.5, 1.005),
            frameon=False,
        )

    sns.despine()
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94 if multi_series else 0.98))
    out_path = os.path.join(out_dir, f"{_plot_file_stem(spec)}_stats.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_ci_plot(metric_frame: pd.DataFrame, spec: dict[str, Any], out_dir: str) -> str | None:
    plot_frame = metric_frame.loc[metric_frame["plot_key"] == spec["key"]].copy()
    if plot_frame.empty:
        return None

    trace_entries = _trace_entries_in_order(plot_frame)
    if not trace_entries:
        return None
    trace_order = [trace_label for _trace_type, trace_label in trace_entries]
    metric_labels = [label for _column, label in spec["series"] if label in plot_frame["metric_label"].unique()]
    if not metric_labels:
        metric_labels = plot_frame["metric_label"].dropna().astype(str).drop_duplicates().tolist()
    if not metric_labels:
        return None

    figure_height = max(4.0, 0.8 * len(trace_order) + 1.8)
    figure_width = 9.5 if len(metric_labels) == 1 else max(5.3 * len(metric_labels), 10.0)
    figure, axes = plt.subplots(
        1,
        len(metric_labels),
        figsize=(figure_width, figure_height),
        sharey=True,
    )
    axes_list = [axes] if len(metric_labels) == 1 else list(np.asarray(axes, dtype=object))

    for axis, metric_label in zip(axes_list, metric_labels):
        label_frame = plot_frame.loc[plot_frame["metric_label"] == metric_label].copy()
        has_negative = bool((label_frame["value"] < 0.0).any())
        try:
            sns.barplot(
                data=label_frame,
                x="value",
                y="trace_label",
                order=trace_order,
                orient="h",
                estimator=np.mean,
                errorbar=("ci", 95),
                capsize=0.1,
                color=NON_SHIELD_SETUP_COLOR,
                edgecolor="black",
                linewidth=1.0,
                ax=axis,
            )
        except (AttributeError, TypeError):
            sns.barplot(
                data=label_frame,
                x="value",
                y="trace_label",
                order=trace_order,
                orient="h",
                estimator=np.mean,
                ci=95,
                capsize=0.1,
                errcolor="#808080",
                errwidth=1.2,
                color=NON_SHIELD_SETUP_COLOR,
                edgecolor="black",
                linewidth=1.0,
                ax=axis,
            )

        _style_axis_bars(axis, trace_entries=trace_entries, num_series=1)
        _style_error_bars(axis, color="#808080")
        _mark_zero_bars(axis)
        _annotate_bar_values(axis, y_offset_factor=-0.42)
        axis.set_xlabel(spec["x_label"])
        axis.set_ylabel("")
        axis.grid(axis="x", linestyle="--", alpha=0.25)
        if len(metric_labels) > 1:
            axis.set_title(metric_label)
        if axis is axes_list[0]:
            axis.set_yticks(np.arange(len(trace_order), dtype=np.float64), labels=trace_order)
            axis.tick_params(axis="y", left=True, labelleft=True)
        else:
            axis.set_yticks(np.arange(len(trace_order), dtype=np.float64))
            axis.tick_params(axis="y", left=False, labelleft=False)
        if has_negative:
            axis.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        _style_axis_spines(axis)

    if len(metric_labels) > 1 and str(spec.get("title", "")).strip():
        figure.suptitle(f"{spec['title']} (95% CI)")
        figure.tight_layout(rect=(0.12, 0.0, 1.0, 0.95))
    else:
        figure.tight_layout()
    sns.despine()
    out_path = os.path.join(out_dir, f"{_plot_file_stem(spec)}_ci95.png")
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return out_path


def _save_controller_decision_time_plot(metric_frame: pd.DataFrame, spec: dict[str, Any], out_dir: str) -> str | None:
    plot_frame = metric_frame.loc[
        (metric_frame["plot_key"] == spec["key"]) & (metric_frame["stat"] == "avg")
    ].copy()
    if plot_frame.empty:
        return None

    trace_entries = _trace_entries_in_order(plot_frame)
    if not trace_entries:
        return None
    trace_order = [trace_label for _trace_type, trace_label in trace_entries]
    metric_labels = [label for _column, label in spec["series"] if label in plot_frame["metric_label"].unique()]
    if not metric_labels:
        return None

    figure, axes = plt.subplots(
        1,
        len(metric_labels),
        figsize=(5.3 * len(metric_labels), max(4.0, 0.55 * len(trace_order) + 1.8)),
        sharey=True,
    )
    axes_list = [axes] if len(metric_labels) == 1 else list(np.asarray(axes, dtype=object))
    y_positions = np.arange(len(trace_order), dtype=np.float64)

    for axis, metric_label in zip(axes_list, metric_labels):
        label_frame = plot_frame.loc[plot_frame["metric_label"] == metric_label].copy()
        value_by_trace = {
            str(row["trace_label"]): float(row["value"])
            for row in label_frame.to_dict(orient="records")
        }
        metric_values = [value_by_trace.get(trace_label, np.nan) for trace_label in trace_order]
        bar_colors = [_setup_bar_color(trace_type) for trace_type, _trace_label in trace_entries]
        axis.barh(
            y_positions,
            metric_values,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.8,
        )
        if axis is axes_list[0]:
            axis.set_yticks(y_positions, labels=trace_order)
            axis.tick_params(axis="y", left=True, labelleft=True)
        else:
            axis.set_yticks(y_positions)
            axis.tick_params(axis="y", left=False, labelleft=False)
        axis.set_xlabel(spec["x_label"])
        axis.set_title(metric_label)
        axis.grid(axis="x", linestyle="--", alpha=0.25)
        _mark_zero_bars(axis)
        _annotate_bar_values(axis)
        _style_axis_spines(axis)

    axes_list[0].set_yticks(y_positions, labels=trace_order)
    axes_list[0].tick_params(axis="y", left=True, labelleft=True)
    axes_list[0].set_ylabel("")
    # sns.despine()
    #* Remove the top and right spines for a cleaner look, but keep the left spine for better readability of the y-axis labels.
    for axis in axes_list:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(spec["title"])
    figure.tight_layout(rect=(0.12, 0.0, 1.0, 0.97))
    out_path = os.path.join(out_dir, f"{_plot_file_stem(spec)}_stats.png")
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return out_path


def _save_controller_decision_time_ci_plot(metric_frame: pd.DataFrame, spec: dict[str, Any], out_dir: str) -> str | None:
    plot_frame = metric_frame.loc[metric_frame["plot_key"] == spec["key"]].copy()
    if plot_frame.empty:
        return None

    trace_entries = _trace_entries_in_order(plot_frame)
    if not trace_entries:
        return None
    trace_order = [trace_label for _trace_type, trace_label in trace_entries]
    figure, axis = plt.subplots(
        1,
        1,
        figsize=(9.5, max(4.0, 0.8 * len(trace_order) + 1.8)),
    )
    legend_handles = [
        Patch(facecolor=TIMING_POLICY_COLOR, edgecolor="black", label="Policy"),
        Patch(facecolor=TIMING_SHIELD_COLOR, edgecolor="black", label="Shield"),
    ]
    bar_height = 0.72

    y_positions = np.arange(len(trace_order), dtype=np.float64)
    policy_means: list[float] = []
    shield_means: list[float] = []
    total_means: list[float] = []
    total_ci_lows: list[float] = []
    total_ci_highs: list[float] = []
    for trace_type, trace_label in trace_entries:
        trace_frame = plot_frame.loc[plot_frame["trace_label"] == trace_label].copy()
        policy_values = pd.to_numeric(trace_frame["policy_value"], errors="coerce").to_numpy(dtype=np.float64)
        shield_values = pd.to_numeric(trace_frame["shield_value"], errors="coerce").to_numpy(dtype=np.float64)
        total_values = policy_values + shield_values
        policy_values = policy_values[np.isfinite(policy_values)]
        shield_values = shield_values[np.isfinite(shield_values)]
        total_values = total_values[np.isfinite(total_values)]
        policy_mean = float(np.mean(policy_values)) if policy_values.size > 0 else 0.0
        shield_mean = float(np.mean(shield_values)) if shield_values.size > 0 else 0.0
        total_ci = _bootstrap_mean_ci(
            total_values.tolist(),
            confidence_pct=_TIMING_CONFIDENCE_PCT,
            seed_key=f"{trace_type}:mean",
        )
        policy_means.append(policy_mean)
        shield_means.append(shield_mean)
        total_means.append(float(total_ci["mean"]))
        total_ci_lows.append(float(total_ci["low"]))
        total_ci_highs.append(float(total_ci["high"]))

    axis.barh(
        y_positions,
        policy_means,
        height=bar_height,
        color=TIMING_POLICY_COLOR,
        edgecolor="black",
        linewidth=0.8,
        label="Policy",
    )
    axis.barh(
        y_positions,
        shield_means,
        left=policy_means,
        height=bar_height,
        color=TIMING_SHIELD_COLOR,
        edgecolor="black",
        linewidth=0.8,
        label="Shield",
    )
    for y_position, total_mean, total_ci_low, total_ci_high in zip(
        y_positions.tolist(),
        total_means,
        total_ci_lows,
        total_ci_highs,
    ):
        axis.errorbar(
            float(total_mean),
            float(y_position),
            xerr=np.asarray(
                [
                    [max(float(total_mean) - float(total_ci_low), 0.0)],
                    [max(float(total_ci_high) - float(total_mean), 0.0)],
                ],
                dtype=np.float64,
            ),
            fmt="none",
            ecolor=TIMING_ERROR_BAR_COLOR,
            elinewidth=1.2,
            capsize=4.0,
            capthick=1.2,
            zorder=6,
        )
    _annotate_total_ci_values(
        axis,
        total_values=total_means,
        ci_high_values=total_ci_highs,
        y_positions=y_positions.tolist(),
        y_offset_factor=-0.42,
        bar_height=bar_height,
    )
    axis.set_yticks(y_positions, labels=trace_order)
    axis.set_xlabel("Decision Time [ms]")
    axis.set_ylabel("")
    axis.grid(axis="x", linestyle="--", alpha=0.25)
    _style_axis_spines(axis)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    figure.suptitle("Controller Decision Time by Setup (95% CI)")
    figure.tight_layout(rect=(0.12, 0.0, 1.0, 0.95))
    out_path = os.path.join(out_dir, f"{_plot_file_stem(spec)}_ci95.png")
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-path",
        type=str,
        required=True,
        help="Path to a single eval_summary.json file or an eval output root directory containing per-setup subdirectories.",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    summary_path = os.path.abspath(os.path.expanduser(args.summary_path))
    summary_payload = _load_summary(summary_path)
    out_dir = _resolve_out_dir(summary_path, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    timing_summary_payloads, timing_summary_paths = _ensure_and_load_timing_summaries(summary_path, summary_payload)

    _set_plot_style()
    metric_frame = _apply_setup_name_map(_apply_setup_value_adjustments(_build_metric_frame(summary_payload)))
    ci_plot_frames: list[tuple[pd.DataFrame, dict[str, Any]]] = []
    for plot_spec in PLOT_SPECS:
        if str(plot_spec.get("render", "")) == "controller_decision_time":
            continue
        plot_frame = _build_ci_metric_frame(summary_payload, plot_key=str(plot_spec["key"]))
        if plot_frame is None:
            continue
        ci_plot_frames.append(
            (
                _apply_setup_name_map(_apply_setup_value_adjustments(plot_frame)),
                plot_spec,
            )
        )
    controller_decision_time_ci_frame = _build_controller_decision_time_timing_frame(
        timing_summary_payloads,
        fallback_summary_payload=summary_payload,
    )
    if controller_decision_time_ci_frame is not None:
        controller_decision_time_ci_frame = _apply_setup_name_map(controller_decision_time_ci_frame)

    output_paths = [
        path
        for path in (_save_metric_plot(metric_frame, spec, out_dir) for spec in PLOT_SPECS)
        if path is not None
    ]
    for plot_frame, plot_spec in ci_plot_frames:
        ci_path = _save_ci_plot(plot_frame, plot_spec, out_dir)
        if ci_path is not None:
            output_paths.append(ci_path)
    if controller_decision_time_ci_frame is not None:
        controller_ci_path = _save_controller_decision_time_ci_plot(
            controller_decision_time_ci_frame,
            _spec_by_key("controller_decision_time"),
            out_dir,
        )
        if controller_ci_path is not None:
            output_paths.append(controller_ci_path)

    manifest_path = os.path.join(out_dir, "plot_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "summary_path": summary_path,
                "summary_paths": list(summary_payload.get("summary_paths", [])),
                "timing_summary_paths": timing_summary_paths,
                "out_dir": out_dir,
                "setup_name_map": SETUP_NAME_MAP,
                "setup_value_adjustment_pct": SETUP_VALUE_ADJUSTMENT_PCT,
                "plots": output_paths,
            },
            file_obj,
            indent=2,
            sort_keys=True,
        )

    for path in output_paths:
        print(path)
    print(manifest_path)


if __name__ == "__main__":  # pragma: no cover
    main()
