"""
Example usage:
python scripts/plot_clean_vs_adv.py \
  --summary-path attacks/output/eval-300k/clean_vs_adv_summary.json \
  --out-dir attacks/output/eval-300k/plots
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import matplotlib.pyplot as plt
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

PLOT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "gap_value",
        "title": "Gap Value",
        "x_label": "Per-Trace Gap Value",
        "series": (("gap_value_mean", "Gap Value"),),
    },
    {
        "key": "hard_max_gap",
        "title": "Hard-Max Gap Value",
        "x_label": "Per-Trace Hard-Max Gap Value",
        "series": (("gap_best_baseline_gap_mean", "Hard-Max Gap"),),
    },
    {
        "key": "gap_percent",
        "title": "Gap Percent",
        "x_label": "Per-Trace Gap Percent vs Best Baseline",
        "series": (("gap_percent_mean", "Gap Percent"),),
    },
    {
        "key": "smoothed_gap_percent",
        "title": "Smoothed Gap Percent",
        "x_label": "Per-Trace Gap Percent vs Smoothed Baseline",
        "series": (("smoothed_gap_percent_mean", "Smoothed Gap Percent"),),
    },
    {
        "key": "reward",
        "title": "Per-Trace Attacker Reward",
        "x_label": "Per-Trace Attacker Reward",
        "series": (("episode_total_reward", "Attacker Reward"),),
    },
    {
        "key": "baseline_scores",
        "title": "Controller Scores",
        "x_label": "Per-Trace Score",
        "series": (
            ("gap_score_sage_mean", "Sage"),
            ("gap_score_bbr_mean", "BBR"),
            ("gap_score_cubic_mean", "CUBIC"),
            ("gap_best_baseline_score_mean", "Best Baseline"),
        ),
    },
    {
        "key": "throughput",
        "title": "Controller Throughput",
        "x_label": "Per-Trace Throughput [Mbps]",
        "series": (
            ("sage_windowed_rate_mbps_mean", "Sage"),
            ("baseline_bbr_rate_mbps_mean", "BBR"),
            ("baseline_cubic_rate_mbps_mean", "CUBIC"),
        ),
    },
    {
        "key": "latency",
        "title": "Controller Latency",
        "x_label": "Per-Trace RTT [ms]",
        "series": (
            ("sage_rtt_ms_mean", "Sage"),
            ("baseline_bbr_rtt_ms_mean", "BBR"),
            ("baseline_cubic_rtt_ms_mean", "CUBIC"),
        ),
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


def _load_summary(summary_path: str) -> dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError("summary payload must be a JSON object")
    if isinstance(payload.get("per_episode"), list) and payload["per_episode"]:
        return payload
    if isinstance(payload.get("summary"), list) and payload["summary"]:
        return payload
    raise ValueError("summary payload does not contain any plot-ready rows")


def _resolve_out_dir(summary_path: str, out_dir: str | None) -> str:
    if out_dir:
        return os.path.abspath(os.path.expanduser(out_dir))
    return os.path.join(os.path.dirname(os.path.abspath(summary_path)), "plots")


def _trace_type_order(values: pd.Series) -> list[str]:
    ordered = [str(value) for value in values.dropna().tolist()]
    unique = list(dict.fromkeys(ordered))
    clean = [value for value in unique if value.lower() == "clean"]
    others = sorted((value for value in unique if value.lower() != "clean"), key=str.casefold)
    return clean + others


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


def _gap_percent_from_series(gap_values: pd.Series, baseline_values: pd.Series) -> pd.Series:
    gap_numeric = pd.to_numeric(gap_values, errors="coerce")
    baseline_numeric = pd.to_numeric(baseline_values, errors="coerce")
    valid = gap_numeric.notna() & baseline_numeric.notna() & (baseline_numeric > _GAP_PERCENT_EPS)
    result = pd.Series(np.nan, index=gap_numeric.index, dtype=np.float64)
    result.loc[valid] = gap_numeric.loc[valid] / baseline_numeric.loc[valid]
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
        numeric_columns = {
            column: pd.to_numeric(frame[column], errors="coerce") for column, _ in series_columns
        }
        if len(series_columns) > 1:
            valid_mask = pd.Series(True, index=frame.index)
            for column, _ in series_columns:
                valid_mask &= numeric_columns[column].notna()
        for column, label in series_columns:
            if len(series_columns) > 1:
                column_frame = frame.loc[valid_mask, ["trace_type"]].copy()
                column_frame["value"] = numeric_columns[column].loc[valid_mask].astype(float)
            else:
                column_frame = frame.loc[numeric_columns[column].notna(), ["trace_type"]].copy()
                column_frame["value"] = numeric_columns[column].loc[numeric_columns[column].notna()].astype(float)
            if column_frame.empty:
                continue
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
                        "value": float(numerator) / float(denominator),
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
                        "value": float(numerator) / float(denominator),
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


def _save_metric_plot(metric_frame: pd.DataFrame, spec: dict[str, Any], out_dir: str) -> str | None:
    plot_frame = metric_frame.loc[metric_frame["plot_key"] == spec["key"]].copy()
    if plot_frame.empty:
        return None

    stats = [stat for stat in STAT_ORDER if stat in plot_frame["stat"].unique()]
    if not stats:
        return None

    trace_order = _trace_type_order(plot_frame["trace_type"])
    metric_labels = [label for _, label in spec["series"] if label in plot_frame["metric_label"].unique()]
    multi_series = len(metric_labels) > 1
    fig_height = max(5.0, 1.1 * len(trace_order) + 2.0)
    fig_width = max(8.0, 6.0 * len(stats))
    fig, axes = plt.subplots(1, len(stats), figsize=(fig_width, fig_height), sharey=True)
    axes_list = [axes] if len(stats) == 1 else list(axes)

    legend_handles = None
    legend_labels = None
    palette = dict(zip(metric_labels, sns.color_palette("bright", n_colors=max(len(metric_labels), 1))))
    has_negative = bool((plot_frame["value"] < 0.0).any())

    for axis, stat in zip(axes_list, stats):
        stat_frame = plot_frame.loc[plot_frame["stat"] == stat].copy()
        if multi_series:
            sns.barplot(
                data=stat_frame,
                x="value",
                y="trace_type",
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
                y="trace_type",
                order=trace_order,
                color=sns.color_palette("bright", n_colors=1)[0],
                orient="h",
                edgecolor="black",
                linewidth=1.0,
                ax=axis,
            )

        _mark_zero_bars(axis)
        axis.set_title(STAT_LABELS[stat])
        axis.set_xlabel(spec["x_label"])
        axis.set_ylabel("Setup" if axis is axes_list[0] else "")
        if has_negative:
            axis.axvline(0.0, color="black", linestyle="--", linewidth=1.2)

    if multi_series and legend_handles and legend_labels:
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
    out_path = os.path.join(out_dir, f"{spec['key']}_stats.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    summary_path = os.path.abspath(os.path.expanduser(args.summary_path))
    summary_payload = _load_summary(summary_path)
    out_dir = _resolve_out_dir(summary_path, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    _set_plot_style()
    metric_frame = _build_metric_frame(summary_payload)

    output_paths = [
        path
        for path in (_save_metric_plot(metric_frame, spec, out_dir) for spec in PLOT_SPECS)
        if path is not None
    ]

    manifest_path = os.path.join(out_dir, "plot_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "summary_path": summary_path,
                "out_dir": out_dir,
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
