"""
Example usage:
PYTHON_BIN=.venv/bin/python
$PYTHON_BIN scripts/plot_clean_vs_adv.py \
  --summary-path attacks/output/eval/clean_vs_adv_summary.json \
  --out-dir attacks/output/eval/plots
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRIC_SPECS: tuple[dict[str, str], ...] = (
    {
        "name": "reward",
        "clean": "clean_reward",
        "adv": "adv_reward",
        "delta": "delta_reward",
        "label": "Episode Reward",
    },
    {
        "name": "sage_reward_mean",
        "clean": "clean_sage_reward_mean",
        "adv": "adv_sage_reward_mean",
        "delta": "delta_sage_reward_mean",
        "label": "Mean Sage Reward",
    },
    {
        "name": "external_score_mean",
        "clean": "clean_external_score_mean",
        "adv": "adv_external_score_mean",
        "delta": "delta_external_score_mean",
        "label": "Mean External Score",
    },
    {
        "name": "rate_mean_mbps",
        "clean": "clean_rate_mean_mbps",
        "adv": "adv_rate_mean_mbps",
        "delta": "delta_rate_mean_mbps",
        "label": "Mean Delivery Rate (Mbps)",
    },
    {
        "name": "rtt_mean_ms",
        "clean": "clean_rtt_mean_ms",
        "adv": "adv_rtt_mean_ms",
        "delta": "delta_rtt_mean_ms",
        "label": "Mean RTT (ms)",
    },
    {
        "name": "loss_mean_mbps",
        "clean": "clean_loss_mean_mbps",
        "adv": "adv_loss_mean_mbps",
        "delta": "delta_loss_mean_mbps",
        "label": "Mean Loss (Mbps)",
    },
    {
        "name": "departure_mean_mbps",
        "clean": "clean_departure_mean_mbps",
        "adv": "adv_departure_mean_mbps",
        "delta": "delta_departure_mean_mbps",
        "label": "Mean Downlink Departure Rate (Mbps)",
    },
)

NEW_METRIC_SPECS: tuple[dict[str, Any], ...] = (
    {"columns": ("episode_total_reward",), "label": "Episode Reward"},
    {"columns": ("reward_mean",), "label": "Mean Step Reward"},
    {"columns": ("sage_reward_mean", "victim_reward_mean"), "label": "Mean Sage Reward"},
    {"columns": ("sage_external_score_mean", "victim_external_score_mean"), "label": "Mean External Score"},
    {"columns": ("sage_windowed_rate_mbps_mean", "victim_windowed_rate_mbps_mean"), "label": "Mean Delivery Rate (Mbps)"},
    {"columns": ("sage_rtt_ms_mean", "victim_rtt_ms_mean"), "label": "Mean RTT (ms)"},
    {"columns": ("sage_loss_mbps_mean", "victim_loss_mbps_mean"), "label": "Mean Loss (Mbps)"},
    {"columns": ("mm_down_departure_rate_mbps_mean",), "label": "Mean Downlink Departure Rate (Mbps)"},
)


def _set_plot_style() -> None:
    sns.set_style("ticks", {"grid.linestyle": ":"})
    plt.rcParams["axes.grid"] = True
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.titlesize"] = 22
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["figure.titlesize"] = 24


def _load_summary(summary_path: str) -> dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError("summary payload must be a JSON object")
    rows = payload.get("rows")
    per_episode = payload.get("per_episode")
    if isinstance(rows, list) and rows:
        return payload
    if isinstance(per_episode, list) and per_episode:
        return payload
    summary_rows = payload.get("summary")
    if not isinstance(summary_rows, list) or not summary_rows:
        raise ValueError("summary payload does not contain any rows")
    return payload


def _resolve_out_dir(summary_path: str, out_dir: str | None) -> str:
    if out_dir:
        return os.path.abspath(os.path.expanduser(out_dir))
    return os.path.join(os.path.dirname(os.path.abspath(summary_path)), "plots")


def _build_rows_frame(summary_payload: dict[str, Any]) -> pd.DataFrame:
    source_rows = summary_payload.get("rows")
    if isinstance(source_rows, list) and source_rows:
        frame = pd.DataFrame(source_rows).copy()
        if "trace_id" not in frame.columns:
            raise ValueError("summary rows are missing trace_id")
        frame["trace_label"] = frame["trace_id"].astype(str)
        return frame

    frame = pd.DataFrame(summary_payload["per_episode"]).copy()
    if "episode_id" not in frame.columns or "trace_type" not in frame.columns:
        raise ValueError("per-episode summary rows are missing episode_id or trace_type")
    frame["trace_label"] = frame["episode_id"].astype(str)
    return frame


def _build_aggregate_frame(rows_frame: pd.DataFrame) -> pd.DataFrame:
    if "trace_type" in rows_frame.columns:
        records: list[dict[str, Any]] = []
        for spec in NEW_METRIC_SPECS:
            column = next((name for name in spec["columns"] if name in rows_frame.columns), None)
            if column is None:
                continue
            if column not in rows_frame.columns:
                continue
            grouped = rows_frame.groupby("trace_type")[column].mean()
            for trace_type, value in grouped.items():
                records.append({"metric": spec["label"], "condition": str(trace_type), "value": float(value)})
        if not records:
            raise ValueError("summary rows do not contain any recognized metric columns")
        return pd.DataFrame.from_records(records)

    records: list[dict[str, Any]] = []
    for spec in METRIC_SPECS:
        clean_col = spec["clean"]
        adv_col = spec["adv"]
        if clean_col not in rows_frame.columns or adv_col not in rows_frame.columns:
            continue
        records.append({"metric": spec["label"], "condition": "Clean", "value": float(rows_frame[clean_col].mean())})
        records.append({"metric": spec["label"], "condition": "Adversarial", "value": float(rows_frame[adv_col].mean())})
    if not records:
        raise ValueError("summary rows do not contain any recognized metric columns")
    return pd.DataFrame.from_records(records)


def _delta_palette(values: pd.Series) -> list[str]:
    return ["#b2182b" if float(value) < 0.0 else "#2166ac" for value in values]


def _save_aggregate_plot(aggregate_frame: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(16, 10))
    conditions = [str(value) for value in aggregate_frame["condition"].drop_duplicates().tolist()]
    palette = sns.color_palette("deep", n_colors=max(len(conditions), 1))
    sns.barplot(
        data=aggregate_frame,
        x="metric",
        y="value",
        hue="condition",
        palette={condition: palette[index] for index, condition in enumerate(conditions)},
        ax=ax,
    )
    ax.set_title("Aggregate Evaluation Performance")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Value Across Test Traces")
    ax.tick_params(axis="x", rotation=25)
    sns.despine()
    fig.tight_layout()
    out_path = os.path.join(out_dir, "aggregate_clean_vs_adv.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_delta_plots(rows_frame: pd.DataFrame, out_dir: str) -> list[str]:
    output_paths: list[str] = []
    if "trace_type" in rows_frame.columns:
        return output_paths
    delta_specs = (
        ("delta_external_score_mean", "Per-Trace Delta External Score"),
        ("delta_reward", "Per-Trace Delta Episode Reward"),
        ("delta_rate_mean_mbps", "Per-Trace Delta Delivery Rate (Mbps)"),
        ("delta_rtt_mean_ms", "Per-Trace Delta RTT (ms)"),
    )
    available_specs = [(column, title) for column, title in delta_specs if column in rows_frame.columns]
    if not available_specs:
        return output_paths

    fig_height = max(18, 0.45 * len(rows_frame))
    fig, axes = plt.subplots(len(available_specs), 1, figsize=(18, fig_height), squeeze=False)

    for axis, (column, title) in zip(axes[:, 0], available_specs):
        delta_frame = rows_frame[["trace_label", column]].sort_values(by=column, ascending=True)
        sns.barplot(
            data=delta_frame,
            x=column,
            y="trace_label",
            palette=_delta_palette(delta_frame[column]),
            orient="h",
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel(column.replace("_", " "))
        axis.set_ylabel("")
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.5)

    sns.despine()
    fig.tight_layout()
    out_path = os.path.join(out_dir, "per_trace_deltas.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(out_path)
    return output_paths


def _save_scatter_plots(rows_frame: pd.DataFrame, out_dir: str) -> list[str]:
    output_paths: list[str] = []
    if "trace_type" in rows_frame.columns:
        return output_paths
    scatter_specs = (
        ("clean_reward", "adv_reward", "Episode Reward"),
        ("clean_external_score_mean", "adv_external_score_mean", "Mean External Score"),
        ("clean_rate_mean_mbps", "adv_rate_mean_mbps", "Mean Delivery Rate (Mbps)"),
        ("clean_rtt_mean_ms", "adv_rtt_mean_ms", "Mean RTT (ms)"),
    )
    available_specs = [(clean, adv, label) for clean, adv, label in scatter_specs if clean in rows_frame.columns and adv in rows_frame.columns]
    if not available_specs:
        return output_paths

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes_flat = axes.flatten()

    for axis, (clean_col, adv_col, label) in zip(axes_flat, available_specs):
        sns.scatterplot(data=rows_frame, x=clean_col, y=adv_col, s=120, color="#4c72b0", ax=axis)
        min_val = min(float(rows_frame[clean_col].min()), float(rows_frame[adv_col].min()))
        max_val = max(float(rows_frame[clean_col].max()), float(rows_frame[adv_col].max()))
        axis.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1.5)
        axis.set_title(label)
        axis.set_xlabel(f"Clean {label}")
        axis.set_ylabel(f"Adversarial {label}")

    for axis in axes_flat[len(available_specs) :]:
        axis.axis("off")

    sns.despine()
    fig.tight_layout()
    out_path = os.path.join(out_dir, "clean_vs_adv_scatter.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(out_path)
    return output_paths


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
    rows_frame = _build_rows_frame(summary_payload)
    aggregate_frame = _build_aggregate_frame(rows_frame)

    output_paths = [
        _save_aggregate_plot(aggregate_frame, out_dir),
        *_save_delta_plots(rows_frame, out_dir),
        *_save_scatter_plots(rows_frame, out_dir),
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
