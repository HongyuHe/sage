"""
Run this after `scripts/generate_online_adv_traces.py`.

Example usage:
Baseline methods are inferred from the saved training config / generated manifest.

time python scripts/eval_sage_clean_vs_adv.py \
  --generated-manifest attacks/adv_traces/hotnets19-loss_50ms_300k/generated_manifest.json \
  --out-dir attacks/output/eval-300k \
  --skip-clean-rollout \
  --wandb --wandb-tags v4 --wandb-project sage-gap-eval

time python scripts/eval_sage_clean_vs_adv.py \
  --generated-manifest attacks/adv_traces/gap-constrained-1baseline_300k/generated_manifest.json \
  --out-dir attacks/output/eval-300k \
  --skip-clean-rollout \
  --wandb --wandb-tags v4 --wandb-project sage-gap-eval

time python scripts/eval_sage_clean_vs_adv.py \
  --generated-manifest attacks/adv_traces/gap-constrained-2baselines_300k/generated_manifest.json \
  --out-dir attacks/output/eval-300k \
  --skip-clean-rollout \
  --wandb --wandb-tags v4 --wandb-project sage-gap-eval
  
time python scripts/eval_sage_clean_vs_adv.py \
  --generated-manifest attacks/adv_traces/gap-constrained-3baselines_300k/generated_manifest.json \
  --out-dir attacks/output/eval-300k \
  --skip-clean-rollout \
  --wandb --wandb-tags v4 --wandb-project sage-gap-eval
  
time python scripts/eval_sage_clean_vs_adv.py \
  --generated-manifest attacks/adv_traces/gap-constrained-3b-hard_200k/generated_manifest.json \
  --out-dir attacks/output/eval-300k \
  --skip-clean-rollout \
  --wandb --wandb-tags v4 --wandb-project sage-gap-eval

time python scripts/eval_sage_clean_vs_adv.py \
  --generated-manifest attacks/adv_traces/gap-unconstrained_300k/generated_manifest.json \
  --out-dir attacks/output/eval-300k \
  --skip-clean-rollout \
  --wandb --wandb-tags v4 --wandb-project sage-gap-eval

time python scripts/eval_sage_clean_vs_adv.py \
  --test-manifest attacks/test/manifest.json \
  --config-path attacks/models/gap_adv_20260321_gap-constrained-bbr_300k_50ms.config.json \
  --out-dir attacks/output/eval-300k \
  --clean-only-rollout \
  --wandb --wandb-tags v4 --wandb-project sage-gap-eval

time python scripts/eval_sage_clean_vs_adv.py \
  --generated-manifest attacks/adv_traces/gap-constrained-3baselines_300k/generated_manifest.json \
  --out-dir attacks/output/eval-300k \
  --skip-clean-rollout \
  --shield-rules-file attacks/output/shield-rules/gap-constrained-3baselines_300k/sage_directional_shield_rules.json \
  --wandb --wandb-tags v4 --wandb-project sage-gap-eval
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
import sys
from typing import Any

import numpy as np

from attacks.envs import ParallelGapAttackEnv, baseline_methods_from_config
from attacks.online import SageLaunchConfig, acquire_run_namespace


if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scripts._trace_attack_common import (
        IndependentAttackEnv,
        attack_bounds_from_config,
        build_clean_action_schedule,
        expand_attack_bounds_for_bandwidth,
        load_mahimahi_trace_schedule,
        load_trace_entries,
        materialize_trace_splits,
        print_wandb_run_links,
        repo_root_from_script,
        resolve_repo_path,
        run_online_policy_episode,
        save_json,
        try_import_wandb,
        utc_now_iso,
    )
else:
    from ._trace_attack_common import (
        IndependentAttackEnv,
        attack_bounds_from_config,
        build_clean_action_schedule,
        expand_attack_bounds_for_bandwidth,
        load_mahimahi_trace_schedule,
        load_trace_entries,
        materialize_trace_splits,
        print_wandb_run_links,
        repo_root_from_script,
        resolve_repo_path,
        run_online_policy_episode,
        save_json,
        try_import_wandb,
        utc_now_iso,
    )


def _ensure_test_manifest(repo_root: str, manifest_path: str) -> str:
    resolved = resolve_repo_path(repo_root, manifest_path)
    if os.path.exists(resolved):
        return resolved
    materialize_trace_splits(
        repo_root=repo_root,
        train_root=resolve_repo_path(repo_root, "attacks/train"),
        test_root=resolve_repo_path(repo_root, "attacks/test"),
    )
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"missing test manifest: {resolved}")
    return resolved


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return dict(json.load(file_obj))


def _load_training_config(repo_root: str, generated_manifest: dict[str, Any], config_path: str | None) -> dict[str, Any]:
    if config_path is not None:
        resolved = resolve_repo_path(repo_root, config_path)
        return _load_json(resolved)

    manifest_training_config = generated_manifest.get("training_config_path")
    if manifest_training_config:
        return _load_json(resolve_repo_path(repo_root, str(manifest_training_config)))

    generated_entries = generated_manifest.get("generated_entries", [])
    if not generated_entries:
        raise ValueError("generated manifest has no generated entries")
    first_schedule = resolve_repo_path(repo_root, generated_entries[0]["schedule_path"])
    schedule_payload = _load_json(first_schedule)
    training_config_path = schedule_payload.get("training_config_path")
    if not training_config_path:
        raise ValueError("schedule payload does not include training_config_path")
    return _load_json(resolve_repo_path(repo_root, str(training_config_path)))


def _trace_set_name(generated_manifest_path: str, generated_manifest: dict[str, Any]) -> str:
    manifest_name = generated_manifest.get("trace_set_name")
    if isinstance(manifest_name, str) and manifest_name.strip():
        return manifest_name.strip()
    parent_dir = os.path.dirname(os.path.abspath(generated_manifest_path))
    return os.path.basename(parent_dir) or "generated"


def _trace_set_name_from_config_path(config_path: str | None) -> str:
    if not config_path:
        return "clean-only"
    basename = os.path.basename(str(config_path))
    if basename.endswith(".config.json"):
        basename = basename[: -len(".config.json")]
    else:
        basename = os.path.splitext(basename)[0]
    return basename or "clean-only"


def _result_trace_label(base_name: str, *, shield_enabled: bool) -> str:
    name = str(base_name).strip() or "eval"
    if bool(shield_enabled):
        return f"{name}_shield"
    return name


_CONTROLLER_TIMING_LOG_FILENAME = "sage-controller-timing.jsonl"


def _expand_legacy_saved_action(action: np.ndarray, config_payload: dict[str, Any]) -> np.ndarray:
    raw = np.asarray(action, dtype=np.float32).reshape(-1)
    inner_bounds = attack_bounds_from_config(config_payload)
    inner_low = np.asarray(inner_bounds.low, dtype=np.float32).reshape(-1)
    inner_high = np.asarray(inner_bounds.high, dtype=np.float32).reshape(-1)
    if raw.shape[0] == inner_low.shape[0]:
        return np.clip(raw, inner_low, inner_high).astype(np.float32, copy=False)

    shared_bandwidth_action = (
        config_payload.get("attack_shared_bw_min_mbps") is not None
        and config_payload.get("attack_shared_bw_max_mbps") is not None
    )
    shared_loss_action = (
        config_payload.get("attack_shared_loss_min") is not None
        and config_payload.get("attack_shared_loss_max") is not None
    )
    shared_bin_loss_action = (
        config_payload.get("attack_shared_bin_loss_min_rate") is not None
        and config_payload.get("attack_shared_bin_loss_max_rate") is not None
    )
    shared_delay_action = (
        config_payload.get("attack_shared_delay_min_ms") is not None
        and config_payload.get("attack_shared_delay_max_ms") is not None
    )
    log_shared_bandwidth_action = bool(shared_bandwidth_action) and str(
        config_payload.get("policy_action_transform", "")
    ) == "log_unit_interval_shared_bandwidth"

    def _shared_bounds(low_a: float, high_a: float, low_b: float, high_b: float, label: str) -> tuple[float, float]:
        shared_low = max(float(low_a), float(low_b))
        shared_high = min(float(high_a), float(high_b))
        if shared_low > shared_high:
            raise ValueError(f"{label} shared bounds do not overlap")
        return float(shared_low), float(shared_high)

    outer_low: list[float] = []
    outer_high: list[float] = []
    if shared_bandwidth_action:
        if log_shared_bandwidth_action:
            outer_low.append(0.0)
            outer_high.append(1.0)
        else:
            shared_low, shared_high = _shared_bounds(inner_low[0], inner_high[0], inner_low[1], inner_high[1], "bandwidth")
            outer_low.append(shared_low)
            outer_high.append(shared_high)
    else:
        outer_low.extend([float(inner_low[0]), float(inner_low[1])])
        outer_high.extend([float(inner_high[0]), float(inner_high[1])])
    if shared_loss_action or shared_bin_loss_action:
        shared_low, shared_high = _shared_bounds(inner_low[2], inner_high[2], inner_low[3], inner_high[3], "loss")
        outer_low.append(shared_low)
        outer_high.append(shared_high)
    else:
        outer_low.extend([float(inner_low[2]), float(inner_low[3])])
        outer_high.extend([float(inner_high[2]), float(inner_high[3])])
    if shared_delay_action:
        shared_low, shared_high = _shared_bounds(inner_low[4], inner_high[4], inner_low[5], inner_high[5], "delay")
        outer_low.append(shared_low)
        outer_high.append(shared_high)
    else:
        outer_low.extend([float(inner_low[4]), float(inner_low[5])])
        outer_high.extend([float(inner_high[4]), float(inner_high[5])])

    outer_low_arr = np.asarray(outer_low, dtype=np.float32)
    outer_high_arr = np.asarray(outer_high, dtype=np.float32)
    if raw.shape[0] != outer_low_arr.shape[0]:
        raise ValueError(
            f"expected saved action with {outer_low_arr.shape[0]} or {inner_low.shape[0]} dims, got {raw.shape[0]}"
        )
    clipped = np.clip(raw, outer_low_arr, outer_high_arr)
    index = 0
    if shared_bandwidth_action:
        if log_shared_bandwidth_action:
            shared_min = max(float(config_payload.get("attack_shared_bw_min_mbps", inner_low[0])), 1e-6)
            shared_max = max(float(config_payload.get("attack_shared_bw_max_mbps", inner_high[0])), shared_min)
            log_min = float(np.log(shared_min))
            log_span = max(float(np.log(shared_max)) - log_min, 1e-6)
            shared_bandwidth_mbps = float(np.exp(log_min + float(clipped[index]) * log_span))
        else:
            shared_bandwidth_mbps = float(clipped[index])
        uplink_bw = shared_bandwidth_mbps
        downlink_bw = shared_bandwidth_mbps
        index += 1
    else:
        uplink_bw = float(clipped[index])
        downlink_bw = float(clipped[index + 1])
        index += 2

    if shared_loss_action or shared_bin_loss_action:
        uplink_loss = float(clipped[index])
        downlink_loss = float(clipped[index])
        index += 1
    else:
        uplink_loss = float(clipped[index])
        downlink_loss = float(clipped[index + 1])
        index += 2

    if shared_delay_action:
        uplink_delay = float(clipped[index])
        downlink_delay = float(clipped[index])
    else:
        uplink_delay = float(clipped[index])
        downlink_delay = float(clipped[index + 1])

    return np.asarray(
        [uplink_bw, downlink_bw, uplink_loss, downlink_loss, uplink_delay, downlink_delay],
        dtype=np.float32,
    )


def _load_action_schedule(schedule_payload: dict[str, Any], *, config_payload: dict[str, Any]) -> list[np.ndarray]:
    actions: list[np.ndarray] = []
    for step in schedule_payload.get("steps", []):
        if isinstance(step, dict) and isinstance(step.get("effective_action"), list):
            actions.append(np.asarray(step["effective_action"], dtype=np.float32))
            continue
        if isinstance(step, dict) and isinstance(step.get("action"), list):
            actions.append(_expand_legacy_saved_action(np.asarray(step["action"], dtype=np.float32), config_payload))
            continue
    return actions


def _resolved_launch_config(
    *,
    config_payload: dict[str, Any],
    run_namespace,
) -> SageLaunchConfig:
    return replace(
        SageLaunchConfig(
            sage_script="sage_rl/sage.sh",
            latency_ms=int(config_payload.get("latency_ms", 25)),
            port=int(config_payload.get("port", 5101)),
            downlink_trace="wired48",
            uplink_trace="wired48",
            iteration_id=int(config_payload.get("iteration_id", 0)),
            qsize_packets=int(config_payload.get("qsize_packets", 128)),
            env_bw_mbps=int(config_payload.get("env_bw_mbps", 48)),
            bw2_mbps=int(config_payload.get("bw2_mbps", 48)),
            trace_period_s=int(config_payload.get("trace_period_s", 7)),
            first_time_mode=int(config_payload.get("sage_mode", 0)),
            log_prefix=str(config_payload.get("log_prefix", "adv-eval")),
            duration_seconds=int(config_payload.get("duration_seconds", 60)),
            actor_id=int(config_payload.get("actor_id", 900)),
            duration_steps=int(config_payload.get("duration_steps", 6000)),
            num_flows=int(config_payload.get("num_flows", 1)),
            save_logs=int(config_payload.get("save_logs", 0)),
            analyze_logs=int(config_payload.get("analyze_logs", 0)),
            mm_adv_bin=config_payload.get("mm_adv_bin"),
            initial_uplink_loss=float(config_payload.get("init_uplink_loss", 0.0)),
            initial_downlink_loss=float(config_payload.get("init_downlink_loss", 0.0)),
            initial_uplink_delay_ms=config_payload.get("init_uplink_delay_ms"),
            initial_downlink_delay_ms=config_payload.get("init_downlink_delay_ms"),
            initial_uplink_queue_packets=config_payload.get("init_uplink_queue_packets"),
            initial_downlink_queue_packets=config_payload.get("init_downlink_queue_packets"),
        ),
        port=int(run_namespace.port_base),
        actor_id=int(run_namespace.actor_id_base),
    )


def _numeric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "avg": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def _current_sage_episode_dir(env: Any) -> str | None:
    children = getattr(env, "_children", None)
    if isinstance(children, dict):
        sage_child = children.get("sage")
        if sage_child is not None and hasattr(sage_child, "_episode_dir"):
            try:
                return str(sage_child._episode_dir())
            except Exception:
                return None
    inner_env = getattr(env, "_inner_env", None)
    if inner_env is not None and hasattr(inner_env, "_episode_dir"):
        try:
            return str(inner_env._episode_dir())
        except Exception:
            return None
    if hasattr(env, "_episode_dir"):
        try:
            return str(env._episode_dir())
        except Exception:
            return None
    return None


def _load_controller_timing_metrics(env: Any) -> dict[str, float]:
    episode_dir = _current_sage_episode_dir(env)
    if not episode_dir:
        return {}
    log_path = os.path.join(str(episode_dir), _CONTROLLER_TIMING_LOG_FILENAME)
    if not os.path.exists(log_path):
        return {}

    controller_values: list[float] = []
    policy_values: list[float] = []
    shield_values: list[float] = []
    with open(log_path, "r", encoding="utf-8") as file_obj:
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
            if isinstance(controller_value, (int, float, np.integer, np.floating)) and np.isfinite(float(controller_value)):
                controller_values.append(float(controller_value))
            if isinstance(policy_value, (int, float, np.integer, np.floating)) and np.isfinite(float(policy_value)):
                policy_values.append(float(policy_value))
            if isinstance(shield_value, (int, float, np.integer, np.floating)) and np.isfinite(float(shield_value)):
                shield_values.append(float(shield_value))

    if not controller_values:
        return {}

    controller_stats = _numeric_summary(controller_values)
    policy_stats = _numeric_summary(policy_values)
    shield_stats = _numeric_summary(shield_values)
    return {
        "controller_decision_count": float(len(controller_values)),
        "controller_decision_time_ms-avg": float(controller_stats["avg"]),
        "controller_decision_time_ms-p50": float(controller_stats["p50"]),
        "controller_decision_time_ms-p95": float(controller_stats["p95"]),
        "policy_decision_time_ms-avg": float(policy_stats["avg"]),
        "policy_decision_time_ms-p50": float(policy_stats["p50"]),
        "policy_decision_time_ms-p95": float(policy_stats["p95"]),
        "shield_decision_time_ms-avg": float(shield_stats["avg"]),
        "shield_decision_time_ms-p50": float(shield_stats["p50"]),
        "shield_decision_time_ms-p95": float(shield_stats["p95"]),
    }


def _write_controller_decision_time_plot(
    *,
    per_episode_rows: list[dict[str, float | str]],
    out_dir: str,
) -> str | None:
    metrics = (
        ("controller_decision_time_ms-avg", "Per-Trace Mean"),
        ("controller_decision_time_ms-p50", "Per-Trace P50"),
        ("controller_decision_time_ms-p95", "Per-Trace P95"),
    )
    trace_types = sorted(
        {
            str(row.get("trace_type"))
            for row in per_episode_rows
            if any(isinstance(row.get(metric_name), (int, float)) for metric_name, _ in metrics)
        }
    )
    if not trace_types:
        return None

    plot_rows: list[tuple[str, list[float]]] = []
    for trace_type in trace_types:
        values: list[float] = []
        typed_rows = [row for row in per_episode_rows if str(row.get("trace_type")) == trace_type]
        for metric_name, _label in metrics:
            metric_values = [
                float(row[metric_name])
                for row in typed_rows
                if isinstance(row.get(metric_name), (int, float, np.integer, np.floating))
            ]
            if not metric_values:
                break
            values.append(float(np.mean(np.asarray(metric_values, dtype=np.float64))))
        if len(values) == len(metrics):
            plot_rows.append((trace_type, values))
    if not plot_rows:
        return None

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    figure, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), max(4.0, 0.55 * len(plot_rows) + 1.8)))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes], dtype=object)
    color = "#377eb8"
    y_positions = np.arange(len(plot_rows), dtype=np.float64)
    labels = [trace_type for trace_type, _values in plot_rows]
    for axis, metric_index in zip(axes.tolist(), range(len(metrics))):
        metric_name, panel_title = metrics[metric_index]
        metric_values = [row_values[metric_index] for _trace_type, row_values in plot_rows]
        axis.barh(
            y_positions,
            metric_values,
            color=color,
            edgecolor="black",
            linewidth=0.8,
        )
        axis.set_yticks(y_positions)
        axis.set_yticklabels(labels if metric_index == 0 else [])
        axis.set_xlabel("Decision Time [ms]")
        axis.set_title(panel_title)
        axis.grid(axis="x", linestyle="--", alpha=0.25)
        for y_position, metric_value in zip(y_positions.tolist(), metric_values):
            axis.text(float(metric_value), float(y_position), f" {metric_value:.3f}", va="center", ha="left", fontsize=8)
    axes[0].set_ylabel("Setup")
    figure.suptitle("Controller Decision Time by Setup")
    figure.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "controller_decision_time_stats.png")
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return out_path


_STEP_AGGREGATE_RECORD_KEYS: dict[str, str] = {
    "reward": "reward",
    "sage_reward": "sage_reward",
    "sage_score": "sage_score",
    "sage_score_rate_norm": "sage_score_rate_norm",
    "sage_score_rtt_norm": "sage_score_rtt_norm",
    "sage_score_loss_norm": "sage_score_loss_norm",
    "sage_score_rate_contrib": "sage_score_rate_contrib",
    "sage_score_rtt_contrib": "sage_score_rtt_contrib",
    "sage_score_loss_penalty": "sage_score_loss_penalty",
    "sage_current_delivery_rate_mbps": "sage_current_delivery_rate_mbps",
    "sage_windowed_rate_mbps": "sage_windowed_rate_mbps",
    "sage_rtt_ms": "sage_rtt_ms",
    "sage_loss_mbps": "sage_loss_mbps",
    "gap_score_sage": "gap_score_sage",
    "gap_score_cubic": "gap_score_cubic",
    "gap_score_bbr": "gap_score_bbr",
    "gap_score_sage_rate_norm": "gap_score_sage_rate_norm",
    "gap_score_sage_rtt_norm": "gap_score_sage_rtt_norm",
    "gap_score_sage_loss_norm": "gap_score_sage_loss_norm",
    "gap_score_sage_rate_contrib": "gap_score_sage_rate_contrib",
    "gap_score_sage_rtt_contrib": "gap_score_sage_rtt_contrib",
    "gap_score_sage_loss_penalty": "gap_score_sage_loss_penalty",
    "gap_score_cubic_rate_norm": "gap_score_cubic_rate_norm",
    "gap_score_cubic_rtt_norm": "gap_score_cubic_rtt_norm",
    "gap_score_cubic_loss_norm": "gap_score_cubic_loss_norm",
    "gap_score_cubic_rate_contrib": "gap_score_cubic_rate_contrib",
    "gap_score_cubic_rtt_contrib": "gap_score_cubic_rtt_contrib",
    "gap_score_cubic_loss_penalty": "gap_score_cubic_loss_penalty",
    "gap_score_bbr_rate_norm": "gap_score_bbr_rate_norm",
    "gap_score_bbr_rtt_norm": "gap_score_bbr_rtt_norm",
    "gap_score_bbr_loss_norm": "gap_score_bbr_loss_norm",
    "gap_score_bbr_rate_contrib": "gap_score_bbr_rate_contrib",
    "gap_score_bbr_rtt_contrib": "gap_score_bbr_rtt_contrib",
    "gap_score_bbr_loss_penalty": "gap_score_bbr_loss_penalty",
    "gap_score_reno": "gap_score_reno",
    "gap_score_reno_rate_norm": "gap_score_reno_rate_norm",
    "gap_score_reno_rtt_norm": "gap_score_reno_rtt_norm",
    "gap_score_reno_loss_norm": "gap_score_reno_loss_norm",
    "gap_score_reno_rate_contrib": "gap_score_reno_rate_contrib",
    "gap_score_reno_rtt_contrib": "gap_score_reno_rtt_contrib",
    "gap_score_reno_loss_penalty": "gap_score_reno_loss_penalty",
    "gap_baseline_score": "gap_baseline_score",
    "gap_baseline_weight_reno": "gap_baseline_weight_reno",
    "gap_baseline_weight_cubic": "gap_baseline_weight_cubic",
    "gap_baseline_weight_bbr": "gap_baseline_weight_bbr",
    "gap_best_baseline_score": "gap_best_baseline_score",
    "gap_best_baseline_gap": "gap_best_baseline_gap",
    "gap_best_baseline_wins": "gap_best_baseline_gap_positive_fraction",
    "gap_value": "gap_value",
    "gap_reward": "gap_reward",
    "baseline_reno_rtt_ms": "baseline_reno_rtt_ms",
    "baseline_reno_rate_mbps": "baseline_reno_rate_mbps",
    "baseline_cubic_rtt_ms": "baseline_cubic_rtt_ms",
    "baseline_cubic_rate_mbps": "baseline_cubic_rate_mbps",
    "baseline_bbr_rtt_ms": "baseline_bbr_rtt_ms",
    "baseline_bbr_rate_mbps": "baseline_bbr_rate_mbps",
    "attacker_shared_bw_mbps": "replay_shared_bw_mbps",
    "attacker_uplink_bw_mbps": "replay_uplink_bw_mbps",
    "attacker_downlink_bw_mbps": "replay_downlink_bw_mbps",
    "mm_up_applied_bw_mbps": "mm_up_applied_bw_mbps",
    "mm_down_applied_bw_mbps": "mm_down_applied_bw_mbps",
    "mm_up_departure_rate_mbps": "mm_up_departure_rate_mbps",
    "mm_down_departure_rate_mbps": "mm_down_departure_rate_mbps",
}


def _summary_stat_key(stat_name: str) -> str:
    return str(stat_name)


def _aggregate_step_record_metrics(step_records: list[dict[str, Any]]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for source_key, target_key in _STEP_AGGREGATE_RECORD_KEYS.items():
        values = [
            float(record[source_key])
            for record in step_records
            if isinstance(record.get(source_key), (int, float, np.floating, np.integer))
        ]
        if not values:
            continue
        stats = _numeric_summary(values)
        payload[f"{target_key}-avg"] = float(stats["avg"])
        payload[f"{target_key}-p50"] = float(stats["p50"])
        payload[f"{target_key}-p95"] = float(stats["p95"])
    return payload


def _augment_result_metrics(result) -> Any:
    step_aggregates = _aggregate_step_record_metrics(result.step_records)
    if not step_aggregates:
        return result
    return replace(
        result,
        metrics={
            **result.metrics,
            **step_aggregates,
        },
    )


def _rename_eval_bandwidth_metrics(result) -> Any:
    rename_map = {
        "attacker_shared_bw_mbps": "replay_shared_bw_mbps",
        "attacker_uplink_bw_mbps": "replay_uplink_bw_mbps",
        "attacker_downlink_bw_mbps": "replay_downlink_bw_mbps",
    }
    renamed_step_records: list[dict[str, Any]] = []
    for record in result.step_records:
        updated = dict(record)
        for source_key, target_key in rename_map.items():
            if source_key in updated:
                updated[target_key] = updated.pop(source_key)
        renamed_step_records.append(updated)

    renamed_metrics = dict(result.metrics)
    for source_key, target_key in rename_map.items():
        if source_key in renamed_metrics:
            renamed_metrics[target_key] = renamed_metrics.pop(source_key)
        for suffix in ("-avg", "-p50", "-p95"):
            metric_key = f"{source_key}{suffix}"
            if metric_key in renamed_metrics:
                renamed_metrics[f"{target_key}{suffix}"] = renamed_metrics.pop(metric_key)

    return replace(
        result,
        metrics=renamed_metrics,
        step_records=renamed_step_records,
    )


def _episode_row(trace_type: str, episode_id: str, result) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "trace_type": trace_type,
        "episode_id": episode_id,
        "episode_total_reward": float(result.total_reward),
        "episode_num_steps": float(result.num_steps),
    }
    for key, value in result.metrics.items():
        row[str(key)] = float(value)
    return row


def _summary_rows(per_episode_rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    trace_types = sorted({str(item["trace_type"]) for item in per_episode_rows})
    for trace_type in trace_types:
        typed_rows = [row for row in per_episode_rows if str(row["trace_type"]) == trace_type]
        metric_names = sorted(
            {
                key
                for row in typed_rows
                for key, value in row.items()
                if key not in {"trace_type", "episode_id"} and isinstance(value, (int, float))
            }
        )
        for metric in metric_names:
            stats = _numeric_summary([float(row[metric]) for row in typed_rows if isinstance(row.get(metric), (int, float))])
            rows.append(
                {
                    "trace_type": trace_type,
                    "metric": metric,
                    "avg": stats["avg"],
                    "p50": stats["p50"],
                    "p95": stats["p95"],
                }
            )
    return rows


def _summary_payload_for_trace_type(summary_rows: list[dict[str, float | str]], trace_type: str) -> dict[str, float]:
    return {
        f"{row['metric']}-{_summary_stat_key(stat)}": float(row[stat])
        for row in summary_rows
        if str(row["trace_type"]) == trace_type
        for stat in ["avg", "p50", "p95"]
    }


def _max_bandwidth_from_schedules(action_schedules: list[list[np.ndarray]]) -> float:
    max_bw = 0.0
    for schedule in action_schedules:
        for action in schedule:
            if action.shape[0] < 2:
                continue
            max_bw = max(max_bw, float(action[0]), float(action[1]))
    return float(max_bw)


def _configured_adv_bandwidth_max(config_payload: dict[str, Any]) -> float | None:
    effective_high = config_payload.get("effective_action_high")
    if isinstance(effective_high, list) and len(effective_high) >= 2:
        try:
            return float(max(float(effective_high[0]), float(effective_high[1])))
        except (TypeError, ValueError):
            return None
    shared_max = config_payload.get("attack_shared_bw_max_mbps")
    if shared_max is not None:
        try:
            return float(shared_max)
        except (TypeError, ValueError):
            return None
    return None


def _validate_adversarial_schedule_bounds(
    *,
    episode_id: str,
    action_schedule: list[np.ndarray],
    config_payload: dict[str, Any],
    tolerance_mbps: float = 1e-3,
) -> None:
    configured_max = _configured_adv_bandwidth_max(config_payload)
    if configured_max is None:
        return
    schedule_max = _max_bandwidth_from_schedules([action_schedule])
    if schedule_max <= configured_max + float(tolerance_mbps):
        return
    raise RuntimeError(
        f"adversarial schedule '{episode_id}' exceeds configured max bandwidth: "
        f"schedule_max={schedule_max:.3f} Mbps > configured_max={configured_max:.3f} Mbps"
    )


def _annotate_replay_bandwidth_metrics(result, *, expected_max_bw_mbps: float) -> Any:
    max_requested = 0.0
    max_applied = 0.0
    for record in result.step_records:
        for key in ("attacker_shared_bw_mbps", "attacker_uplink_bw_mbps", "attacker_downlink_bw_mbps"):
            value = record.get(key)
            if isinstance(value, (int, float, np.floating, np.integer)):
                max_requested = max(max_requested, float(value))
        for key in ("mm_up_applied_bw_mbps", "mm_down_applied_bw_mbps"):
            value = record.get(key)
            if isinstance(value, (int, float, np.floating, np.integer)):
                max_applied = max(max_applied, float(value))
    return replace(
        result,
        metrics={
            **result.metrics,
            "replay_expected_max_bw_mbps": float(expected_max_bw_mbps),
            "replay_requested_max_bw_mbps": float(max_requested),
            "replay_applied_max_bw_mbps": float(max_applied),
        },
    )


def _assert_replay_applied_bandwidth_sane(
    *,
    trace_type: str,
    episode_id: str,
    result,
    tolerance_mbps: float = 1e-3,
) -> None:
    if trace_type == "clean":
        return
    expected_max = float(result.metrics.get("replay_expected_max_bw_mbps", 0.0))
    applied_max = float(result.metrics.get("replay_applied_max_bw_mbps", 0.0))
    requested_max = float(result.metrics.get("replay_requested_max_bw_mbps", 0.0))
    ceiling = max(expected_max, requested_max)
    if applied_max <= ceiling + float(tolerance_mbps):
        return
    raise RuntimeError(
        f"adversarial replay '{episode_id}' applied bandwidth above schedule bounds: "
        f"applied_max={applied_max:.3f} Mbps, requested_max={requested_max:.3f} Mbps, "
        f"expected_max={expected_max:.3f} Mbps"
    )


def _uses_parallel_gap_eval(config_payload: dict[str, Any]) -> bool:
    attack_mode = str(config_payload.get("attack_mode", "independent"))
    return attack_mode in {"independent", "independent_gap"}


def _default_attack_delay_ms(config_payload: dict[str, Any], *, direction: str) -> float:
    init_value = config_payload.get(f"init_{direction}_delay_ms")
    if init_value is not None:
        return float(init_value)
    return float(config_payload.get("latency_ms", 25.0))


def _load_existing_eval_summary(path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    if not os.path.exists(path):
        return [], [], None
    try:
        payload = _load_json(path)
    except Exception:
        return [], [], None
    per_episode = [
        dict(row)
        for row in payload.get("per_episode", [])
        if isinstance(row, dict)
    ]
    evaluation_runs = [
        dict(row)
        for row in payload.get("evaluation_runs", [])
        if isinstance(row, dict)
    ]
    created_at_utc = payload.get("created_at_utc")
    if not isinstance(created_at_utc, str):
        created_at_utc = None
    return per_episode, evaluation_runs, created_at_utc


def _log_episode_to_wandb(
    wandb,
    run,
    *,
    trace_type: str,
    trace_index: int,
    episode_id: str,
    result,
    global_step: int,
) -> int:
    for record in result.step_records:
        payload = {key: float(value) for key, value in record.items() if isinstance(value, (int, float)) and key != "step"}
        payload["trace_index"] = float(trace_index)
        payload["episode_step"] = float(record.get("step", 0))
        payload["trace_type"] = trace_type
        run.log(payload, step=global_step)
        global_step += 1

    episode_payload = {
        key: float(value)
        for key, value in result.metrics.items()
        if isinstance(value, (int, float, np.floating, np.integer))
    }
    episode_payload["trace_index"] = float(trace_index)
    episode_payload["episode_total_reward"] = float(result.total_reward)
    episode_payload["episode_num_steps"] = float(result.num_steps)
    episode_payload["trace_type"] = trace_type
    run.log(episode_payload, step=max(global_step - 1, 0))
    run.summary[f"episodes/{episode_id}"] = float(result.total_reward)
    return global_step


def _evaluate_trace_set(
    *,
    trace_type: str,
    repo_root: str,
    runtime_dir: str,
    config_payload: dict[str, Any],
    launch_config: SageLaunchConfig,
    bounds,
    schedules: list[tuple[str, list[np.ndarray]]],
    wandb,
    wandb_run,
) -> list[Any]:
    if _uses_parallel_gap_eval(config_payload):
        baseline_methods = baseline_methods_from_config(config_payload)
        env = ParallelGapAttackEnv(
            repo_root=repo_root,
            launch_config=launch_config,
            bounds=bounds,
            obs_history_len=int(config_payload.get("obs_history_len", 4)),
            attack_interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
            max_episode_steps=int(config_payload.get("episode_steps", 6000)),
            launch_timeout_s=float(config_payload.get("launch_timeout_s", 90.0)),
            step_timeout_s=float(config_payload.get("step_timeout_s", 10.0)),
            runtime_dir=runtime_dir,
            baseline_gap_alpha=float(config_payload.get("baseline_gap_alpha", 2.0)),
            baseline_hard_max=bool(config_payload.get("baseline_hard_max", False)),
            baseline_methods=baseline_methods,
            smooth_penalty_scale=float(config_payload.get("smooth_penalty_scale", 0.0)),
            sync_guard_ms=float(config_payload.get("sync_guard_ms", 25.0)),
            launch_retries=int(config_payload.get("gap_launch_retries", 6)),
            shared_bin_loss_enabled=bool(config_payload.get("shared_bin_loss_enabled", False)),
            shared_bin_loss_bin_ms=float(config_payload.get("shared_bin_loss_bin_ms", 5.0)),
        )
    else:
        env = IndependentAttackEnv(
            repo_root=repo_root,
            launch_config=launch_config,
            bounds=bounds,
            obs_history_len=int(config_payload.get("obs_history_len", 4)),
            attack_interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
            max_episode_steps=int(config_payload.get("episode_steps", 6000)),
            launch_timeout_s=float(config_payload.get("launch_timeout_s", 90.0)),
            step_timeout_s=float(config_payload.get("step_timeout_s", 10.0)),
            runtime_dir=runtime_dir,
            shared_bandwidth_action=(
                config_payload.get("attack_shared_bw_min_mbps") is not None
                and config_payload.get("attack_shared_bw_max_mbps") is not None
            ),
            shared_loss_action=(
                config_payload.get("attack_shared_loss_min") is not None
                and config_payload.get("attack_shared_loss_max") is not None
            ),
            shared_bin_loss_action=(
                config_payload.get("attack_shared_bin_loss_min_rate") is not None
                and config_payload.get("attack_shared_bin_loss_max_rate") is not None
            ),
            shared_bin_loss_bin_ms=float(config_payload.get("shared_bin_loss_bin_ms", 5.0)),
            shared_delay_action=(
                config_payload.get("attack_shared_delay_min_ms") is not None
                and config_payload.get("attack_shared_delay_max_ms") is not None
            ),
            smooth_penalty_scale=float(config_payload.get("smooth_penalty_scale", 0.0)),
        )
    results: list[Any] = []
    global_step = 0
    try:
        for trace_index, (episode_id, action_schedule) in enumerate(schedules):
            if not action_schedule:
                continue
            if trace_type != "clean":
                _validate_adversarial_schedule_bounds(
                    episode_id=episode_id,
                    action_schedule=action_schedule,
                    config_payload=config_payload,
                )
            result = run_online_policy_episode(
                env,
                action_fn=lambda observation, info, step, schedule=action_schedule: schedule[min(step, len(schedule) - 1)],
                max_steps=len(action_schedule),
                episode_id=episode_id,
            )
            controller_timing_metrics = _load_controller_timing_metrics(env)
            if controller_timing_metrics:
                result = replace(
                    result,
                    metrics={
                        **result.metrics,
                        **controller_timing_metrics,
                    },
                )
            result = _augment_result_metrics(result)
            result = _annotate_replay_bandwidth_metrics(
                result,
                expected_max_bw_mbps=_max_bandwidth_from_schedules([action_schedule]),
            )
            result = _rename_eval_bandwidth_metrics(result)
            _assert_replay_applied_bandwidth_sane(
                trace_type=trace_type,
                episode_id=episode_id,
                result=result,
            )
            results.append(result)
            if wandb is not None and wandb_run is not None:
                global_step = _log_episode_to_wandb(
                    wandb,
                    wandb_run,
                    trace_type=trace_type,
                    trace_index=trace_index,
                    episode_id=episode_id,
                    result=result,
                    global_step=global_step,
                )
    finally:
        env.close()
    return results


def _init_wandb_run(
    wandb,
    *,
    args,
    run_name: str,
    group_name: str,
    generated_manifest_path: str,
    test_manifest_path: str,
    trace_count: int,
    baseline_methods: tuple[str, ...],
) -> Any | None:
    if wandb is None:
        return None
    run = wandb.init(
        project=str(args.wandb_project),
        entity=args.wandb_entity,
        name=run_name,
        group=group_name,
        mode=str(args.wandb_mode),
        tags=[tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()],
        config={
            "generated_manifest_path": generated_manifest_path,
            "test_manifest_path": test_manifest_path,
            "trace_count": trace_count,
            "trace_type": run_name,
            "baseline_methods": list(baseline_methods),
            "shield_rules_file": str(getattr(args, "shield_rules_file", "") or ""),
            "shield_action_delta": float(getattr(args, "shield_action_delta", 0.0)),
            "shield_consecutive_risk": int(getattr(args, "shield_consecutive_risk", 0)),
            "shield_cooldown_steps": int(getattr(args, "shield_cooldown_steps", 0)),
        },
        reinit=True,
    )
    print_wandb_run_links(
        run,
        entity=args.wandb_entity,
        project=str(args.wandb_project),
    )
    return run


def _write_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=str, default=repo_root_from_script(__file__))
    parser.add_argument("--generated-manifest", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--test-manifest", type=str, default="attacks/test/manifest.json")
    parser.add_argument("--skip-clean-rollout", action="store_true")
    parser.add_argument("--clean-only-rollout", action="store_true")
    parser.add_argument("--shield-rules-file", type=str, default=None)
    parser.add_argument("--shield-action-delta", type=float, default=0.15)
    parser.add_argument("--shield-consecutive-risk", type=int, default=2)
    parser.add_argument("--shield-cooldown-steps", type=int, default=2)
    parser.add_argument("--shield-log-path", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default="attacks/runtime")
    parser.add_argument("--out-dir", type=str, default="attacks/output/eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="sage-online-adv")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-tags", type=str, default="")
    args = parser.parse_args()
    if bool(args.skip_clean_rollout) and bool(args.clean_only_rollout):
        raise ValueError("--skip-clean-rollout and --clean-only-rollout cannot be used together")
    if args.generated_manifest is None and not bool(args.clean_only_rollout):
        raise ValueError("--generated-manifest is required unless --clean-only-rollout is set")
    if args.generated_manifest is None and args.config_path is None:
        raise ValueError("--config-path is required when running --clean-only-rollout without --generated-manifest")

    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    if args.generated_manifest is not None:
        generated_manifest_path = resolve_repo_path(repo_root, str(args.generated_manifest))
        generated_manifest = _load_json(generated_manifest_path)
        config_payload = _load_training_config(repo_root, generated_manifest, args.config_path)
        trace_set_name = _trace_set_name(generated_manifest_path, generated_manifest)
    else:
        generated_manifest_path = ""
        generated_manifest = {}
        config_payload = _load_json(resolve_repo_path(repo_root, str(args.config_path)))
        trace_set_name = _trace_set_name_from_config_path(args.config_path)
    shield_enabled = bool(args.shield_rules_file)
    clean_trace_label = _result_trace_label("clean", shield_enabled=shield_enabled)
    adv_trace_label = _result_trace_label(trace_set_name, shield_enabled=shield_enabled)
    run_clean_rollout = not bool(args.skip_clean_rollout)
    run_adv_rollout = not bool(args.clean_only_rollout)
    test_manifest_path = (
        _ensure_test_manifest(repo_root, str(args.test_manifest))
        if run_clean_rollout
        else resolve_repo_path(repo_root, str(args.test_manifest))
    )
    test_entries = load_trace_entries(test_manifest_path) if run_clean_rollout else []
    generated_entries = list(generated_manifest.get("generated_entries", [])) if run_adv_rollout else []
    use_parallel_gap_eval = _uses_parallel_gap_eval(config_payload)
    baseline_methods = baseline_methods_from_config(config_payload)
    clean_uplink_delay_ms = _default_attack_delay_ms(config_payload, direction="uplink")
    clean_downlink_delay_ms = _default_attack_delay_ms(config_payload, direction="downlink")

    clean_schedules: list[tuple[str, list[np.ndarray]]] = []
    if run_clean_rollout:
        for entry in test_entries:
            schedule = load_mahimahi_trace_schedule(
                entry.copied_path,
                interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
            )
            clean_schedules.append(
                (
                    entry.name,
                    build_clean_action_schedule(
                        schedule,
                        uplink_delay_ms=clean_uplink_delay_ms if use_parallel_gap_eval else 0.0,
                        downlink_delay_ms=clean_downlink_delay_ms if use_parallel_gap_eval else 0.0,
                    ),
                )
            )

    adv_schedules: list[tuple[str, list[np.ndarray]]] = []
    if run_adv_rollout:
        for index, generated_entry in enumerate(generated_entries):
            schedule_path = resolve_repo_path(repo_root, str(generated_entry["schedule_path"]))
            schedule_payload = _load_json(schedule_path)
            actions = _load_action_schedule(schedule_payload, config_payload=config_payload)
            if not actions:
                continue
            episode_id = str(generated_entry.get("trace_name") or generated_entry.get("trace_id") or f"generated-{index:03d}")
            adv_schedules.append((episode_id, actions))

    if not clean_schedules and run_clean_rollout:
        raise RuntimeError("no clean schedules were produced from the test manifest")
    if run_adv_rollout and not adv_schedules:
        raise RuntimeError("no adversarial schedules were found in the generated manifest")

    base_bounds = attack_bounds_from_config(config_payload)
    clean_replay_bounds = expand_attack_bounds_for_bandwidth(
        base_bounds,
        _max_bandwidth_from_schedules([actions for _, actions in clean_schedules]) if clean_schedules else 0.0,
    )
    adv_replay_bounds = (
        expand_attack_bounds_for_bandwidth(
            base_bounds,
            _max_bandwidth_from_schedules([actions for _, actions in adv_schedules]),
        )
        if adv_schedules
        else base_bounds
    )

    run_namespace = acquire_run_namespace(
        repo_root=repo_root,
        runtime_dir=str(args.runtime_dir),
        actor_id=int(config_payload.get("actor_id", 900)),
        port=int(config_payload.get("port", 5101)),
        label=str(args.wandb_name or f"eval-{trace_set_name}"),
        ports_per_run=(len(baseline_methods) + 1) if use_parallel_gap_eval else 1,
    )
    launch_config = replace(
        _resolved_launch_config(config_payload=config_payload, run_namespace=run_namespace),
        controller_timing_log_enabled=True,
    )
    if args.shield_rules_file:
        launch_config = replace(
            launch_config,
            shield_rules_file=str(args.shield_rules_file),
            shield_action_delta=float(args.shield_action_delta),
            shield_consecutive_risk=int(args.shield_consecutive_risk),
            shield_cooldown_steps=int(args.shield_cooldown_steps),
            shield_log_path=(str(args.shield_log_path) if args.shield_log_path is not None else None),
        )

    wandb = None
    if bool(args.wandb):
        wandb = try_import_wandb()
        if wandb is None:
            raise RuntimeError("--wandb was set but the wandb package is unavailable")

    group_name = str(args.wandb_name or trace_set_name)
    clean_run = None
    clean_results: list[Any] = []
    if clean_schedules:
        clean_run = _init_wandb_run(
            wandb,
            args=args,
            run_name=clean_trace_label,
            group_name=group_name,
            generated_manifest_path=generated_manifest_path,
            test_manifest_path=test_manifest_path,
            trace_count=len(clean_schedules),
            baseline_methods=baseline_methods,
        )
        clean_results = _evaluate_trace_set(
            trace_type=clean_trace_label,
            repo_root=repo_root,
            runtime_dir=os.path.join(run_namespace.runtime_dir, clean_trace_label),
            config_payload=config_payload,
            launch_config=launch_config,
            bounds=clean_replay_bounds,
            schedules=clean_schedules,
            wandb=wandb,
            wandb_run=clean_run,
        )
        if clean_run is not None:
            clean_summary_rows = _summary_rows(
                [
                    _episode_row(clean_trace_label, episode_id, result)
                    for episode_id, result in zip([item[0] for item in clean_schedules], clean_results)
                ]
            )
            clean_run.summary.update(_summary_payload_for_trace_type(clean_summary_rows, clean_trace_label))
            clean_run.summary["trace_count"] = float(len(clean_results))
            clean_run.finish()
            clean_run = None

    adv_run = None
    adv_results: list[Any] = []
    if adv_schedules:
        adv_run = _init_wandb_run(
            wandb,
            args=args,
            run_name=adv_trace_label,
            group_name=group_name,
            generated_manifest_path=generated_manifest_path,
            test_manifest_path=test_manifest_path,
            trace_count=len(adv_schedules),
            baseline_methods=baseline_methods,
        )
        adv_results = _evaluate_trace_set(
            trace_type=adv_trace_label,
            repo_root=repo_root,
            runtime_dir=os.path.join(run_namespace.runtime_dir, adv_trace_label),
            config_payload=config_payload,
            launch_config=replace(
                launch_config,
                actor_id=int(launch_config.actor_id) + 5000,
                port=int(launch_config.port) + 500,
            ),
            bounds=adv_replay_bounds,
            schedules=adv_schedules,
            wandb=wandb,
            wandb_run=adv_run,
        )

    run_namespace.release()

    if run_adv_rollout and not adv_results:
        raise RuntimeError("evaluation did not produce adversarial results")
    if run_clean_rollout and clean_schedules and not clean_results:
        raise RuntimeError("evaluation did not produce clean results")

    per_episode_rows = []
    per_episode_rows.extend(
        _episode_row(clean_trace_label, episode_id, result)
        for episode_id, result in zip([item[0] for item in clean_schedules], clean_results)
    )
    per_episode_rows.extend(
        _episode_row(adv_trace_label, episode_id, result)
        for episode_id, result in zip([item[0] for item in adv_schedules], adv_results)
    )
    summary_rows = _summary_rows(per_episode_rows)

    out_dir = resolve_repo_path(repo_root, str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    summary_json_path = os.path.join(out_dir, "clean_vs_adv_summary.json")
    summary_csv_path = os.path.join(out_dir, "clean_vs_adv_summary.csv")
    episodes_csv_path = os.path.join(out_dir, "clean_vs_adv_episode_metrics.csv")
    existing_per_episode_rows, evaluation_runs, created_at_utc = _load_existing_eval_summary(summary_json_path)
    current_eval_run = {
        "created_at_utc": utc_now_iso(),
        "generated_manifest_path": generated_manifest_path,
        "test_manifest_path": test_manifest_path,
        "trace_set_name": adv_trace_label,
        "baseline_methods": list(baseline_methods),
        "skip_clean_rollout": bool(args.skip_clean_rollout),
        "clean_only_rollout": bool(args.clean_only_rollout),
        "clean_episode_count": int(len(clean_results)),
        "adv_episode_count": int(len(adv_results)),
    }
    combined_per_episode_rows = list(existing_per_episode_rows) + list(per_episode_rows)
    combined_summary_rows = _summary_rows(combined_per_episode_rows)

    save_json(
        summary_json_path,
        {
            "created_at_utc": created_at_utc or current_eval_run["created_at_utc"],
            "updated_at_utc": current_eval_run["created_at_utc"],
            "generated_manifest_path": generated_manifest_path,
            "test_manifest_path": test_manifest_path,
            "trace_set_name": adv_trace_label,
            "baseline_methods": list(baseline_methods),
            "evaluation_runs": evaluation_runs + [current_eval_run],
            "per_episode": combined_per_episode_rows,
            "summary": combined_summary_rows,
        },
    )
    _write_csv(
        summary_csv_path,
        combined_summary_rows,
        fieldnames=["trace_type", "metric", "avg", "p50", "p95"],
    )

    episode_fieldnames = sorted({key for row in combined_per_episode_rows for key in row.keys()})
    _write_csv(episodes_csv_path, combined_per_episode_rows, fieldnames=episode_fieldnames)
    controller_timing_plot_path = _write_controller_decision_time_plot(
        per_episode_rows=combined_per_episode_rows,
        out_dir=out_dir,
    )

    if adv_run is not None:
        adv_run.summary.update(_summary_payload_for_trace_type(summary_rows, adv_trace_label))
        adv_run.summary["trace_count"] = float(len(adv_results))
        if controller_timing_plot_path is not None:
            adv_run.summary["controller_timing_plot_path"] = str(controller_timing_plot_path)
        adv_run.finish()

    print(summary_csv_path)


if __name__ == "__main__":  # pragma: no cover
    main()
