"""
Generate a per-step Sage shield-learning dataset from clean and adversarial replays.

Example usage:
time python scripts/gen_sage_shield_dataset.py \
  --generated-manifest attacks/adv_traces/gap-constrained-3baselines_300k/generated_manifest.json \
  --clean-manifest attacks/train/manifest.json \
  --out-dir attacks/output/shield-dataset/gap-constrained-3baselines_300k

time python scripts/gen_sage_shield_dataset.py \
  --generated-manifest attacks/adv_traces/bugged/rl-constrained-300k/generated_manifest.json \
  --out-dir attacks/output/shield-dataset/rl-constrained-300k \
  --adv-only-rollout
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
import sys
from typing import Any, Iterable

import numpy as np

from attacks.envs import ParallelGapAttackEnv, baseline_methods_from_config
from attacks.online import SageLaunchConfig, acquire_run_namespace
from sage_rl.shield.features import FEATURE_COLUMNS, ShieldFeatureTracker
from sage_rl.shield.labels import best_baseline_method, hard_gap_percent


if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scripts._trace_attack_common import (
        attack_bounds_from_config,
        build_clean_action_schedule,
        expand_attack_bounds_for_bandwidth,
        load_mahimahi_trace_schedule,
        load_trace_entries,
        materialize_trace_splits,
        repo_root_from_script,
        resolve_repo_path,
        save_json,
        utc_now_iso,
    )
else:
    from ._trace_attack_common import (
        attack_bounds_from_config,
        build_clean_action_schedule,
        expand_attack_bounds_for_bandwidth,
        load_mahimahi_trace_schedule,
        load_trace_entries,
        materialize_trace_splits,
        repo_root_from_script,
        resolve_repo_path,
        save_json,
        utc_now_iso,
    )


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return dict(json.load(file_obj))


def _resolve_existing_path(repo_root: str, path: str | None) -> str | None:
    if not path:
        return None
    resolved = resolve_repo_path(repo_root, str(path))
    return resolved if os.path.exists(resolved) else None


def _search_for_basename(repo_root: str, basename: str, *, search_roots: Iterable[str]) -> str | None:
    target = str(basename).strip()
    if not target:
        return None
    for root in search_roots:
        resolved_root = resolve_repo_path(repo_root, root)
        if os.path.isfile(resolved_root) and os.path.basename(resolved_root) == target:
            return resolved_root
        if not os.path.isdir(resolved_root):
            continue
        for dirpath, _, filenames in os.walk(resolved_root):
            if target in filenames:
                return os.path.join(dirpath, target)
    return None


def _ensure_split_manifest(repo_root: str, manifest_path: str) -> str:
    resolved = resolve_repo_path(repo_root, manifest_path)
    if os.path.exists(resolved):
        return resolved
    materialize_trace_splits(
        repo_root=repo_root,
        train_root=resolve_repo_path(repo_root, "attacks/train"),
        test_root=resolve_repo_path(repo_root, "attacks/test"),
    )
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"missing clean manifest: {resolved}")
    return resolved


def _trace_set_name(generated_manifest_path: str, generated_manifest: dict[str, Any]) -> str:
    manifest_name = generated_manifest.get("trace_set_name")
    if isinstance(manifest_name, str) and manifest_name.strip():
        return manifest_name.strip()
    parent_dir = os.path.dirname(os.path.abspath(generated_manifest_path))
    return os.path.basename(parent_dir) or "generated"


def _resolve_schedule_path(
    repo_root: str,
    *,
    generated_manifest_path: str,
    generated_entry: dict[str, Any],
) -> str:
    manifest_dir = os.path.dirname(os.path.abspath(generated_manifest_path))
    schedule_path = generated_entry.get("schedule_path")
    bundle_dir = generated_entry.get("bundle_dir")
    candidate_paths: list[str] = []
    if schedule_path:
        candidate_paths.append(str(schedule_path))
    if bundle_dir:
        candidate_paths.append(os.path.join(str(bundle_dir), "schedule.json"))

    bundle_leaf = ""
    if bundle_dir:
        bundle_leaf = os.path.basename(os.path.normpath(str(bundle_dir)))
    elif schedule_path:
        bundle_leaf = os.path.basename(os.path.dirname(str(schedule_path)))
    if bundle_leaf:
        candidate_paths.append(os.path.join(manifest_dir, bundle_leaf, "schedule.json"))

    seen: set[str] = set()
    for candidate in candidate_paths:
        normalized = os.path.normpath(str(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved = _resolve_existing_path(repo_root, normalized)
        if resolved is not None:
            return resolved

    if bundle_leaf:
        for dirpath, _, filenames in os.walk(manifest_dir):
            if os.path.basename(dirpath) == bundle_leaf and "schedule.json" in filenames:
                return os.path.join(dirpath, "schedule.json")

    raise FileNotFoundError(
        f"missing schedule bundle for trace entry {generated_entry.get('trace_id') or generated_entry.get('trace_name') or '<unknown>'}"
    )


def _load_training_config(
    repo_root: str,
    generated_manifest_path: str,
    generated_manifest: dict[str, Any],
    config_path: str | None,
) -> tuple[str, dict[str, Any]]:
    search_roots = (
        "attacks/models",
        "attacks/models/bugged",
        "attacks/output/models",
        "attacks/output/models/archive",
    )
    if config_path is not None:
        resolved = _resolve_existing_path(repo_root, config_path)
        if resolved is not None:
            return resolved, _load_json(resolved)
        relocated = _search_for_basename(repo_root, os.path.basename(str(config_path)), search_roots=search_roots)
        if relocated is not None:
            return relocated, _load_json(relocated)
    manifest_training_config = generated_manifest.get("training_config_path")
    if manifest_training_config:
        resolved = _resolve_existing_path(repo_root, str(manifest_training_config))
        if resolved is not None:
            return resolved, _load_json(resolved)
        relocated = _search_for_basename(
            repo_root,
            os.path.basename(str(manifest_training_config)),
            search_roots=search_roots,
        )
        if relocated is not None:
            return relocated, _load_json(relocated)
    generated_entries = generated_manifest.get("generated_entries", [])
    if not generated_entries:
        raise ValueError("generated manifest has no generated entries")
    schedule_payload = _load_json(
        _resolve_schedule_path(
            repo_root,
            generated_manifest_path=generated_manifest_path,
            generated_entry=dict(generated_entries[0]),
        )
    )
    training_config_path = schedule_payload.get("training_config_path")
    if not training_config_path:
        raise ValueError("schedule payload does not include training_config_path")
    resolved = _resolve_existing_path(repo_root, str(training_config_path))
    if resolved is not None:
        return resolved, _load_json(resolved)
    relocated = _search_for_basename(
        repo_root,
        os.path.basename(str(training_config_path)),
        search_roots=search_roots,
    )
    if relocated is not None:
        return relocated, _load_json(relocated)
    raise FileNotFoundError(f"missing training config: {resolve_repo_path(repo_root, str(training_config_path))}")


def _default_attack_delay_ms(config_payload: dict[str, Any], *, direction: str) -> float:
    init_value = config_payload.get(f"init_{direction}_delay_ms")
    if init_value is not None:
        return float(init_value)
    return float(config_payload.get("latency_ms", 25.0))


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

    def _shared_bounds(low_a: float, high_a: float, low_b: float, high_b: float) -> tuple[float, float]:
        shared_low = max(float(low_a), float(low_b))
        shared_high = min(float(high_a), float(high_b))
        if shared_low > shared_high:
            raise ValueError("shared bounds do not overlap")
        return float(shared_low), float(shared_high)

    outer_low: list[float] = []
    outer_high: list[float] = []
    if shared_bandwidth_action:
        if log_shared_bandwidth_action:
            outer_low.append(0.0)
            outer_high.append(1.0)
        else:
            shared_low, shared_high = _shared_bounds(inner_low[0], inner_high[0], inner_low[1], inner_high[1])
            outer_low.append(shared_low)
            outer_high.append(shared_high)
    else:
        outer_low.extend([float(inner_low[0]), float(inner_low[1])])
        outer_high.extend([float(inner_high[0]), float(inner_high[1])])

    if shared_loss_action or shared_bin_loss_action:
        shared_low, shared_high = _shared_bounds(inner_low[2], inner_high[2], inner_low[3], inner_high[3])
        outer_low.append(shared_low)
        outer_high.append(shared_high)
    else:
        outer_low.extend([float(inner_low[2]), float(inner_low[3])])
        outer_high.extend([float(inner_high[2]), float(inner_high[3])])

    if shared_delay_action:
        shared_low, shared_high = _shared_bounds(inner_low[4], inner_high[4], inner_low[5], inner_high[5])
        outer_low.append(shared_low)
        outer_high.append(shared_high)
    else:
        outer_low.extend([float(inner_low[4]), float(inner_low[5])])
        outer_high.extend([float(inner_high[4]), float(inner_high[5])])

    clipped = np.clip(raw, np.asarray(outer_low, dtype=np.float32), np.asarray(outer_high, dtype=np.float32))
    index = 0
    if shared_bandwidth_action:
        if log_shared_bandwidth_action:
            shared_min = max(float(config_payload.get("attack_shared_bw_min_mbps", inner_low[0])), 1e-6)
            shared_max = max(float(config_payload.get("attack_shared_bw_max_mbps", inner_high[0])), shared_min)
            log_min = float(np.log(shared_min))
            log_span = max(float(np.log(shared_max)) - log_min, 1e-6)
            shared_bandwidth = float(np.exp(log_min + float(clipped[index]) * log_span))
        else:
            shared_bandwidth = float(clipped[index])
        uplink_bw = shared_bandwidth
        downlink_bw = shared_bandwidth
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
    return actions


def _resolved_launch_config(*, config_payload: dict[str, Any], run_namespace) -> SageLaunchConfig:
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
            bw2_mbps=int(config_payload.get("bw2_mbps", config_payload.get("env_bw_mbps", 48))),
            trace_period_s=int(config_payload.get("trace_period_s", 7)),
            first_time_mode=0,
            log_prefix=str(config_payload.get("log_prefix", "adv")),
            duration_seconds=int(config_payload.get("duration_seconds", 60)),
            actor_id=int(config_payload.get("actor_id", 0)),
            duration_steps=int(config_payload.get("duration_steps", 6000)),
            num_flows=int(config_payload.get("num_flows", 1)),
            save_logs=int(config_payload.get("save_logs", 0)),
            analyze_logs=int(config_payload.get("analyze_logs", 0)),
            mm_adv_bin=str(config_payload.get("mm_adv_bin")) if config_payload.get("mm_adv_bin") else None,
        ),
        port=int(run_namespace.port_base),
        actor_id=int(run_namespace.actor_id_base),
    )


def _max_bandwidth_from_schedules(schedules: Iterable[list[np.ndarray]]) -> float:
    maximum = 0.0
    for action_schedule in schedules:
        for action in action_schedule:
            array = np.asarray(action, dtype=np.float32).reshape(-1)
            if array.shape[0] >= 2:
                maximum = max(maximum, float(array[0]), float(array[1]))
    return float(maximum)


def _collect_episode_rows(
    *,
    env: ParallelGapAttackEnv,
    trace_type: str,
    setup_name: str,
    episode_id: str,
    action_schedule: list[np.ndarray],
    baseline_methods: tuple[str, ...],
    history_len: int,
) -> list[dict[str, Any]]:
    tracker = ShieldFeatureTracker(history_len=history_len)
    rows: list[dict[str, Any]] = []
    _observation, _info = env.reset()

    for step_index in range(len(action_schedule)):
        action = np.asarray(action_schedule[min(step_index, len(action_schedule) - 1)], dtype=np.float32)
        _observation, reward, terminated, truncated, info = env.step(action)
        row: dict[str, Any] = {
            "setup": str(setup_name),
            "trace_type": str(trace_type),
            "episode_id": str(episode_id),
            "episode_step": int(step_index),
            "reward": float(reward),
            "attacker_reward": float(info.get("attacker/reward", reward)),
            "has_env_error": 1 if "env/error" in info else 0,
            "env_bootstrap_placeholder": int(float(info.get("env/bootstrap_placeholder", 0.0)) > 0.0),
            "env_nonfinite_sage_values": float(info.get("env/nonfinite_sage_values", 0.0)),
        }
        row.update(tracker.update_from_info(info))
        row["hard_gap_value"] = float(info.get("gap/best_baseline_gap", float("nan")))
        row["hard_baseline_score"] = float(info.get("gap/best_baseline_score", float("nan")))
        row["hard_gap_percent"] = float(
            hard_gap_percent(
                best_baseline_gap=float(row["hard_gap_value"]),
                best_baseline_score=float(row["hard_baseline_score"]),
            )
        )
        row["smoothed_gap_value"] = float(info.get("gap/value", float("nan")))
        row["smoothed_baseline_score"] = float(info.get("gap/baseline_score", float("nan")))
        row["smoothed_gap_percent"] = float(
            hard_gap_percent(
                best_baseline_gap=float(row["smoothed_gap_value"]),
                best_baseline_score=float(row["smoothed_baseline_score"]),
            )
        )
        row["sage_score"] = float(info.get("gap/score_sage", float("nan")))
        row["sage_previous_action"] = float(info.get("sage/previous_action", float("nan")))
        row["sage_reward"] = float(info.get("sage/reward", float("nan")))

        for method in baseline_methods:
            row[f"gap_score_{method}"] = float(info.get(f"gap/score_{method}", float("nan")))
            row[f"baseline_rate_{method}_mbps"] = float(info.get(f"baseline/{method}_rate_mbps", float("nan")))
            row[f"baseline_rtt_{method}_ms"] = float(info.get(f"baseline/{method}_rtt_ms", float("nan")))
            row[f"baseline_previous_action_{method}"] = float(
                info.get(f"baseline/{method}_previous_action", float("nan"))
            )

        method = best_baseline_method(row, baseline_methods=baseline_methods)
        row["best_baseline_method"] = str(method or "")
        row["best_baseline_previous_action"] = float(
            row.get(f"baseline_previous_action_{method}", float("nan")) if method else float("nan")
        )
        rows.append(row)

        if terminated or truncated:
            break

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Sage shield-learning dataset from clean/adversarial replays.")
    parser.add_argument("--repo-root", type=str, default=repo_root_from_script(__file__))
    parser.add_argument("--generated-manifest", required=True)
    parser.add_argument(
        "--clean-manifest",
        default="attacks/train/manifest.json",
        help="Manifest for benign clean traces used to build the shield dataset. Defaults to the training split.",
    )
    parser.add_argument(
        "--test-manifest",
        dest="clean_manifest_legacy",
        default=None,
        help="Deprecated alias for --clean-manifest.",
    )
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--runtime-dir", type=str, default="attacks/runtime")
    parser.add_argument("--skip-clean-rollout", action="store_true")
    parser.add_argument(
        "--adv-only-rollout",
        action="store_true",
        help="Alias for --skip-clean-rollout; replay only adversarial schedules and skip clean traces.",
    )
    parser.add_argument("--clean-only-rollout", action="store_true")
    parser.add_argument("--feature-history-len", type=int, default=4)
    args = parser.parse_args()

    skip_clean_rollout = bool(args.skip_clean_rollout) or bool(args.adv_only_rollout)
    if skip_clean_rollout and bool(args.clean_only_rollout):
        raise ValueError("cannot combine clean-only rollout with skip-clean/adv-only rollout")

    repo_root = os.path.abspath(str(args.repo_root))
    out_dir = resolve_repo_path(repo_root, str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    generated_manifest_path = resolve_repo_path(repo_root, str(args.generated_manifest))
    generated_manifest = _load_json(generated_manifest_path)
    config_path, config_payload = _load_training_config(
        repo_root,
        generated_manifest_path,
        generated_manifest,
        args.config_path,
    )
    trace_set_name = _trace_set_name(generated_manifest_path, generated_manifest)
    baseline_methods = baseline_methods_from_config(config_payload)
    clean_manifest_arg = str(args.clean_manifest_legacy or args.clean_manifest)

    run_clean_rollout = not skip_clean_rollout
    run_adv_rollout = not bool(args.clean_only_rollout)
    clean_manifest_path = (
        _ensure_split_manifest(repo_root, clean_manifest_arg)
        if run_clean_rollout
        else resolve_repo_path(repo_root, clean_manifest_arg)
    )

    clean_entries = load_trace_entries(clean_manifest_path) if run_clean_rollout else []
    generated_entries = list(generated_manifest.get("generated_entries", [])) if run_adv_rollout else []

    clean_schedules: list[tuple[str, list[np.ndarray]]] = []
    if run_clean_rollout:
        clean_uplink_delay_ms = _default_attack_delay_ms(config_payload, direction="uplink")
        clean_downlink_delay_ms = _default_attack_delay_ms(config_payload, direction="downlink")
        for entry in clean_entries:
            schedule = load_mahimahi_trace_schedule(
                entry.copied_path,
                interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
            )
            clean_schedules.append(
                (
                    entry.name,
                    build_clean_action_schedule(
                        schedule,
                        uplink_delay_ms=clean_uplink_delay_ms,
                        downlink_delay_ms=clean_downlink_delay_ms,
                    ),
                )
            )

    adv_schedules: list[tuple[str, list[np.ndarray]]] = []
    if run_adv_rollout:
        for index, generated_entry in enumerate(generated_entries):
            schedule_payload = _load_json(
                _resolve_schedule_path(
                    repo_root,
                    generated_manifest_path=generated_manifest_path,
                    generated_entry=dict(generated_entry),
                )
            )
            actions = _load_action_schedule(schedule_payload, config_payload=config_payload)
            if not actions:
                continue
            episode_id = str(generated_entry.get("trace_name") or generated_entry.get("trace_id") or f"generated-{index:03d}")
            adv_schedules.append((episode_id, actions))

    base_bounds = attack_bounds_from_config(config_payload)
    all_schedules = [actions for _, actions in clean_schedules] + [actions for _, actions in adv_schedules]
    replay_bounds = expand_attack_bounds_for_bandwidth(base_bounds, _max_bandwidth_from_schedules(all_schedules))

    run_namespace = acquire_run_namespace(
        repo_root=repo_root,
        runtime_dir=str(args.runtime_dir),
        actor_id=int(config_payload.get("actor_id", 900)),
        port=int(config_payload.get("port", 5101)),
        label=f"shield-dataset-{trace_set_name}",
        ports_per_run=len(baseline_methods) + 1,
    )
    resolved_runtime_dir = run_namespace.runtime_dir
    launch_config = _resolved_launch_config(config_payload=config_payload, run_namespace=run_namespace)

    fieldnames = [
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
    csv_path = os.path.join(out_dir, "sage_shield_dataset.csv")
    summary_payload = {
        "created_at_utc": utc_now_iso(),
        "repo_root": repo_root,
        "generated_manifest_path": generated_manifest_path,
        "training_config_path": config_path,
        "clean_manifest_path": clean_manifest_path,
        "runtime_dir_resolved": resolved_runtime_dir,
        "trace_set_name": trace_set_name,
        "baseline_methods": list(baseline_methods),
        "feature_history_len": int(args.feature_history_len),
        "feature_columns": list(FEATURE_COLUMNS),
        "csv_path": os.path.relpath(csv_path, repo_root),
        "num_clean_episodes": len(clean_schedules),
        "num_adv_episodes": len(adv_schedules),
    }

    total_rows = 0
    env = ParallelGapAttackEnv(
        repo_root=repo_root,
        launch_config=launch_config,
        bounds=replay_bounds,
        obs_history_len=int(config_payload.get("obs_history_len", 4)),
        attack_interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
        max_episode_steps=int(config_payload.get("episode_steps", 6000)),
        launch_timeout_s=float(config_payload.get("launch_timeout_s", 90.0)),
        step_timeout_s=float(config_payload.get("step_timeout_s", 10.0)),
        runtime_dir=resolved_runtime_dir,
        baseline_gap_alpha=float(config_payload.get("baseline_gap_alpha", 2.0)),
        baseline_hard_max=bool(config_payload.get("baseline_hard_max", False)),
        baseline_methods=baseline_methods,
        smooth_penalty_scale=float(config_payload.get("smooth_penalty_scale", 0.0)),
        sync_guard_ms=float(config_payload.get("sync_guard_ms", 25.0)),
        launch_retries=int(config_payload.get("gap_launch_retries", 6)),
        shared_bin_loss_enabled=bool(config_payload.get("shared_bin_loss_enabled", False)),
        shared_bin_loss_bin_ms=float(config_payload.get("shared_bin_loss_bin_ms", 5.0)),
    )
    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()

            for trace_type, schedules in (("clean", clean_schedules), ("adv", adv_schedules)):
                for episode_id, action_schedule in schedules:
                    rows = _collect_episode_rows(
                        env=env,
                        trace_type=trace_type,
                        setup_name=trace_set_name if trace_type == "adv" else "clean",
                        episode_id=episode_id,
                        action_schedule=action_schedule,
                        baseline_methods=baseline_methods,
                        history_len=int(args.feature_history_len),
                    )
                    for row in rows:
                        writer.writerow(row)
                    total_rows += len(rows)
    finally:
        env.close()

    summary_payload["num_rows"] = int(total_rows)
    save_json(os.path.join(out_dir, "sage_shield_dataset_meta.json"), summary_payload)
    print(csv_path)


if __name__ == "__main__":
    main()
