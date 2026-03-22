"""
Generate a per-trace explanation dataset for clean and adversarial schedules.

Example usage:
time python scripts/gen_trace_explanation_dataset.py \
  --generated-manifest attacks/adv_traces/gap-constrained-3baselines_300k/generated_manifest.json \
  --clean-manifest attacks/train/manifest.json \
  --out-dir attacks/output/trace-explanations/gap-constrained-3baselines_300k

time python scripts/gen_trace_explanation_dataset.py \
  --generated-manifest attacks/adv_traces/gap-constrained-1baseline_300k/generated_manifest.json \
  --clean-manifest attacks/train/manifest.json \
  --out-dir attacks/output/trace-explanations/gap-constrained-1baseline_300k
  
time python scripts/gen_trace_explanation_dataset.py \
  --generated-manifest attacks/adv_traces/gap-unconstrained_300k/generated_manifest.json \
  --clean-manifest attacks/train/manifest.json \
  --out-dir attacks/output/trace-explanations/gap-unconstrained_300k
  
time python scripts/gen_trace_explanation_dataset.py \
  --generated-manifest attacks/adv_traces/hotnets19-loss_50ms_300k/generated_manifest.json \
  --clean-manifest attacks/train/manifest.json \
  --out-dir attacks/output/trace-explanations/hotnets19-loss_50ms_300k
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Iterable

import numpy as np

from attacks.analysis import (
    DEFAULT_SHARED_WINDOW_STEPS,
    extract_trace_explanation_features,
    normalize_trace_explanation_window_steps,
    trace_explanation_feature_columns,
    trace_explanation_feature_descriptions,
)
from attacks.envs import ParallelGapAttackEnv, baseline_methods_from_config
from attacks.online import acquire_run_namespace

if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scripts._trace_attack_common import (
        attack_bounds_from_config,
        build_clean_action_schedule,
        expand_attack_bounds_for_bandwidth,
        load_mahimahi_trace_schedule,
        load_trace_entries,
        repo_root_from_script,
        resolve_repo_path,
        save_json,
        utc_now_iso,
    )
    from scripts.gen_sage_shield_dataset import (
        _default_attack_delay_ms,
        _ensure_split_manifest,
        _load_action_schedule,
        _load_json,
        _load_training_config,
        _max_bandwidth_from_schedules,
        _resolve_schedule_path,
        _resolved_launch_config,
        _trace_set_name,
    )
else:
    from ._trace_attack_common import (
        attack_bounds_from_config,
        build_clean_action_schedule,
        expand_attack_bounds_for_bandwidth,
        load_mahimahi_trace_schedule,
        load_trace_entries,
        repo_root_from_script,
        resolve_repo_path,
        save_json,
        utc_now_iso,
    )
    from .gen_sage_shield_dataset import (
        _default_attack_delay_ms,
        _ensure_split_manifest,
        _load_action_schedule,
        _load_json,
        _load_training_config,
        _max_bandwidth_from_schedules,
        _resolve_schedule_path,
        _resolved_launch_config,
        _trace_set_name,
    )


_SUMMARY_METRIC_KEYS: tuple[str, ...] = (
    "hard_gap_value",
    "hard_gap_percent",
    "hard_baseline_score",
    "smoothed_gap_value",
    "smoothed_gap_percent",
    "smoothed_baseline_score",
    "sage_score",
    "attacker_reward",
    "best_minus_sage_rate_contrib",
    "best_minus_sage_rtt_contrib",
    "sage_minus_best_loss_penalty",
)


def _parse_window_steps_arg(value: str | None) -> tuple[int, ...]:
    if value is None or not str(value).strip():
        return normalize_trace_explanation_window_steps(DEFAULT_SHARED_WINDOW_STEPS)
    return normalize_trace_explanation_window_steps(
        [int(part.strip()) for part in str(value).split(",") if part.strip()]
    )


def _aggregate_summary(values: list[float], *, prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if array.size == 0:
        array = np.zeros(1, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_p50": float(np.percentile(array, 50.0)),
        f"{prefix}_p95": float(np.percentile(array, 95.0)),
        f"{prefix}_max": float(np.max(array)),
    }


def _best_method_for_step(info: dict[str, Any], *, baseline_methods: tuple[str, ...]) -> str | None:
    best_method: str | None = None
    best_score = float("-inf")
    for method in baseline_methods:
        score = float(info.get(f"gap/score_{method}", float("nan")))
        if not np.isfinite(score):
            continue
        if best_method is None or score > best_score:
            best_method = str(method)
            best_score = float(score)
    return best_method


def _replay_trace_summary(
    *,
    env: ParallelGapAttackEnv,
    action_schedule: list[np.ndarray],
    episode_id: str,
    trace_type: str,
    setup_name: str,
    baseline_methods: tuple[str, ...],
    attack_interval_ms: float,
    baseline_methods_key: str,
    attack_mode: str,
    shared_window_steps: tuple[int, ...],
) -> dict[str, Any]:
    _observation, _info = env.reset()
    per_metric: dict[str, list[float]] = {key: [] for key in _SUMMARY_METRIC_KEYS}
    best_method_counts = {method: 0 for method in baseline_methods}
    env_error_steps = 0
    terminated_early = 0

    for step_index, step_action in enumerate(action_schedule):
        _observation, reward, terminated, truncated, info = env.step(np.asarray(step_action, dtype=np.float32))
        hard_gap_value = float(info.get("gap/best_baseline_gap", 0.0))
        hard_baseline_score = float(info.get("gap/best_baseline_score", 0.0))
        smoothed_gap_value = float(info.get("gap/value", 0.0))
        smoothed_baseline_score = float(info.get("gap/baseline_score", 0.0))
        per_metric["hard_gap_value"].append(hard_gap_value)
        per_metric["hard_gap_percent"].append(100.0 * hard_gap_value / max(hard_baseline_score, 1e-6))
        per_metric["hard_baseline_score"].append(hard_baseline_score)
        per_metric["smoothed_gap_value"].append(smoothed_gap_value)
        per_metric["smoothed_gap_percent"].append(100.0 * smoothed_gap_value / max(smoothed_baseline_score, 1e-6))
        per_metric["smoothed_baseline_score"].append(smoothed_baseline_score)
        per_metric["sage_score"].append(float(info.get("gap/score_sage", 0.0)))
        per_metric["attacker_reward"].append(float(info.get("attacker/reward", reward)))

        best_method = _best_method_for_step(info, baseline_methods=baseline_methods)
        if best_method is not None:
            best_method_counts[best_method] += 1
            best_rate = float(info.get(f"gap/score_{best_method}_rate_contrib", 0.0))
            best_rtt = float(info.get(f"gap/score_{best_method}_rtt_contrib", 0.0))
            best_loss = float(info.get(f"gap/score_{best_method}_loss_penalty", 0.0))
        else:
            best_rate = 0.0
            best_rtt = 0.0
            best_loss = 0.0
        per_metric["best_minus_sage_rate_contrib"].append(best_rate - float(info.get("gap/score_sage_rate_contrib", 0.0)))
        per_metric["best_minus_sage_rtt_contrib"].append(best_rtt - float(info.get("gap/score_sage_rtt_contrib", 0.0)))
        per_metric["sage_minus_best_loss_penalty"].append(float(info.get("gap/score_sage_loss_penalty", 0.0)) - best_loss)

        if "env/error" in info:
            env_error_steps += 1
        if terminated or truncated:
            if step_index + 1 < len(action_schedule):
                terminated_early = 1
            break

    summary: dict[str, Any] = {
        "setup": str(setup_name),
        "trace_type": str(trace_type),
        "episode_id": str(episode_id),
        "num_replay_steps": int(len(per_metric["hard_gap_value"])),
        "attack_interval_ms": float(attack_interval_ms),
        "baseline_methods_key": str(baseline_methods_key),
        "attack_mode": str(attack_mode),
        "env_error_steps": int(env_error_steps),
        "terminated_early": int(terminated_early),
        "hard_gap_positive_fraction": float(
            np.mean(np.asarray(per_metric["hard_gap_value"], dtype=np.float64) > 0.0)
        ) if per_metric["hard_gap_value"] else 0.0,
    }
    for metric_name, values in per_metric.items():
        summary.update(_aggregate_summary(values, prefix=metric_name))

    total_best_counts = max(sum(best_method_counts.values()), 1)
    dominant_best_method = max(best_method_counts.items(), key=lambda item: item[1])[0] if best_method_counts else ""
    summary["dominant_best_baseline_method"] = str(dominant_best_method)
    for method in baseline_methods:
        summary[f"best_baseline_fraction_{method}"] = float(best_method_counts.get(method, 0)) / float(total_best_counts)

    summary.update(
        extract_trace_explanation_features(
            action_schedule,
            attack_interval_ms=float(attack_interval_ms),
            baseline_methods_key=str(baseline_methods_key),
            attack_mode=str(attack_mode),
            shared_window_steps=shared_window_steps,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a per-trace explanation dataset from clean and adversarial schedules.")
    parser.add_argument("--repo-root", type=str, default=repo_root_from_script(__file__))
    parser.add_argument("--generated-manifest", required=True)
    parser.add_argument("--clean-manifest", default="attacks/train/manifest.json")
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--runtime-dir", type=str, default="attacks/runtime")
    parser.add_argument(
        "--window-steps",
        type=str,
        default=",".join(str(value) for value in DEFAULT_SHARED_WINDOW_STEPS),
        help="Comma-separated shared-bandwidth sliding-window lengths in replay steps for windowed trace features.",
    )
    parser.add_argument("--skip-clean-rollout", action="store_true")
    parser.add_argument("--adv-only-rollout", action="store_true")
    parser.add_argument("--clean-only-rollout", action="store_true")
    args = parser.parse_args()

    skip_clean_rollout = bool(args.skip_clean_rollout) or bool(args.adv_only_rollout)
    if skip_clean_rollout and bool(args.clean_only_rollout):
        raise ValueError("cannot combine clean-only rollout with skip-clean/adv-only rollout")

    repo_root = os.path.abspath(str(args.repo_root))
    out_dir = resolve_repo_path(repo_root, str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    shared_window_steps = _parse_window_steps_arg(args.window_steps)
    feature_columns = trace_explanation_feature_columns(shared_window_steps)
    feature_descriptions = trace_explanation_feature_descriptions(shared_window_steps)

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
    baseline_methods_key = "+".join(str(method) for method in baseline_methods)
    attack_mode = str(config_payload.get("attack_mode", generated_manifest.get("attack_mode", "independent_gap")))

    run_clean_rollout = not skip_clean_rollout
    run_adv_rollout = not bool(args.clean_only_rollout)
    clean_manifest_path = (
        _ensure_split_manifest(repo_root, str(args.clean_manifest))
        if run_clean_rollout
        else resolve_repo_path(repo_root, str(args.clean_manifest))
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
        label=f"trace-explanations-{trace_set_name}",
        ports_per_run=len(baseline_methods) + 1,
    )
    resolved_runtime_dir = run_namespace.runtime_dir
    launch_config = _resolved_launch_config(config_payload=config_payload, run_namespace=run_namespace)

    rows: list[dict[str, Any]] = []
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
        for trace_type, schedules in (("clean", clean_schedules), ("adv", adv_schedules)):
            for episode_id, action_schedule in schedules:
                rows.append(
                    _replay_trace_summary(
                        env=env,
                        action_schedule=action_schedule,
                        episode_id=episode_id,
                        trace_type=trace_type,
                        setup_name=trace_set_name if trace_type == "adv" else "clean",
                        baseline_methods=baseline_methods,
                        attack_interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
                        baseline_methods_key=baseline_methods_key,
                        attack_mode=attack_mode,
                        shared_window_steps=shared_window_steps,
                    )
                )
    finally:
        env.close()

    if not rows:
        raise RuntimeError("no trace explanation rows were generated")

    fieldnames = list(rows[0].keys())
    csv_path = os.path.join(out_dir, "trace_explanation_dataset.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    feature_desc_path = os.path.join(out_dir, "trace_explanation_feature_descriptions.json")
    save_json(feature_desc_path, feature_descriptions)
    summary_payload = {
        "created_at_utc": utc_now_iso(),
        "repo_root": repo_root,
        "generated_manifest_path": generated_manifest_path,
        "training_config_path": config_path,
        "clean_manifest_path": clean_manifest_path,
        "runtime_dir_resolved": resolved_runtime_dir,
        "trace_set_name": trace_set_name,
        "baseline_methods": list(baseline_methods),
        "baseline_methods_key": baseline_methods_key,
        "attack_mode": attack_mode,
        "shared_window_steps": list(shared_window_steps),
        "feature_columns": list(feature_columns),
        "feature_description_path": os.path.relpath(feature_desc_path, repo_root),
        "csv_path": os.path.relpath(csv_path, repo_root),
        "num_rows": len(rows),
        "num_clean_episodes": len(clean_schedules),
        "num_adv_episodes": len(adv_schedules),
    }
    save_json(os.path.join(out_dir, "trace_explanation_dataset_meta.json"), summary_payload)
    print(csv_path)


if __name__ == "__main__":
    main()
