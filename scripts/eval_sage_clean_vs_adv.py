"""
Run this after `scripts/generate_online_adv_traces.py`.

Example usage:
time python scripts/eval_sage_clean_vs_adv.py \
  --generated-manifest attacks/adv_traces/rl-unconstrained-30k/generated_manifest.json \
  --test-manifest attacks/test/manifest.json \
  --out-dir attacks/output/eval \
  --wandb --wandb-project sage-gap-eval
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


def _load_action_schedule(schedule_payload: dict[str, Any]) -> list[np.ndarray]:
    actions: list[np.ndarray] = []
    for step in schedule_payload.get("steps", []):
        if isinstance(step, dict) and isinstance(step.get("effective_action"), list):
            actions.append(np.asarray(step["effective_action"], dtype=np.float32))
            continue
        if isinstance(step, dict) and isinstance(step.get("action"), list):
            actions.append(np.asarray(step["action"], dtype=np.float32))
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
    "gap_baseline_score": "gap_baseline_score",
    "gap_value": "gap_value",
    "gap_reward": "gap_reward",
    "baseline_cubic_rate_mbps": "baseline_cubic_rate_mbps",
    "baseline_bbr_rate_mbps": "baseline_bbr_rate_mbps",
    "attacker_uplink_bw_mbps": "attacker_uplink_bw_mbps",
    "attacker_downlink_bw_mbps": "attacker_downlink_bw_mbps",
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


def _max_bandwidth_from_schedules(action_schedules: list[list[np.ndarray]]) -> float:
    max_bw = 0.0
    for schedule in action_schedules:
        for action in schedule:
            if action.shape[0] < 2:
                continue
            max_bw = max(max_bw, float(action[0]), float(action[1]))
    return float(max_bw)


def _log_episode_to_wandb(wandb, run, *, trace_index: int, episode_id: str, result, global_step: int) -> int:
    for record in result.step_records:
        payload = {key: float(value) for key, value in record.items() if isinstance(value, (int, float)) and key != "step"}
        payload["trace_index"] = float(trace_index)
        payload["episode_step"] = float(record.get("step", 0))
        wandb.log(payload, step=global_step)
        global_step += 1

    episode_payload = {
        key: float(value)
        for key, value in result.metrics.items()
        if isinstance(value, (int, float, np.floating, np.integer))
    }
    episode_payload["trace_index"] = float(trace_index)
    episode_payload["episode_total_reward"] = float(result.total_reward)
    episode_payload["episode_num_steps"] = float(result.num_steps)
    wandb.log(episode_payload, step=max(global_step - 1, 0))
    run.summary[f"episodes/{episode_id}"] = float(result.total_reward)
    return global_step


def _evaluate_trace_set(
    *,
    repo_root: str,
    runtime_dir: str,
    config_payload: dict[str, Any],
    launch_config: SageLaunchConfig,
    bounds,
    schedules: list[tuple[str, list[np.ndarray]]],
    wandb,
    wandb_run,
) -> list[Any]:
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
            result = run_online_policy_episode(
                env,
                action_fn=lambda observation, info, step, schedule=action_schedule: schedule[min(step, len(schedule) - 1)],
                max_steps=len(action_schedule),
                episode_id=episode_id,
            )
            result = _augment_result_metrics(result)
            results.append(result)
            if wandb is not None and wandb_run is not None:
                global_step = _log_episode_to_wandb(
                    wandb,
                    wandb_run,
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
) -> Any | None:
    if wandb is None:
        return None
    return wandb.init(
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
        },
        reinit=True,
    )


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
    parser.add_argument("--generated-manifest", type=str, required=True)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--test-manifest", type=str, default="attacks/test/manifest.json")
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

    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    generated_manifest_path = resolve_repo_path(repo_root, str(args.generated_manifest))
    generated_manifest = _load_json(generated_manifest_path)
    config_payload = _load_training_config(repo_root, generated_manifest, args.config_path)
    test_manifest_path = _ensure_test_manifest(repo_root, str(args.test_manifest))
    test_entries = load_trace_entries(test_manifest_path)
    generated_entries = list(generated_manifest.get("generated_entries", []))
    trace_set_name = _trace_set_name(generated_manifest_path, generated_manifest)

    clean_schedules: list[tuple[str, list[np.ndarray]]] = []
    for entry in test_entries:
        schedule = load_mahimahi_trace_schedule(
            entry.copied_path,
            interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
        )
        clean_schedules.append((entry.name, build_clean_action_schedule(schedule)))

    adv_schedules: list[tuple[str, list[np.ndarray]]] = []
    for index, generated_entry in enumerate(generated_entries):
        schedule_path = resolve_repo_path(repo_root, str(generated_entry["schedule_path"]))
        schedule_payload = _load_json(schedule_path)
        actions = _load_action_schedule(schedule_payload)
        if not actions:
            continue
        episode_id = str(generated_entry.get("trace_name") or generated_entry.get("trace_id") or f"generated-{index:03d}")
        adv_schedules.append((episode_id, actions))

    if not clean_schedules:
        raise RuntimeError("no clean schedules were produced from the test manifest")
    if not adv_schedules:
        raise RuntimeError("no adversarial schedules were found in the generated manifest")

    base_bounds = attack_bounds_from_config(config_payload)
    replay_bounds = expand_attack_bounds_for_bandwidth(
        base_bounds,
        max(
            _max_bandwidth_from_schedules([actions for _, actions in clean_schedules]),
            _max_bandwidth_from_schedules([actions for _, actions in adv_schedules]),
        ),
    )

    run_namespace = acquire_run_namespace(
        repo_root=repo_root,
        runtime_dir=str(args.runtime_dir),
        actor_id=int(config_payload.get("actor_id", 900)),
        port=int(config_payload.get("port", 5101)),
        label=str(args.wandb_name or f"eval-{trace_set_name}"),
    )
    launch_config = _resolved_launch_config(config_payload=config_payload, run_namespace=run_namespace)

    wandb = None
    if bool(args.wandb):
        wandb = try_import_wandb()
        if wandb is None:
            raise RuntimeError("--wandb was set but the wandb package is unavailable")

    group_name = str(args.wandb_name or trace_set_name)
    clean_run = _init_wandb_run(
        wandb,
        args=args,
        run_name="clean",
        group_name=group_name,
        generated_manifest_path=generated_manifest_path,
        test_manifest_path=test_manifest_path,
        trace_count=len(clean_schedules),
    )
    clean_results = _evaluate_trace_set(
        repo_root=repo_root,
        runtime_dir=os.path.join(run_namespace.runtime_dir, "clean"),
        config_payload=config_payload,
        launch_config=launch_config,
        bounds=replay_bounds,
        schedules=clean_schedules,
        wandb=wandb,
        wandb_run=clean_run,
    )

    adv_run = _init_wandb_run(
        wandb,
        args=args,
        run_name=trace_set_name,
        group_name=group_name,
        generated_manifest_path=generated_manifest_path,
        test_manifest_path=test_manifest_path,
        trace_count=len(adv_schedules),
    )
    adv_results = _evaluate_trace_set(
        repo_root=repo_root,
        runtime_dir=os.path.join(run_namespace.runtime_dir, trace_set_name),
        config_payload=config_payload,
        launch_config=replace(
            launch_config,
            actor_id=int(launch_config.actor_id) + 5000,
            port=int(launch_config.port) + 500,
        ),
        bounds=replay_bounds,
        schedules=adv_schedules,
        wandb=wandb,
        wandb_run=adv_run,
    )

    run_namespace.release()

    if not clean_results or not adv_results:
        raise RuntimeError("evaluation did not produce both clean and adversarial results")

    per_episode_rows = []
    per_episode_rows.extend(_episode_row("clean", episode_id, result) for episode_id, result in zip([item[0] for item in clean_schedules], clean_results))
    per_episode_rows.extend(
        _episode_row(trace_set_name, episode_id, result) for episode_id, result in zip([item[0] for item in adv_schedules], adv_results)
    )
    summary_rows = _summary_rows(per_episode_rows)

    out_dir = resolve_repo_path(repo_root, str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    summary_json_path = os.path.join(out_dir, "clean_vs_adv_summary.json")
    summary_csv_path = os.path.join(out_dir, "clean_vs_adv_summary.csv")
    episodes_csv_path = os.path.join(out_dir, "clean_vs_adv_episode_metrics.csv")

    save_json(
        summary_json_path,
        {
            "created_at_utc": utc_now_iso(),
            "generated_manifest_path": generated_manifest_path,
            "test_manifest_path": test_manifest_path,
            "trace_set_name": trace_set_name,
            "per_episode": per_episode_rows,
            "summary": summary_rows,
        },
    )
    _write_csv(
        summary_csv_path,
        summary_rows,
        fieldnames=["trace_type", "metric", "avg", "p50", "p95"],
    )

    episode_fieldnames = sorted({key for row in per_episode_rows for key in row.keys()})
    _write_csv(episodes_csv_path, per_episode_rows, fieldnames=episode_fieldnames)

    if clean_run is not None:
        summary_payload = {
            f"{row['metric']}-{_summary_stat_key(stat)}": float(row[stat])
            for row in summary_rows
            if str(row["trace_type"]) == "clean"
            for stat in ["avg", "p50", "p95"]
        }
        clean_run.summary.update(summary_payload)
        clean_run.summary["trace_count"] = float(len(clean_results))
        clean_run.finish()
    if adv_run is not None:
        summary_payload = {
            f"{row['metric']}-{_summary_stat_key(stat)}": float(row[stat])
            for row in summary_rows
            if str(row["trace_type"]) == trace_set_name
            for stat in ["avg", "p50", "p95"]
        }
        adv_run.summary.update(summary_payload)
        adv_run.summary["trace_count"] = float(len(adv_results))
        adv_run.finish()

    print(summary_csv_path)


if __name__ == "__main__":  # pragma: no cover
    main()
