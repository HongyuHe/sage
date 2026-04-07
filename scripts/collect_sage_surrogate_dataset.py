"""
Collect clean-trace replay labels for the Sage clean-trace surrogate.

Example usage:
time python scripts/collect_sage_surrogate_dataset.py \
  --config-path attacks/models/gap_adv_20260321_gap-constrained-all-loss_50ms_300k.config.json \
  --clean-manifest attacks/test/manifest.json \
  --out attacks/output/surrogate-datasets/gap-constrained-all-loss_50ms_300k/clean_trace_surrogate_dataset.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from typing import Any

import numpy as np

from attacks.envs import ParallelGapAttackEnv, baseline_methods_from_config
from attacks.online import SageLaunchConfig, acquire_run_namespace
from attacks.surrogate import (
    load_clean_trace_sequences,
    pad_1d_sequences,
    replay_bounds_for_action_schedules,
)


if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scripts._trace_attack_common import (
        attack_bounds_from_config,
        materialize_trace_splits,
        repo_root_from_script,
        resolve_repo_path,
        run_online_policy_episode,
        save_json,
        utc_now_iso,
    )
else:
    from ._trace_attack_common import (
        attack_bounds_from_config,
        materialize_trace_splits,
        repo_root_from_script,
        resolve_repo_path,
        run_online_policy_episode,
        save_json,
        utc_now_iso,
    )


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _load_training_config(repo_root: str, config_path: str) -> tuple[str, dict[str, Any]]:
    resolved = resolve_repo_path(repo_root, config_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"missing training config: {resolved}")
    return resolved, _load_json(resolved)


def _ensure_clean_manifest(repo_root: str, manifest_path: str) -> str:
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
            log_prefix=str(config_payload.get("log_prefix", "surrogate-dataset")),
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


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect clean-trace replay labels for a Sage trace surrogate.")
    parser.add_argument("--repo-root", type=str, default=repo_root_from_script(__file__))
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--clean-manifest", type=str, default="attacks/test/manifest.json")
    parser.add_argument("--out", type=str, required=True, help="Output .npz path")
    parser.add_argument("--runtime-dir", type=str, default="attacks/runtime")
    parser.add_argument("--num-traces", type=int, default=-1)
    args = parser.parse_args()

    repo_root = os.path.abspath(str(args.repo_root))
    out_path = resolve_repo_path(repo_root, str(args.out))
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    config_path, config_payload = _load_training_config(repo_root, str(args.config_path))
    attack_mode = str(config_payload.get("attack_mode", "independent_gap"))
    if attack_mode != "independent_gap":
        raise ValueError(
            f"clean-trace surrogate collection currently supports only attack_mode=independent_gap, got {attack_mode!r}"
        )

    clean_manifest_path = _ensure_clean_manifest(repo_root, str(args.clean_manifest))
    sequences = load_clean_trace_sequences(
        manifest_path=clean_manifest_path,
        config_payload=config_payload,
        limit=int(args.num_traces),
    )
    if not sequences:
        raise RuntimeError("no clean trace sequences were loaded from the manifest")

    baseline_methods = baseline_methods_from_config(config_payload)
    replay_bounds = replay_bounds_for_action_schedules(
        attack_bounds_from_config(config_payload),
        [sequence.to_action_schedule() for sequence in sequences],
    )
    run_namespace = acquire_run_namespace(
        repo_root=repo_root,
        runtime_dir=str(args.runtime_dir),
        actor_id=int(config_payload.get("actor_id", 900)),
        port=int(config_payload.get("port", 5101)),
        label=f"surrogate-dataset-{os.path.splitext(os.path.basename(out_path))[0]}",
        ports_per_run=len(baseline_methods) + 1,
    )
    resolved_runtime_dir = run_namespace.runtime_dir

    per_trace_rows: list[dict[str, Any]] = []
    env: ParallelGapAttackEnv | None = None
    try:
        env = ParallelGapAttackEnv(
            repo_root=repo_root,
            launch_config=_resolved_launch_config(config_payload=config_payload, run_namespace=run_namespace),
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
        for trace_index, sequence in enumerate(sequences):
            action_schedule = sequence.to_action_schedule()
            result = run_online_policy_episode(
                env,
                action_fn=lambda observation, info, step, schedule=action_schedule: schedule[min(step, len(schedule) - 1)],
                max_steps=len(action_schedule),
                episode_id=sequence.trace_name,
            )
            per_trace_rows.append(
                {
                    "trace_index": int(trace_index),
                    "trace_id": str(sequence.trace_id),
                    "trace_name": str(sequence.trace_name),
                    "num_steps": int(result.num_steps),
                    "episode_total_reward": float(result.total_reward),
                    "reward_mean": _metric_value(result.metrics, "reward_mean"),
                    "gap_reward_mean": _metric_value(result.metrics, "gap_reward_mean"),
                    "gap_value_mean": _metric_value(result.metrics, "gap_value_mean"),
                    "gap_best_baseline_gap_mean": _metric_value(result.metrics, "gap_best_baseline_gap_mean"),
                    "gap_baseline_score_mean": _metric_value(result.metrics, "gap_baseline_score_mean"),
                    "gap_best_baseline_score_mean": _metric_value(result.metrics, "gap_best_baseline_score_mean"),
                    "gap_best_baseline_wins_mean": _metric_value(result.metrics, "gap_best_baseline_wins_mean"),
                }
            )
    finally:
        if env is not None:
            env.close()
        run_namespace.release()

    if len(per_trace_rows) != len(sequences):
        raise RuntimeError("replay rows did not align with clean trace sequences")

    x_shared_bw, x_len = pad_1d_sequences([sequence.shared_bandwidth_mbps for sequence in sequences])
    x_shared_loss, loss_len = pad_1d_sequences([sequence.shared_loss_rate for sequence in sequences])
    if not np.array_equal(x_len, loss_len):
        raise RuntimeError("bandwidth and loss lengths diverged")

    np.savez_compressed(
        out_path,
        trace_id=np.asarray([sequence.trace_id for sequence in sequences]).astype(str),
        trace_name=np.asarray([sequence.trace_name for sequence in sequences]).astype(str),
        source_relative_path=np.asarray(
            [str(sequence.source_trace.get("relative_path", "")) for sequence in sequences]
        ).astype(str),
        X_shared_bw=x_shared_bw.astype(np.float32),
        X_shared_loss=x_shared_loss.astype(np.float32),
        X_len=x_len.astype(np.int64),
        Y_num_steps=np.asarray([row["num_steps"] for row in per_trace_rows], dtype=np.int64),
        Y_episode_total_reward=np.asarray([row["episode_total_reward"] for row in per_trace_rows], dtype=np.float32),
        Y_reward_mean=np.asarray([row["reward_mean"] for row in per_trace_rows], dtype=np.float32),
        Y_gap_reward_mean=np.asarray([row["gap_reward_mean"] for row in per_trace_rows], dtype=np.float32),
        Y_gap_value_mean=np.asarray([row["gap_value_mean"] for row in per_trace_rows], dtype=np.float32),
        Y_gap_best_baseline_gap_mean=np.asarray(
            [row["gap_best_baseline_gap_mean"] for row in per_trace_rows],
            dtype=np.float32,
        ),
        Y_gap_baseline_score_mean=np.asarray(
            [row["gap_baseline_score_mean"] for row in per_trace_rows],
            dtype=np.float32,
        ),
        Y_gap_best_baseline_score_mean=np.asarray(
            [row["gap_best_baseline_score_mean"] for row in per_trace_rows],
            dtype=np.float32,
        ),
        Y_gap_best_baseline_wins_mean=np.asarray(
            [row["gap_best_baseline_wins_mean"] for row in per_trace_rows],
            dtype=np.float32,
        ),
        attack_interval_ms=np.asarray(float(config_payload.get("attack_interval_ms", 100.0)), dtype=np.float32),
        shared_bin_loss_enabled=np.asarray(bool(config_payload.get("shared_bin_loss_enabled", False))),
        training_config_path=np.asarray(config_path),
        clean_manifest_path=np.asarray(clean_manifest_path),
    )

    meta_path = os.path.join(out_dir, "clean_trace_surrogate_dataset_meta.json")
    save_json(
        meta_path,
        {
            "created_at_utc": utc_now_iso(),
            "repo_root": repo_root,
            "training_config_path": config_path,
            "clean_manifest_path": clean_manifest_path,
            "runtime_dir_resolved": resolved_runtime_dir,
            "npz_path": os.path.relpath(out_path, repo_root),
            "num_traces": len(sequences),
            "baseline_methods": list(baseline_methods),
            "attack_mode": attack_mode,
            "max_bandwidth_mbps": float(
                max(float(np.max(sequence.shared_bandwidth_mbps)) for sequence in sequences)
            ),
            "per_trace": per_trace_rows,
        },
    )
    print(out_path)


if __name__ == "__main__":
    main()
