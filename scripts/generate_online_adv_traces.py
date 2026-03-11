"""
Example usage:
python scripts/generate_online_adv_traces.py \
  --model-path attacks/models/gap_adv_20260310_hotnets19_30k.zip \
  --test-manifest attacks/test/manifest.json \
  --out-dir attacks/adv_traces/hotnets19-30k \
  --wandb

python scripts/generate_online_adv_traces.py \
  --model-path attacks/models/gap_adv_20260310_rl-unconstrained_30k.zip \
  --test-manifest attacks/test/manifest.json \
  --out-dir attacks/adv_traces/rl-unconstrained-30k \
  --wandb

Run this after `scripts/train_online_attacker.py`.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys
from typing import Any

import numpy as np

from attacks.envs import ParallelGapAttackEnv
from attacks.online import SageLaunchConfig, acquire_run_namespace


if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scripts._trace_attack_common import (
        IndependentAttackEnv,
        TraceConditionedAttackEnv,
        attack_bounds_from_config,
        load_trace_entries,
        materialize_trace_splits,
        repo_root_from_script,
        resolve_repo_path,
        run_online_policy_episode,
        run_policy_episode,
        save_json,
        try_import_wandb,
        utc_now_iso,
        write_bandwidth_trace,
    )
else:
    from ._trace_attack_common import (
        IndependentAttackEnv,
        TraceConditionedAttackEnv,
        attack_bounds_from_config,
        load_trace_entries,
        materialize_trace_splits,
        repo_root_from_script,
        resolve_repo_path,
        run_online_policy_episode,
        run_policy_episode,
        save_json,
        try_import_wandb,
        utc_now_iso,
        write_bandwidth_trace,
    )


def _require_sb3():
    try:
        from stable_baselines3 import PPO  # type: ignore

        return PPO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("stable-baselines3 is required for adversarial trace generation") from exc


def _load_config(repo_root: str, model_path: str, config_path: str | None) -> tuple[str, dict[str, Any]]:
    if config_path is None:
        candidate = os.path.splitext(model_path)[0] + ".config.json"
        if not os.path.exists(candidate):
            candidate = model_path.replace(".zip", ".config.json")
        config_path = candidate
    resolved = resolve_repo_path(repo_root, config_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"missing training config: {resolved}")
    with open(resolved, "r", encoding="utf-8") as file_obj:
        return resolved, dict(json.load(file_obj))


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


def _attack_mode(config_payload: dict[str, Any]) -> str:
    return str(config_payload.get("attack_mode", "trace_conditioned"))


def _trace_set_name(out_dir: str) -> str:
    return os.path.basename(os.path.abspath(out_dir.rstrip(os.sep))) or "generated"


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
            log_prefix=str(config_payload.get("log_prefix", "adv-generate")),
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


def _legacy_trace_conditioned_generation(
    *,
    args,
    repo_root: str,
    config_path: str,
    config_payload: dict[str, Any],
    test_entries,
    model,
    launch_config: SageLaunchConfig,
    runtime_dir: str,
    out_dir: str,
    wandb,
) -> list[dict[str, Any]]:
    env = TraceConditionedAttackEnv(
        repo_root=repo_root,
        trace_entries=test_entries,
        launch_config=launch_config,
        obs_history_len=int(config_payload.get("obs_history_len", 4)),
        attack_interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
        max_episode_steps=int(config_payload.get("episode_steps", 6000)),
        launch_timeout_s=float(config_payload.get("launch_timeout_s", 90.0)),
        step_timeout_s=float(config_payload.get("step_timeout_s", 10.0)),
        runtime_dir=runtime_dir,
        sample_mode="round_robin",
        seed=int(args.seed),
        bw_scale_min=float(config_payload.get("bw_scale_min", 0.1)),
        bw_scale_max=float(config_payload.get("bw_scale_max", 2.0)),
        effective_bw_cap_mbps=float(config_payload.get("effective_bw_cap_mbps", 2000.0)),
        loss_min=0.0,
        loss_max=float(config_payload.get("loss_max", 0.15)),
        delay_min_ms=0.0,
        delay_max_ms=float(config_payload.get("delay_max_ms", 150.0)),
        reward_rate_weight=float(config_payload.get("reward_rate_weight", 1.0)),
        reward_rtt_weight=float(config_payload.get("reward_rtt_weight", 0.05)),
        reward_loss_weight=float(config_payload.get("reward_loss_weight", 2.0)),
        smooth_penalty_scale=float(config_payload.get("smooth_penalty_scale", 0.0)),
    )

    generated_entries: list[dict[str, Any]] = []
    try:
        for trace_index, entry in enumerate(test_entries):
            def policy_fn(observation: np.ndarray, info: dict[str, Any], step: int) -> np.ndarray:
                action, _ = model.predict(observation, deterministic=bool(args.deterministic))
                return np.asarray(action, dtype=np.float32)

            result = run_policy_episode(env, action_fn=policy_fn, trace_index=trace_index)
            bundle_dir = os.path.join(out_dir, f"{trace_index:03d}-{entry.trace_id}")
            os.makedirs(bundle_dir, exist_ok=True)

            uplink_effective = [float(record["effective_action"][0]) for record in result.step_records]
            downlink_effective = [float(record["effective_action"][1]) for record in result.step_records]
            uplink_trace_path = os.path.join(bundle_dir, "uplink.trace")
            downlink_trace_path = os.path.join(bundle_dir, "downlink.trace")
            write_bandwidth_trace(
                bandwidth_mbps=uplink_effective,
                interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
                out_path=uplink_trace_path,
            )
            write_bandwidth_trace(
                bandwidth_mbps=downlink_effective,
                interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
                out_path=downlink_trace_path,
            )

            schedule_payload = {
                "created_at_utc": utc_now_iso(),
                "attack_mode": "trace_conditioned",
                "model_path": resolve_repo_path(repo_root, str(args.model_path)),
                "training_config_path": config_path,
                "source_trace": entry.to_dict(),
                "attack_interval_ms": float(config_payload.get("attack_interval_ms", 100.0)),
                "num_steps": int(result.num_steps),
                "metrics": result.metrics,
                "steps": result.step_records,
            }
            schedule_path = os.path.join(bundle_dir, "schedule.json")
            save_json(schedule_path, schedule_payload)

            generated_entry = {
                "trace_id": entry.trace_id,
                "trace_name": entry.name,
                "source_trace": entry.to_dict(),
                "bundle_dir": os.path.relpath(bundle_dir, repo_root),
                "schedule_path": os.path.relpath(schedule_path, repo_root),
                "uplink_trace_path": os.path.relpath(uplink_trace_path, repo_root),
                "downlink_trace_path": os.path.relpath(downlink_trace_path, repo_root),
                "metrics": result.metrics,
            }
            generated_entries.append(generated_entry)

            if wandb is not None:
                payload = {
                    "generate/trace_index": float(trace_index),
                    "generate/num_steps": float(result.num_steps),
                    "generate/episode_total_reward": float(result.total_reward),
                    "generate/trace_source_is_pantheon": 1.0 if entry.source_group == "pantheon" else 0.0,
                }
                for key, value in result.metrics.items():
                    payload[f"generate/{key}"] = float(value)
                wandb.log(payload, step=trace_index)
    finally:
        env.close()
    return generated_entries


def _independent_generation(
    *,
    args,
    repo_root: str,
    config_path: str,
    config_payload: dict[str, Any],
    model,
    launch_config: SageLaunchConfig,
    runtime_dir: str,
    out_dir: str,
    test_entries,
    wandb,
) -> list[dict[str, Any]]:
    attack_mode = _attack_mode(config_payload)
    use_gap_objective = attack_mode == "independent_gap"
    bounds = attack_bounds_from_config(config_payload)
    if use_gap_objective:
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
            smooth_penalty_scale=float(config_payload.get("smooth_penalty_scale", 0.0)),
            sync_guard_ms=float(config_payload.get("sync_guard_ms", 25.0)),
            launch_retries=int(config_payload.get("gap_launch_retries", 6)),
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
            reward_rate_weight=float(config_payload.get("reward_rate_weight", 1.0)),
            reward_rtt_weight=float(config_payload.get("reward_rtt_weight", 0.05)),
            reward_loss_weight=float(config_payload.get("reward_loss_weight", 2.0)),
            smooth_penalty_scale=float(config_payload.get("smooth_penalty_scale", 0.0)),
        )

    num_generated_traces = len(test_entries)
    if int(args.num_traces) > 0:
        num_generated_traces = int(args.num_traces)

    generated_entries: list[dict[str, Any]] = []
    try:
        for trace_index in range(num_generated_traces):
            trace_id = f"generated-{trace_index:03d}"

            def policy_fn(observation: np.ndarray, info: dict[str, Any], step: int) -> np.ndarray:
                #* Independent trace generation always samples the policy stochastically.
                action, _ = model.predict(observation, deterministic=False)
                return np.asarray(action, dtype=np.float32)

            result = run_online_policy_episode(
                env,
                action_fn=policy_fn,
                max_steps=int(config_payload.get("episode_steps", 6000)),
                episode_id=trace_id,
            )
            bundle_dir = os.path.join(out_dir, f"{trace_index:03d}-{trace_id}")
            os.makedirs(bundle_dir, exist_ok=True)

            uplink_trace_path = os.path.join(bundle_dir, "uplink.trace")
            downlink_trace_path = os.path.join(bundle_dir, "downlink.trace")
            write_bandwidth_trace(
                bandwidth_mbps=[
                    float(record.get("attacker_uplink_bw_mbps", record.get("uplink_bw_mbps", record["action"][0])))
                    for record in result.step_records
                ],
                interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
                out_path=uplink_trace_path,
            )
            write_bandwidth_trace(
                bandwidth_mbps=[
                    float(record.get("attacker_downlink_bw_mbps", record.get("downlink_bw_mbps", record["action"][1])))
                    for record in result.step_records
                ],
                interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
                out_path=downlink_trace_path,
            )

            schedule_payload = {
                "created_at_utc": utc_now_iso(),
                "attack_mode": attack_mode,
                "trace_id": trace_id,
                "trace_name": trace_id,
                "model_path": resolve_repo_path(repo_root, str(args.model_path)),
                "training_config_path": config_path,
                "attack_interval_ms": float(config_payload.get("attack_interval_ms", 100.0)),
                "num_steps": int(result.num_steps),
                "metrics": result.metrics,
                "steps": result.step_records,
            }
            schedule_path = os.path.join(bundle_dir, "schedule.json")
            save_json(schedule_path, schedule_payload)

            generated_entry = {
                "trace_id": trace_id,
                "trace_name": trace_id,
                "bundle_dir": os.path.relpath(bundle_dir, repo_root),
                "schedule_path": os.path.relpath(schedule_path, repo_root),
                "uplink_trace_path": os.path.relpath(uplink_trace_path, repo_root),
                "downlink_trace_path": os.path.relpath(downlink_trace_path, repo_root),
                "metrics": result.metrics,
            }
            generated_entries.append(generated_entry)

            if wandb is not None:
                payload = {
                    "generate/trace_index": float(trace_index),
                    "generate/num_steps": float(result.num_steps),
                    "generate/episode_total_reward": float(result.total_reward),
                    "generate/policy_deterministic": 0.0,
                }
                for key, value in result.metrics.items():
                    payload[f"generate/{key}"] = float(value)
                wandb.log(payload, step=trace_index)
    finally:
        env.close()
    return generated_entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=str, default=repo_root_from_script(__file__))
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--test-manifest", type=str, default="attacks/test/manifest.json")
    parser.add_argument("--out-dir", type=str, default="attacks/output/generated_traces")
    parser.add_argument("--runtime-dir", type=str, default="attacks/runtime")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--num-traces", type=int, default=-1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="sage-online-adv-gen")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-tags", type=str, default="")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    model_path = resolve_repo_path(repo_root, str(args.model_path))
    config_path, config_payload = _load_config(repo_root, model_path, args.config_path)
    manifest_path = _ensure_test_manifest(repo_root, str(args.test_manifest))
    test_entries = load_trace_entries(manifest_path)
    if int(args.num_traces) > 0 and _attack_mode(config_payload) == "trace_conditioned":
        test_entries = test_entries[: int(args.num_traces)]
    PPO = _require_sb3()
    model = PPO.load(model_path)

    out_dir = resolve_repo_path(repo_root, str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    trace_set_name = _trace_set_name(out_dir)
    run_namespace = acquire_run_namespace(
        repo_root=repo_root,
        runtime_dir=str(args.runtime_dir),
        actor_id=int(config_payload.get("actor_id", 900)),
        port=int(config_payload.get("port", 5101)),
        label=str(args.wandb_name or trace_set_name or "generate-online-adv"),
        ports_per_run=8 if _attack_mode(config_payload) == "independent_gap" else 1,
    )
    resolved_runtime_dir = run_namespace.runtime_dir

    wandb = None
    wandb_run = None
    try:
        if bool(args.wandb):
            wandb = try_import_wandb()
            if wandb is None:
                raise RuntimeError("--wandb was set but the wandb package is unavailable")
            wandb_run = wandb.init(
                project=str(args.wandb_project),
                entity=args.wandb_entity,
                name=args.wandb_name,
                mode=str(args.wandb_mode),
                tags=[tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()],
                config={
                    "model_path": model_path,
                    "config_path": config_path,
                    "test_manifest_resolved": manifest_path,
                    "num_test_traces": len(test_entries),
                    "policy_sampling_deterministic": (
                        False if _attack_mode(config_payload) in {"independent", "independent_gap"} else bool(args.deterministic)
                    ),
                    "trace_set_name": trace_set_name,
                    "attack_mode": _attack_mode(config_payload),
                    "runtime_dir_resolved": resolved_runtime_dir,
                    "run_id": run_namespace.run_id,
                },
            )

        launch_config = _resolved_launch_config(config_payload=config_payload, run_namespace=run_namespace)

        if _attack_mode(config_payload) == "trace_conditioned":
            generated_entries = _legacy_trace_conditioned_generation(
                args=args,
                repo_root=repo_root,
                config_path=config_path,
                config_payload=config_payload,
                test_entries=test_entries,
                model=model,
                launch_config=launch_config,
                runtime_dir=resolved_runtime_dir,
                out_dir=out_dir,
                wandb=wandb,
            )
        else:
            generated_entries = _independent_generation(
                args=args,
                repo_root=repo_root,
                config_path=config_path,
                config_payload=config_payload,
                model=model,
                launch_config=launch_config,
                runtime_dir=resolved_runtime_dir,
                out_dir=out_dir,
                test_entries=test_entries,
                wandb=wandb,
            )

        manifest_payload = {
            "created_at_utc": utc_now_iso(),
            "repo_root": repo_root,
            "trace_set_name": trace_set_name,
            "attack_mode": _attack_mode(config_payload),
            "model_path": model_path,
            "training_config_path": config_path,
            "test_manifest_resolved": manifest_path,
            "attack_interval_ms": float(config_payload.get("attack_interval_ms", 100.0)),
            "num_reference_test_traces": len(test_entries),
            "num_generated_traces": len(generated_entries),
            "generated_entries": generated_entries,
        }
        generated_manifest_path = os.path.join(out_dir, "generated_manifest.json")
        save_json(generated_manifest_path, manifest_payload)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "artifact/generated_manifest_path": generated_manifest_path,
                    "artifact/generated_trace_count": float(len(generated_entries)),
                }
            )
        print(generated_manifest_path)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        run_namespace.release()


if __name__ == "__main__":  # pragma: no cover
    main()
