"""
Generate random online attack traces in the same bundle / manifest format as
`scripts/generate_online_adv_traces.py`.

Example usage:
time python scripts/generate_random_online_traces.py \
  --config-path attacks/models/gap_adv_20260321_gap-constrained-all-loss_50ms_300k.config.json \
  --test-manifest attacks/test/manifest.json \
  --out-dir attacks/adv_traces/random_gap-constrained-all-loss_50ms_300k \
  --seed 42 \
  --shared-bw-min-mbps 5 --shared-bw-max-mbps 150 \
  --shared-loss-min 0.0 --shared-loss-max 0.02 \
  --segment-steps-min 1 --segment-steps-max 8

time python scripts/generate_random_online_traces.py \
  --config-path attacks/output/checkpoints/gap_adv_20260321-005628_gap-unconstrained-all-loss-20260321-055628-p1647083-s0001.config.json \
  --test-manifest attacks/test/manifest.json \
  --out-dir attacks/adv_traces/random_gap-unconstrained-all-loss_50ms_300k \
  --seed 42 \
  --shared-bw-min-mbps 0.5 --shared-bw-max-mbps 2000 \
  --shared-loss-min 0.0 --shared-loss-max 0.5 \
  --segment-steps-min 1 --segment-steps-max 8

This preserves the rollout environment, attack interval, episode length, and
schedule serialization used for adversarial generation. The only difference is
that actions are sampled randomly within controlled ranges instead of coming
from a trained policy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
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
        load_trace_entries,
        materialize_trace_splits,
        print_wandb_run_links,
        repo_root_from_script,
        resolve_repo_path,
        run_online_policy_episode,
        save_json,
        try_import_wandb,
        utc_now_iso,
        write_bandwidth_trace,
    )
else:
    from ._trace_attack_common import (
        IndependentAttackEnv,
        attack_bounds_from_config,
        load_trace_entries,
        materialize_trace_splits,
        print_wandb_run_links,
        repo_root_from_script,
        resolve_repo_path,
        run_online_policy_episode,
        save_json,
        try_import_wandb,
        utc_now_iso,
        write_bandwidth_trace,
    )


@dataclass(frozen=True)
class RandomEffectiveRanges:
    low: np.ndarray
    high: np.ndarray
    sample_shared_bandwidth: bool
    sample_shared_loss: bool
    fixed_uplink_delay_ms: float
    fixed_downlink_delay_ms: float


def _load_config(repo_root: str, config_path: str) -> tuple[str, dict[str, Any]]:
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
            log_prefix=str(config_payload.get("log_prefix", "random-generate")),
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


def _default_delay_ms(launch_config: SageLaunchConfig, *, direction: str) -> float:
    configured = getattr(launch_config, f"initial_{direction}_delay_ms")
    if configured is not None:
        return float(configured)
    return float(launch_config.latency_ms)


def _shared_bandwidth_enabled(config_payload: dict[str, Any]) -> bool:
    attack_mode = _attack_mode(config_payload)
    return attack_mode == "independent_gap" or (
        config_payload.get("attack_shared_bw_min_mbps") is not None
        and config_payload.get("attack_shared_bw_max_mbps") is not None
    )


def _shared_loss_enabled(config_payload: dict[str, Any]) -> bool:
    return (
        config_payload.get("attack_shared_loss_min") is not None
        and config_payload.get("attack_shared_loss_max") is not None
    ) or bool(config_payload.get("shared_bin_loss_enabled", False))


def _validate_segment_bounds(segment_steps_min: int, segment_steps_max: int) -> tuple[int, int]:
    if int(segment_steps_min) < 1:
        raise ValueError("--segment-steps-min must be >= 1")
    if int(segment_steps_max) < int(segment_steps_min):
        raise ValueError("--segment-steps-max must be >= --segment-steps-min")
    return int(segment_steps_min), int(segment_steps_max)


def _validate_override_range(
    *,
    name: str,
    low: float,
    high: float,
    base_low: float,
    base_high: float,
) -> tuple[float, float]:
    if float(low) > float(high):
        raise ValueError(f"{name} min must be <= max")
    if float(low) < float(base_low) - 1e-9 or float(high) > float(base_high) + 1e-9:
        raise ValueError(
            f"{name} range [{low}, {high}] lies outside config bounds [{base_low}, {base_high}]"
        )
    return float(low), float(high)


def _apply_pair_override(
    *,
    low: np.ndarray,
    high: np.ndarray,
    indices: tuple[int, int],
    min_value: float | None,
    max_value: float | None,
    name: str,
) -> None:
    if min_value is None and max_value is None:
        return
    if min_value is None or max_value is None:
        raise ValueError(f"{name} min and max must be set together")
    idx_a, idx_b = indices
    validated_low, validated_high = _validate_override_range(
        name=name,
        low=float(min_value),
        high=float(max_value),
        base_low=max(float(low[idx_a]), float(low[idx_b])),
        base_high=min(float(high[idx_a]), float(high[idx_b])),
    )
    low[idx_a] = validated_low
    low[idx_b] = validated_low
    high[idx_a] = validated_high
    high[idx_b] = validated_high


def _apply_single_override(
    *,
    low: np.ndarray,
    high: np.ndarray,
    index: int,
    min_value: float | None,
    max_value: float | None,
    name: str,
) -> None:
    if min_value is None and max_value is None:
        return
    if min_value is None or max_value is None:
        raise ValueError(f"{name} min and max must be set together")
    validated_low, validated_high = _validate_override_range(
        name=name,
        low=float(min_value),
        high=float(max_value),
        base_low=float(low[index]),
        base_high=float(high[index]),
    )
    low[index] = validated_low
    high[index] = validated_high


def _random_effective_ranges(
    *,
    config_payload: dict[str, Any],
    launch_config: SageLaunchConfig,
    args,
) -> RandomEffectiveRanges:
    bounds = attack_bounds_from_config(config_payload)
    low = np.asarray(
        [
            float(bounds.uplink_bw_mbps[0]),
            float(bounds.downlink_bw_mbps[0]),
            float(bounds.uplink_loss[0]),
            float(bounds.downlink_loss[0]),
            _default_delay_ms(launch_config, direction="uplink"),
            _default_delay_ms(launch_config, direction="downlink"),
        ],
        dtype=np.float32,
    )
    high = np.asarray(
        [
            float(bounds.uplink_bw_mbps[1]),
            float(bounds.downlink_bw_mbps[1]),
            float(bounds.uplink_loss[1]),
            float(bounds.downlink_loss[1]),
            _default_delay_ms(launch_config, direction="uplink"),
            _default_delay_ms(launch_config, direction="downlink"),
        ],
        dtype=np.float32,
    )

    shared_bandwidth = _shared_bandwidth_enabled(config_payload)
    shared_loss = _shared_loss_enabled(config_payload)

    if (
        args.shared_bw_min_mbps is not None or args.shared_bw_max_mbps is not None
    ) and any(
        value is not None
        for value in (
            args.uplink_bw_min_mbps,
            args.uplink_bw_max_mbps,
            args.downlink_bw_min_mbps,
            args.downlink_bw_max_mbps,
        )
    ):
        raise ValueError("shared bandwidth overrides are mutually exclusive with per-direction bandwidth overrides")
    if (
        args.shared_loss_min is not None or args.shared_loss_max is not None
    ) and any(
        value is not None
        for value in (
            args.uplink_loss_min,
            args.uplink_loss_max,
            args.downlink_loss_min,
            args.downlink_loss_max,
        )
    ):
        raise ValueError("shared loss overrides are mutually exclusive with per-direction loss overrides")

    if shared_bandwidth or args.shared_bw_min_mbps is not None or args.shared_bw_max_mbps is not None:
        _apply_pair_override(
            low=low,
            high=high,
            indices=(0, 1),
            min_value=args.shared_bw_min_mbps if args.shared_bw_min_mbps is not None else float(low[0]),
            max_value=args.shared_bw_max_mbps if args.shared_bw_max_mbps is not None else float(high[0]),
            name="shared bandwidth",
        )
        shared_bandwidth = True
    else:
        _apply_single_override(
            low=low,
            high=high,
            index=0,
            min_value=args.uplink_bw_min_mbps,
            max_value=args.uplink_bw_max_mbps,
            name="uplink bandwidth",
        )
        _apply_single_override(
            low=low,
            high=high,
            index=1,
            min_value=args.downlink_bw_min_mbps,
            max_value=args.downlink_bw_max_mbps,
            name="downlink bandwidth",
        )

    if shared_loss or args.shared_loss_min is not None or args.shared_loss_max is not None:
        _apply_pair_override(
            low=low,
            high=high,
            indices=(2, 3),
            min_value=args.shared_loss_min if args.shared_loss_min is not None else float(low[2]),
            max_value=args.shared_loss_max if args.shared_loss_max is not None else float(high[2]),
            name="shared loss",
        )
        shared_loss = True
    else:
        _apply_single_override(
            low=low,
            high=high,
            index=2,
            min_value=args.uplink_loss_min,
            max_value=args.uplink_loss_max,
            name="uplink loss",
        )
        _apply_single_override(
            low=low,
            high=high,
            index=3,
            min_value=args.downlink_loss_min,
            max_value=args.downlink_loss_max,
            name="downlink loss",
        )

    if np.any(low > high):
        raise ValueError("random effective-action ranges are invalid after applying overrides")
    return RandomEffectiveRanges(
        low=low.astype(np.float32, copy=False),
        high=high.astype(np.float32, copy=False),
        sample_shared_bandwidth=bool(shared_bandwidth),
        sample_shared_loss=bool(shared_loss),
        fixed_uplink_delay_ms=float(low[4]),
        fixed_downlink_delay_ms=float(low[5]),
    )


class RandomSchedulePolicy:
    def __init__(
        self,
        *,
        env,
        rng: np.random.Generator,
        ranges: RandomEffectiveRanges,
        segment_steps_min: int,
        segment_steps_max: int,
    ) -> None:
        self._env = env
        self._rng = rng
        self._ranges = ranges
        self._segment_steps_min = int(segment_steps_min)
        self._segment_steps_max = int(segment_steps_max)
        self._remaining_segment_steps = 0
        self._current_policy_action: np.ndarray | None = None

    def _sample_uniform(self, low: float, high: float) -> float:
        if abs(float(high) - float(low)) <= 1e-12:
            return float(low)
        return float(self._rng.uniform(float(low), float(high)))

    def _sample_effective_action(self) -> np.ndarray:
        action = np.zeros((6,), dtype=np.float32)
        if self._ranges.sample_shared_bandwidth:
            shared_bw = self._sample_uniform(self._ranges.low[0], self._ranges.high[0])
            action[0] = float(shared_bw)
            action[1] = float(shared_bw)
        else:
            action[0] = self._sample_uniform(self._ranges.low[0], self._ranges.high[0])
            action[1] = self._sample_uniform(self._ranges.low[1], self._ranges.high[1])
        if self._ranges.sample_shared_loss:
            shared_loss = self._sample_uniform(self._ranges.low[2], self._ranges.high[2])
            action[2] = float(shared_loss)
            action[3] = float(shared_loss)
        else:
            action[2] = self._sample_uniform(self._ranges.low[2], self._ranges.high[2])
            action[3] = self._sample_uniform(self._ranges.low[3], self._ranges.high[3])
        action[4] = float(self._ranges.fixed_uplink_delay_ms)
        action[5] = float(self._ranges.fixed_downlink_delay_ms)
        return action.astype(np.float32, copy=False)

    def _policy_action_from_effective(self, effective_action: np.ndarray) -> np.ndarray:
        if not hasattr(self._env, "_expand_effective_action"):
            raise TypeError("random online trace generation only supports independent or independent_gap envs")
        _, policy_action = self._env._expand_effective_action(np.asarray(effective_action, dtype=np.float32))
        return np.asarray(policy_action, dtype=np.float32)

    def __call__(self, observation: np.ndarray, info: dict[str, Any], step: int) -> np.ndarray:
        if self._current_policy_action is None or self._remaining_segment_steps <= 0:
            effective_action = self._sample_effective_action()
            self._current_policy_action = self._policy_action_from_effective(effective_action)
            self._remaining_segment_steps = int(
                self._rng.integers(self._segment_steps_min, self._segment_steps_max + 1)
            )
        self._remaining_segment_steps -= 1
        return np.asarray(self._current_policy_action, dtype=np.float32)


def _build_env(
    *,
    repo_root: str,
    config_payload: dict[str, Any],
    launch_config: SageLaunchConfig,
    runtime_dir: str,
):
    attack_mode = _attack_mode(config_payload)
    bounds = attack_bounds_from_config(config_payload)
    if attack_mode == "independent_gap":
        baseline_methods = baseline_methods_from_config(config_payload)
        return ParallelGapAttackEnv(
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
    if attack_mode == "independent":
        return IndependentAttackEnv(
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
    raise ValueError(
        f"random online trace generation only supports independent and independent_gap attack modes, got {attack_mode}"
    )


def _write_generated_bundle(
    *,
    repo_root: str,
    out_dir: str,
    trace_index: int,
    trace_id: str,
    result,
    config_path: str,
    config_payload: dict[str, Any],
    args,
) -> dict[str, Any]:
    bundle_dir = os.path.join(out_dir, f"{trace_index:03d}-{trace_id}")
    os.makedirs(bundle_dir, exist_ok=True)

    uplink_trace_path = os.path.join(bundle_dir, "uplink.trace")
    downlink_trace_path = os.path.join(bundle_dir, "downlink.trace")
    write_bandwidth_trace(
        bandwidth_mbps=[
            float(
                record.get(
                    "attacker_uplink_bw_mbps",
                    record.get(
                        "uplink_bw_mbps",
                        (
                            record.get("effective_action", record["action"])[0]
                            if record.get("effective_action", record["action"])
                            else 0.0
                        ),
                    ),
                )
            )
            for record in result.step_records
        ],
        interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
        out_path=uplink_trace_path,
    )
    write_bandwidth_trace(
        bandwidth_mbps=[
            float(
                record.get(
                    "attacker_downlink_bw_mbps",
                    record.get(
                        "downlink_bw_mbps",
                        (
                            record.get("effective_action", record["action"])[1]
                            if len(record.get("effective_action", record["action"])) > 1
                            else record.get("effective_action", record["action"])[0]
                        ),
                    ),
                )
            )
            for record in result.step_records
        ],
        interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
        out_path=downlink_trace_path,
    )

    schedule_payload = {
        "created_at_utc": utc_now_iso(),
        "attack_mode": _attack_mode(config_payload),
        "trace_id": trace_id,
        "trace_name": trace_id,
        "model_path": None,
        "training_config_path": config_path,
        "attack_interval_ms": float(config_payload.get("attack_interval_ms", 100.0)),
        "shared_bin_loss_enabled": bool(config_payload.get("shared_bin_loss_enabled", False)),
        "shared_bin_loss_bin_ms": float(config_payload.get("shared_bin_loss_bin_ms", 5.0)),
        "num_steps": int(result.num_steps),
        "metrics": result.metrics,
        "steps": result.step_records,
    }
    schedule_path = os.path.join(bundle_dir, "schedule.json")
    save_json(schedule_path, schedule_payload)

    return {
        "trace_id": trace_id,
        "trace_name": trace_id,
        "bundle_dir": os.path.relpath(bundle_dir, repo_root),
        "schedule_path": os.path.relpath(schedule_path, repo_root),
        "uplink_trace_path": os.path.relpath(uplink_trace_path, repo_root),
        "downlink_trace_path": os.path.relpath(downlink_trace_path, repo_root),
        "metrics": result.metrics,
    }


def _generate_random_traces(
    *,
    args,
    repo_root: str,
    config_path: str,
    config_payload: dict[str, Any],
    launch_config: SageLaunchConfig,
    runtime_dir: str,
    out_dir: str,
    num_generated_traces: int,
    wandb,
) -> list[dict[str, Any]]:
    env = _build_env(
        repo_root=repo_root,
        config_payload=config_payload,
        launch_config=launch_config,
        runtime_dir=runtime_dir,
    )
    ranges = _random_effective_ranges(
        config_payload=config_payload,
        launch_config=launch_config,
        args=args,
    )
    segment_steps_min, segment_steps_max = _validate_segment_bounds(args.segment_steps_min, args.segment_steps_max)

    generated_entries: list[dict[str, Any]] = []
    try:
        for trace_index in range(int(num_generated_traces)):
            trace_id = f"generated-{trace_index:03d}"
            rng = np.random.default_rng(int(args.seed) + int(trace_index))
            policy = RandomSchedulePolicy(
                env=env,
                rng=rng,
                ranges=ranges,
                segment_steps_min=segment_steps_min,
                segment_steps_max=segment_steps_max,
            )
            result = run_online_policy_episode(
                env,
                action_fn=policy,
                max_steps=int(config_payload.get("episode_steps", 6000)),
                episode_id=trace_id,
            )
            generated_entry = _write_generated_bundle(
                repo_root=repo_root,
                out_dir=out_dir,
                trace_index=trace_index,
                trace_id=trace_id,
                result=result,
                config_path=config_path,
                config_payload=config_payload,
                args=args,
            )
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
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--test-manifest", type=str, default="attacks/test/manifest.json")
    parser.add_argument("--out-dir", type=str, default="attacks/output/random_traces")
    parser.add_argument("--runtime-dir", type=str, default="attacks/runtime")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-traces", type=int, default=-1)
    parser.add_argument("--segment-steps-min", type=int, default=1)
    parser.add_argument("--segment-steps-max", type=int, default=1)
    parser.add_argument("--shared-bw-min-mbps", type=float, default=None)
    parser.add_argument("--shared-bw-max-mbps", type=float, default=None)
    parser.add_argument("--uplink-bw-min-mbps", type=float, default=None)
    parser.add_argument("--uplink-bw-max-mbps", type=float, default=None)
    parser.add_argument("--downlink-bw-min-mbps", type=float, default=None)
    parser.add_argument("--downlink-bw-max-mbps", type=float, default=None)
    parser.add_argument("--shared-loss-min", type=float, default=None)
    parser.add_argument("--shared-loss-max", type=float, default=None)
    parser.add_argument("--uplink-loss-min", type=float, default=None)
    parser.add_argument("--uplink-loss-max", type=float, default=None)
    parser.add_argument("--downlink-loss-min", type=float, default=None)
    parser.add_argument("--downlink-loss-max", type=float, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="sage-online-random-gen")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-tags", type=str, default="")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    config_path, config_payload = _load_config(repo_root, str(args.config_path))
    attack_mode = _attack_mode(config_payload)
    if attack_mode not in {"independent", "independent_gap"}:
        raise ValueError(
            f"random online trace generation only supports independent and independent_gap configs, got {attack_mode}"
        )

    manifest_path = _ensure_test_manifest(repo_root, str(args.test_manifest))
    test_entries = load_trace_entries(manifest_path)
    num_generated_traces = len(test_entries) if int(args.num_traces) <= 0 else int(args.num_traces)
    out_dir = resolve_repo_path(repo_root, str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    trace_set_name = _trace_set_name(out_dir)
    baseline_methods = baseline_methods_from_config(config_payload)
    run_namespace = acquire_run_namespace(
        repo_root=repo_root,
        runtime_dir=str(args.runtime_dir),
        actor_id=int(config_payload.get("actor_id", 900)),
        port=int(config_payload.get("port", 5101)),
        label=str(args.wandb_name or trace_set_name or "generate-random-online"),
        ports_per_run=(len(baseline_methods) + 1 if attack_mode == "independent_gap" else 1),
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
                    "config_path": config_path,
                    "test_manifest_resolved": manifest_path,
                    "num_reference_test_traces": len(test_entries),
                    "num_generated_traces": int(num_generated_traces),
                    "trace_set_name": trace_set_name,
                    "attack_mode": attack_mode,
                    "baseline_methods": list(baseline_methods),
                    "runtime_dir_resolved": resolved_runtime_dir,
                    "run_id": run_namespace.run_id,
                    "generator_type": "random",
                    "random_seed": int(args.seed),
                    "segment_steps_min": int(args.segment_steps_min),
                    "segment_steps_max": int(args.segment_steps_max),
                    "shared_bw_min_mbps": args.shared_bw_min_mbps,
                    "shared_bw_max_mbps": args.shared_bw_max_mbps,
                    "shared_loss_min": args.shared_loss_min,
                    "shared_loss_max": args.shared_loss_max,
                },
            )
            print_wandb_run_links(
                wandb_run,
                entity=args.wandb_entity,
                project=str(args.wandb_project),
            )

        launch_config = _resolved_launch_config(config_payload=config_payload, run_namespace=run_namespace)
        generated_entries = _generate_random_traces(
            args=args,
            repo_root=repo_root,
            config_path=config_path,
            config_payload=config_payload,
            launch_config=launch_config,
            runtime_dir=resolved_runtime_dir,
            out_dir=out_dir,
            num_generated_traces=num_generated_traces,
            wandb=wandb,
        )

        manifest_payload = {
            "created_at_utc": utc_now_iso(),
            "repo_root": repo_root,
            "trace_set_name": trace_set_name,
            "attack_mode": attack_mode,
            "baseline_methods": list(baseline_methods),
            "model_path": None,
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
