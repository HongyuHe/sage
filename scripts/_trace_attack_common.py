from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import shutil
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from attacks.envs.online_sage_env import AttackBounds, OnlineSageAttackEnv
from attacks.online import SageLaunchConfig


def _require_gym() -> tuple[Any, Any]:
    try:
        import gymnasium as gym  # type: ignore
        from gymnasium import spaces  # type: ignore

        return gym, spaces
    except Exception:
        import gym  # type: ignore
        from gym import spaces  # type: ignore

        return gym, spaces


gym, spaces = _require_gym()

TOP_LEVEL_TRACE_DIR = "traces"
PANTHEON_TRACE_DIR = "ccBench/pantheon-modified/src/experiments/traces"
PACKET_SIZE_BYTES = 1500
BASE_OBS_FEATURE_DIM = 69 + 1 + 14 + 6
CONTEXT_DIM = 5


def _is_retryable_sage_launch_error(exc: RuntimeError) -> bool:
    message = str(exc)
    retry_markers = (
        "Address already in use",
        "timed out waiting for Sage keys file",
        "no initial observation became available",
        "Sage never produced a real observation",
        "no response (OK_Signal)",
    )
    return any(marker in message for marker in retry_markers)


def repo_root_from_script(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(path), ".."))


def resolve_repo_path(repo_root: str, path: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(repo_root, expanded))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TraceEntry:
    trace_id: str
    split: str
    source_group: str
    name: str
    source_path: str
    copied_path: str
    relative_path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "split": self.split,
            "source_group": self.source_group,
            "name": self.name,
            "source_path": self.source_path,
            "copied_path": self.copied_path,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TraceEntry":
        return cls(
            trace_id=str(payload["trace_id"]),
            split=str(payload["split"]),
            source_group=str(payload["source_group"]),
            name=str(payload["name"]),
            source_path=str(payload["source_path"]),
            copied_path=str(payload["copied_path"]),
            relative_path=str(payload["relative_path"]),
            size_bytes=int(payload["size_bytes"]),
            sha256=str(payload["sha256"]),
        )


@dataclass(frozen=True)
class TraceSchedule:
    interval_ms: int
    bandwidth_mbps: np.ndarray
    last_timestamp_ms: int
    total_packets: int

    @property
    def num_steps(self) -> int:
        return int(self.bandwidth_mbps.shape[0])

    @property
    def mean_bandwidth_mbps(self) -> float:
        if self.bandwidth_mbps.size == 0:
            return 0.0
        return float(np.mean(self.bandwidth_mbps))

    @property
    def max_bandwidth_mbps(self) -> float:
        if self.bandwidth_mbps.size == 0:
            return 0.0
        return float(np.max(self.bandwidth_mbps))


@dataclass(frozen=True)
class EpisodeResult:
    trace_entry: TraceEntry
    num_steps: int
    total_reward: float
    metrics: dict[str, float]
    step_records: list[dict[str, Any]]


@dataclass(frozen=True)
class OnlineEpisodeResult:
    episode_id: str
    num_steps: int
    total_reward: float
    metrics: dict[str, float]
    step_records: list[dict[str, Any]]


def _iter_trace_files(directory: str) -> list[Path]:
    path = Path(directory)
    return sorted(item for item in path.iterdir() if item.is_file())


def _copy_trace_entry(
    *,
    repo_root: str,
    split_root: str,
    split_name: str,
    source_group: str,
    source_path: str,
) -> TraceEntry:
    source = Path(source_path)
    destination_dir = Path(split_root) / source_group
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    shutil.copy2(str(source), str(destination))
    relative_path = os.path.relpath(destination, repo_root)
    trace_id = f"{source_group}__{source.name}"
    return TraceEntry(
        trace_id=trace_id,
        split=split_name,
        source_group=source_group,
        name=source.name,
        source_path=str(source.resolve()),
        copied_path=str(destination.resolve()),
        relative_path=relative_path,
        size_bytes=int(destination.stat().st_size),
        sha256=_sha256_file(str(destination)),
    )


def materialize_trace_splits(
    *,
    repo_root: str,
    train_root: str,
    test_root: str,
) -> dict[str, Any]:
    repo_root = os.path.abspath(repo_root)
    top_dir = resolve_repo_path(repo_root, TOP_LEVEL_TRACE_DIR)
    pantheon_dir = resolve_repo_path(repo_root, PANTHEON_TRACE_DIR)

    top_level_files = _iter_trace_files(top_dir)
    pantheon_files = _iter_trace_files(pantheon_dir)
    top_names = {item.name for item in top_level_files}
    pantheon_names = {item.name for item in pantheon_files}
    overlap_names = top_names & pantheon_names

    train_entries: list[TraceEntry] = []
    test_entries: list[TraceEntry] = []

    for item in pantheon_files:
        train_entries.append(
            _copy_trace_entry(
                repo_root=repo_root,
                split_root=train_root,
                split_name="train",
                source_group="pantheon",
                source_path=str(item),
            )
        )

    for item in top_level_files:
        if item.name in overlap_names:
            train_entries.append(
                _copy_trace_entry(
                    repo_root=repo_root,
                    split_root=train_root,
                    split_name="train",
                    source_group="top_level",
                    source_path=str(item),
                )
            )
        else:
            test_entries.append(
                _copy_trace_entry(
                    repo_root=repo_root,
                    split_root=test_root,
                    split_name="test",
                    source_group="top_level",
                    source_path=str(item),
                )
            )

    train_manifest = {
        "created_at_utc": utc_now_iso(),
        "repo_root": repo_root,
        "split": "train",
        "counts": {
            "entries": len(train_entries),
            "pantheon_entries": sum(1 for entry in train_entries if entry.source_group == "pantheon"),
            "top_level_entries": sum(1 for entry in train_entries if entry.source_group == "top_level"),
        },
        "entries": [entry.to_dict() for entry in train_entries],
    }
    test_manifest = {
        "created_at_utc": utc_now_iso(),
        "repo_root": repo_root,
        "split": "test",
        "counts": {
            "entries": len(test_entries),
            "top_level_entries": sum(1 for entry in test_entries if entry.source_group == "top_level"),
        },
        "entries": [entry.to_dict() for entry in test_entries],
    }
    combined_manifest = {
        "created_at_utc": utc_now_iso(),
        "repo_root": repo_root,
        "top_level_trace_dir": os.path.relpath(top_dir, repo_root),
        "pantheon_trace_dir": os.path.relpath(pantheon_dir, repo_root),
        "counts": {
            "top_level_total": len(top_level_files),
            "pantheon_total": len(pantheon_files),
            "overlap_names": len(overlap_names),
            "train_entries": len(train_entries),
            "test_entries": len(test_entries),
        },
        "overlap_names": sorted(overlap_names),
        "train_manifest": os.path.relpath(os.path.join(train_root, "manifest.json"), repo_root),
        "test_manifest": os.path.relpath(os.path.join(test_root, "manifest.json"), repo_root),
    }

    Path(train_root).mkdir(parents=True, exist_ok=True)
    Path(test_root).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(train_root, "manifest.json"), "w", encoding="utf-8") as file_obj:
        json.dump(train_manifest, file_obj, indent=2, sort_keys=True)
    with open(os.path.join(test_root, "manifest.json"), "w", encoding="utf-8") as file_obj:
        json.dump(test_manifest, file_obj, indent=2, sort_keys=True)
    with open(os.path.join(repo_root, "attacks", "trace_split_manifest.json"), "w", encoding="utf-8") as file_obj:
        json.dump(combined_manifest, file_obj, indent=2, sort_keys=True)
    return combined_manifest


def load_trace_entries(manifest_path: str) -> list[TraceEntry]:
    with open(manifest_path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    entries = payload.get("entries", [])
    return [TraceEntry.from_dict(item) for item in entries]


def load_mahimahi_trace_schedule(path: str, *, interval_ms: float) -> TraceSchedule:
    interval = max(int(round(float(interval_ms))), 1)
    factor = float(PACKET_SIZE_BYTES * 8.0 / 1_000_000.0) / (float(interval) / 1000.0)
    timestamps: list[int] = []

    with open(path, "r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            timestamp_ms = int(line)
            if timestamp_ms <= 0:
                continue
            timestamps.append(timestamp_ms)

    if not timestamps:
        return TraceSchedule(
            interval_ms=interval,
            bandwidth_mbps=np.asarray([0.0], dtype=np.float32),
            last_timestamp_ms=0,
            total_packets=0,
        )

    timestamps_array = np.asarray(timestamps, dtype=np.int64)
    last_timestamp_ms = int(timestamps_array[-1])
    super_period_ms = (last_timestamp_ms * interval) // math.gcd(last_timestamp_ms, interval)
    repeats = max(int(super_period_ms // last_timestamp_ms), 1)
    num_buckets = max(int(super_period_ms // interval), 1)
    bucket_counts = np.zeros((num_buckets,), dtype=np.int64)

    for repeat_idx in range(repeats):
        shifted = timestamps_array + repeat_idx * last_timestamp_ms
        bucket_indices = ((shifted - 1) // interval).astype(np.int64, copy=False)
        np.add.at(bucket_counts, bucket_indices, 1)

    return TraceSchedule(
        interval_ms=interval,
        bandwidth_mbps=(bucket_counts.astype(np.float32) * factor),
        last_timestamp_ms=int(last_timestamp_ms),
        total_packets=int(timestamps_array.shape[0] * repeats),
    )


def write_bandwidth_trace(
    *,
    bandwidth_mbps: Sequence[float],
    interval_ms: float,
    out_path: str,
) -> None:
    interval = max(int(round(float(interval_ms))), 1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    packet_credit = 0.0
    with open(out_path, "w", encoding="utf-8") as file_obj:
        for index, rate in enumerate(bandwidth_mbps):
            packet_credit += max(float(rate), 0.0) * float(interval) / 12.0
            packet_count = int(math.floor(packet_credit + 1e-9))
            packet_credit -= packet_count
            if packet_count <= 0:
                continue
            window_start = index * interval + 1
            for packet_idx in range(packet_count):
                offset = int((packet_idx * interval) / max(packet_count, 1))
                file_obj.write(f"{window_start + offset}\n")


def neutral_residual_action() -> np.ndarray:
    return np.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def build_clean_action_schedule(
    schedule: TraceSchedule,
    *,
    uplink_loss: float = 0.0,
    downlink_loss: float = 0.0,
    uplink_delay_ms: float = 0.0,
    downlink_delay_ms: float = 0.0,
) -> list[np.ndarray]:
    actions: list[np.ndarray] = []
    for bandwidth in np.asarray(schedule.bandwidth_mbps, dtype=np.float32).tolist():
        actions.append(
            np.asarray(
                [
                    float(max(bandwidth, 0.0)),
                    float(max(bandwidth, 0.0)),
                    float(max(uplink_loss, 0.0)),
                    float(max(downlink_loss, 0.0)),
                    float(max(uplink_delay_ms, 0.0)),
                    float(max(downlink_delay_ms, 0.0)),
                ],
                dtype=np.float32,
            )
        )
    return actions


def attack_bounds_from_config(config_payload: dict[str, Any]) -> AttackBounds:
    low = config_payload.get("action_space_low")
    high = config_payload.get("action_space_high")
    if isinstance(low, list) and isinstance(high, list) and len(low) == 6 and len(high) == 6:
        return AttackBounds(
            uplink_bw_mbps=(float(low[0]), float(high[0])),
            downlink_bw_mbps=(float(low[1]), float(high[1])),
            uplink_loss=(float(low[2]), float(high[2])),
            downlink_loss=(float(low[3]), float(high[3])),
            uplink_delay_ms=(float(low[4]), float(high[4])),
            downlink_delay_ms=(float(low[5]), float(high[5])),
        )

    effective_bw_cap_mbps = float(config_payload.get("effective_bw_cap_mbps", 2000.0))
    loss_max = float(config_payload.get("loss_max", 0.15))
    delay_max_ms = float(config_payload.get("delay_max_ms", 150.0))
    return AttackBounds(
        uplink_bw_mbps=(
            float(config_payload.get("attack_uplink_bw_min_mbps", 0.0)),
            float(config_payload.get("attack_uplink_bw_max_mbps", effective_bw_cap_mbps)),
        ),
        downlink_bw_mbps=(
            float(config_payload.get("attack_downlink_bw_min_mbps", 0.0)),
            float(config_payload.get("attack_downlink_bw_max_mbps", effective_bw_cap_mbps)),
        ),
        uplink_loss=(
            float(config_payload.get("attack_uplink_loss_min", 0.0)),
            float(config_payload.get("attack_uplink_loss_max", loss_max)),
        ),
        downlink_loss=(
            float(config_payload.get("attack_downlink_loss_min", 0.0)),
            float(config_payload.get("attack_downlink_loss_max", loss_max)),
        ),
        uplink_delay_ms=(
            float(config_payload.get("attack_uplink_delay_min_ms", 0.0)),
            float(config_payload.get("attack_uplink_delay_max_ms", delay_max_ms)),
        ),
        downlink_delay_ms=(
            float(config_payload.get("attack_downlink_delay_min_ms", 0.0)),
            float(config_payload.get("attack_downlink_delay_max_ms", delay_max_ms)),
        ),
    )


def expand_attack_bounds_for_bandwidth(bounds: AttackBounds, max_bandwidth_mbps: float) -> AttackBounds:
    target_max = max(float(max_bandwidth_mbps), bounds.uplink_bw_mbps[1], bounds.downlink_bw_mbps[1])
    return AttackBounds(
        uplink_bw_mbps=(bounds.uplink_bw_mbps[0], target_max),
        downlink_bw_mbps=(bounds.downlink_bw_mbps[0], target_max),
        uplink_loss=bounds.uplink_loss,
        downlink_loss=bounds.downlink_loss,
        uplink_delay_ms=bounds.uplink_delay_ms,
        downlink_delay_ms=bounds.downlink_delay_ms,
    )


class TraceConditionedAttackEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        repo_root: str,
        trace_entries: Sequence[TraceEntry],
        launch_config: SageLaunchConfig,
        online_attack_bounds: AttackBounds | None = None,
        obs_history_len: int = 4,
        attack_interval_ms: float = 100.0,
        max_episode_steps: int = 6000,
        launch_timeout_s: float = 90.0,
        step_timeout_s: float = 10.0,
        runtime_dir: str = "attacks/runtime",
        sample_mode: str = "random",
        seed: int = 7,
        bw_scale_min: float = 0.1,
        bw_scale_max: float = 2.0,
        effective_bw_cap_mbps: float = 2000.0,
        loss_min: float = 0.0,
        loss_max: float = 0.15,
        delay_min_ms: float = 0.0,
        delay_max_ms: float = 150.0,
        reward_rate_weight: float = 1.0,
        reward_rtt_weight: float = 0.05,
        reward_loss_weight: float = 2.0,
        smooth_penalty_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if not trace_entries:
            raise ValueError("trace_entries must not be empty")

        self.repo_root = os.path.abspath(repo_root)
        self.trace_entries = list(trace_entries)
        self.launch_config = launch_config
        self.online_attack_bounds = online_attack_bounds
        self.obs_history_len = int(obs_history_len)
        self.attack_interval_ms = float(attack_interval_ms)
        self.max_episode_steps = int(max_episode_steps)
        self.launch_timeout_s = float(launch_timeout_s)
        self.step_timeout_s = float(step_timeout_s)
        self.runtime_dir = runtime_dir
        self.sample_mode = str(sample_mode)
        self.seed = int(seed)
        self.bw_scale_min = float(bw_scale_min)
        self.bw_scale_max = float(bw_scale_max)
        self.effective_bw_cap_mbps = float(effective_bw_cap_mbps)
        self.loss_min = float(loss_min)
        self.loss_max = float(loss_max)
        self.delay_min_ms = float(delay_min_ms)
        self.delay_max_ms = float(delay_max_ms)
        self.reward_rate_weight = float(reward_rate_weight)
        self.reward_rtt_weight = float(reward_rtt_weight)
        self.reward_loss_weight = float(reward_loss_weight)
        self.smooth_penalty_scale = float(smooth_penalty_scale)
        self._rng = random.Random(self.seed)
        self._trace_cursor = 0
        self._trace_usage: dict[str, int] = {}
        self._schedule_cache: dict[str, TraceSchedule] = {}
        self._active_trace_entry: TraceEntry | None = None
        self._active_schedule: TraceSchedule | None = None
        self._active_env: OnlineSageAttackEnv | None = None
        self._episode_step = 0
        self._launch_index = 0
        self._last_residual_action = neutral_residual_action()

        obs_dim = self.obs_history_len * BASE_OBS_FEATURE_DIM + CONTEXT_DIM
        obs_high = np.full((obs_dim,), 1e9, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.asarray(
                [
                    self.bw_scale_min,
                    self.bw_scale_min,
                    self.loss_min,
                    self.loss_min,
                    self.delay_min_ms,
                    self.delay_min_ms,
                ],
                dtype=np.float32,
            ),
            high=np.asarray(
                [
                    self.bw_scale_max,
                    self.bw_scale_max,
                    self.loss_max,
                    self.loss_max,
                    self.delay_max_ms,
                    self.delay_max_ms,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    def close(self) -> None:
        if self._active_env is not None:
            self._active_env.close()
            self._active_env = None

    def trace_usage_counts(self) -> dict[str, int]:
        return dict(self._trace_usage)

    def _load_schedule(self, entry: TraceEntry) -> TraceSchedule:
        cached = self._schedule_cache.get(entry.trace_id)
        if cached is not None:
            return cached
        schedule = load_mahimahi_trace_schedule(entry.copied_path, interval_ms=self.attack_interval_ms)
        self._schedule_cache[entry.trace_id] = schedule
        return schedule

    def _select_trace_entry(self, options: dict[str, Any] | None) -> TraceEntry:
        if options is not None:
            if "trace_index" in options:
                return self.trace_entries[int(options["trace_index"])]
            if "trace_id" in options:
                target = str(options["trace_id"])
                for entry in self.trace_entries:
                    if entry.trace_id == target:
                        return entry
                raise KeyError(f"unknown trace_id: {target}")
        if self.sample_mode == "round_robin":
            entry = self.trace_entries[self._trace_cursor % len(self.trace_entries)]
            self._trace_cursor += 1
            return entry
        return self.trace_entries[self._rng.randrange(len(self.trace_entries))]

    def _reserve_launch_port(self, preferred_port: int, max_tries: int = 256) -> int:
        candidate = max(int(preferred_port), 1024)
        for offset in range(max(int(max_tries), 1)):
            port = candidate + offset
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("0.0.0.0", port))
                except OSError:
                    continue
                return int(port)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("0.0.0.0", 0))
            return int(sock.getsockname()[1])

    def _make_launch_config(self, entry: TraceEntry, schedule: TraceSchedule, *, launch_index: int) -> SageLaunchConfig:
        first_bw = float(schedule.bandwidth_mbps[0]) if schedule.bandwidth_mbps.size else 0.0
        mean_bw = max(float(schedule.mean_bandwidth_mbps), 1.0)
        episode_steps = max(1, int(self.max_episode_steps))
        duration_seconds = max(
            int(self.launch_config.duration_seconds),
            int(math.ceil((episode_steps * self.attack_interval_ms) / 1000.0)) + 2,
        )
        #* Keep each training job on a stable preferred port and only drift locally
        #* when the previous episode has not fully released its socket yet.
        port = self._reserve_launch_port(int(self.launch_config.port))
        actor_id = max(int(self.launch_config.actor_id) + int(launch_index), 0)
        iteration_id = int(self.launch_config.iteration_id) + int(launch_index)
        return replace(
            self.launch_config,
            port=port,
            downlink_trace=entry.name,
            uplink_trace=entry.name,
            iteration_id=iteration_id,
            env_bw_mbps=max(1, int(round(mean_bw))),
            bw2_mbps=max(1, int(round(mean_bw))),
            duration_seconds=duration_seconds,
            duration_steps=episode_steps,
            actor_id=actor_id,
            initial_uplink_bw_mbps=first_bw,
            initial_downlink_bw_mbps=first_bw,
        )

    def _make_inner_env(self, entry: TraceEntry, schedule: TraceSchedule, *, launch_index: int) -> OnlineSageAttackEnv:
        bounds = self.online_attack_bounds or AttackBounds(
            uplink_bw_mbps=(0.0, self.effective_bw_cap_mbps),
            downlink_bw_mbps=(0.0, self.effective_bw_cap_mbps),
            uplink_loss=(self.loss_min, self.loss_max),
            downlink_loss=(self.loss_min, self.loss_max),
            uplink_delay_ms=(self.delay_min_ms, self.delay_max_ms),
            downlink_delay_ms=(self.delay_min_ms, self.delay_max_ms),
        )
        return OnlineSageAttackEnv(
            repo_root=self.repo_root,
            launch_config=self._make_launch_config(entry, schedule, launch_index=launch_index),
            bounds=bounds,
            obs_history_len=self.obs_history_len,
            attack_interval_ms=self.attack_interval_ms,
            max_episode_steps=max(1, int(self.max_episode_steps)),
            launch_timeout_s=self.launch_timeout_s,
            step_timeout_s=self.step_timeout_s,
            smooth_penalty_scale=0.0,
            reward_scale=1.0,
            runtime_dir=self.runtime_dir,
        )

    def _context_features(self, step_index: int) -> np.ndarray:
        assert self._active_schedule is not None
        schedule = self._active_schedule
        period = max(schedule.num_steps, 1)
        current_idx = int(step_index) % period
        next_idx = (current_idx + 1) % period
        current_bw = float(schedule.bandwidth_mbps[current_idx]) if schedule.bandwidth_mbps.size else 0.0
        next_bw = float(schedule.bandwidth_mbps[next_idx]) if schedule.bandwidth_mbps.size else current_bw
        progress = float(current_idx) / float(max(period - 1, 1))
        return np.asarray(
            [
                current_bw,
                current_bw,
                next_bw,
                next_bw,
                progress,
            ],
            dtype=np.float32,
        )

    def _augment_observation(self, observation: np.ndarray, step_index: int) -> np.ndarray:
        return np.concatenate(
            [np.asarray(observation, dtype=np.float32), self._context_features(step_index)],
            axis=0,
        ).astype(np.float32, copy=False)

    def _compose_effective_action(self, residual_action: np.ndarray) -> np.ndarray:
        assert self._active_schedule is not None
        clipped = np.clip(np.asarray(residual_action, dtype=np.float32), self.action_space.low, self.action_space.high)
        schedule = self._active_schedule
        idx = int(self._episode_step) % max(schedule.num_steps, 1)
        base_up = float(schedule.bandwidth_mbps[idx]) if schedule.bandwidth_mbps.size else 0.0
        base_down = base_up
        effective_up = base_up * float(clipped[0]) if base_up > 0.0 else 0.0
        effective_down = base_down * float(clipped[1]) if base_down > 0.0 else 0.0
        return np.asarray(
            [
                min(max(effective_up, 0.0), self.effective_bw_cap_mbps),
                min(max(effective_down, 0.0), self.effective_bw_cap_mbps),
                float(clipped[2]),
                float(clipped[3]),
                float(clipped[4]),
                float(clipped[5]),
            ],
            dtype=np.float32,
        )

    def _reward_from_info(self, info: dict[str, Any], residual_action: np.ndarray) -> tuple[float, float]:
        rate = float(info.get("sage/windowed_delivery_rate_mbps", 0.0))
        rtt = float(info.get("sage/current_rtt_ms", 0.0))
        loss = float(info.get("sage/current_loss_mbps", 0.0))
        external_score = self.reward_rate_weight * rate - self.reward_rtt_weight * rtt - self.reward_loss_weight * loss
        smooth_penalty = float(np.mean(np.abs(residual_action - self._last_residual_action)))
        reward = -external_score - self.smooth_penalty_scale * smooth_penalty
        info["sage/external_score"] = float(external_score)
        info["attacker/smooth_penalty"] = float(smooth_penalty)
        return float(reward), float(external_score)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        try:
            super().reset(seed=seed)
        except TypeError:
            if seed is not None and hasattr(self, "seed"):
                self.seed(seed)
        self.close()

        entry = self._select_trace_entry(options)
        schedule = self._load_schedule(entry)
        self._active_trace_entry = entry
        self._active_schedule = schedule
        self._episode_step = 0
        self._last_residual_action = neutral_residual_action()
        self._trace_usage[entry.trace_id] = self._trace_usage.get(entry.trace_id, 0) + 1

        launch_error: RuntimeError | None = None
        observation = None
        info: dict[str, Any] | None = None
        for _ in range(8):
            launch_index = self._launch_index
            self._launch_index += 1
            self._active_env = self._make_inner_env(entry, schedule, launch_index=launch_index)
            try:
                observation, info = self._active_env.reset(seed=seed)
                launch_error = None
                break
            except RuntimeError as exc:
                launch_error = exc
                self._active_env.close()
                self._active_env = None
                if not _is_retryable_sage_launch_error(exc):
                    raise
        if launch_error is not None:
            raise launch_error

        assert observation is not None
        assert info is not None
        info = dict(info)
        active_launch_config = self._active_env.launch_config
        info.update(
            {
                "trace/launch_port": float(active_launch_config.port),
                "trace/launch_actor_id": float(active_launch_config.actor_id),
                "trace/source_is_pantheon": 1.0 if entry.source_group == "pantheon" else 0.0,
                "trace/mean_bw_mbps": float(schedule.mean_bandwidth_mbps),
                "trace/max_bw_mbps": float(schedule.max_bandwidth_mbps),
                "trace/period_steps": float(schedule.num_steps),
                "trace/progress": 0.0,
            }
        )
        return self._augment_observation(observation, 0), info

    def step(self, action):
        if self._active_env is None or self._active_trace_entry is None or self._active_schedule is None:
            raise RuntimeError("environment is not initialized; call reset() first")

        residual_action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        effective_action = self._compose_effective_action(residual_action)
        observation, _, terminated, truncated, info = self._active_env.step(effective_action)
        info = dict(info)
        reward, external_score = self._reward_from_info(info, residual_action)

        period = max(self._active_schedule.num_steps, 1)
        idx = int(self._episode_step) % period
        base_bw = float(self._active_schedule.bandwidth_mbps[idx]) if self._active_schedule.bandwidth_mbps.size else 0.0
        info.update(
            {
                "trace/source_is_pantheon": 1.0 if self._active_trace_entry.source_group == "pantheon" else 0.0,
                "trace/mean_bw_mbps": float(self._active_schedule.mean_bandwidth_mbps),
                "trace/max_bw_mbps": float(self._active_schedule.max_bandwidth_mbps),
                "trace/period_steps": float(self._active_schedule.num_steps),
                "trace/progress": float(idx) / float(max(period - 1, 1)),
                "trace/base_uplink_bw_mbps": float(base_bw),
                "trace/base_downlink_bw_mbps": float(base_bw),
                "attacker/uplink_bw_scale": float(residual_action[0]),
                "attacker/downlink_bw_scale": float(residual_action[1]),
                "attacker/effective_uplink_bw_mbps": float(effective_action[0]),
                "attacker/effective_downlink_bw_mbps": float(effective_action[1]),
                "attacker/reward": float(reward),
                "sage/external_score": float(external_score),
            }
        )
        self._last_residual_action = residual_action.astype(np.float32, copy=True)
        self._episode_step += 1

        return self._augment_observation(observation, self._episode_step), reward, terminated, truncated, info


class IndependentAttackEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        repo_root: str,
        launch_config: SageLaunchConfig,
        bounds: AttackBounds | None = None,
        obs_history_len: int = 4,
        attack_interval_ms: float = 100.0,
        max_episode_steps: int = 120,
        launch_timeout_s: float = 90.0,
        step_timeout_s: float = 10.0,
        runtime_dir: str = "attacks/runtime",
        reward_rate_weight: float = 1.0,
        reward_rtt_weight: float = 0.05,
        reward_loss_weight: float = 2.0,
        smooth_penalty_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.repo_root = os.path.abspath(repo_root)
        self._base_launch_config = launch_config
        self._bounds = bounds
        self._obs_history_len = int(obs_history_len)
        self._attack_interval_ms = float(attack_interval_ms)
        self._launch_timeout_s = float(launch_timeout_s)
        self._step_timeout_s = float(step_timeout_s)
        self._runtime_dir = runtime_dir
        self.reward_rate_weight = float(reward_rate_weight)
        self.reward_rtt_weight = float(reward_rtt_weight)
        self.reward_loss_weight = float(reward_loss_weight)
        self.smooth_penalty_scale = float(smooth_penalty_scale)
        self.max_episode_steps = max(1, int(max_episode_steps))
        self._episode_step = 0
        self._last_action: np.ndarray | None = None
        self._launch_index = 0

        self._inner_env = self._make_inner_env(launch_index=self._launch_index)
        self._launch_index += 1
        self.action_space = self._inner_env.action_space
        inner_obs_shape = self._inner_env.observation_space.shape
        if len(inner_obs_shape) != 1:
            raise ValueError("IndependentAttackEnv only supports flat observations")
        obs_high = np.full((int(inner_obs_shape[0]) + 1,), 1e9, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

    @property
    def launch_config(self) -> SageLaunchConfig:
        return self._inner_env.launch_config

    def _reserve_launch_port(self, preferred_port: int, max_tries: int = 256) -> int:
        candidate = max(int(preferred_port), 1024)
        for offset in range(max(int(max_tries), 1)):
            port = candidate + offset
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("0.0.0.0", port))
                except OSError:
                    continue
                return int(port)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("0.0.0.0", 0))
            return int(sock.getsockname()[1])

    def _make_launch_config(self, *, launch_index: int) -> SageLaunchConfig:
        preferred_port = int(self._base_launch_config.port)
        port = self._reserve_launch_port(preferred_port)
        actor_id = max(int(self._base_launch_config.actor_id) + int(launch_index), 0)
        iteration_id = int(self._base_launch_config.iteration_id) + int(launch_index)
        return replace(
            self._base_launch_config,
            port=port,
            actor_id=actor_id,
            iteration_id=iteration_id,
        )

    def _make_inner_env(self, *, launch_index: int) -> OnlineSageAttackEnv:
        return OnlineSageAttackEnv(
            repo_root=self.repo_root,
            launch_config=self._make_launch_config(launch_index=launch_index),
            bounds=self._bounds,
            obs_history_len=self._obs_history_len,
            attack_interval_ms=self._attack_interval_ms,
            max_episode_steps=self.max_episode_steps,
            launch_timeout_s=self._launch_timeout_s,
            step_timeout_s=self._step_timeout_s,
            smooth_penalty_scale=0.0,
            reward_scale=1.0,
            runtime_dir=self._runtime_dir,
        )

    def _is_retryable_launch_error(self, exc: RuntimeError) -> bool:
        return _is_retryable_sage_launch_error(exc)

    def _progress_feature(self) -> np.ndarray:
        progress = float(self._episode_step) / float(max(self.max_episode_steps, 1))
        return np.asarray([min(max(progress, 0.0), 1.0)], dtype=np.float32)

    def _augment_observation(self, observation: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [np.asarray(observation, dtype=np.float32), self._progress_feature()],
            axis=0,
        ).astype(np.float32, copy=False)

    def _normalized_action(self, action: np.ndarray) -> np.ndarray:
        denom = np.maximum(self.action_space.high - self.action_space.low, 1e-6)
        return ((action - self.action_space.low) / denom).astype(np.float32, copy=False)

    def _reward_from_info(self, info: dict[str, Any], action: np.ndarray) -> tuple[float, float, float]:
        rate = float(info.get("sage/windowed_delivery_rate_mbps", 0.0))
        rtt = float(info.get("sage/current_rtt_ms", 0.0))
        loss = float(info.get("sage/current_loss_mbps", 0.0))
        external_score = self.reward_rate_weight * rate - self.reward_rtt_weight * rtt - self.reward_loss_weight * loss
        previous_action = self._last_action if self._last_action is not None else np.asarray(action, dtype=np.float32)
        smooth_penalty = float(
            np.mean(
                np.abs(
                    self._normalized_action(np.asarray(action, dtype=np.float32))
                    - self._normalized_action(np.asarray(previous_action, dtype=np.float32))
                )
            )
        )
        reward = -external_score - self.smooth_penalty_scale * smooth_penalty
        return float(reward), float(external_score), float(smooth_penalty)

    def close(self) -> None:
        self._inner_env.close()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        launch_error: RuntimeError | None = None
        observation = None
        info = None
        for _ in range(8):
            try:
                observation, info = self._inner_env.reset(seed=seed, options=options)
                launch_error = None
                break
            except RuntimeError as exc:
                launch_error = exc
                if not self._is_retryable_launch_error(exc):
                    raise
                self._inner_env.close()
                self._inner_env = self._make_inner_env(launch_index=self._launch_index)
                self._launch_index += 1

        if launch_error is not None:
            raise launch_error
        if observation is None or info is None:
            raise RuntimeError("IndependentAttackEnv reset failed without launch diagnostics")

        self._episode_step = 0
        self._last_action = np.asarray(self._inner_env._default_action(), dtype=np.float32)
        info = dict(info)
        info["episode/progress"] = 0.0
        info["trace/launch_port"] = float(self._inner_env.launch_config.port)
        info["trace/launch_actor_id"] = float(self._inner_env.launch_config.actor_id)
        return self._augment_observation(observation), info

    def step(self, action):
        clipped = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        observation, _, terminated, truncated, info = self._inner_env.step(clipped)
        reward, external_score, smooth_penalty = self._reward_from_info(dict(info), clipped)
        self._episode_step += 1
        self._last_action = clipped.astype(np.float32, copy=True)
        info = dict(info)
        info["sage/external_score"] = float(external_score)
        info["attacker/reward"] = float(reward)
        info["attacker/smooth_penalty"] = float(smooth_penalty)
        info["episode/progress"] = float(self._progress_feature()[0])
        return self._augment_observation(observation), float(reward), terminated, truncated, info


class EpisodeAccumulator:
    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._count = 0

    def add(self, info: dict[str, Any]) -> None:
        self._count += 1
        for key, value in info.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                self._sums[str(key)] = self._sums.get(str(key), 0.0) + float(value)

    def summary(self) -> dict[str, float]:
        if self._count <= 0:
            return {}
        return {f"{key}_mean": value / float(self._count) for key, value in self._sums.items()}


def run_policy_episode(
    env: TraceConditionedAttackEnv,
    *,
    action_fn: Callable[[np.ndarray, dict[str, Any], int], np.ndarray],
    trace_index: int | None = None,
    trace_id: str | None = None,
    max_steps: int | None = None,
) -> EpisodeResult:
    options: dict[str, Any] = {}
    if trace_index is not None:
        options["trace_index"] = int(trace_index)
    if trace_id is not None:
        options["trace_id"] = str(trace_id)

    observation, info = env.reset(options=options if options else None)
    accumulator = EpisodeAccumulator()
    total_reward = 0.0
    step_records: list[dict[str, Any]] = []
    accumulator.add(info)
    step_index = 0

    while True:
        action = np.asarray(action_fn(observation, info, step_index), dtype=np.float32)
        next_observation, reward, terminated, truncated, info = env.step(action)
        accumulator.add(info)
        total_reward += float(reward)
        step_records.append(
            {
                "step": int(step_index),
                "residual_action": [float(x) for x in action.tolist()],
                "effective_action": [
                    float(info.get("attacker/effective_uplink_bw_mbps", 0.0)),
                    float(info.get("attacker/effective_downlink_bw_mbps", 0.0)),
                    float(action[2]),
                    float(action[3]),
                    float(action[4]),
                    float(action[5]),
                ],
                "base_uplink_bw_mbps": float(info.get("trace/base_uplink_bw_mbps", 0.0)),
                "base_downlink_bw_mbps": float(info.get("trace/base_downlink_bw_mbps", 0.0)),
                "sage_reward": float(info.get("sage/reward", 0.0)),
                "sage_external_score": float(info.get("sage/external_score", 0.0)),
                "sage_rtt_ms": float(info.get("sage/current_rtt_ms", 0.0)),
                "sage_windowed_rate_mbps": float(info.get("sage/windowed_delivery_rate_mbps", 0.0)),
                "sage_loss_mbps": float(info.get("sage/current_loss_mbps", 0.0)),
            }
        )
        observation = next_observation
        step_index += 1
        if terminated or truncated:
            break
        if max_steps is not None and step_index >= int(max_steps):
            break

    if env._active_trace_entry is None:
        raise RuntimeError("trace entry disappeared during rollout")
    metrics = accumulator.summary()
    metrics["episode_total_reward"] = float(total_reward)
    metrics["episode_num_steps"] = float(step_index)
    return EpisodeResult(
        trace_entry=env._active_trace_entry,
        num_steps=int(step_index),
        total_reward=float(total_reward),
        metrics=metrics,
        step_records=step_records,
    )


def _plain_step_record(
    *,
    step_index: int,
    action: np.ndarray,
    reward: float,
    info: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "step": int(step_index),
        "action": [float(x) for x in np.asarray(action, dtype=np.float32).tolist()],
        "reward": float(reward),
    }
    key_map = {
        "attacker/uplink_bw_mbps": "attacker_uplink_bw_mbps",
        "attacker/downlink_bw_mbps": "attacker_downlink_bw_mbps",
        "attacker/uplink_loss": "attacker_uplink_loss",
        "attacker/downlink_loss": "attacker_downlink_loss",
        "attacker/uplink_delay_ms": "attacker_uplink_delay_ms",
        "attacker/downlink_delay_ms": "attacker_downlink_delay_ms",
        "attacker/reward": "reward",
        "attacker/smooth_penalty": "attacker_smooth_penalty",
        "sage/reward": "sage_reward",
        "sage/external_score": "sage_external_score",
        "sage/current_rtt_ms": "sage_rtt_ms",
        "sage/current_rttvar_ms": "sage_rttvar_ms",
        "sage/current_delivery_rate_mbps": "sage_current_delivery_rate_mbps",
        "sage/windowed_delivery_rate_mbps": "sage_windowed_rate_mbps",
        "sage/max_windowed_delivery_rate_mbps": "sage_max_windowed_rate_mbps",
        "sage/current_loss_mbps": "sage_loss_mbps",
        "sage/current_min_rtt_ratio": "sage_min_rtt_ratio",
        "mm/up_applied_bw_mbps": "mm_up_applied_bw_mbps",
        "mm/up_applied_loss_rate": "mm_up_applied_loss_rate",
        "mm/up_applied_delay_ms": "mm_up_applied_delay_ms",
        "mm/up_queue_delay_ms": "mm_up_queue_delay_ms",
        "mm/up_departure_rate_mbps": "mm_up_departure_rate_mbps",
        "mm/down_applied_bw_mbps": "mm_down_applied_bw_mbps",
        "mm/down_applied_loss_rate": "mm_down_applied_loss_rate",
        "mm/down_applied_delay_ms": "mm_down_applied_delay_ms",
        "mm/down_queue_delay_ms": "mm_down_queue_delay_ms",
        "mm/down_departure_rate_mbps": "mm_down_departure_rate_mbps",
        "gap/base_rtt_ms": "gap_base_rtt_ms",
        "gap/path_cap_mbps": "gap_path_cap_mbps",
        "gap/score_sage": "gap_score_sage",
        "gap/score_cubic": "gap_score_cubic",
        "gap/score_bbr": "gap_score_bbr",
        "gap/baseline_score": "gap_baseline_score",
        "gap/value": "gap_value",
        "gap/reward": "gap_reward",
        "baseline/cubic_rtt_ms": "baseline_cubic_rtt_ms",
        "baseline/bbr_rtt_ms": "baseline_bbr_rtt_ms",
        "baseline/cubic_rate_mbps": "baseline_cubic_rate_mbps",
        "baseline/bbr_rate_mbps": "baseline_bbr_rate_mbps",
        "episode/progress": "progress",
    }
    for source_key, target_key in key_map.items():
        value = info.get(source_key)
        if isinstance(value, (int, float, np.floating, np.integer)):
            record[target_key] = float(value)
    for key, value in info.items():
        if not isinstance(value, (int, float, np.floating, np.integer)):
            continue
        sanitized = str(key).replace("/", "_")
        if (
            sanitized.startswith("attacker_")
            or sanitized.startswith("sage_")
            or sanitized.startswith("mm_")
            or sanitized.startswith("gap_")
            or sanitized.startswith("baseline_")
            or sanitized.startswith("episode_")
        ):
            if sanitized not in record:
                record[sanitized] = float(value)
    return record


def run_online_policy_episode(
    env: gym.Env,
    *,
    action_fn: Callable[[np.ndarray, dict[str, Any], int], np.ndarray],
    max_steps: int | None = None,
    reset_options: dict[str, Any] | None = None,
    episode_id: str | None = None,
) -> OnlineEpisodeResult:
    observation, info = env.reset(options=reset_options)
    accumulator = EpisodeAccumulator()
    total_reward = 0.0
    step_records: list[dict[str, Any]] = []
    step_index = 0

    while True:
        action = np.asarray(action_fn(observation, dict(info), step_index), dtype=np.float32)
        next_observation, reward, terminated, truncated, info = env.step(action)
        record = _plain_step_record(step_index=step_index, action=action, reward=float(reward), info=dict(info))
        accumulator.add({key: value for key, value in record.items() if isinstance(value, (int, float)) and key != "step"})
        total_reward += float(reward)
        step_records.append(record)
        observation = next_observation
        step_index += 1
        if terminated or truncated:
            break
        if max_steps is not None and step_index >= int(max_steps):
            break

    metrics = accumulator.summary()
    metrics["episode_total_reward"] = float(total_reward)
    metrics["episode_num_steps"] = float(step_index)
    return OnlineEpisodeResult(
        episode_id=str(episode_id or f"episode-{int(step_index):05d}"),
        num_steps=int(step_index),
        total_reward=float(total_reward),
        metrics=metrics,
        step_records=step_records,
    )


def save_json(path: str, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)


def try_import_wandb() -> Any | None:
    try:
        import wandb  # type: ignore

        if not hasattr(wandb, "init"):
            raise ImportError("wandb import did not resolve to the package")
        return wandb
    except Exception:
        return None


def numeric_info_payload(info: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in info.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            payload[str(key)] = float(value)
    return payload
