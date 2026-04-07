from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from attacks.envs import AttackBounds
from scripts._trace_attack_common import (
    build_clean_action_schedule,
    load_mahimahi_trace_schedule,
    load_trace_entries,
)


@dataclass(frozen=True)
class CleanTraceSequence:
    trace_id: str
    trace_name: str
    source_trace: dict[str, Any]
    shared_bandwidth_mbps: np.ndarray
    shared_loss_rate: np.ndarray
    uplink_delay_ms: float
    downlink_delay_ms: float

    @property
    def num_steps(self) -> int:
        return int(self.shared_bandwidth_mbps.shape[0])

    def to_action_schedule(self) -> list[np.ndarray]:
        return shared_components_to_action_schedule(
            shared_bandwidth_mbps=self.shared_bandwidth_mbps,
            shared_loss_rate=self.shared_loss_rate,
            uplink_delay_ms=self.uplink_delay_ms,
            downlink_delay_ms=self.downlink_delay_ms,
        )


def default_attack_delay_ms(config_payload: dict[str, Any], *, direction: str) -> float:
    init_value = config_payload.get(f"init_{direction}_delay_ms")
    if init_value is not None:
        return float(init_value)
    return float(config_payload.get("latency_ms", 25.0))


def action_schedule_to_shared_components(
    action_schedule: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if not action_schedule:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            0.0,
            0.0,
        )

    shared_bandwidth: list[float] = []
    shared_loss: list[float] = []
    uplink_delays: list[float] = []
    downlink_delays: list[float] = []
    for action in action_schedule:
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw.shape[0] < 6:
            raise ValueError(f"expected effective action with 6 values, got shape={raw.shape}")
        shared_bandwidth.append(float(min(float(raw[0]), float(raw[1]))))
        shared_loss.append(float(min(float(raw[2]), float(raw[3]))))
        uplink_delays.append(float(raw[4]))
        downlink_delays.append(float(raw[5]))

    uplink_delay_ms = float(np.median(np.asarray(uplink_delays, dtype=np.float32)))
    downlink_delay_ms = float(np.median(np.asarray(downlink_delays, dtype=np.float32)))
    return (
        np.asarray(shared_bandwidth, dtype=np.float32),
        np.asarray(shared_loss, dtype=np.float32),
        uplink_delay_ms,
        downlink_delay_ms,
    )


def shared_components_to_action_schedule(
    *,
    shared_bandwidth_mbps: Sequence[float] | np.ndarray,
    shared_loss_rate: Sequence[float] | np.ndarray,
    uplink_delay_ms: float,
    downlink_delay_ms: float,
) -> list[np.ndarray]:
    bandwidth = np.asarray(shared_bandwidth_mbps, dtype=np.float32).reshape(-1)
    loss = np.asarray(shared_loss_rate, dtype=np.float32).reshape(-1)
    if bandwidth.shape != loss.shape:
        raise ValueError("shared_bandwidth_mbps and shared_loss_rate must have matching shape")

    actions: list[np.ndarray] = []
    for bw_value, loss_value in zip(bandwidth.tolist(), loss.tolist()):
        actions.append(
            np.asarray(
                [
                    float(max(bw_value, 0.0)),
                    float(max(bw_value, 0.0)),
                    float(np.clip(loss_value, 0.0, 1.0)),
                    float(np.clip(loss_value, 0.0, 1.0)),
                    float(max(uplink_delay_ms, 0.0)),
                    float(max(downlink_delay_ms, 0.0)),
                ],
                dtype=np.float32,
            )
        )
    return actions


def pad_1d_sequences(sequences: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not sequences:
        raise ValueError("sequences must not be empty")

    lengths = np.asarray([int(np.asarray(sequence).shape[0]) for sequence in sequences], dtype=np.int64)
    if np.any(lengths <= 0):
        raise ValueError("all sequences must have positive length")

    max_len = int(np.max(lengths))
    padded = np.zeros((len(sequences), max_len), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        values = np.asarray(sequence, dtype=np.float32).reshape(-1)
        padded[index, : values.shape[0]] = values
    return padded, lengths


def replay_bounds_for_action_schedules(
    base_bounds: AttackBounds,
    action_schedules: Sequence[Sequence[np.ndarray]],
) -> AttackBounds:
    flattened: list[np.ndarray] = []
    for schedule in action_schedules:
        if not schedule:
            continue
        schedule_array = np.asarray([np.asarray(action, dtype=np.float32).reshape(-1) for action in schedule], dtype=np.float32)
        if schedule_array.ndim != 2 or schedule_array.shape[1] < 6:
            raise ValueError("action schedules must contain 6D effective actions")
        flattened.append(schedule_array[:, :6])
    if not flattened:
        return base_bounds

    values = np.concatenate(flattened, axis=0)
    return AttackBounds(
        uplink_bw_mbps=(
            min(float(base_bounds.uplink_bw_mbps[0]), float(np.min(values[:, 0]))),
            max(float(base_bounds.uplink_bw_mbps[1]), float(np.max(values[:, 0]))),
        ),
        downlink_bw_mbps=(
            min(float(base_bounds.downlink_bw_mbps[0]), float(np.min(values[:, 1]))),
            max(float(base_bounds.downlink_bw_mbps[1]), float(np.max(values[:, 1]))),
        ),
        uplink_loss=(
            min(float(base_bounds.uplink_loss[0]), float(np.min(values[:, 2]))),
            max(float(base_bounds.uplink_loss[1]), float(np.max(values[:, 2]))),
        ),
        downlink_loss=(
            min(float(base_bounds.downlink_loss[0]), float(np.min(values[:, 3]))),
            max(float(base_bounds.downlink_loss[1]), float(np.max(values[:, 3]))),
        ),
        uplink_delay_ms=(
            min(float(base_bounds.uplink_delay_ms[0]), float(np.min(values[:, 4]))),
            max(float(base_bounds.uplink_delay_ms[1]), float(np.max(values[:, 4]))),
        ),
        downlink_delay_ms=(
            min(float(base_bounds.downlink_delay_ms[0]), float(np.min(values[:, 5]))),
            max(float(base_bounds.downlink_delay_ms[1]), float(np.max(values[:, 5]))),
        ),
    )


def bandwidth_projection_bounds(
    shared_bw: np.ndarray,
    lengths: np.ndarray,
    *,
    eps_abs: float,
    eps_rel: float,
) -> tuple[float, float]:
    values = np.asarray(shared_bw, dtype=np.float32)
    seq_lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if values.ndim != 2:
        raise ValueError("shared_bw must have shape [N, L]")
    if seq_lengths.shape != (values.shape[0],):
        raise ValueError("lengths must have shape [N]")

    mask = np.arange(values.shape[1], dtype=np.int64)[None, :] < seq_lengths[:, None]
    valid = values[mask]
    if valid.size == 0:
        raise ValueError("no valid bandwidth values were provided")

    lower = np.full(valid.shape, -np.inf, dtype=np.float32)
    upper = np.full(valid.shape, np.inf, dtype=np.float32)
    constrained = False
    if float(eps_abs) > 0.0:
        lower = np.maximum(lower, valid - float(eps_abs))
        upper = np.minimum(upper, valid + float(eps_abs))
        constrained = True
    if float(eps_rel) > 0.0:
        lower = np.maximum(lower, valid * float(max(0.0, 1.0 - float(eps_rel))))
        upper = np.minimum(upper, valid * float(1.0 + float(eps_rel)))
        constrained = True
    if not constrained:
        lower = valid.copy()
        upper = valid.copy()
    return float(max(np.min(lower), 0.0)), float(max(np.max(upper), 0.0))


def load_clean_trace_sequences(
    *,
    manifest_path: str,
    config_payload: dict[str, Any],
    limit: int = -1,
) -> list[CleanTraceSequence]:
    entries = load_trace_entries(str(manifest_path))
    if int(limit) > 0:
        entries = entries[: int(limit)]

    interval_ms = float(config_payload.get("attack_interval_ms", 100.0))
    max_steps = max(int(config_payload.get("episode_steps", 6000)), 1)
    clean_uplink_delay_ms = default_attack_delay_ms(config_payload, direction="uplink")
    clean_downlink_delay_ms = default_attack_delay_ms(config_payload, direction="downlink")

    sequences: list[CleanTraceSequence] = []
    for entry in entries:
        schedule = load_mahimahi_trace_schedule(entry.copied_path, interval_ms=interval_ms)
        action_schedule = build_clean_action_schedule(
            schedule,
            uplink_delay_ms=clean_uplink_delay_ms,
            downlink_delay_ms=clean_downlink_delay_ms,
        )[:max_steps]
        if not action_schedule:
            continue
        shared_bandwidth, shared_loss, uplink_delay_ms, downlink_delay_ms = action_schedule_to_shared_components(
            action_schedule
        )
        sequences.append(
            CleanTraceSequence(
                trace_id=str(entry.trace_id),
                trace_name=str(entry.name),
                source_trace=entry.to_dict(),
                shared_bandwidth_mbps=shared_bandwidth,
                shared_loss_rate=shared_loss,
                uplink_delay_ms=uplink_delay_ms,
                downlink_delay_ms=downlink_delay_ms,
            )
        )
    return sequences
