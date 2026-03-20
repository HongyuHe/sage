from __future__ import annotations

from collections import deque
from typing import Mapping, Sequence

import numpy as np

from attacks.online.shm import DEFAULT_OBS_COLS


#* Shared semantic feature names used by the Sage shield tooling.
CURRENT_VALUE_COLUMNS: tuple[str, ...] = (
    "current_rtt_ms",
    "current_rttvar_ms",
    "current_delivery_rate_mbps",
    "windowed_delivery_rate_mbps",
    "max_windowed_delivery_rate_mbps",
    "current_loss_mbps",
    "current_min_rtt_ratio",
    "previous_action",
    "time_delta_ms",
    "delivery_growth_ratio",
    "max_delivery_growth_ratio",
)

HISTORY_VALUE_COLUMNS: tuple[str, ...] = (
    "current_rtt_ms",
    "windowed_delivery_rate_mbps",
    "current_loss_mbps",
    "current_min_rtt_ratio",
    "previous_action",
)

DERIVED_VALUE_COLUMNS: tuple[str, ...] = (
    "rtt_inflation",
    "windowed_vs_max_rate_ratio",
    "loss_to_windowed_rate_ratio",
)

FEATURE_COLUMNS: tuple[str, ...] = CURRENT_VALUE_COLUMNS + DERIVED_VALUE_COLUMNS + tuple(
    f"{name}_{suffix}"
    for name in HISTORY_VALUE_COLUMNS
    for suffix in ("avg", "min", "max", "delta")
)

_OBS_RAW_INDEX_TO_NAME: dict[int, str] = {
    2: "current_rtt_ms",
    3: "current_rttvar_ms",
    7: "current_delivery_rate_mbps",
    65: "time_delta_ms",
    66: "current_min_rtt_ratio",
    67: "current_loss_mbps",
    69: "delivery_growth_ratio",
    71: "windowed_delivery_rate_mbps",
    73: "max_delivery_growth_ratio",
    74: "max_windowed_delivery_rate_mbps",
    76: "previous_action",
}

_OBS_RAW_INDEX_SCALE: dict[int, float] = {
    2: 100.0,
    7: 100.0,
    67: 100.0,
    71: 100.0,
    74: 100.0,
}


def _obs_index_map(obs_cols: Sequence[int]) -> dict[int, int]:
    return {int(raw_index): int(position) for position, raw_index in enumerate(obs_cols)}


def _sanitize_float(value: object, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return float(numeric)


def current_values_from_observation(
    observation: np.ndarray | Sequence[float],
    *,
    obs_cols: Sequence[int] = DEFAULT_OBS_COLS,
) -> dict[str, float]:
    observation_array = np.asarray(observation, dtype=np.float32).reshape(-1)
    obs_index = _obs_index_map(obs_cols)
    values = {name: 0.0 for name in CURRENT_VALUE_COLUMNS}
    for raw_index, feature_name in _OBS_RAW_INDEX_TO_NAME.items():
        position = obs_index.get(int(raw_index))
        if position is None or position >= observation_array.shape[0]:
            continue
        scale = float(_OBS_RAW_INDEX_SCALE.get(int(raw_index), 1.0))
        values[str(feature_name)] = _sanitize_float(observation_array[int(position)] * scale)
    return values


def current_values_from_info(info: Mapping[str, object]) -> dict[str, float]:
    return {
        "current_rtt_ms": _sanitize_float(info.get("sage/current_rtt_ms")),
        "current_rttvar_ms": _sanitize_float(info.get("sage/current_rttvar_ms")),
        "current_delivery_rate_mbps": _sanitize_float(info.get("sage/current_delivery_rate_mbps")),
        "windowed_delivery_rate_mbps": _sanitize_float(info.get("sage/windowed_delivery_rate_mbps")),
        "max_windowed_delivery_rate_mbps": _sanitize_float(info.get("sage/max_windowed_delivery_rate_mbps")),
        "current_loss_mbps": _sanitize_float(info.get("sage/current_loss_mbps")),
        "current_min_rtt_ratio": _sanitize_float(info.get("sage/current_min_rtt_ratio")),
        "previous_action": _sanitize_float(info.get("sage/previous_action")),
        "time_delta_ms": _sanitize_float(info.get("sage/time_delta_ms")),
        "delivery_growth_ratio": _sanitize_float(info.get("sage/delivery_growth_ratio")),
        "max_delivery_growth_ratio": _sanitize_float(info.get("sage/max_delivery_growth_ratio")),
    }


class ShieldFeatureTracker:
    def __init__(self, *, history_len: int = 4) -> None:
        self.history_len = max(int(history_len), 1)
        self._history = {
            name: deque(maxlen=self.history_len)
            for name in HISTORY_VALUE_COLUMNS
        }

    def reset(self) -> None:
        for history in self._history.values():
            history.clear()

    def _derived_values(self, current_values: Mapping[str, float]) -> dict[str, float]:
        windowed_rate = max(_sanitize_float(current_values.get("windowed_delivery_rate_mbps")), 0.0)
        max_windowed_rate = max(_sanitize_float(current_values.get("max_windowed_delivery_rate_mbps")), 1e-6)
        current_loss = max(_sanitize_float(current_values.get("current_loss_mbps")), 0.0)
        min_rtt_ratio = max(_sanitize_float(current_values.get("current_min_rtt_ratio")), 1e-6)
        return {
            "rtt_inflation": float(1.0 / min_rtt_ratio),
            "windowed_vs_max_rate_ratio": float(windowed_rate / max_windowed_rate),
            "loss_to_windowed_rate_ratio": float(current_loss / max(windowed_rate, 1e-6)),
        }

    def update_from_current_values(self, current_values: Mapping[str, float]) -> dict[str, float]:
        features = {
            name: _sanitize_float(current_values.get(name))
            for name in CURRENT_VALUE_COLUMNS
        }
        features.update(self._derived_values(features))

        for history_name in HISTORY_VALUE_COLUMNS:
            self._history[str(history_name)].append(float(features[str(history_name)]))

        for history_name in HISTORY_VALUE_COLUMNS:
            history_values = np.asarray(list(self._history[str(history_name)]), dtype=np.float64)
            if history_values.size == 0:
                history_values = np.asarray([0.0], dtype=np.float64)
            features[f"{history_name}_avg"] = float(np.mean(history_values))
            features[f"{history_name}_min"] = float(np.min(history_values))
            features[f"{history_name}_max"] = float(np.max(history_values))
            features[f"{history_name}_delta"] = float(history_values[-1] - history_values[0])

        return {name: float(features.get(name, 0.0)) for name in FEATURE_COLUMNS}

    def update_from_observation(
        self,
        observation: np.ndarray | Sequence[float],
        *,
        obs_cols: Sequence[int] = DEFAULT_OBS_COLS,
    ) -> dict[str, float]:
        return self.update_from_current_values(
            current_values_from_observation(observation, obs_cols=obs_cols)
        )

    def update_from_info(self, info: Mapping[str, object]) -> dict[str, float]:
        return self.update_from_current_values(current_values_from_info(info))
