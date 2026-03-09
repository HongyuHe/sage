from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any, Sequence

import numpy as np
import sysv_ipc

DEFAULT_OBS_COLS: tuple[int, ...] = (
    2, 3, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76,
)


@dataclass(frozen=True)
class SageStep:
    rid: int
    raw: np.ndarray
    observation: np.ndarray
    reward: float
    previous_action: float


def is_placeholder_step(step: SageStep, *, atol: float = 1e-9) -> bool:
    #* Sage writes bootstrap/fallback placeholder observations with all feature
    #* signals zeroed and only reward/previous_action optionally populated.
    raw = np.asarray(step.raw, dtype=np.float64)
    if raw.size < 2:
        return True
    signal_slice = raw[:-2]
    if signal_slice.size == 0:
        return True
    return bool(np.all(np.abs(signal_slice) <= float(atol)))


def wait_for_keys_file(path: str, timeout_s: float = 60.0, poll_interval_s: float = 0.1) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout_s)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if "mem_r" in payload and "mem_w" in payload:
                    return payload
            except Exception as exc:
                last_error = exc
        time.sleep(float(poll_interval_s))
    if last_error is not None:
        raise TimeoutError(f"timed out waiting for Sage keys file {path}: {last_error}") from last_error
    raise TimeoutError(f"timed out waiting for Sage keys file {path}")


class SageSharedMemoryReader:
    def __init__(
        self,
        mem_r_key: int,
        *,
        obs_cols: Sequence[int] = DEFAULT_OBS_COLS,
        input_dim: int = 77,
        poll_interval_s: float = 0.01,
    ) -> None:
        self.mem_r_key = int(mem_r_key)
        self.input_dim = int(input_dim)
        self.poll_interval_s = float(poll_interval_s)
        self.obs_cols = np.asarray(tuple(int(x) for x in obs_cols), dtype=np.int64)
        self._mem = sysv_ipc.SharedMemory(self.mem_r_key)
        self._last_rid: int | None = None

    @classmethod
    def from_keys_file(
        cls,
        path: str,
        *,
        timeout_s: float = 60.0,
        obs_cols: Sequence[int] = DEFAULT_OBS_COLS,
        input_dim: int = 77,
        poll_interval_s: float = 0.01,
    ) -> "SageSharedMemoryReader":
        payload = wait_for_keys_file(path, timeout_s=timeout_s, poll_interval_s=poll_interval_s)
        return cls(
            int(payload["mem_r"]),
            obs_cols=obs_cols,
            input_dim=int(payload.get("input_dim", input_dim)),
            poll_interval_s=poll_interval_s,
        )

    def close(self) -> None:
        try:
            self._mem.detach()
        except Exception:
            pass

    def _parse_payload(self, payload: bytes) -> SageStep | None:
        raw_text = payload.decode("utf-8", errors="ignore")
        end_idx = raw_text.find("\0")
        if end_idx >= 0:
            raw_text = raw_text[:end_idx]
        raw_text = raw_text.strip()
        if not raw_text:
            return None

        values = np.fromstring(raw_text, dtype=np.float64, sep=" ")
        if values.size < (self.input_dim + 1):
            return None

        rid = int(values[0])
        raw = values[1 : 1 + self.input_dim].astype(np.float32, copy=False)
        obs = raw[self.obs_cols].astype(np.float32, copy=False)
        reward = float(raw[-2])
        previous_action = float(raw[76]) if raw.shape[0] > 76 else 0.0
        return SageStep(rid=rid, raw=raw, observation=obs, reward=reward, previous_action=previous_action)

    def read_latest(self, *, require_new: bool = True, timeout_s: float = 5.0) -> SageStep:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            try:
                step = self._parse_payload(self._mem.read())
            except sysv_ipc.ExistentialError as exc:
                raise RuntimeError("Sage shared memory disappeared") from exc
            if step is None:
                time.sleep(self.poll_interval_s)
                continue
            if require_new and self._last_rid is not None and step.rid == self._last_rid:
                time.sleep(self.poll_interval_s)
                continue
            self._last_rid = int(step.rid)
            return step
        raise TimeoutError("timed out waiting for Sage observation")
