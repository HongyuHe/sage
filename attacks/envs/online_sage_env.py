from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import os
import time
from typing import Any

import numpy as np

from attacks.mahimahi import DirectionConfig, MahimahiControlClient
from attacks.online import SageLaunchConfig, SageSharedMemoryReader, SageStep, is_placeholder_step, launch_sage


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

_APPLIED_CONFIG_TOLERANCE = 1e-3


def _monotonic_ms() -> float:
    if hasattr(time, "clock_gettime_ns") and hasattr(time, "CLOCK_MONOTONIC"):
        return float(time.clock_gettime_ns(time.CLOCK_MONOTONIC)) / 1_000_000.0
    return float(time.monotonic_ns()) / 1_000_000.0


@dataclass(frozen=True)
class AttackBounds:
    uplink_bw_mbps: tuple[float, float] = (0.5, 192.0)
    downlink_bw_mbps: tuple[float, float] = (0.5, 192.0)
    uplink_loss: tuple[float, float] = (0.0, 0.15)
    downlink_loss: tuple[float, float] = (0.0, 0.15)
    uplink_delay_ms: tuple[float, float] = (0.0, 150.0)
    downlink_delay_ms: tuple[float, float] = (0.0, 150.0)

    @property
    def low(self) -> np.ndarray:
        return np.asarray(
            [
                self.uplink_bw_mbps[0],
                self.downlink_bw_mbps[0],
                self.uplink_loss[0],
                self.downlink_loss[0],
                self.uplink_delay_ms[0],
                self.downlink_delay_ms[0],
            ],
            dtype=np.float32,
        )

    @property
    def high(self) -> np.ndarray:
        return np.asarray(
            [
                self.uplink_bw_mbps[1],
                self.downlink_bw_mbps[1],
                self.uplink_loss[1],
                self.downlink_loss[1],
                self.uplink_delay_ms[1],
                self.downlink_delay_ms[1],
            ],
            dtype=np.float32,
        )


class OnlineSageAttackEnv(gym.Env):
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
        smooth_penalty_scale: float = 0.05,
        reward_scale: float = 1.0,
        runtime_dir: str = "attacks/runtime",
    ) -> None:
        super().__init__()

        self.repo_root = os.path.abspath(repo_root)
        minimum_ready_signal_timeout_ms = max(int(float(launch_timeout_s) * 1000.0) + 30_000, 1)
        configured_ready_signal_timeout_ms = (
            int(launch_config.ready_signal_timeout_ms)
            if launch_config.ready_signal_timeout_ms is not None
            else 0
        )
        self.launch_config = replace(
            launch_config,
            ready_signal_timeout_ms=max(configured_ready_signal_timeout_ms, minimum_ready_signal_timeout_ms),
        )
        self.bounds = bounds or AttackBounds()
        self.obs_history_len = int(obs_history_len)
        self.attack_interval_ms = float(attack_interval_ms)
        self.max_episode_steps = int(max_episode_steps)
        self.launch_timeout_s = float(launch_timeout_s)
        self.step_timeout_s = float(step_timeout_s)
        self.smooth_penalty_scale = float(smooth_penalty_scale)
        self.reward_scale = float(reward_scale)
        self.runtime_dir = os.path.abspath(os.path.join(self.repo_root, runtime_dir))

        self._feature_dim = 69 + 1 + 14 + 6
        obs_high = np.full((self.obs_history_len * self._feature_dim,), 1e9, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=self.bounds.low, high=self.bounds.high, dtype=np.float32)

        self._obs_history: deque[np.ndarray] = deque(maxlen=self.obs_history_len)
        self._control_client: MahimahiControlClient | None = None
        self._sage_reader: SageSharedMemoryReader | None = None
        self._sage_process = None
        self._episode_index = 0
        self._steps_taken = 0
        self._last_action = self._default_action()
        self._pending_action: np.ndarray | None = None
        self._last_victim_step: SageStep | None = None
        self._has_real_victim_step = False

    def _recent_process_logs(self, max_lines: int = 40) -> str:
        if self._sage_process is None:
            return ""

        def tail_file(path: str) -> str:
            if not path or not os.path.exists(path):
                return ""
            with open(path, "r", encoding="utf-8", errors="ignore") as file_obj:
                lines = file_obj.readlines()
            if not lines:
                return ""
            return "".join(lines[-max_lines:]).strip()

        stderr_tail = tail_file(self._sage_process.stderr_path)
        stdout_tail = tail_file(self._sage_process.stdout_path)
        chunks: list[str] = []
        if stderr_tail:
            chunks.append(f"[sage stderr]\n{stderr_tail}")
        if stdout_tail:
            chunks.append(f"[sage stdout]\n{stdout_tail}")
        return "\n\n".join(chunks)

    def _raise_launch_error(self, message: str, *, cause: Exception | None = None) -> None:
        details = self._recent_process_logs()
        if details:
            message = f"{message}\n\nRecent Sage logs:\n{details}"
        if cause is not None:
            raise RuntimeError(message) from cause
        raise RuntimeError(message)

    def _default_action(self) -> np.ndarray:
        default = np.asarray(
            [
                self.launch_config.initial_uplink_bw_mbps or float(self.launch_config.env_bw_mbps),
                self.launch_config.initial_downlink_bw_mbps or float(self.launch_config.env_bw_mbps),
                self.launch_config.initial_uplink_loss or 0.0,
                self.launch_config.initial_downlink_loss or 0.0,
                self.launch_config.initial_uplink_delay_ms or float(self.launch_config.latency_ms),
                self.launch_config.initial_downlink_delay_ms or float(self.launch_config.latency_ms),
            ],
            dtype=np.float32,
        )
        return np.clip(default, self.action_space.low, self.action_space.high)

    def _episode_dir(self) -> str:
        return os.path.join(self.runtime_dir, f"episode-{self._episode_index:05d}")

    def _make_direction_configs(
        self,
        action: np.ndarray,
        *,
        episode_step: int = 0,
        effective_after_abs_ms: float = 0.0,
    ) -> tuple[DirectionConfig, DirectionConfig]:
        clipped = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        uplink_queue_packets = int(self.launch_config.initial_uplink_queue_packets or self.launch_config.qsize_packets)
        downlink_queue_packets = int(self.launch_config.initial_downlink_queue_packets or self.launch_config.qsize_packets)
        uplink = DirectionConfig(
            bandwidth_mbps=float(clipped[0]),
            loss_rate=float(clipped[2]),
            delay_ms=float(clipped[4]),
            queue_packets=uplink_queue_packets,
            queue_bytes=int(self.launch_config.initial_uplink_queue_bytes or 0),
            episode_step=max(int(episode_step), 0),
            effective_after_abs_ms=max(float(effective_after_abs_ms), 0.0),
        )
        downlink = DirectionConfig(
            bandwidth_mbps=float(clipped[1]),
            loss_rate=float(clipped[3]),
            delay_ms=float(clipped[5]),
            queue_packets=downlink_queue_packets,
            queue_bytes=int(self.launch_config.initial_downlink_queue_bytes or 0),
            episode_step=max(int(episode_step), 0),
            effective_after_abs_ms=max(float(effective_after_abs_ms), 0.0),
        )
        return uplink, downlink

    def _control_snapshot(self):
        assert self._control_client is not None
        return self._control_client.snapshot()

    def _telemetry_features(self, *, snapshot=None) -> np.ndarray:
        snapshot = snapshot if snapshot is not None else self._control_snapshot()
        up = snapshot.uplink_telemetry
        down = snapshot.downlink_telemetry
        return np.nan_to_num(
            np.asarray(
                [
                    up.applied_bandwidth_mbps,
                    up.applied_loss_rate,
                    up.applied_delay_ms,
                    float(up.queue_occupancy_packets),
                    float(up.queue_occupancy_bytes),
                    up.queue_delay_ms,
                    up.departure_rate_mbps,
                    down.applied_bandwidth_mbps,
                    down.applied_loss_rate,
                    down.applied_delay_ms,
                    float(down.queue_occupancy_packets),
                    float(down.queue_occupancy_bytes),
                    down.queue_delay_ms,
                    down.departure_rate_mbps,
                ],
                dtype=np.float32,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)

    def _applied_step(self, *, snapshot) -> int:
        up_step = int(round(float(snapshot.uplink_telemetry.applied_step)))
        down_step = int(round(float(snapshot.downlink_telemetry.applied_step)))
        return min(up_step, down_step)

    def _snapshot_matches_action(self, *, snapshot, action: np.ndarray) -> bool:
        requested = np.asarray(action, dtype=np.float32)
        up = snapshot.uplink_telemetry
        down = snapshot.downlink_telemetry
        return (
            abs(float(up.applied_bandwidth_mbps) - float(requested[0])) <= _APPLIED_CONFIG_TOLERANCE
            and abs(float(down.applied_bandwidth_mbps) - float(requested[1])) <= _APPLIED_CONFIG_TOLERANCE
            and abs(float(up.applied_loss_rate) - float(requested[2])) <= _APPLIED_CONFIG_TOLERANCE
            and abs(float(down.applied_loss_rate) - float(requested[3])) <= _APPLIED_CONFIG_TOLERANCE
            and abs(float(up.applied_delay_ms) - float(requested[4])) <= _APPLIED_CONFIG_TOLERANCE
            and abs(float(down.applied_delay_ms) - float(requested[5])) <= _APPLIED_CONFIG_TOLERANCE
        )

    def _sage_metrics(self, step: SageStep) -> dict[str, float]:
        raw = np.asarray(step.raw, dtype=np.float32)
        current_rtt_ms = float(raw[2] * 100.0) if raw.shape[0] > 2 else 0.0
        current_rttvar_ms = float(raw[3]) if raw.shape[0] > 3 else 0.0
        current_delivery_rate_mbps = float(raw[7] * 100.0) if raw.shape[0] > 7 else 0.0
        time_delta_ms = float(raw[65]) if raw.shape[0] > 65 else 0.0
        current_min_rtt_ratio = float(raw[66]) if raw.shape[0] > 66 else 0.0
        current_loss_mbps = float(raw[67] * 100.0) if raw.shape[0] > 67 else 0.0
        delivery_growth_ratio = float(raw[69]) if raw.shape[0] > 69 else 0.0
        windowed_delivery_rate_mbps = float(raw[71] * 100.0) if raw.shape[0] > 71 else current_delivery_rate_mbps
        max_delivery_growth_ratio = float(raw[73]) if raw.shape[0] > 73 else 0.0
        max_windowed_delivery_rate_mbps = float(raw[74] * 100.0) if raw.shape[0] > 74 else 0.0
        return {
            "sage/current_rtt_ms": current_rtt_ms,
            "sage/current_rttvar_ms": current_rttvar_ms,
            "sage/current_delivery_rate_mbps": current_delivery_rate_mbps,
            "sage/time_delta_ms": time_delta_ms,
            "sage/windowed_delivery_rate_mbps": windowed_delivery_rate_mbps,
            "sage/max_windowed_delivery_rate_mbps": max_windowed_delivery_rate_mbps,
            "sage/current_loss_mbps": current_loss_mbps,
            "sage/current_min_rtt_ratio": current_min_rtt_ratio,
            "sage/delivery_growth_ratio": delivery_growth_ratio,
            "sage/max_delivery_growth_ratio": max_delivery_growth_ratio,
        }

    def _normalized_action(self, action: np.ndarray) -> np.ndarray:
        denom = np.maximum(self.action_space.high - self.action_space.low, 1e-6)
        return ((action - self.action_space.low) / denom).astype(np.float32, copy=False)

    def _build_feature(self, step: SageStep, action: np.ndarray, *, telemetry: np.ndarray | None = None) -> np.ndarray:
        telemetry_features = telemetry if telemetry is not None else self._telemetry_features()
        feature = np.concatenate(
            [
                np.asarray(step.observation, dtype=np.float32),
                np.asarray([step.reward], dtype=np.float32),
                telemetry_features,
                self._normalized_action(action),
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        return np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def _stacked_observation(self) -> np.ndarray:
        return np.concatenate(list(self._obs_history), axis=0).astype(np.float32, copy=False)

    def _is_observable_step(self, step: SageStep) -> bool:
        #* Treat non-zero reward as usable even if all feature slots are still zero.
        #* This avoids startup deadlocks when Sage emits sparse early samples.
        if not is_placeholder_step(step):
            return True
        return abs(float(step.reward)) > 1e-9

    def _wait_for_initial_real_step(self, *, timeout_s: float) -> tuple[SageStep, bool]:
        if self._sage_reader is None or self._sage_process is None:
            raise RuntimeError("Sage is not initialized")

        remaining_timeout_s = max(float(timeout_s), 0.0)
        initial_timeout = min(max(remaining_timeout_s, 0.05), 1.0)
        deadline = time.monotonic() + remaining_timeout_s
        step = self._sage_reader.read_latest(require_new=False, timeout_s=initial_timeout)
        if self._is_observable_step(step):
            return step, False

        last_step = step
        while time.monotonic() < deadline:
            if self._sage_process.poll() is not None:
                break
            remaining = max(deadline - time.monotonic(), 0.0)
            if remaining <= 0.0:
                break
            try:
                candidate = self._sage_reader.read_latest(require_new=True, timeout_s=min(1.0, remaining))
            except TimeoutError:
                continue
            last_step = candidate
            if self._is_observable_step(candidate):
                return candidate, False

        if self._sage_process.poll() is not None:
            self._raise_launch_error(
                f"Sage never produced a real observation within {remaining_timeout_s:.1f}s; "
                f"last rid={int(last_step.rid)} remained a placeholder"
            )

        #* Keep training alive: bootstrap with the latest placeholder and let
        #* step() transition to real victim samples when they appear.
        return last_step, True

    def _cleanup(self) -> None:
        if self._sage_reader is not None:
            self._sage_reader.close()
            self._sage_reader = None
        if self._sage_process is not None:
            self._sage_process.terminate()
            self._sage_process = None
        if self._control_client is not None:
            self._control_client.close()
            self._control_client = None

    def close(self) -> None:
        self._cleanup()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        try:
            super().reset(seed=seed)
        except TypeError:
            if seed is not None and hasattr(self, "seed"):
                self.seed(seed)
        self._cleanup()

        self._steps_taken = 0
        self._episode_index += 1
        self._last_action = self._default_action()
        self._pending_action = None
        self._last_victim_step = None
        self._has_real_victim_step = False
        episode_dir = self._episode_dir()
        os.makedirs(episode_dir, exist_ok=True)

        control_file = os.path.join(episode_dir, "mahimahi.control")
        keys_file = os.path.join(episode_dir, "sage.keys.json")
        #* Prevent attaching to a stale shared-memory key from a previous failed run.
        try:
            if os.path.exists(keys_file):
                os.remove(keys_file)
        except OSError:
            pass
        uplink, downlink = self._make_direction_configs(self._last_action)
        self._control_client = MahimahiControlClient(
            control_file,
            label=f"sage-adv-{self._episode_index}",
            initial_uplink=uplink,
            initial_downlink=downlink,
        )
        self._sage_process = launch_sage(
            self.repo_root,
            self.launch_config,
            control_file=control_file,
            keys_file=keys_file,
            runtime_dir=episode_dir,
        )
        launch_started_at = time.monotonic()
        try:
            self._sage_reader = SageSharedMemoryReader.from_keys_file(keys_file, timeout_s=self.launch_timeout_s)
            remaining_launch_timeout_s = max(self.launch_timeout_s - (time.monotonic() - launch_started_at), 0.0)
            initial_step, placeholder_bootstrap = self._wait_for_initial_real_step(
                timeout_s=remaining_launch_timeout_s
            )
        except Exception as exc:
            exit_code = self._sage_process.poll()
            prefix = "failed to launch Sage"
            if exit_code is not None:
                prefix = f"Sage exited during reset with code {exit_code}"
            self._raise_launch_error(
                f"{prefix}; no initial observation became available within {self.launch_timeout_s:.1f}s",
                cause=exc,
            )

        self._obs_history.clear()
        self._last_victim_step = initial_step
        self._has_real_victim_step = not bool(placeholder_bootstrap)
        snapshot = self._control_snapshot()
        telemetry = self._telemetry_features(snapshot=snapshot)
        initial_feature = self._build_feature(initial_step, self._last_action, telemetry=telemetry)
        for _ in range(self.obs_history_len):
            self._obs_history.append(np.asarray(initial_feature, dtype=np.float32))

        info = {
            "sage/reward": float(initial_step.reward),
            "sage/rid": int(initial_step.rid),
            "sage/previous_action": float(initial_step.previous_action),
            "env/nonfinite_sage_values": float(initial_step.nonfinite_count),
            "env/bootstrap_placeholder": 1.0 if placeholder_bootstrap else 0.0,
            "mm/up_applied_step": float(snapshot.uplink_telemetry.applied_step),
            "mm/down_applied_step": float(snapshot.downlink_telemetry.applied_step),
            "mm/up_applied_effective_after_abs_ms": float(
                snapshot.uplink_telemetry.applied_effective_after_abs_ms
            ),
            "mm/down_applied_effective_after_abs_ms": float(
                snapshot.downlink_telemetry.applied_effective_after_abs_ms
            ),
        }
        if placeholder_bootstrap:
            info["env/error"] = "sage_initial_placeholder_bootstrap"
        info.update(self._sage_metrics(initial_step))
        return self._stacked_observation(), info

    def apply_action(
        self,
        action,
        *,
        episode_step: int = 0,
        effective_after_abs_ms: float = 0.0,
    ) -> np.ndarray:
        if self._control_client is None or self._sage_reader is None or self._sage_process is None:
            raise RuntimeError("environment is not initialized; call reset() first")

        clipped = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        uplink, downlink = self._make_direction_configs(
            clipped,
            episode_step=episode_step,
            effective_after_abs_ms=effective_after_abs_ms,
        )
        self._control_client.update(uplink=uplink, downlink=downlink)
        self._pending_action = clipped.astype(np.float32, copy=True)
        return clipped

    def collect_step(
        self,
        *,
        expected_episode_step: int | None = None,
        expected_action: np.ndarray | None = None,
        strict: bool = False,
        deadline_abs_ms: float | None = None,
        timeout_s: float | None = None,
    ):
        if self._control_client is None or self._sage_reader is None or self._sage_process is None:
            raise RuntimeError("environment is not initialized; call reset() first")

        pending_action = (
            np.asarray(self._pending_action, dtype=np.float32)
            if self._pending_action is not None
            else np.asarray(self._last_action, dtype=np.float32)
        )
        reference_action = (
            np.asarray(expected_action, dtype=np.float32)
            if expected_action is not None
            else pending_action
        )
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        previous_step = self._last_victim_step
        victim_step = None
        snapshot = None

        if self._sage_process.poll() is not None:
            terminated = True
            truncated = True
        else:
            deadline = (
                float(deadline_abs_ms)
                if deadline_abs_ms is not None
                else _monotonic_ms() + 1000.0 * float(timeout_s if timeout_s is not None else self.step_timeout_s)
            )
            while _monotonic_ms() < deadline and self._sage_process.poll() is None:
                remaining_s = max((deadline - _monotonic_ms()) / 1000.0, 1e-3)
                try:
                    candidate = self._sage_reader.read_latest(
                        require_new=True,
                        timeout_s=min(remaining_s, float(timeout_s if timeout_s is not None else self.step_timeout_s)),
                    )
                except TimeoutError:
                    continue

                if not self._is_observable_step(candidate):
                    if strict:
                        continue
                    victim_step = previous_step
                    info["env/error"] = "sage_placeholder_observation_reused_last"
                    break

                snapshot = self._control_snapshot()
                if not self._snapshot_matches_action(snapshot=snapshot, action=reference_action):
                    continue
                if expected_episode_step is not None and self._applied_step(snapshot=snapshot) < int(expected_episode_step):
                    continue

                victim_step = candidate
                break

            if victim_step is None and previous_step is not None and not strict:
                victim_step = previous_step
                info["env/error"] = "sage_observation_timeout_reused_last"

        if victim_step is None:
            self._pending_action = None
            if "env/error" not in info:
                info["env/error"] = (
                    "sage_process_exited" if self._sage_process.poll() is not None else "sage_observation_timeout"
                )
            obs = self._stacked_observation()
            return obs, 0.0, terminated, True if strict else truncated, info

        if snapshot is None:
            snapshot = self._control_snapshot()
        self._last_victim_step = victim_step
        if self._is_observable_step(victim_step):
            self._has_real_victim_step = True

        smooth_penalty = float(
            np.mean(np.abs(self._normalized_action(pending_action) - self._normalized_action(self._last_action)))
        )
        reward = float(-self.reward_scale * victim_step.reward - self.smooth_penalty_scale * smooth_penalty)

        telemetry = self._telemetry_features(snapshot=snapshot)
        self._last_action = pending_action.astype(np.float32, copy=True)
        self._pending_action = None
        self._obs_history.append(self._build_feature(victim_step, self._last_action, telemetry=telemetry))
        self._steps_taken += 1
        if self._steps_taken >= self.max_episode_steps:
            truncated = True
        uplink = snapshot.uplink
        downlink = snapshot.downlink
        info.update(
            {
                "sage/reward": float(victim_step.reward),
                "sage/rid": int(victim_step.rid),
                "sage/previous_action": float(victim_step.previous_action),
                "env/nonfinite_sage_values": float(victim_step.nonfinite_count),
                "attacker/reward": float(reward),
                "attacker/smooth_penalty": float(smooth_penalty),
                "attacker/uplink_bw_mbps": float(self._last_action[0]),
                "attacker/downlink_bw_mbps": float(self._last_action[1]),
                "attacker/uplink_loss": float(self._last_action[2]),
                "attacker/downlink_loss": float(self._last_action[3]),
                "attacker/uplink_delay_ms": float(self._last_action[4]),
                "attacker/downlink_delay_ms": float(self._last_action[5]),
                "mm/up_queue_packets": float(uplink.queue_packets),
                "mm/down_queue_packets": float(downlink.queue_packets),
                "mm/up_applied_bw_mbps": float(telemetry[0]),
                "mm/up_applied_loss_rate": float(telemetry[1]),
                "mm/up_applied_delay_ms": float(telemetry[2]),
                "mm/up_queue_delay_ms": float(telemetry[5]),
                "mm/up_departure_rate_mbps": float(telemetry[6]),
                "mm/down_applied_bw_mbps": float(telemetry[7]),
                "mm/down_applied_loss_rate": float(telemetry[8]),
                "mm/down_applied_delay_ms": float(telemetry[9]),
                "mm/down_queue_delay_ms": float(telemetry[12]),
                "mm/down_departure_rate_mbps": float(telemetry[13]),
                "mm/up_applied_step": float(snapshot.uplink_telemetry.applied_step),
                "mm/down_applied_step": float(snapshot.downlink_telemetry.applied_step),
                "mm/up_applied_effective_after_abs_ms": float(
                    snapshot.uplink_telemetry.applied_effective_after_abs_ms
                ),
                "mm/down_applied_effective_after_abs_ms": float(
                    snapshot.downlink_telemetry.applied_effective_after_abs_ms
                ),
            }
        )
        info.update(self._sage_metrics(victim_step))
        return self._stacked_observation(), reward, terminated, truncated, info

    def step(self, action):
        self.apply_action(action)
        time.sleep(self.attack_interval_ms / 1000.0)
        return self.collect_step(strict=False, timeout_s=self.step_timeout_s)
