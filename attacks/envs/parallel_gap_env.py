from __future__ import annotations

from dataclasses import replace
import os
import socket
import time
from typing import Any

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

_BASE_OBS_FEATURE_DIM = 69 + 1 + 14 + 6
_SYNC_TOLERANCE_MS = 5.0
_CONFIG_TOLERANCE = 1e-3
_RETRYABLE_LAUNCH_MARKERS = (
    "Address already in use",
    "timed out waiting for Sage keys file",
    "no initial observation became available",
    "Sage never produced a real observation",
    "no response (OK_Signal)",
)


def _monotonic_ms() -> float:
    if hasattr(time, "clock_gettime_ns") and hasattr(time, "CLOCK_MONOTONIC"):
        return float(time.clock_gettime_ns(time.CLOCK_MONOTONIC)) / 1_000_000.0
    return float(time.monotonic_ns()) / 1_000_000.0


def _sleep_until(target_ms: float) -> None:
    remaining_ms = float(target_ms) - _monotonic_ms()
    if remaining_ms > 0.0:
        time.sleep(remaining_ms / 1000.0)


def _is_retryable_launch_error(exc: RuntimeError) -> bool:
    return any(marker in str(exc) for marker in _RETRYABLE_LAUNCH_MARKERS)


class ParallelGapAttackEnv(gym.Env):
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
        baseline_gap_alpha: float = 2.0,
        smooth_penalty_scale: float = 0.0,
        sync_guard_ms: float = 25.0,
        launch_retries: int = 6,
    ) -> None:
        super().__init__()
        self.repo_root = os.path.abspath(repo_root)
        self._base_launch_config = launch_config
        self._bounds = bounds or AttackBounds()
        self._obs_history_len = int(obs_history_len)
        self._attack_interval_ms = float(attack_interval_ms)
        self._max_episode_steps = max(1, int(max_episode_steps))
        self._launch_timeout_s = float(launch_timeout_s)
        self._step_timeout_s = float(step_timeout_s)
        self._runtime_dir = runtime_dir
        self._baseline_gap_alpha = float(baseline_gap_alpha)
        self._smooth_penalty_scale = float(smooth_penalty_scale)
        self._sync_guard_ms = max(float(sync_guard_ms), 1.0)
        self._launch_retries = max(int(launch_retries), 1)

        self._children: dict[str, OnlineSageAttackEnv] = {}
        self._launch_generation = 0
        self._episode_step = 0
        self._episode_anchor_abs_ms = 0.0
        self._base_rtt_ms = max(float(self._base_launch_config.latency_ms) * 2.0, 1.0)
        self.action_space = spaces.Box(low=self._bounds.low, high=self._bounds.high, dtype=np.float32)
        obs_dim = self._obs_history_len * _BASE_OBS_FEATURE_DIM + 3
        obs_high = np.full((obs_dim,), 1e9, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        self._last_action = self._default_action()
        self._last_sage_observation: np.ndarray | None = None
        self._previous_gap_features = np.zeros((3,), dtype=np.float32)
        self._bootstrap_pending_roles: set[str] = set()

    def _default_action(self) -> np.ndarray:
        return np.clip(
            np.asarray(
                [
                    self._base_launch_config.initial_uplink_bw_mbps or float(self._base_launch_config.env_bw_mbps),
                    self._base_launch_config.initial_downlink_bw_mbps or float(self._base_launch_config.env_bw_mbps),
                    self._base_launch_config.initial_uplink_loss or 0.0,
                    self._base_launch_config.initial_downlink_loss or 0.0,
                    self._base_launch_config.initial_uplink_delay_ms or float(self._base_launch_config.latency_ms),
                    self._base_launch_config.initial_downlink_delay_ms or float(self._base_launch_config.latency_ms),
                ],
                dtype=np.float32,
            ),
            self.action_space.low,
            self.action_space.high,
        )

    def _normalized_action(self, action: np.ndarray) -> np.ndarray:
        denom = np.maximum(self.action_space.high - self.action_space.low, 1e-6)
        return ((np.asarray(action, dtype=np.float32) - self.action_space.low) / denom).astype(np.float32, copy=False)

    def _reserve_launch_port(self, preferred_port: int, *, taken_ports: set[int], max_tries: int = 256) -> int:
        candidate = max(int(preferred_port), 1024)
        for offset in range(max(int(max_tries), 1)):
            port = candidate + offset
            if int(port) in taken_ports:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("0.0.0.0", port))
                except OSError:
                    continue
                taken_ports.add(int(port))
                return int(port)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("0.0.0.0", 0))
            port = int(sock.getsockname()[1])
            taken_ports.add(port)
            return port

    def _augment_observation(self, observation: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [np.asarray(observation, dtype=np.float32), self._previous_gap_features],
            axis=0,
        ).astype(np.float32, copy=False)

    def _child_runtime_dir(self, role: str) -> str:
        return os.path.join(str(self._runtime_dir), role)

    def _build_child(
        self,
        *,
        role: str,
        scheme: str,
        controller_mode: str,
        port_offset: int,
        taken_ports: set[int],
    ) -> OnlineSageAttackEnv:
        generation_offset = self._launch_generation * 10
        preferred_port = int(self._base_launch_config.port) + int(port_offset)
        resolved_port = self._reserve_launch_port(preferred_port, taken_ports=taken_ports)
        launch_config = replace(
            self._base_launch_config,
            scheme=str(scheme),
            controller_mode=str(controller_mode),
            port=resolved_port,
            actor_id=max(int(self._base_launch_config.actor_id) + generation_offset + int(port_offset), 0),
            iteration_id=int(self._base_launch_config.iteration_id) + generation_offset,
        )
        return OnlineSageAttackEnv(
            repo_root=self.repo_root,
            launch_config=launch_config,
            bounds=self._bounds,
            obs_history_len=self._obs_history_len,
            attack_interval_ms=self._attack_interval_ms,
            max_episode_steps=self._max_episode_steps,
            launch_timeout_s=self._launch_timeout_s,
            step_timeout_s=self._step_timeout_s,
            smooth_penalty_scale=0.0,
            reward_scale=1.0,
            runtime_dir=self._child_runtime_dir(role),
        )

    def _create_children(self) -> dict[str, OnlineSageAttackEnv]:
        self._launch_generation += 1
        taken_ports: set[int] = set()
        return {
            "sage": self._build_child(
                role="sage",
                scheme="pure",
                controller_mode="sage",
                port_offset=0,
                taken_ports=taken_ports,
            ),
            "cubic": self._build_child(
                role="cubic",
                scheme="cubic",
                controller_mode="kernel_cc",
                port_offset=1,
                taken_ports=taken_ports,
            ),
            "bbr": self._build_child(
                role="bbr",
                scheme="bbr",
                controller_mode="kernel_cc",
                port_offset=2,
                taken_ports=taken_ports,
            ),
        }

    def _applied_config_matches(self, info: dict[str, Any], action: np.ndarray) -> bool:
        requested = np.asarray(action, dtype=np.float32)
        return (
            abs(float(info.get("mm/up_applied_bw_mbps", -1.0)) - float(requested[0])) <= _CONFIG_TOLERANCE
            and abs(float(info.get("mm/down_applied_bw_mbps", -1.0)) - float(requested[1])) <= _CONFIG_TOLERANCE
            and abs(float(info.get("mm/up_applied_loss_rate", -1.0)) - float(requested[2])) <= _CONFIG_TOLERANCE
            and abs(float(info.get("mm/down_applied_loss_rate", -1.0)) - float(requested[3])) <= _CONFIG_TOLERANCE
            and abs(float(info.get("mm/up_applied_delay_ms", -1.0)) - float(requested[4])) <= _CONFIG_TOLERANCE
            and abs(float(info.get("mm/down_applied_delay_ms", -1.0)) - float(requested[5])) <= _CONFIG_TOLERANCE
        )

    def _sync_matches(
        self,
        info: dict[str, Any],
        *,
        action: np.ndarray,
        expected_step: int,
        effective_after_abs_ms: float,
    ) -> bool:
        up_step = int(round(float(info.get("mm/up_applied_step", -1.0))))
        down_step = int(round(float(info.get("mm/down_applied_step", -1.0))))
        step_matches = up_step == int(expected_step) and down_step == int(expected_step)

        up_effective_after = float(info.get("mm/up_applied_effective_after_abs_ms", -1.0))
        down_effective_after = float(info.get("mm/down_applied_effective_after_abs_ms", -1.0))
        effective_after_matches = (
            abs(up_effective_after - float(effective_after_abs_ms)) <= _SYNC_TOLERANCE_MS
            and abs(down_effective_after - float(effective_after_abs_ms)) <= _SYNC_TOLERANCE_MS
        )
        if step_matches and effective_after_matches:
            return True
        return self._applied_config_matches(info, action)

    def _score_from_info(self, info: dict[str, Any], *, path_cap_mbps: float) -> float:
        current_rtt_ms = max(float(info.get("sage/current_rtt_ms", 0.0)), 1e-6)
        windowed_rate_mbps = max(float(info.get("sage/windowed_delivery_rate_mbps", 0.0)), 0.0)
        current_loss_mbps = max(float(info.get("sage/current_loss_mbps", 0.0)), 0.0)
        rtt_term = min(self._base_rtt_ms / current_rtt_ms, 1.0)
        rate_term = max(windowed_rate_mbps - self._baseline_gap_alpha * current_loss_mbps, 0.0) / max(
            float(path_cap_mbps),
            1e-6,
        )
        return float(rtt_term * rate_term * rate_term)

    def close(self) -> None:
        for child in self._children.values():
            child.close()
        self._children = {}
        self._bootstrap_pending_roles = set()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        try:
            super().reset(seed=seed)
        except TypeError:
            if seed is not None and hasattr(self, "seed"):
                self.seed(seed)
        self.close()

        launch_error: RuntimeError | None = None
        for _ in range(self._launch_retries):
            children = self._create_children()
            try:
                initial_obs: dict[str, np.ndarray] = {}
                initial_infos: dict[str, dict[str, Any]] = {}
                placeholder_roles: list[str] = []
                for role, child in children.items():
                    observation, info = child.reset(seed=seed, options=options if role == "sage" else None)
                    if float(info.get("env/bootstrap_placeholder", 0.0)) > 0.0:
                        placeholder_roles.append(str(role))
                    initial_obs[role] = np.asarray(observation, dtype=np.float32)
                    initial_infos[role] = dict(info)
            except RuntimeError as exc:
                launch_error = exc
                for child in children.values():
                    child.close()
                if not _is_retryable_launch_error(exc):
                    raise
                continue

            self._children = children
            self._episode_step = 0
            self._episode_anchor_abs_ms = _monotonic_ms() + self._sync_guard_ms
            self._last_action = self._default_action()
            self._previous_gap_features = np.zeros((3,), dtype=np.float32)
            self._last_sage_observation = initial_obs["sage"]
            self._bootstrap_pending_roles = set(placeholder_roles)
            positive_rtts = [
                float(info.get("sage/current_rtt_ms", 0.0)) for info in initial_infos.values() if info.get("sage/current_rtt_ms")
            ]
            if positive_rtts:
                self._base_rtt_ms = max(min(positive_rtts), 1.0)
            info = dict(initial_infos["sage"])
            info.update(
                {
                    "env/bootstrap_placeholder": 1.0 if "sage" in placeholder_roles else 0.0,
                    "env/bootstrap_placeholder_children": float(len(placeholder_roles)),
                    "gap/base_rtt_ms": float(self._base_rtt_ms),
                    "gap/score_cubic": 0.0,
                    "gap/score_bbr": 0.0,
                    "gap/score_sage": 0.0,
                    "gap/baseline_score": 0.0,
                    "gap/reward": 0.0,
                    "gap/value": 0.0,
                }
            )
            if placeholder_roles:
                info["env/bootstrap_placeholder_roles"] = ",".join(sorted(placeholder_roles))
            return self._augment_observation(initial_obs["sage"]), info

        if launch_error is not None:
            raise launch_error
        raise RuntimeError("ParallelGapAttackEnv reset failed without diagnostics")

    def step(self, action):
        if not self._children:
            raise RuntimeError("environment is not initialized; call reset() first")

        clipped = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        bootstrap_mode = bool(self._bootstrap_pending_roles)
        expected_step = self._episode_step + 1
        scheduled_effective_after_abs_ms = self._episode_anchor_abs_ms + self._episode_step * self._attack_interval_ms
        effective_after_abs_ms = max(scheduled_effective_after_abs_ms, _monotonic_ms() + self._sync_guard_ms)
        if effective_after_abs_ms > scheduled_effective_after_abs_ms:
            self._episode_anchor_abs_ms = effective_after_abs_ms - self._episode_step * self._attack_interval_ms
        collect_after_abs_ms = effective_after_abs_ms + self._attack_interval_ms
        collect_deadline_abs_ms = collect_after_abs_ms + 1000.0 * self._step_timeout_s

        try:
            for child in self._children.values():
                child.apply_action(
                    clipped,
                    episode_step=expected_step,
                    effective_after_abs_ms=effective_after_abs_ms,
                )

            _sleep_until(collect_after_abs_ms)

            results: dict[str, tuple[np.ndarray, float, bool, bool, dict[str, Any]]] = {}
            sync_failed = False
            child_failed = False
            for role, child in self._children.items():
                observation, _, terminated, truncated, info = child.collect_step(
                    expected_episode_step=expected_step,
                    expected_action=clipped,
                    strict=not bootstrap_mode,
                    deadline_abs_ms=collect_deadline_abs_ms,
                    timeout_s=max((collect_deadline_abs_ms - _monotonic_ms()) / 1000.0, 1e-3),
                )
                results[role] = (np.asarray(observation, dtype=np.float32), 0.0, terminated, truncated, dict(info))
                if terminated or truncated:
                    child_failed = True
                if bootstrap_mode:
                    if bool(getattr(child, "_has_real_victim_step", False)):
                        self._bootstrap_pending_roles.discard(str(role))
                elif not self._sync_matches(
                    dict(info),
                    action=clipped,
                    expected_step=expected_step,
                    effective_after_abs_ms=effective_after_abs_ms,
                ):
                    sync_failed = True
                elif bool(getattr(child, "_has_real_victim_step", False)):
                    self._bootstrap_pending_roles.discard(str(role))
        except RuntimeError as exc:
            fallback_obs = (
                np.asarray(self._last_sage_observation, dtype=np.float32)
                if self._last_sage_observation is not None
                else np.zeros((self._obs_history_len * _BASE_OBS_FEATURE_DIM,), dtype=np.float32)
            )
            return self._augment_observation(fallback_obs), 0.0, False, True, {
                "env/error": str(exc),
                "gap/base_rtt_ms": float(self._base_rtt_ms),
            }

        sage_obs, _, sage_terminated, sage_truncated, sage_info = results["sage"]
        self._last_sage_observation = sage_obs
        terminated = sage_terminated or any(item[2] for item in results.values())
        truncated = sage_truncated or any(item[3] for item in results.values()) or expected_step >= self._max_episode_steps

        if child_failed or sync_failed:
            info = dict(sage_info)
            info["env/error"] = "parallel_gap_sync_failure" if sync_failed else str(
                info.get("env/error", "parallel_gap_child_failure")
            )
            for role, (_, _, _, _, child_info) in results.items():
                info[f"sync/{role}_up_step"] = float(child_info.get("mm/up_applied_step", -1.0))
                info[f"sync/{role}_down_step"] = float(child_info.get("mm/down_applied_step", -1.0))
                info[f"sync/{role}_up_after_abs_ms"] = float(
                    child_info.get("mm/up_applied_effective_after_abs_ms", -1.0)
                )
                info[f"sync/{role}_down_after_abs_ms"] = float(
                    child_info.get("mm/down_applied_effective_after_abs_ms", -1.0)
                )
            info["gap/reward"] = 0.0
            info["gap/value"] = 0.0
            info["gap/base_rtt_ms"] = float(self._base_rtt_ms)
            return self._augment_observation(sage_obs), 0.0, terminated, True, info

        if self._bootstrap_pending_roles:
            self._episode_step = expected_step
            self._last_action = clipped.astype(np.float32, copy=True)
            info = dict(sage_info)
            info.update(
                {
                    "attacker/reward": 0.0,
                    "gap/base_rtt_ms": float(self._base_rtt_ms),
                    "gap/reward": 0.0,
                    "gap/value": 0.0,
                    "env/bootstrap_pending_roles": ",".join(sorted(self._bootstrap_pending_roles)),
                    "env/bootstrap_pending_children": float(len(self._bootstrap_pending_roles)),
                    "episode/progress": float(self._episode_step) / float(max(self._max_episode_steps, 1)),
                }
            )
            return self._augment_observation(sage_obs), 0.0, terminated, truncated, info

        path_cap_mbps = max(min(float(clipped[0]), float(clipped[1])), 1e-6)
        score_sage = self._score_from_info(sage_info, path_cap_mbps=path_cap_mbps)
        score_cubic = self._score_from_info(results["cubic"][4], path_cap_mbps=path_cap_mbps)
        score_bbr = self._score_from_info(results["bbr"][4], path_cap_mbps=path_cap_mbps)
        baseline_score = max(score_cubic, score_bbr)
        smooth_penalty = float(
            np.mean(np.abs(self._normalized_action(clipped) - self._normalized_action(self._last_action)))
        )
        reward = float(baseline_score - score_sage - self._smooth_penalty_scale * smooth_penalty)
        gap_value = float(baseline_score - score_sage)

        self._episode_step = expected_step
        self._last_action = clipped.astype(np.float32, copy=True)
        self._previous_gap_features = np.asarray([score_cubic, score_bbr, gap_value], dtype=np.float32)

        info = dict(sage_info)
        info.update(
            {
                "attacker/reward": float(reward),
                "attacker/smooth_penalty": float(smooth_penalty),
                "gap/base_rtt_ms": float(self._base_rtt_ms),
                "gap/path_cap_mbps": float(path_cap_mbps),
                "gap/score_sage": float(score_sage),
                "gap/score_cubic": float(score_cubic),
                "gap/score_bbr": float(score_bbr),
                "gap/baseline_score": float(baseline_score),
                "gap/value": float(gap_value),
                "gap/reward": float(reward),
                "baseline/cubic_rtt_ms": float(results["cubic"][4].get("sage/current_rtt_ms", 0.0)),
                "baseline/bbr_rtt_ms": float(results["bbr"][4].get("sage/current_rtt_ms", 0.0)),
                "baseline/cubic_rate_mbps": float(results["cubic"][4].get("sage/windowed_delivery_rate_mbps", 0.0)),
                "baseline/bbr_rate_mbps": float(results["bbr"][4].get("sage/windowed_delivery_rate_mbps", 0.0)),
                "episode/progress": float(self._episode_step) / float(max(self._max_episode_steps, 1)),
            }
        )
        return self._augment_observation(sage_obs), reward, terminated, truncated, info
