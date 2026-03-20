from __future__ import annotations

from dataclasses import replace
import os
import socket
import time
from typing import Any

import numpy as np

from attacks.envs.baseline_utils import BASELINE_CONTROLLER_SPECS, normalize_baseline_methods
from attacks.envs.online_sage_env import AttackBounds, OnlineSageAttackEnv
from attacks.envs.score_utils import bounded_linear_score_terms_from_info
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
        baseline_hard_max: bool = False,
        baseline_methods: tuple[str, ...] | list[str] | None = None,
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
        self._baseline_hard_max = bool(baseline_hard_max)
        self._baseline_methods = normalize_baseline_methods(baseline_methods)
        self._gap_obs_feature_dim = 8 + 4 * len(self._baseline_methods)
        self._smooth_penalty_scale = float(smooth_penalty_scale)
        self._sync_guard_ms = max(float(sync_guard_ms), 1.0)
        self._launch_retries = max(int(launch_retries), 1)

        self._children: dict[str, OnlineSageAttackEnv] = {}
        self._launch_generation = 0
        self._episode_step = 0
        self._episode_anchor_abs_ms = 0.0
        self._base_rtt_ms = max(float(self._base_launch_config.latency_ms) * 2.0, 1.0)
        self._shared_bandwidth_min_mbps = max(
            float(self._bounds.uplink_bw_mbps[0]),
            float(self._bounds.downlink_bw_mbps[0]),
        )
        self._shared_bandwidth_max_mbps = min(
            float(self._bounds.uplink_bw_mbps[1]),
            float(self._bounds.downlink_bw_mbps[1]),
        )
        if self._shared_bandwidth_min_mbps > self._shared_bandwidth_max_mbps:
            raise ValueError("shared bottleneck bandwidth bounds do not overlap")
        self._log_shared_bandwidth_min = float(np.log(max(self._shared_bandwidth_min_mbps, 1e-6)))
        self._log_shared_bandwidth_max = float(np.log(max(self._shared_bandwidth_max_mbps, 1e-6)))
        self._log_shared_bandwidth_span = max(
            self._log_shared_bandwidth_max - self._log_shared_bandwidth_min,
            1e-6,
        )
        self._fixed_uplink_delay_ms = float(
            self._base_launch_config.initial_uplink_delay_ms
            if self._base_launch_config.initial_uplink_delay_ms is not None
            else self._base_launch_config.latency_ms
        )
        self._fixed_downlink_delay_ms = float(
            self._base_launch_config.initial_downlink_delay_ms
            if self._base_launch_config.initial_downlink_delay_ms is not None
            else self._base_launch_config.latency_ms
        )
        self._effective_bounds = AttackBounds(
            uplink_bw_mbps=tuple(self._bounds.uplink_bw_mbps),
            downlink_bw_mbps=tuple(self._bounds.downlink_bw_mbps),
            uplink_loss=tuple(self._bounds.uplink_loss),
            downlink_loss=tuple(self._bounds.downlink_loss),
            uplink_delay_ms=tuple(self._bounds.uplink_delay_ms),
            downlink_delay_ms=tuple(self._bounds.downlink_delay_ms),
        )
        self.action_space = spaces.Box(
            low=np.asarray([0.0], dtype=np.float32),
            high=np.asarray([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        obs_dim = self._obs_history_len * _BASE_OBS_FEATURE_DIM + self._gap_obs_feature_dim
        obs_high = np.full((obs_dim,), 1e9, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        self._last_policy_action = self._default_policy_action()
        self._last_sage_observation: np.ndarray | None = None
        self._previous_gap_features = np.zeros((self._gap_obs_feature_dim,), dtype=np.float32)
        self._bootstrap_pending_roles: set[str] = set()

    def _default_shared_bandwidth_mbps(self) -> float:
        initial_uplink_bw = float(
            self._base_launch_config.initial_uplink_bw_mbps
            if self._base_launch_config.initial_uplink_bw_mbps is not None
            else self._base_launch_config.env_bw_mbps
        )
        initial_downlink_bw = float(
            self._base_launch_config.initial_downlink_bw_mbps
            if self._base_launch_config.initial_downlink_bw_mbps is not None
            else self._base_launch_config.env_bw_mbps
        )
        return float(
            np.clip(
                min(initial_uplink_bw, initial_downlink_bw),
                self._shared_bandwidth_min_mbps,
                self._shared_bandwidth_max_mbps,
            )
        )

    def _shared_bandwidth_from_policy_action(self, policy_action: np.ndarray) -> float:
        fraction = float(np.clip(np.asarray(policy_action, dtype=np.float32).reshape(-1)[0], 0.0, 1.0))
        return float(np.exp(self._log_shared_bandwidth_min + fraction * self._log_shared_bandwidth_span))

    def _policy_action_from_bandwidth(self, shared_bandwidth_mbps: float) -> np.ndarray:
        clipped_bandwidth = float(
            np.clip(shared_bandwidth_mbps, self._shared_bandwidth_min_mbps, self._shared_bandwidth_max_mbps)
        )
        fraction = (np.log(max(clipped_bandwidth, 1e-6)) - self._log_shared_bandwidth_min) / self._log_shared_bandwidth_span
        return np.asarray([float(np.clip(fraction, 0.0, 1.0))], dtype=np.float32)

    def _effective_action_from_policy(self, policy_action: np.ndarray) -> np.ndarray:
        shared_bandwidth_mbps = self._shared_bandwidth_from_policy_action(policy_action)
        return np.asarray(
            [
                shared_bandwidth_mbps,
                shared_bandwidth_mbps,
                0.0,
                0.0,
                self._fixed_uplink_delay_ms,
                self._fixed_downlink_delay_ms,
            ],
            dtype=np.float32,
        )

    def _expand_effective_action(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        effective_low = self._effective_bounds.low
        effective_high = self._effective_bounds.high
        if raw.shape[0] == effective_low.shape[0]:
            effective_action = np.clip(raw, effective_low, effective_high).astype(np.float32, copy=False)
            shared_bandwidth_mbps = float(min(float(effective_action[0]), float(effective_action[1])))
            policy_action = np.clip(
                self._policy_action_from_bandwidth(shared_bandwidth_mbps),
                self.action_space.low,
                self.action_space.high,
            ).astype(np.float32, copy=False)
            return effective_action, policy_action
        if raw.shape[0] != self.action_space.low.shape[0]:
            raise ValueError(
                f"expected action with {self.action_space.low.shape[0]} or {effective_low.shape[0]} dims, got {raw.shape[0]}"
            )
        policy_action = np.clip(raw, self.action_space.low, self.action_space.high).astype(np.float32, copy=False)
        return self._effective_action_from_policy(policy_action), policy_action

    def _default_policy_action(self) -> np.ndarray:
        return self._policy_action_from_bandwidth(self._default_shared_bandwidth_mbps())

    def _normalized_bandwidth_fraction(self, policy_action: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(policy_action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32, copy=False)

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

    def _gap_feature_vector(
        self,
        *,
        sage_score_terms: dict[str, float],
        baseline_score_terms: dict[str, dict[str, float]],
        baseline_score: float,
        gap_value: float,
        policy_action: np.ndarray,
    ) -> np.ndarray:
        values: list[float] = [float(sage_score_terms["score"])]
        values.extend(float(baseline_score_terms[method]["score"]) for method in self._baseline_methods)
        values.extend(
            [
                float(baseline_score),
                float(gap_value),
                float(sage_score_terms["rate_norm"]),
                float(sage_score_terms["rtt_norm"]),
                float(sage_score_terms["loss_norm"]),
            ]
        )
        for method in self._baseline_methods:
            values.extend(
                [
                    float(baseline_score_terms[method]["rate_norm"]),
                    float(baseline_score_terms[method]["rtt_norm"]),
                    float(baseline_score_terms[method]["loss_norm"]),
                ]
            )
        values.extend(
            [
                float(self._normalized_bandwidth_fraction(policy_action)[0]),
                float(self._shared_bandwidth_from_policy_action(policy_action)),
            ]
        )
        return np.asarray(values, dtype=np.float32)

    def _smoothed_baseline_score(self, *, baseline_scores: dict[str, float]) -> tuple[float, dict[str, float]]:
        scores = np.asarray([float(baseline_scores[method]) for method in self._baseline_methods], dtype=np.float64)
        if bool(self._baseline_hard_max):
            max_score = float(np.max(scores))
            winner_mask = np.isclose(scores, max_score, atol=1e-9)
            weights = winner_mask.astype(np.float64)
            weights /= max(float(np.sum(weights)), 1e-12)
        elif abs(self._baseline_gap_alpha) <= 1e-9:
            weights = np.full((len(self._baseline_methods),), 1.0 / float(len(self._baseline_methods)), dtype=np.float64)
        else:
            logits = self._baseline_gap_alpha * (scores - float(np.max(scores)))
            weights = np.exp(np.clip(logits, -60.0, 0.0))
            weights /= max(float(np.sum(weights)), 1e-12)
        baseline_score = float(np.dot(weights, scores))
        return baseline_score, {
            method: float(weights[index]) for index, method in enumerate(self._baseline_methods)
        }

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
            initial_uplink_bw_mbps=self._default_shared_bandwidth_mbps(),
            initial_downlink_bw_mbps=self._default_shared_bandwidth_mbps(),
            initial_uplink_loss=0.0,
            initial_downlink_loss=0.0,
            initial_uplink_delay_ms=self._fixed_uplink_delay_ms,
            initial_downlink_delay_ms=self._fixed_downlink_delay_ms,
        )
        return OnlineSageAttackEnv(
            repo_root=self.repo_root,
            launch_config=launch_config,
            bounds=self._effective_bounds,
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
        children = {
            "sage": self._build_child(
                role="sage",
                scheme="pure",
                controller_mode="sage",
                port_offset=0,
                taken_ports=taken_ports,
            )
        }
        for port_offset, method in enumerate(self._baseline_methods, start=1):
            scheme, controller_mode = BASELINE_CONTROLLER_SPECS[method]
            children[method] = self._build_child(
                role=method,
                scheme=scheme,
                controller_mode=controller_mode,
                port_offset=port_offset,
                taken_ports=taken_ports,
            )
        return children

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

    def _score_terms_from_info(self, info: dict[str, Any], *, path_cap_mbps: float) -> dict[str, float]:
        return bounded_linear_score_terms_from_info(
            info,
            base_rtt_ms=self._base_rtt_ms,
            path_cap_mbps=path_cap_mbps,
        )

    def _baseline_metric_defaults(self) -> dict[str, float]:
        defaults: dict[str, float] = {}
        for method in self._baseline_methods:
            defaults[f"gap/score_{method}"] = 0.0
            defaults[f"gap/score_{method}_rate_norm"] = 0.0
            defaults[f"gap/score_{method}_rtt_norm"] = 0.0
            defaults[f"gap/score_{method}_loss_norm"] = 0.0
            defaults[f"gap/score_{method}_rate_contrib"] = 0.0
            defaults[f"gap/score_{method}_rtt_contrib"] = 0.0
            defaults[f"gap/score_{method}_loss_penalty"] = 0.0
            defaults[f"gap/baseline_weight_{method}"] = 0.0
            defaults[f"baseline/{method}_rtt_ms"] = 0.0
            defaults[f"baseline/{method}_rate_mbps"] = 0.0
            defaults[f"baseline/{method}_previous_action"] = 0.0
        return defaults

    def _zero_gap_step_metrics(
        self,
        *,
        effective_action: np.ndarray,
        policy_action: np.ndarray,
    ) -> dict[str, float]:
        shared_bandwidth_mbps = float(min(float(effective_action[0]), float(effective_action[1])))
        path_cap_mbps = max(shared_bandwidth_mbps, 1e-6)
        return {
            "attacker/reward": 0.0,
            "attacker/smooth_penalty": 0.0,
            "attacker/shared_bw_mbps": float(shared_bandwidth_mbps),
            "attacker/shared_bw_fraction": float(policy_action.reshape(-1)[0]),
            "attacker/uplink_bw_mbps": float(effective_action[0]),
            "attacker/downlink_bw_mbps": float(effective_action[1]),
            "attacker/uplink_loss": float(effective_action[2]),
            "attacker/downlink_loss": float(effective_action[3]),
            "attacker/uplink_delay_ms": float(effective_action[4]),
            "attacker/downlink_delay_ms": float(effective_action[5]),
            "mm/up_queue_packets": 0.0,
            "mm/down_queue_packets": 0.0,
            "mm/up_applied_bw_mbps": 0.0,
            "mm/up_applied_loss_rate": 0.0,
            "mm/up_applied_delay_ms": 0.0,
            "mm/up_queue_delay_ms": 0.0,
            "mm/up_departure_rate_mbps": 0.0,
            "mm/down_applied_bw_mbps": 0.0,
            "mm/down_applied_loss_rate": 0.0,
            "mm/down_applied_delay_ms": 0.0,
            "mm/down_queue_delay_ms": 0.0,
            "mm/down_departure_rate_mbps": 0.0,
            "mm/up_applied_step": -1.0,
            "mm/down_applied_step": -1.0,
            "mm/up_applied_effective_after_abs_ms": -1.0,
            "mm/down_applied_effective_after_abs_ms": -1.0,
            "sage/score": 0.0,
            "sage/score_rate_norm": 0.0,
            "sage/score_rtt_norm": 0.0,
            "sage/score_loss_norm": 0.0,
            "sage/score_rate_contrib": 0.0,
            "sage/score_rtt_contrib": 0.0,
            "sage/score_loss_penalty": 0.0,
            "gap/base_rtt_ms": float(self._base_rtt_ms),
            "gap/path_cap_mbps": float(path_cap_mbps),
            "gap/score_sage": 0.0,
            "gap/score_sage_rate_norm": 0.0,
            "gap/score_sage_rtt_norm": 0.0,
            "gap/score_sage_loss_norm": 0.0,
            "gap/score_sage_rate_contrib": 0.0,
            "gap/score_sage_rtt_contrib": 0.0,
            "gap/score_sage_loss_penalty": 0.0,
            "gap/baseline_score": 0.0,
            "gap/best_baseline_score": 0.0,
            "gap/best_baseline_gap": 0.0,
            "gap/best_baseline_wins": 0.0,
            "gap/reward": 0.0,
            "gap/value": 0.0,
            **self._baseline_metric_defaults(),
        }

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
            self._last_policy_action = self._default_policy_action()
            self._previous_gap_features = np.zeros((self._gap_obs_feature_dim,), dtype=np.float32)
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
                    "sage/score": 0.0,
                    "sage/score_rate_norm": 0.0,
                    "sage/score_rtt_norm": 0.0,
                    "sage/score_loss_norm": 0.0,
                    "sage/score_rate_contrib": 0.0,
                    "sage/score_rtt_contrib": 0.0,
                    "sage/score_loss_penalty": 0.0,
                    "gap/score_sage": 0.0,
                    "gap/baseline_score": 0.0,
                    "gap/best_baseline_score": 0.0,
                    "gap/best_baseline_gap": 0.0,
                    "gap/best_baseline_wins": 0.0,
                    "gap/reward": 0.0,
                    "gap/value": 0.0,
                    **self._baseline_metric_defaults(),
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

        effective_action, policy_action = self._expand_effective_action(np.asarray(action, dtype=np.float32))
        shared_bandwidth_mbps = float(min(float(effective_action[0]), float(effective_action[1])))
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
                    effective_action,
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
                    expected_action=effective_action,
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
                    action=effective_action,
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
            info = self._zero_gap_step_metrics(
                effective_action=effective_action,
                policy_action=policy_action,
            )
            info["env/error"] = str(exc)
            return self._augment_observation(fallback_obs), 0.0, False, True, info

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
            info.update(
                self._zero_gap_step_metrics(
                    effective_action=effective_action,
                    policy_action=policy_action,
                )
            )
            return self._augment_observation(sage_obs), 0.0, terminated, True, info

        if self._bootstrap_pending_roles:
            self._episode_step = expected_step
            self._last_policy_action = policy_action.astype(np.float32, copy=True)
            info = dict(sage_info)
            info.update(
                {
                    **self._zero_gap_step_metrics(
                        effective_action=effective_action,
                        policy_action=policy_action,
                    ),
                    "env/bootstrap_pending_roles": ",".join(sorted(self._bootstrap_pending_roles)),
                    "env/bootstrap_pending_children": float(len(self._bootstrap_pending_roles)),
                    "episode/progress": float(self._episode_step) / float(max(self._max_episode_steps, 1)),
                }
            )
            return self._augment_observation(sage_obs), 0.0, terminated, truncated, info

        path_cap_mbps = max(min(float(effective_action[0]), float(effective_action[1])), 1e-6)
        sage_score_terms = self._score_terms_from_info(sage_info, path_cap_mbps=path_cap_mbps)
        score_sage = float(sage_score_terms["score"])
        baseline_score_terms = {
            method: self._score_terms_from_info(results[method][4], path_cap_mbps=path_cap_mbps)
            for method in self._baseline_methods
        }
        baseline_scores = {
            method: float(score_terms["score"]) for method, score_terms in baseline_score_terms.items()
        }
        baseline_score, baseline_weights = self._smoothed_baseline_score(
            baseline_scores=baseline_scores,
        )
        best_baseline_score = float(max(baseline_scores.values()))
        best_baseline_gap = float(best_baseline_score - score_sage)
        best_baseline_wins = 1.0 if best_baseline_gap > 0.0 else 0.0
        gap_value = float(baseline_score - score_sage)
        smooth_penalty = float(
            np.mean(
                np.abs(
                    self._normalized_bandwidth_fraction(policy_action)
                    - self._normalized_bandwidth_fraction(self._last_policy_action)
                )
            )
        )
        reward = float(gap_value * baseline_score - self._smooth_penalty_scale * smooth_penalty)

        self._episode_step = expected_step
        self._last_policy_action = policy_action.astype(np.float32, copy=True)
        self._previous_gap_features = self._gap_feature_vector(
            sage_score_terms=sage_score_terms,
            baseline_score_terms=baseline_score_terms,
            baseline_score=baseline_score,
            gap_value=gap_value,
            policy_action=policy_action,
        )

        baseline_info_payload: dict[str, float] = {}
        for method in self._baseline_methods:
            score_terms = baseline_score_terms[method]
            baseline_info_payload.update(
                {
                    f"gap/score_{method}": float(baseline_scores[method]),
                    f"gap/score_{method}_rate_norm": float(score_terms["rate_norm"]),
                    f"gap/score_{method}_rtt_norm": float(score_terms["rtt_norm"]),
                    f"gap/score_{method}_loss_norm": float(score_terms["loss_norm"]),
                    f"gap/score_{method}_rate_contrib": float(score_terms["rate_contrib"]),
                    f"gap/score_{method}_rtt_contrib": float(score_terms["rtt_contrib"]),
                    f"gap/score_{method}_loss_penalty": float(score_terms["loss_penalty"]),
                    f"gap/baseline_weight_{method}": float(baseline_weights[method]),
                    f"baseline/{method}_rtt_ms": float(results[method][4].get("sage/current_rtt_ms", 0.0)),
                    f"baseline/{method}_rate_mbps": float(results[method][4].get("sage/windowed_delivery_rate_mbps", 0.0)),
                    f"baseline/{method}_previous_action": float(results[method][4].get("sage/previous_action", 0.0)),
                }
            )

        info = dict(sage_info)
        info.update(
            {
                "attacker/reward": float(reward),
                "attacker/smooth_penalty": float(smooth_penalty),
                "attacker/shared_bw_mbps": float(shared_bandwidth_mbps),
                "attacker/shared_bw_fraction": float(policy_action.reshape(-1)[0]),
                "attacker/uplink_bw_mbps": float(effective_action[0]),
                "attacker/downlink_bw_mbps": float(effective_action[1]),
                "attacker/uplink_loss": float(effective_action[2]),
                "attacker/downlink_loss": float(effective_action[3]),
                "attacker/uplink_delay_ms": float(effective_action[4]),
                "attacker/downlink_delay_ms": float(effective_action[5]),
                "sage/score": float(score_sage),
                "sage/score_rate_norm": float(sage_score_terms["rate_norm"]),
                "sage/score_rtt_norm": float(sage_score_terms["rtt_norm"]),
                "sage/score_loss_norm": float(sage_score_terms["loss_norm"]),
                "sage/score_rate_contrib": float(sage_score_terms["rate_contrib"]),
                "sage/score_rtt_contrib": float(sage_score_terms["rtt_contrib"]),
                "sage/score_loss_penalty": float(sage_score_terms["loss_penalty"]),
                "gap/base_rtt_ms": float(self._base_rtt_ms),
                "gap/path_cap_mbps": float(path_cap_mbps),
                "gap/score_sage": float(score_sage),
                "gap/score_sage_rate_norm": float(sage_score_terms["rate_norm"]),
                "gap/score_sage_rtt_norm": float(sage_score_terms["rtt_norm"]),
                "gap/score_sage_loss_norm": float(sage_score_terms["loss_norm"]),
                "gap/score_sage_rate_contrib": float(sage_score_terms["rate_contrib"]),
                "gap/score_sage_rtt_contrib": float(sage_score_terms["rtt_contrib"]),
                "gap/score_sage_loss_penalty": float(sage_score_terms["loss_penalty"]),
                "gap/baseline_score": float(baseline_score),
                "gap/best_baseline_score": float(best_baseline_score),
                "gap/best_baseline_gap": float(best_baseline_gap),
                "gap/best_baseline_wins": float(best_baseline_wins),
                "gap/value": float(gap_value),
                "gap/reward": float(reward),
                **baseline_info_payload,
                "episode/progress": float(self._episode_step) / float(max(self._max_episode_steps, 1)),
            }
        )
        return self._augment_observation(sage_obs), reward, terminated, truncated, info
