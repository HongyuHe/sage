from __future__ import annotations

from dataclasses import dataclass
import os
import signal
import shutil
import stat
import subprocess
from typing import Mapping


def _resolve_path(repo_root: str, path: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(repo_root, expanded))


def _is_setuid_root_binary(path: str) -> bool:
    try:
        st = os.stat(path)
    except OSError:
        return False
    return bool(st.st_mode & stat.S_ISUID) and int(st.st_uid) == 0 and os.access(path, os.X_OK)


def _resolve_mm_adv_bin(repo_root: str, configured_path: str | None) -> str:
    if configured_path is not None:
        resolved = _resolve_path(repo_root, configured_path)
        if not _is_setuid_root_binary(resolved):
            raise RuntimeError(
                "mm-adv-net must be installed setuid root. "
                f"Configured path is not usable: {resolved}. "
                "Install it with `sudo make install` under ccBench/mahimahi, or point "
                "--mm-adv-bin at an installed setuid-root mm-adv-net binary."
            )
        return resolved

    candidates: list[str] = []
    path_candidate = shutil.which("mm-adv-net")
    if path_candidate:
        candidates.append(path_candidate)
    repo_candidate = os.path.join(repo_root, "ccBench", "mahimahi", "src", "frontend", "mm-adv-net")
    if os.path.exists(repo_candidate):
        candidates.append(repo_candidate)

    for candidate in candidates:
        resolved = _resolve_path(repo_root, candidate)
        if _is_setuid_root_binary(resolved):
            return resolved

    searched = ", ".join(_resolve_path(repo_root, candidate) for candidate in candidates) if candidates else "<none>"
    raise RuntimeError(
        "Could not find a usable setuid-root mm-adv-net binary. "
        f"Searched: {searched}. "
        "The in-repo build artifact is not sufficient by itself; install Mahimahi with "
        "`sudo make install` so mm-adv-net is owned by root and marked setuid."
    )


@dataclass(frozen=True)
class SageLaunchConfig:
    sage_script: str = "sage_rl/sage.sh"
    scheme: str = "pure"
    controller_mode: str = "sage"
    startup_stagger_ms: int | None = 0
    ready_signal_timeout_ms: int | None = 180_000
    latency_ms: int = 25
    port: int = 5101
    downlink_trace: str = "wired48"
    uplink_trace: str = "wired48"
    iteration_id: int = 0
    qsize_packets: int = 128
    env_bw_mbps: int = 48
    bw2_mbps: int = 48
    trace_period_s: int = 7
    first_time_mode: int = 0
    log_prefix: str = "adv"
    duration_seconds: int = 60
    actor_id: int = 0
    duration_steps: int = 6000
    num_flows: int = 1
    save_logs: int = 0
    analyze_logs: int = 0
    mm_adv_bin: str | None = None
    initial_uplink_bw_mbps: float | None = None
    initial_downlink_bw_mbps: float | None = None
    initial_uplink_loss: float | None = None
    initial_downlink_loss: float | None = None
    initial_uplink_delay_ms: float | None = None
    initial_downlink_delay_ms: float | None = None
    initial_uplink_queue_packets: int | None = None
    initial_downlink_queue_packets: int | None = None
    initial_uplink_queue_bytes: int | None = None
    initial_downlink_queue_bytes: int | None = None
    shield_rules_file: str | None = None
    shield_action_delta: float | None = None
    shield_consecutive_risk: int | None = None
    shield_cooldown_steps: int | None = None
    shield_log_path: str | None = None
    controller_timing_log_enabled: bool = False
    controller_timing_log_path: str | None = None

    def make_command(self, repo_root: str) -> list[str]:
        script = _resolve_path(repo_root, self.sage_script)
        return [
            "bash",
            script,
            str(int(self.latency_ms)),
            str(int(self.port)),
            str(self.downlink_trace),
            str(self.uplink_trace),
            str(int(self.iteration_id)),
            str(int(self.qsize_packets)),
            str(int(self.env_bw_mbps)),
            str(int(self.bw2_mbps)),
            str(int(self.trace_period_s)),
            str(int(self.first_time_mode)),
            str(self.log_prefix),
            str(int(self.duration_seconds)),
            str(int(self.actor_id)),
            str(int(self.duration_steps)),
            str(int(self.num_flows)),
            str(int(self.save_logs)),
            str(int(self.analyze_logs)),
        ]

    def env_overrides(self, repo_root: str, control_file: str, keys_file: str) -> dict[str, str]:
        env = {
            "SAGE_MM_ADV_CONTROL_FILE": os.path.abspath(control_file),
            "SAGE_ATTACK_KEYS_FILE": os.path.abspath(keys_file),
            "SAGE_SCHEME": str(self.scheme),
            "SAGE_CONTROLLER_MODE": str(self.controller_mode),
        }
        if self.startup_stagger_ms is not None:
            env["SAGE_STARTUP_STAGGER_MS"] = str(int(self.startup_stagger_ms))
        if self.ready_signal_timeout_ms is not None:
            env["SAGE_READY_SIGNAL_TIMEOUT_MS"] = str(int(self.ready_signal_timeout_ms))
        env["SAGE_MM_ADV_BIN"] = _resolve_mm_adv_bin(repo_root, self.mm_adv_bin)

        optional_values: Mapping[str, float | int | None] = {
            "SAGE_MM_ADV_UPLINK_BW": self.initial_uplink_bw_mbps,
            "SAGE_MM_ADV_DOWNLINK_BW": self.initial_downlink_bw_mbps,
            "SAGE_MM_ADV_UPLINK_LOSS": self.initial_uplink_loss,
            "SAGE_MM_ADV_DOWNLINK_LOSS": self.initial_downlink_loss,
            "SAGE_MM_ADV_UPLINK_DELAY_MS": self.initial_uplink_delay_ms,
            "SAGE_MM_ADV_DOWNLINK_DELAY_MS": self.initial_downlink_delay_ms,
            "SAGE_MM_ADV_UPLINK_QUEUE_PACKETS": self.initial_uplink_queue_packets,
            "SAGE_MM_ADV_DOWNLINK_QUEUE_PACKETS": self.initial_downlink_queue_packets,
            "SAGE_MM_ADV_UPLINK_QUEUE_BYTES": self.initial_uplink_queue_bytes,
            "SAGE_MM_ADV_DOWNLINK_QUEUE_BYTES": self.initial_downlink_queue_bytes,
            "SAGE_SHIELD_ACTION_DELTA": self.shield_action_delta,
            "SAGE_SHIELD_CONSECUTIVE_RISK": self.shield_consecutive_risk,
            "SAGE_SHIELD_COOLDOWN_STEPS": self.shield_cooldown_steps,
        }
        for key, value in optional_values.items():
            if value is not None:
                env[key] = str(value)
        if self.shield_rules_file:
            env["SAGE_SHIELD_RULES_FILE"] = _resolve_path(repo_root, self.shield_rules_file)
        if self.shield_log_path:
            env["SAGE_SHIELD_LOG_PATH"] = _resolve_path(repo_root, self.shield_log_path)
        if self.controller_timing_log_enabled:
            env["SAGE_CONTROLLER_TIMING_LOG_ENABLED"] = "1"
        if self.controller_timing_log_path:
            env["SAGE_CONTROLLER_TIMING_LOG_PATH"] = _resolve_path(repo_root, self.controller_timing_log_path)
        return env


class SageProcess:
    def __init__(
        self,
        process: subprocess.Popen[str],
        *,
        stdout_handle,
        stderr_handle,
        stdout_path: str,
        stderr_path: str,
    ) -> None:
        self.process = process
        self.stdout_handle = stdout_handle
        self.stderr_handle = stderr_handle
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path

    def poll(self) -> int | None:
        return self.process.poll()

    def wait(self, timeout: float | None = None) -> int:
        return self.process.wait(timeout=timeout)

    def terminate(self, timeout_s: float = 10.0) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=timeout_s)
            except Exception:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.stdout_handle.close()
        self.stderr_handle.close()


def launch_sage(
    repo_root: str,
    launch_config: SageLaunchConfig,
    *,
    control_file: str,
    keys_file: str,
    runtime_dir: str,
    extra_env: Mapping[str, str] | None = None,
) -> SageProcess:
    repo_root = os.path.abspath(repo_root)
    runtime_dir = os.path.abspath(runtime_dir)
    os.makedirs(runtime_dir, exist_ok=True)
    stdout_path = os.path.join(runtime_dir, "sage.stdout.log")
    stderr_path = os.path.join(runtime_dir, "sage.stderr.log")
    stdout_handle = open(stdout_path, "w", encoding="utf-8")
    stderr_handle = open(stderr_path, "w", encoding="utf-8")

    env = os.environ.copy()
    env.update(launch_config.env_overrides(repo_root, control_file=control_file, keys_file=keys_file))
    if (
        launch_config.shield_rules_file
        and "SAGE_SHIELD_LOG_PATH" not in env
    ):
        env["SAGE_SHIELD_LOG_PATH"] = os.path.join(runtime_dir, "sage-shield-runtime.jsonl")
    if (
        launch_config.controller_timing_log_enabled
        and "SAGE_CONTROLLER_TIMING_LOG_PATH" not in env
    ):
        env["SAGE_CONTROLLER_TIMING_LOG_PATH"] = os.path.join(runtime_dir, "sage-controller-timing.jsonl")
    if extra_env is not None:
        env.update({str(k): str(v) for k, v in extra_env.items()})

    process = subprocess.Popen(
        launch_config.make_command(repo_root),
        cwd=os.path.join(repo_root, "sage_rl"),
        env=env,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        start_new_session=True,
    )
    return SageProcess(
        process,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
