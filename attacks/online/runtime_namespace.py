from __future__ import annotations

from dataclasses import dataclass
import atexit
import errno
import fcntl
import json
import os
import re
import socket
import time
from typing import Any


def _resolve_path(repo_root: str, path: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(repo_root, expanded))


def _sanitize_label(label: str | None) -> str:
    if label is None:
        return "run"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(label).strip().lower()).strip("._-")
    return cleaned[:48] or "run"


def _linux_process_start_ticks(pid: int) -> int | None:
    try:
        with open(f"/proc/{int(pid)}/stat", "r", encoding="utf-8") as file_obj:
            content = file_obj.read().strip()
    except OSError:
        return None
    right_paren = content.rfind(")")
    if right_paren < 0:
        return None
    fields = content[right_paren + 2 :].split()
    if len(fields) <= 19:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def _process_is_alive(pid: int, proc_start_ticks: int | None) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return exc.errno == errno.EPERM

    if proc_start_ticks is None:
        return True
    live_start_ticks = _linux_process_start_ticks(int(pid))
    if live_start_ticks is None:
        return True
    return int(live_start_ticks) == int(proc_start_ticks)


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp-{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _port_ranges_overlap(
    first_port_base: int,
    first_ports_per_run: int,
    second_port_base: int,
    second_ports_per_run: int,
) -> bool:
    first_start = int(first_port_base)
    first_end = first_start + max(int(first_ports_per_run), 1) - 1
    second_start = int(second_port_base)
    second_end = second_start + max(int(second_ports_per_run), 1) - 1
    return not (first_end < second_start or second_end < first_start)


def _port_block_is_available(port_base: int, ports_per_run: int) -> bool:
    sockets: list[socket.socket] = []
    try:
        for port in range(int(port_base), int(port_base) + max(int(ports_per_run), 1)):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", int(port)))
            sock.listen(1)
            sockets.append(sock)
        return True
    except OSError:
        return False
    finally:
        for sock in sockets:
            try:
                sock.close()
            except OSError:
                pass


@dataclass(frozen=True)
class RunNamespace:
    run_id: str
    slot: int
    runtime_parent_dir: str
    runtime_dir: str
    actor_id_base: int
    port_base: int
    ports_per_run: int
    lease_path: str


class RunNamespaceLease:
    def __init__(
        self,
        namespace: RunNamespace,
        *,
        pid: int,
        proc_start_ticks: int | None,
        lock_path: str,
    ) -> None:
        self.namespace = namespace
        self._pid = int(pid)
        self._proc_start_ticks = proc_start_ticks
        self._lock_path = lock_path
        self._released = False
        atexit.register(self.release)

    @property
    def run_id(self) -> str:
        return self.namespace.run_id

    @property
    def runtime_dir(self) -> str:
        return self.namespace.runtime_dir

    @property
    def actor_id_base(self) -> int:
        return int(self.namespace.actor_id_base)

    @property
    def port_base(self) -> int:
        return int(self.namespace.port_base)

    @property
    def slot(self) -> int:
        return int(self.namespace.slot)

    def metadata(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "slot": self.slot,
            "runtime_parent_dir": self.namespace.runtime_parent_dir,
            "runtime_dir": self.runtime_dir,
            "actor_id_base": self.actor_id_base,
            "port_base": self.port_base,
            "ports_per_run": int(self.namespace.ports_per_run),
            "lease_path": self.namespace.lease_path,
        }

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            os.makedirs(os.path.dirname(self._lock_path), exist_ok=True)
            with open(self._lock_path, "a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    with open(self.namespace.lease_path, "r", encoding="utf-8") as file_obj:
                        payload = json.load(file_obj)
                except (FileNotFoundError, json.JSONDecodeError, OSError):
                    payload = None
                if isinstance(payload, dict):
                    pid = int(payload.get("pid", -1))
                    proc_start_ticks = payload.get("proc_start_ticks")
                    if pid == self._pid and proc_start_ticks == self._proc_start_ticks:
                        try:
                            os.remove(self.namespace.lease_path)
                        except FileNotFoundError:
                            pass
        except OSError:
            return


def acquire_run_namespace(
    *,
    repo_root: str,
    runtime_dir: str,
    actor_id: int,
    port: int,
    label: str | None = None,
    actor_id_stride: int = 10_000,
    ports_per_run: int = 1,
    max_slots: int = 4096,
) -> RunNamespaceLease:
    runtime_parent_dir = _resolve_path(repo_root, runtime_dir)
    os.makedirs(runtime_parent_dir, exist_ok=True)
    lease_root = os.path.join(runtime_parent_dir, ".run-namespaces")
    os.makedirs(lease_root, exist_ok=True)
    lock_path = os.path.join(lease_root, "index.lock")

    pid = os.getpid()
    proc_start_ticks = _linux_process_start_ticks(pid)
    preferred_port = max(int(port), 1024)
    port_block = max(int(ports_per_run), 1)
    slot_limit = max(1, min(int(max_slots), (65535 - preferred_port + 1) // port_block))
    label_prefix = _sanitize_label(label)

    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        active_slots: set[int] = set()
        live_payloads: list[dict[str, Any]] = []
        for slot in range(slot_limit):
            lease_path = os.path.join(lease_root, f"slot-{slot:04d}.json")
            try:
                with open(lease_path, "r", encoding="utf-8") as file_obj:
                    payload = json.load(file_obj)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                payload = None

            if isinstance(payload, dict):
                live_pid = int(payload.get("pid", -1))
                live_start_ticks = payload.get("proc_start_ticks")
                if _process_is_alive(live_pid, live_start_ticks):
                    active_slots.add(int(slot))
                    live_payloads.append(dict(payload))
                    continue
                try:
                    os.remove(lease_path)
                except FileNotFoundError:
                    pass

        for slot in range(slot_limit):
            if int(slot) in active_slots:
                continue
            lease_path = os.path.join(lease_root, f"slot-{slot:04d}.json")
            candidate_port_base = int(preferred_port) + int(slot) * int(port_block)
            if any(
                _port_ranges_overlap(
                    candidate_port_base,
                    port_block,
                    int(payload.get("port_base", -1)),
                    int(payload.get("ports_per_run", 1)),
                )
                for payload in live_payloads
            ):
                continue
            if not _port_block_is_available(candidate_port_base, port_block):
                continue

            timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            run_id = f"{label_prefix}-{timestamp}-p{pid}-s{slot:04d}"
            namespace = RunNamespace(
                run_id=run_id,
                slot=int(slot),
                runtime_parent_dir=runtime_parent_dir,
                runtime_dir=os.path.join(runtime_parent_dir, run_id),
                actor_id_base=max(int(actor_id) + int(slot) * int(actor_id_stride), 0),
                port_base=candidate_port_base,
                ports_per_run=int(port_block),
                lease_path=lease_path,
            )
            os.makedirs(namespace.runtime_dir, exist_ok=True)
            _write_json_atomic(
                lease_path,
                {
                    "actor_id_base": namespace.actor_id_base,
                    "pid": pid,
                    "port_base": namespace.port_base,
                    "ports_per_run": int(namespace.ports_per_run),
                    "proc_start_ticks": proc_start_ticks,
                    "run_id": namespace.run_id,
                    "runtime_dir": namespace.runtime_dir,
                    "slot": namespace.slot,
                    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            return RunNamespaceLease(
                namespace,
                pid=pid,
                proc_start_ticks=proc_start_ticks,
                lock_path=lock_path,
            )

    raise RuntimeError(
        f"could not allocate a concurrent runtime slot under {runtime_parent_dir}; "
        f"all {slot_limit} slots are already in use"
    )
