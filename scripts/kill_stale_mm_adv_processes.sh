#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLIENT_PATH="${REPO_ROOT}/sage_rl/rl_module/client"
SAGE_BIN_PATH="${REPO_ROOT}/sage_rl/rl_module/sage"
SAGE_SCRIPT_PATH="${REPO_ROOT}/sage_rl/sage.sh"
SERVER_SCRIPT_PATH="${REPO_ROOT}/sage_rl/server.sh"

DRY_RUN=0
NO_SUDO=0
FORCE=0
GRACE_SECONDS=3
RUNTIME_ROOT="${REPO_ROOT}/attacks/runtime/"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run] [--force] [--no-sudo] [--grace-seconds N]

Kill stale Mahimahi/Sage process trees:
  - mm-adv-net (including sudo wrappers and descendants)
  - ${CLIENT_PATH}
  - ${SAGE_BIN_PATH}
  - ${SAGE_SCRIPT_PATH}
  - ${SERVER_SCRIPT_PATH}

Options:
  --dry-run          Show matching processes, do not kill.
  --force            Kill processes even if tied to an active runtime lease.
  --no-sudo          Never use sudo for root-owned processes.
  --grace-seconds N  Wait time between TERM and KILL (default: ${GRACE_SECONDS}).
  -h, --help         Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-sudo)
      NO_SUDO=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --grace-seconds)
      GRACE_SECONDS="${2:-}"
      if [[ -z "${GRACE_SECONDS}" || ! "${GRACE_SECONDS}" =~ ^[0-9]+$ ]]; then
        echo "invalid --grace-seconds value: ${2:-<missing>}" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v ps >/dev/null 2>&1; then
  echo "ps is required but not found" >&2
  exit 1
fi

if ! command -v awk >/dev/null 2>&1; then
  echo "awk is required but not found" >&2
  exit 1
fi

declare -A PID_TO_CMD=()
declare -A PID_TO_USER=()
declare -A PID_TO_PPID=()
declare -A PID_TO_PGID=()
declare -A GROUP_TO_PIDS=()
declare -A MATCHED_GROUPS=()
declare -A MATCHED_SINGLES=()
declare -A GROUP_TO_RUNTIME_DIR=()
declare -A SINGLE_TO_RUNTIME_DIR=()
declare -A ACTIVE_RUNTIME_DIRS=()

append_to_group() {
  local group="$1"
  local pid="$2"
  local current="${GROUP_TO_PIDS[${group}]:-}"
  if [[ -z "${current}" ]]; then
    GROUP_TO_PIDS["${group}"]="${pid}"
  else
    GROUP_TO_PIDS["${group}"]="${current} ${pid}"
  fi
}

load_process_table() {
  while IFS=$'\t' read -r pid ppid pgid user cmd; do
    [[ -z "${pid}" || ! "${pid}" =~ ^[0-9]+$ ]] && continue
    PID_TO_PPID["${pid}"]="${ppid}"
    PID_TO_PGID["${pid}"]="${pgid}"
    PID_TO_USER["${pid}"]="${user}"
    PID_TO_CMD["${pid}"]="${cmd}"
    if [[ -n "${pgid}" && "${pgid}" =~ ^[0-9]+$ ]]; then
      append_to_group "${pgid}" "${pid}"
    fi
  done < <(
    ps -eo pid=,ppid=,pgid=,user=,args= | awk '{
      pid=$1; ppid=$2; pgid=$3; user=$4;
      $1=""; $2=""; $3=""; $4="";
      sub(/^[[:space:]]+/, "", $0);
      printf "%s\t%s\t%s\t%s\t%s\n", pid, ppid, pgid, user, $0;
    }'
  )
}

command_matches() {
  local cmd="$1"
  [[ "${cmd}" == *"mm-adv-net"* ]] && return 0
  [[ "${cmd}" == *"/sage_rl/rl_module/client"* ]] && return 0
  [[ "${cmd}" == *"/sage_rl/rl_module/sage"* ]] && return 0
  [[ "${cmd}" == *"/sage_rl/sage.sh"* ]] && return 0
  [[ "${cmd}" == *"/sage_rl/server.sh"* ]] && return 0
  return 1
}

shorten_cmd() {
  local cmd="$1"
  local max_len=220
  if [[ "${#cmd}" -gt "${max_len}" ]]; then
    printf '%s...' "${cmd:0:${max_len}}"
  else
    printf '%s' "${cmd}"
  fi
}

proc_start_ticks() {
  local pid="$1"
  local stat_path="/proc/${pid}/stat"
  [[ -r "${stat_path}" ]] || return 1
  awk '{print $22}' "${stat_path}" 2>/dev/null || return 1
}

load_active_runtime_dirs() {
  local lease_root="${RUNTIME_ROOT}.run-namespaces"
  [[ -d "${lease_root}" ]] || return 0

  shopt -s nullglob
  local lease_file
  for lease_file in "${lease_root}"/slot-*.json; do
    [[ -f "${lease_file}" ]] || continue

    local runtime_dir=""
    local lease_pid=""
    local lease_ticks=""
    runtime_dir="$(sed -n 's/^[[:space:]]*"runtime_dir"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "${lease_file}" | head -n1)"
    lease_pid="$(sed -n 's/^[[:space:]]*"pid"[[:space:]]*:[[:space:]]*\([0-9]\+\).*/\1/p' "${lease_file}" | head -n1)"
    lease_ticks="$(sed -n 's/^[[:space:]]*"proc_start_ticks"[[:space:]]*:[[:space:]]*\([0-9]\+\).*/\1/p' "${lease_file}" | head -n1)"

    [[ -n "${runtime_dir}" && -n "${lease_pid}" ]] || continue
    [[ -d "/proc/${lease_pid}" ]] || continue

    if [[ -n "${lease_ticks}" ]]; then
      local live_ticks=""
      live_ticks="$(proc_start_ticks "${lease_pid}" || true)"
      [[ -n "${live_ticks}" && "${live_ticks}" == "${lease_ticks}" ]] || continue
    fi

    ACTIVE_RUNTIME_DIRS["${runtime_dir}"]=1
  done
  shopt -u nullglob
}

runtime_dir_is_active() {
  local runtime_dir="$1"
  [[ -n "${runtime_dir}" ]] || return 1

  if [[ -n "${ACTIVE_RUNTIME_DIRS[${runtime_dir}]:-}" ]]; then
    return 0
  fi

  local active_dir
  for active_dir in "${!ACTIVE_RUNTIME_DIRS[@]}"; do
    if [[ "${runtime_dir}" == "${active_dir}" || "${runtime_dir}" == "${active_dir}/"* || "${active_dir}" == "${runtime_dir}/"* ]]; then
      return 0
    fi
  done
  return 1
}

extract_runtime_dir_from_cmd() {
  local cmd="$1"
  if [[ "${cmd}" != *"${RUNTIME_ROOT}"* ]]; then
    return 1
  fi

  local suffix="${cmd#*"${RUNTIME_ROOT}"}"
  suffix="${suffix%% *}"
  suffix="${suffix%%\'}"
  suffix="${suffix%%\"}"
  local full_path="${RUNTIME_ROOT}${suffix}"
  if [[ "${full_path}" == *"/episode-"* ]]; then
    printf '%s\n' "${full_path%%/episode-*}"
    return 0
  fi
  printf '%s\n' "${full_path}"
  return 0
}

runtime_dir_for_pid() {
  local pid="$1"
  local current_pid="${pid}"
  local depth=0

  while [[ -n "${current_pid}" && "${current_pid}" =~ ^[0-9]+$ && "${current_pid}" -gt 1 && "${depth}" -lt 24 ]]; do
    local cmd="${PID_TO_CMD[${current_pid}]:-}"
    if [[ -n "${cmd}" ]]; then
      local runtime_dir=""
      runtime_dir="$(extract_runtime_dir_from_cmd "${cmd}" || true)"
      if [[ -n "${runtime_dir}" ]]; then
        printf '%s\n' "${runtime_dir}"
        return 0
      fi
    fi
    current_pid="${PID_TO_PPID[${current_pid}]:-}"
    depth=$((depth + 1))
  done

  return 1
}

group_requires_sudo() {
  local pgid="$1"
  local pid
  for pid in ${GROUP_TO_PIDS[${pgid}]:-}; do
    local owner="${PID_TO_USER[${pid}]:-}"
    if [[ -n "${owner}" && "${owner}" != "${USER}" ]]; then
      return 0
    fi
  done
  return 1
}

group_alive() {
  local pgid="$1"
  local pid
  for pid in ${GROUP_TO_PIDS[${pgid}]:-}; do
    if [[ -d "/proc/${pid}" ]]; then
      return 0
    fi
  done
  return 1
}

kill_group_signal() {
  local signal="$1"
  local pgid="$2"
  local use_sudo=0

  if [[ "${NO_SUDO}" -ne 1 ]] && group_requires_sudo "${pgid}"; then
    use_sudo=1
  fi

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    if [[ "${use_sudo}" -eq 1 ]]; then
      echo "DRY-RUN: sudo kill -${signal} -- -${pgid}"
    else
      echo "DRY-RUN: kill -${signal} -- -${pgid}"
    fi
    return 0
  fi

  if [[ "${use_sudo}" -eq 1 ]]; then
    sudo kill "-${signal}" -- "-${pgid}" 2>/dev/null || true
  else
    kill "-${signal}" -- "-${pgid}" 2>/dev/null || true
  fi
}

single_requires_sudo() {
  local pid="$1"
  local owner="${PID_TO_USER[${pid}]:-}"
  [[ -n "${owner}" && "${owner}" != "${USER}" ]]
}

single_alive() {
  local pid="$1"
  [[ -d "/proc/${pid}" ]]
}

kill_single_signal() {
  local signal="$1"
  local pid="$2"
  local use_sudo=0

  if [[ "${NO_SUDO}" -ne 1 ]] && single_requires_sudo "${pid}"; then
    use_sudo=1
  fi

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    if [[ "${use_sudo}" -eq 1 ]]; then
      echo "DRY-RUN: sudo kill -${signal} ${pid}"
    else
      echo "DRY-RUN: kill -${signal} ${pid}"
    fi
    return 0
  fi

  if [[ "${use_sudo}" -eq 1 ]]; then
    sudo kill "-${signal}" "${pid}" 2>/dev/null || true
  else
    kill "-${signal}" "${pid}" 2>/dev/null || true
  fi
}

collect_matches() {
  local pid
  for pid in "${!PID_TO_CMD[@]}"; do
    [[ "${pid}" == "$$" || "${pid}" == "$PPID" ]] && continue
    if command_matches "${PID_TO_CMD[${pid}]}"; then
      local pgid="${PID_TO_PGID[${pid}]:-}"
      if [[ -n "${pgid}" && "${pgid}" =~ ^[0-9]+$ ]]; then
        MATCHED_GROUPS["${pgid}"]=1
      else
        MATCHED_SINGLES["${pid}"]=1
      fi
    fi
  done
}

load_process_table
load_active_runtime_dirs
collect_matches

if [[ ${#MATCHED_GROUPS[@]} -eq 0 && ${#MATCHED_SINGLES[@]} -eq 0 ]]; then
  echo "No stale Mahimahi/Sage processes found."
  exit 0
fi

declare -a MATCHED_GROUP_IDS=()
if [[ ${#MATCHED_GROUPS[@]} -gt 0 ]]; then
  mapfile -t MATCHED_GROUP_IDS < <(printf '%s\n' "${!MATCHED_GROUPS[@]}" | sort -n)
fi

declare -a MATCHED_SINGLE_IDS=()
if [[ ${#MATCHED_SINGLES[@]} -gt 0 ]]; then
  mapfile -t MATCHED_SINGLE_IDS < <(printf '%s\n' "${!MATCHED_SINGLES[@]}" | sort -n)
fi

echo "Matched process groups:"
if [[ ${#MATCHED_GROUP_IDS[@]} -eq 0 ]]; then
  echo "  <none>"
else
  for pgid in "${MATCHED_GROUP_IDS[@]}"; do
    runtime_dir=""
    sample_pid=""
    sample_cmd=""
    process_count=0
    pid_list="${GROUP_TO_PIDS[${pgid}]:-}"

    for pid in ${pid_list}; do
      process_count=$((process_count + 1))
      if [[ -z "${sample_pid}" ]]; then
        sample_pid="${pid}"
        sample_cmd="${PID_TO_CMD[${pid}]:-}"
      fi
      if [[ -z "${runtime_dir}" ]]; then
        runtime_dir="$(runtime_dir_for_pid "${pid}" || true)"
      fi
    done

    GROUP_TO_RUNTIME_DIR["${pgid}"]="${runtime_dir}"
    active_flag="no"
    if runtime_dir_is_active "${runtime_dir}"; then
      active_flag="yes"
    fi

    printf "  pgid=%s size=%s active_runtime=%s runtime=%s sample_pid=%s sample_cmd=%s\n" \
      "${pgid}" \
      "${process_count}" \
      "${active_flag}" \
      "${runtime_dir:-<none>}" \
      "${sample_pid:-<none>}" \
      "$(shorten_cmd "${sample_cmd:-<none>}")"
  done
fi

if [[ ${#MATCHED_SINGLE_IDS[@]} -gt 0 ]]; then
  echo
  echo "Matched standalone processes:"
  for pid in "${MATCHED_SINGLE_IDS[@]}"; do
    runtime_dir="$(runtime_dir_for_pid "${pid}" || true)"
    SINGLE_TO_RUNTIME_DIR["${pid}"]="${runtime_dir}"
    active_flag="no"
    if runtime_dir_is_active "${runtime_dir}"; then
      active_flag="yes"
    fi
    printf "  pid=%s user=%s active_runtime=%s runtime=%s cmd=%s\n" \
      "${pid}" \
      "${PID_TO_USER[${pid}]:-unknown}" \
      "${active_flag}" \
      "${runtime_dir:-<none>}" \
      "$(shorten_cmd "${PID_TO_CMD[${pid}]}")"
  done
fi

declare -a TARGET_GROUPS=()
declare -a PROTECTED_GROUPS=()
for pgid in "${MATCHED_GROUP_IDS[@]}"; do
  runtime_dir="${GROUP_TO_RUNTIME_DIR[${pgid}]:-}"
  if runtime_dir_is_active "${runtime_dir}" && [[ "${FORCE}" -ne 1 ]]; then
    PROTECTED_GROUPS+=("${pgid}")
  else
    TARGET_GROUPS+=("${pgid}")
  fi
done

declare -a TARGET_SINGLES=()
declare -a PROTECTED_SINGLES=()
for pid in "${MATCHED_SINGLE_IDS[@]}"; do
  runtime_dir="${SINGLE_TO_RUNTIME_DIR[${pid}]:-}"
  if runtime_dir_is_active "${runtime_dir}" && [[ "${FORCE}" -ne 1 ]]; then
    PROTECTED_SINGLES+=("${pid}")
  else
    TARGET_SINGLES+=("${pid}")
  fi
done

if [[ (${#PROTECTED_GROUPS[@]} -gt 0 || ${#PROTECTED_SINGLES[@]} -gt 0) && "${FORCE}" -ne 1 ]]; then
  echo
  echo "Skipping processes tied to active runtime leases (use --force to kill them):"
  for pgid in "${PROTECTED_GROUPS[@]}"; do
    printf "  pgid=%s runtime=%s\n" \
      "${pgid}" \
      "${GROUP_TO_RUNTIME_DIR[${pgid}]}"
  done
  for pid in "${PROTECTED_SINGLES[@]}"; do
    printf "  pid=%s runtime=%s cmd=%s\n" \
      "${pid}" \
      "${SINGLE_TO_RUNTIME_DIR[${pid}]}" \
      "${PID_TO_CMD[${pid}]}"
  done
fi

if [[ ${#TARGET_GROUPS[@]} -eq 0 && ${#TARGET_SINGLES[@]} -eq 0 ]]; then
  if [[ "${FORCE}" -ne 1 ]]; then
    echo
    echo "No unprotected stale processes to kill. Re-run with --force to kill protected ones."
  else
    echo
    echo "No matching processes remain after filtering."
  fi
  exit 0
fi

for pgid in "${TARGET_GROUPS[@]}"; do
  kill_group_signal TERM "${pgid}"
done
for pid in "${TARGET_SINGLES[@]}"; do
  kill_single_signal TERM "${pid}"
done

if [[ "${DRY_RUN}" -eq 0 ]]; then
  sleep "${GRACE_SECONDS}"
fi

for pgid in "${TARGET_GROUPS[@]}"; do
  if group_alive "${pgid}"; then
    kill_group_signal KILL "${pgid}"
  fi
done
for pid in "${TARGET_SINGLES[@]}"; do
  if single_alive "${pid}"; then
    kill_single_signal KILL "${pid}"
  fi
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run completed."
  exit 0
fi

echo "Cleanup completed."
