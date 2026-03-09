#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLIENT_PATH="${REPO_ROOT}/sage_rl/rl_module/client"

DRY_RUN=0
NO_SUDO=0
GRACE_SECONDS=3

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run] [--no-sudo] [--grace-seconds N]

Kill stale Mahimahi adaptive-network and Sage RL client processes:
  - mm-adv-net (including sudo wrapper processes)
  - ${CLIENT_PATH}

Options:
  --dry-run          Show matching processes, do not kill.
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

if ! command -v pgrep >/dev/null 2>&1; then
  echo "pgrep is required but not found" >&2
  exit 1
fi

declare -A PID_TO_CMD=()
declare -A PID_TO_USER=()

collect_matches() {
  local pattern="$1"
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    local pid="${line%% *}"
    local cmd="${line#* }"
    [[ -z "${pid}" || "${pid}" == "${line}" ]] && continue
    [[ "${pid}" == "$$" || "${pid}" == "$PPID" ]] && continue
    PID_TO_CMD["${pid}"]="${cmd}"
    PID_TO_USER["${pid}"]="$(ps -o user= -p "${pid}" | awk '{print $1}' || true)"
  done < <(pgrep -af -- "${pattern}" || true)
}

collect_matches "mm-adv-net"
collect_matches "${CLIENT_PATH}"

if [[ ${#PID_TO_CMD[@]} -eq 0 ]]; then
  echo "No stale mm-adv-net/client processes found."
  exit 0
fi

echo "Matched processes:"
for pid in "${!PID_TO_CMD[@]}"; do
  printf "  pid=%s user=%s cmd=%s\n" "${pid}" "${PID_TO_USER[${pid}]:-unknown}" "${PID_TO_CMD[${pid}]}"
done | sort -n -k2

kill_signal() {
  local signal="$1"
  local pid="$2"
  local owner="${PID_TO_USER[${pid}]:-}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    if [[ "${owner}" == "${USER}" || -z "${owner}" || "${NO_SUDO}" -eq 1 ]]; then
      echo "DRY-RUN: kill -${signal} ${pid}"
    else
      echo "DRY-RUN: sudo kill -${signal} ${pid}"
    fi
    return 0
  fi

  if [[ "${owner}" == "${USER}" || -z "${owner}" || "${NO_SUDO}" -eq 1 ]]; then
    kill "-${signal}" "${pid}" 2>/dev/null || true
  else
    sudo kill "-${signal}" "${pid}" 2>/dev/null || true
  fi
}

for pid in "${!PID_TO_CMD[@]}"; do
  kill_signal TERM "${pid}"
done

if [[ "${DRY_RUN}" -eq 0 ]]; then
  sleep "${GRACE_SECONDS}"
fi

for pid in "${!PID_TO_CMD[@]}"; do
  if kill -0 "${pid}" 2>/dev/null; then
    kill_signal KILL "${pid}"
  fi
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run completed."
  exit 0
fi

echo "Cleanup completed."
