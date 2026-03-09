#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAHIMAHI_DIR="${REPO_ROOT}/ccBench/mahimahi"
BINARY_PATH="/usr/local/bin/mm-adv-net"

CHECK_ONLY=0
FORCE_REBUILD=0
USE_SUDO=1
VERBOSE=0
JOBS="$(nproc)"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

# check only
./scripts/check_rebuild_mm_adv_net.sh --check-only

# check + rebuild if outdated
./scripts/check_rebuild_mm_adv_net.sh

# force rebuild
./scripts/check_rebuild_mm_adv_net.sh --force

Check whether mm-adv-net is outdated relative to Mahimahi sources, and rebuild if needed.

Options:
  --check-only          Only check status; do not rebuild.
  --force               Rebuild even if up to date.
  --no-sudo             Do not use sudo for install/ldconfig.
  --mahimahi-dir PATH   Mahimahi source directory (default: ${MAHIMAHI_DIR})
  --binary-path PATH    Installed mm-adv-net path (default: ${BINARY_PATH})
  --jobs N              Parallel build jobs (default: nproc)
  --verbose             Print additional diagnostics.
  -h, --help            Show this help.

Exit codes:
  0  up to date (or rebuilt successfully)
  3  outdated (with --check-only)
  1  error
EOF
}

log() {
  printf '[mm-adv-net] %s\n' "$*"
}

debug() {
  if [[ "${VERBOSE}" -eq 1 ]]; then
    printf '[mm-adv-net][debug] %s\n' "$*"
  fi
}

die() {
  printf '[mm-adv-net][error] %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --force)
      FORCE_REBUILD=1
      shift
      ;;
    --no-sudo)
      USE_SUDO=0
      shift
      ;;
    --mahimahi-dir)
      [[ $# -ge 2 ]] || die "missing value for --mahimahi-dir"
      MAHIMAHI_DIR="$2"
      shift 2
      ;;
    --binary-path)
      [[ $# -ge 2 ]] || die "missing value for --binary-path"
      BINARY_PATH="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || die "missing value for --jobs"
      JOBS="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -d "${MAHIMAHI_DIR}" ]] || die "mahimahi directory not found: ${MAHIMAHI_DIR}"
[[ "${JOBS}" =~ ^[0-9]+$ ]] || die "--jobs must be a non-negative integer"

SOURCE_GLOBS=(
  '*.cc'
  '*.hh'
  '*.h'
  'Makefile.am'
  'Makefile.in'
  'configure.ac'
  'configure'
)

latest_source_epoch=""
latest_source_path=""
binary_epoch=""
declare -a outdated_reasons=()

collect_latest_source() {
  local record
  record="$(
    find "${MAHIMAHI_DIR}" \
      \( -path "${MAHIMAHI_DIR}/src/*" -o -path "${MAHIMAHI_DIR}/Makefile.am" -o -path "${MAHIMAHI_DIR}/Makefile.in" -o -path "${MAHIMAHI_DIR}/configure.ac" -o -path "${MAHIMAHI_DIR}/configure" \) \
      -type f \
      \( -name '*.cc' -o -name '*.hh' -o -name '*.h' -o -name 'Makefile.am' -o -name 'Makefile.in' -o -name 'configure.ac' -o -name 'configure' \) \
      -printf '%T@ %p\n' \
      | sort -nr \
      | head -n 1
  )"
  [[ -n "${record}" ]] || die "failed to discover source files under ${MAHIMAHI_DIR}"
  latest_source_epoch="${record%% *}"
  latest_source_path="${record#* }"
}

binary_supports_adaptive_flags() {
  if [[ ! -x "${BINARY_PATH}" ]]; then
    return 1
  fi
  local help_text
  help_text="$("${BINARY_PATH}" --help 2>&1 || true)"
  grep -q -- '--control-file' <<< "${help_text}"
}

check_outdated() {
  outdated_reasons=()
  collect_latest_source

  if [[ ! -x "${BINARY_PATH}" ]]; then
    outdated_reasons+=("installed binary is missing or not executable: ${BINARY_PATH}")
    return 0
  fi

  binary_epoch="$(stat -c '%Y' "${BINARY_PATH}")"
  local latest_source_int
  latest_source_int="$(printf '%.0f' "${latest_source_epoch}")"
  if (( latest_source_int > binary_epoch )); then
    outdated_reasons+=("source is newer than installed binary (${latest_source_path})")
  fi

  if ! binary_supports_adaptive_flags; then
    outdated_reasons+=("installed binary does not expose adaptive control flags")
  fi

  if [[ ${#outdated_reasons[@]} -gt 0 ]]; then
    return 0
  fi
  return 1
}

needs_autotools_refresh() {
  if [[ ! -f "${MAHIMAHI_DIR}/configure" || ! -f "${MAHIMAHI_DIR}/Makefile.in" ]]; then
    return 0
  fi
  if [[ "${MAHIMAHI_DIR}/configure.ac" -nt "${MAHIMAHI_DIR}/configure" ]]; then
    return 0
  fi
  if find "${MAHIMAHI_DIR}" -name 'Makefile.am' -type f -newer "${MAHIMAHI_DIR}/Makefile.in" | grep -q .; then
    return 0
  fi
  return 1
}

run_install_step() {
  if [[ "${USE_SUDO}" -eq 1 ]]; then
    sudo make install
    sudo ldconfig
  else
    make install
    debug "skipping ldconfig (no sudo mode)"
  fi
}

rebuild_binary() {
  log "Rebuilding mm-adv-net from ${MAHIMAHI_DIR}"
  pushd "${MAHIMAHI_DIR}" >/dev/null
  if needs_autotools_refresh; then
    log "Refreshing autotools files (autogen/configure)"
    ./autogen.sh
    ./configure
  elif [[ ! -f "${MAHIMAHI_DIR}/Makefile" ]]; then
    log "Generating Makefile via configure"
    ./configure
  fi
  make -j"${JOBS}"
  run_install_step
  popd >/dev/null
}

summarize_status() {
  collect_latest_source
  log "Latest source file: ${latest_source_path}"
  if [[ -x "${BINARY_PATH}" ]]; then
    log "Installed binary: ${BINARY_PATH}"
    log "Installed binary mtime: $(stat -c '%y' "${BINARY_PATH}")"
  else
    log "Installed binary: missing (${BINARY_PATH})"
  fi
}

summarize_status
if check_outdated; then
  log "mm-adv-net is OUTDATED."
  for reason in "${outdated_reasons[@]}"; do
    log "reason: ${reason}"
  done
  if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    exit 3
  fi
elif [[ "${FORCE_REBUILD}" -eq 1 ]]; then
  log "mm-adv-net appears up to date, but --force was requested."
else
  log "mm-adv-net is up to date."
  exit 0
fi

rebuild_binary
if check_outdated; then
  die "rebuild completed but mm-adv-net still appears outdated"
fi
log "mm-adv-net rebuild complete and up to date."
