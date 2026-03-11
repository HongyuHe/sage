#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: ./scripts/rebuild.sh [--no-sudo] [--jobs N] [--verbose]

Rebuild the installed `mm-adv-net` binary and the Sage userspace binaries.

Options are forwarded to `scripts/check_rebuild_mm_adv_net.sh`.
EOF
}

log() {
  printf '[rebuild] %s\n' "$*"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

log "Rebuilding installed mm-adv-net binary"
"${SCRIPT_DIR}/check_rebuild_mm_adv_net.sh" --force "$@"

log "Rebuilding Sage userspace binaries"
(
  cd "${REPO_ROOT}/sage_rl"
  ./build.sh
)

log "Rebuild complete."
