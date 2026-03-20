#!/usr/bin/env bash
set -euo pipefail

# Post-reboot setup for Sage runtime on kernel 4.19.112-0062.
# This script is idempotent and safe to re-run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAGE_DIR="${SCRIPT_DIR}"
SAGE_RL_DIR="${SAGE_DIR}/sage_rl"
EXPECTED_KERNEL="4.19.112-0062"
PYENV_ROOT="${HOME}/.pyenv"
PYTHON_VERSION="3.8.18"
VENV_LINK="/home/${USER}/venvpy36"
VENV_REAL="${HOME}/venvpy36"

RUN_SMOKE_TEST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-smoke-test) RUN_SMOKE_TEST=1 ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--with-smoke-test]" >&2
      exit 2
      ;;
  esac
  shift
done

log() {
  echo "[post-reboot] $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

append_unique_token() {
  local current="$1"
  local token="$2"

  if [[ -z "${current}" ]]; then
    echo "${token}"
    return
  fi

  case " ${current} " in
    *" ${token} "*) echo "${current}" ;;
    *) echo "${current} ${token}" ;;
  esac
}

setup_home_symlink() {
  # Sage binaries expect /home/$USER/venvpy36/bin/python.
  if [[ ! -e "/home/${USER}" ]]; then
    sudo mkdir -p /home
    sudo ln -s "${HOME}" "/home/${USER}"
  fi
}

install_system_packages() {
  log "Installing system packages"
  sudo apt-get update
  sudo apt-get -y install \
    build-essential curl git libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev libffi-dev liblzma-dev tk-dev xz-utils \
    gnuplot default-jre-headless
}

enable_bbr() {
  log "Enabling TCP BBR alongside Sage's custom 'pure' CCA"

  sudo modprobe tcp_bbr

  local available
  available="$(sysctl -n net.ipv4.tcp_available_congestion_control)"
  if [[ " ${available} " != *" bbr "* ]]; then
    echo "tcp_bbr was loaded but bbr is still not advertised as available: ${available}" >&2
    exit 1
  fi

  local allowed
  allowed="$(sysctl -n net.ipv4.tcp_allowed_congestion_control)"
  allowed="$(append_unique_token "${allowed}" "bbr")"
  allowed="$(append_unique_token "${allowed}" "pure")"
  sudo sysctl -w "net.ipv4.tcp_allowed_congestion_control=${allowed}" >/dev/null

  echo "tcp_bbr" | sudo tee /etc/modules-load.d/tcp_bbr.conf >/dev/null
  printf '%s\n' "net.ipv4.tcp_allowed_congestion_control = ${allowed}" | \
    sudo tee /etc/sysctl.d/99-sage-bbr.conf >/dev/null
}

setup_pyenv_python() {
  log "Installing pyenv and Python ${PYTHON_VERSION}"
  if [[ ! -d "${PYENV_ROOT}" ]]; then
    curl -fsSL https://pyenv.run | bash
  fi

  export PYENV_ROOT
  export PATH="${PYENV_ROOT}/bin:${PATH}"
  eval "$(pyenv init -)"
  pyenv install -s "${PYTHON_VERSION}"
}

setup_runtime_venv() {
  log "Creating runtime venv at ${VENV_REAL}"
  rm -rf "${VENV_REAL}"
  "${PYENV_ROOT}/versions/${PYTHON_VERSION}/bin/python3.8" -m venv "${VENV_REAL}"

  # Ensure hardcoded runtime path exists.
  if [[ ! -e "${VENV_LINK}" ]]; then
    sudo ln -s "${VENV_REAL}" "${VENV_LINK}"
  fi

  # shellcheck disable=SC1090
  source "${VENV_LINK}/bin/activate"
  pip install -U "pip<24" "setuptools<58"
  pip install "wheel==0.36.2"
}

install_python_runtime_deps() {
  log "Installing Python runtime dependencies"
  # shellcheck disable=SC1090
  source "${VENV_LINK}/bin/activate"

  pip install \
    absl-py==0.11.0 ruamel.yaml==0.16.12 tqdm==4.56.0 \
    numpy==1.19.5 pandas==1.1.5 matplotlib==3.3.4 \
    sysv-ipc==1.1.0 dm-env==1.3 dm-tree==0.1.5 dm-sonnet==2.0.0 dm-acme==0.2.0 \
    tensorflow==2.4.1 tensorflow-estimator==2.4.0 tensorflow-probability==0.12.1 \
    trfl==1.1.0 dm-reverb==0.2.0

  # Legacy gym pin required by dm-acme wrappers in this stack.
  pip install gym==0.18.0

  # JAX pin from project requirements (cp38 wheel URL).
  pip install \
    jax==0.2.9 \
    https://storage.googleapis.com/jax-releases/nocuda/jaxlib-0.1.60-cp38-none-manylinux2010_x86_64.whl

  pip install -e "${SAGE_DIR}" --no-deps
}

patch_client_overflow_if_needed() {
  local client_c="${SAGE_RL_DIR}/src/client.c"
  if grep -q "char query\\[100\\]" "${client_c}"; then
    log "Patching client buffer overflow in src/client.c"
    sed -i 's/char query\[100\]/char query[256]/' "${client_c}"
  fi
  # Make memset size consistent with buffer.
  sed -i "s/memset(query,'\\\\0',150);/memset(query,'\\\\0',sizeof(query));/" "${client_c}"
}

rebuild_native_binaries() {
  log "Rebuilding native binaries with generic x86_64 flags and CET disabled"
  pushd "${SAGE_RL_DIR}" >/dev/null

  local cxxflags=(
    -O2
    -march=x86-64
    -mtune=generic
    -fcf-protection=none
    -mno-shstk
  )

  g++ "${cxxflags[@]}" -pthread src/mem-manager.cc src/flow.cc -o mem-manager
  g++ "${cxxflags[@]}" -pthread src/sage.cc src/flow.cc -o sage
  g++ "${cxxflags[@]}" src/client.c -o client
  mv mem-manager client sage rl_module/

  popd >/dev/null
}

fix_permissions_and_models() {
  log "Fixing script/tool permissions and copying model checkpoints"
  pushd "${SAGE_RL_DIR}" >/dev/null
  chmod +x *.sh mm-thr mm-del-file plot-del.sh killall.sh mem_clean.sh
  bash cp_models.sh
  popd >/dev/null
}

sanity_checks() {
  log "Running sanity checks"
  # shellcheck disable=SC1090
  source "${VENV_LINK}/bin/activate"

  uname -r
  if [[ "$(uname -r)" != "${EXPECTED_KERNEL}" ]]; then
    echo "WARNING: running kernel is $(uname -r), expected ${EXPECTED_KERNEL}" >&2
  fi

  sysctl -n net.ipv4.tcp_available_congestion_control | grep -qw bbr
  sysctl -n net.ipv4.tcp_allowed_congestion_control | grep -qw bbr
  sysctl -n net.ipv4.tcp_allowed_congestion_control | grep -qw pure
  command -v mm-link mm-delay mm-loss mm-adv-net >/dev/null
  command -v java >/dev/null
  java -version
  "${VENV_LINK}/bin/python" --version
  "${VENV_LINK}/bin/python" -c "import tensorflow, acme, sonnet, reverb, jax, sysv_ipc, gym; print('python runtime imports ok')"
  "${VENV_LINK}/bin/python" "${SAGE_RL_DIR}/rl_module/test_loading_sage_model.py"

  if [[ ${RUN_SMOKE_TEST} -eq 1 ]]; then
    log "Running sample smoke test (timeout 45s)"
    pushd "${SAGE_RL_DIR}" >/dev/null
    timeout 45 bash run_sample.sh || true
    popd >/dev/null
  fi
}

main() {
  require_cmd sudo
  require_cmd git
  require_cmd curl
  require_cmd g++
  require_cmd python3

  setup_home_symlink
  install_system_packages
  enable_bbr
  setup_pyenv_python
  setup_runtime_venv
  install_python_runtime_deps
  patch_client_overflow_if_needed
  rebuild_native_binaries
  fix_permissions_and_models
  sanity_checks

  log "Done. Sage runtime is ready."
}

main "$@"
