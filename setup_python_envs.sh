#!/usr/bin/env bash
set -euo pipefail

#* Example usage:
#*   bash setup_python_envs.sh
#*   bash setup_python_envs.sh --verify-only
#*   bash setup_python_envs.sh --skip-system-packages --torch-wheel default

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
PYENV_ROOT="${HOME}/.pyenv"
PYTHON_VERSION="3.8.18"
RUNTIME_VENV_REAL="${HOME}/venvpy36"
RUNTIME_VENV_LINK="/home/${USER}/venvpy36"
ATTACK_VENV="${REPO_ROOT}/.venv"
INSTALL_SYSTEM_PACKAGES=1
SETUP_RUNTIME_VENV=1
SETUP_ATTACK_VENV=1
VERIFY_ENVS=1
FORCE_RECREATE=0
TORCH_WHEEL="cpu"

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Provision the Python environments used by Sage:
  1. Runtime env for Sage victim processes at ${RUNTIME_VENV_LINK}
  2. Repo-local attack/orchestration env at ${ATTACK_VENV}

Options:
  --python-version VERSION     Python version managed via pyenv (default: ${PYTHON_VERSION})
  --runtime-venv PATH          Runtime venv real path (default: ${RUNTIME_VENV_REAL})
  --attack-venv PATH           Attack venv path (default: ${ATTACK_VENV})
  --skip-system-packages       Do not apt-install build/runtime prerequisites
  --skip-runtime-venv          Do not create/update the Sage runtime env
  --skip-attack-venv           Do not create/update the repo-local attack env
  --skip-verify                Do not run post-install verification checks
  --verify-only                Skip installation and only verify existing envs
  --force-recreate             Recreate selected venvs from scratch
  --torch-wheel MODE           Torch wheel source for attack env: cpu | default (default: ${TORCH_WHEEL})
  -h, --help                   Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --runtime-venv)
      RUNTIME_VENV_REAL="$2"
      shift 2
      ;;
    --attack-venv)
      ATTACK_VENV="$2"
      shift 2
      ;;
    --skip-system-packages)
      INSTALL_SYSTEM_PACKAGES=0
      shift
      ;;
    --skip-runtime-venv)
      SETUP_RUNTIME_VENV=0
      shift
      ;;
    --skip-attack-venv)
      SETUP_ATTACK_VENV=0
      shift
      ;;
    --skip-verify)
      VERIFY_ENVS=0
      shift
      ;;
    --verify-only)
      INSTALL_SYSTEM_PACKAGES=0
      SETUP_RUNTIME_VENV=0
      SETUP_ATTACK_VENV=0
      VERIFY_ENVS=1
      shift
      ;;
    --force-recreate)
      FORCE_RECREATE=1
      shift
      ;;
    --torch-wheel)
      TORCH_WHEEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

log() {
  echo "[setup-python-envs] $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

setup_home_symlink() {
  if [[ ! -e "/home/${USER}" ]]; then
    log "Creating /home/${USER} -> ${HOME} symlink"
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
    gnuplot
}

activate_pyenv() {
  export PYENV_ROOT
  export PATH="${PYENV_ROOT}/bin:${PATH}"
  eval "$(pyenv init -)"
}

setup_pyenv_python() {
  log "Ensuring pyenv and Python ${PYTHON_VERSION}"
  if [[ ! -d "${PYENV_ROOT}" ]]; then
    curl -fsSL https://pyenv.run | bash
  fi

  activate_pyenv
  pyenv install -s "${PYTHON_VERSION}"
}

python_bin_for_version() {
  local version="$1"
  local major_minor
  major_minor="$(echo "${version}" | awk -F. '{print $1"."$2}')"
  echo "${PYENV_ROOT}/versions/${version}/bin/python${major_minor}"
}

create_or_reuse_venv() {
  local python_bin="$1"
  local venv_path="$2"

  if [[ ${FORCE_RECREATE} -eq 1 && -d "${venv_path}" ]]; then
    log "Recreating ${venv_path}"
    rm -rf "${venv_path}"
  fi

  if [[ ! -d "${venv_path}" ]]; then
    log "Creating venv at ${venv_path}"
    "${python_bin}" -m venv "${venv_path}"
  else
    log "Reusing existing venv at ${venv_path}"
  fi
}

ensure_runtime_link() {
  if [[ -L "${RUNTIME_VENV_LINK}" ]]; then
    local target
    target="$(readlink -f "${RUNTIME_VENV_LINK}")"
    if [[ "${target}" != "$(readlink -f "${RUNTIME_VENV_REAL}")" ]]; then
      echo "Runtime link ${RUNTIME_VENV_LINK} points to ${target}, expected ${RUNTIME_VENV_REAL}" >&2
      exit 1
    fi
    return
  fi

  if [[ -e "${RUNTIME_VENV_LINK}" ]]; then
    echo "Runtime link path exists and is not a symlink: ${RUNTIME_VENV_LINK}" >&2
    exit 1
  fi

  log "Creating runtime symlink ${RUNTIME_VENV_LINK} -> ${RUNTIME_VENV_REAL}"
  sudo ln -s "${RUNTIME_VENV_REAL}" "${RUNTIME_VENV_LINK}"
}

install_runtime_env() {
  local python_bin="$1"
  create_or_reuse_venv "${python_bin}" "${RUNTIME_VENV_REAL}"
  ensure_runtime_link

  log "Installing Sage runtime packages into ${RUNTIME_VENV_REAL}"
  "${RUNTIME_VENV_REAL}/bin/python" -m pip install -U "pip<24" "setuptools<58"
  "${RUNTIME_VENV_REAL}/bin/python" -m pip install "wheel==0.36.2"
  "${RUNTIME_VENV_REAL}/bin/python" -m pip install -r "${REPO_ROOT}/requirements.runtime-py38.txt"
  "${RUNTIME_VENV_REAL}/bin/python" -m pip install -e "${REPO_ROOT}" --no-deps
}

install_attack_env() {
  local python_bin="$1"
  create_or_reuse_venv "${python_bin}" "${ATTACK_VENV}"

  log "Installing attack/orchestration packages into ${ATTACK_VENV}"
  "${ATTACK_VENV}/bin/python" -m pip install -U pip setuptools wheel
  "${ATTACK_VENV}/bin/python" -m pip install -r "${REPO_ROOT}/requirements.attack.txt"

  case "${TORCH_WHEEL}" in
    cpu)
      "${ATTACK_VENV}/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
      ;;
    default)
      "${ATTACK_VENV}/bin/python" -m pip install torch==2.4.1
      ;;
    *)
      echo "Unsupported --torch-wheel value: ${TORCH_WHEEL}" >&2
      exit 2
      ;;
  esac

  "${ATTACK_VENV}/bin/python" -m pip install -e "${REPO_ROOT}" --no-deps
}

verify_runtime_env() {
  if [[ ! -x "${RUNTIME_VENV_LINK}/bin/python" ]]; then
    echo "Runtime Python not found: ${RUNTIME_VENV_LINK}/bin/python" >&2
    exit 1
  fi

  log "Verifying Sage runtime env"
  "${RUNTIME_VENV_LINK}/bin/python" --version
  "${RUNTIME_VENV_LINK}/bin/python" - <<'PY'
import tensorflow, acme, sonnet, reverb, jax, sysv_ipc, gym
print("runtime imports ok")
PY
}

verify_attack_env() {
  if [[ ! -x "${ATTACK_VENV}/bin/python" ]]; then
    echo "Attack Python not found: ${ATTACK_VENV}/bin/python" >&2
    exit 1
  fi

  log "Verifying attack/orchestration env"
  "${ATTACK_VENV}/bin/python" --version
  "${ATTACK_VENV}/bin/python" - <<'PY'
import gymnasium, matplotlib, numpy, pandas, seaborn, stable_baselines3, sysv_ipc, torch, wandb
print("attack env imports ok")
PY
}

main() {
  require_cmd git
  require_cmd curl
  require_cmd python3

  if [[ ${INSTALL_SYSTEM_PACKAGES} -eq 1 || ${SETUP_RUNTIME_VENV} -eq 1 ]]; then
    require_cmd sudo
  fi

  if [[ ${INSTALL_SYSTEM_PACKAGES} -eq 1 ]]; then
    install_system_packages
  fi

  if [[ ${SETUP_RUNTIME_VENV} -eq 1 || ${SETUP_ATTACK_VENV} -eq 1 ]]; then
    setup_pyenv_python
  elif [[ ${VERIFY_ENVS} -eq 1 ]]; then
    log "Skipping pyenv installation because no venv creation was requested"
  fi

  local python_bin=""
  if [[ ${SETUP_RUNTIME_VENV} -eq 1 || ${SETUP_ATTACK_VENV} -eq 1 ]]; then
    python_bin="$(python_bin_for_version "${PYTHON_VERSION}")"
    if [[ ! -x "${python_bin}" ]]; then
      echo "Python binary not found after pyenv setup: ${python_bin}" >&2
      exit 1
    fi
  fi

  if [[ ${SETUP_RUNTIME_VENV} -eq 1 ]]; then
    setup_home_symlink
    install_runtime_env "${python_bin}"
  fi

  if [[ ${SETUP_ATTACK_VENV} -eq 1 ]]; then
    install_attack_env "${python_bin}"
  fi

  if [[ ${VERIFY_ENVS} -eq 1 ]]; then
    if [[ ${SETUP_RUNTIME_VENV} -eq 1 || -x "${RUNTIME_VENV_LINK}/bin/python" ]]; then
      verify_runtime_env
    fi
    if [[ ${SETUP_ATTACK_VENV} -eq 1 || -x "${ATTACK_VENV}/bin/python" ]]; then
      verify_attack_env
    fi
  fi

  log "Done."
}

main "$@"
