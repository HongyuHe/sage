#!/usr/bin/env bash
set -euo pipefail

# Reusable pre-reboot setup for Sage.
# Covers:
# 1) Sage kernel package install + grub default selection
# 2) Mahimahi patched install from ccBench
# 3) Sage build + sysctl setup
# 4) Minimal Python env setup (best-effort on modern Ubuntu)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAGE_DIR="${SCRIPT_DIR}"
LINUX_PATCH_DIR="${SAGE_DIR}/linux-patch"
VENV_DIR="${HOME}/venvpy36"
KERNEL_VER="4.19.112-0062"
KERNEL_IMAGE_DEB="linux-image-${KERNEL_VER}_${KERNEL_VER}-10.00.Custom_amd64.deb"
KERNEL_HEADERS_DEB="linux-headers-${KERNEL_VER}_${KERNEL_VER}-10.00.Custom_amd64.deb"

# Prefer in-repo ccBench (./ccBench). Fall back to legacy ~/ccBench if needed.
if [[ -d "${SAGE_DIR}/ccBench" ]]; then
  CCBENCH_DIR="${SAGE_DIR}/ccBench"
elif [[ -d "${HOME}/ccBench" ]]; then
  CCBENCH_DIR="${HOME}/ccBench"
else
  CCBENCH_DIR="${SAGE_DIR}/ccBench"
fi
MAHIMAHI_DIR="${CCBENCH_DIR}/mahimahi"

SKIP_MAHIMAHI=0
SKIP_PYTHON=0
SKIP_PACKAGES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-mahimahi) SKIP_MAHIMAHI=1 ;;
    --skip-python) SKIP_PYTHON=1 ;;
    --skip-packages) SKIP_PACKAGES=1 ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--skip-mahimahi] [--skip-python] [--skip-packages]" >&2
      exit 2
      ;;
  esac
  shift
done

log() {
  echo "[pre-reboot] $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

apply_text_fix_if_missing() {
  local file="$1"
  local needle="$2"
  local replacement="$3"
  if ! grep -Fq "$replacement" "$file"; then
    sed -i "s|${needle}|${replacement}|g" "$file"
  fi
}

install_base_packages() {
  log "Installing base packages"
  sudo apt-get update
  sudo apt-get -y install \
    python3-pip python3-venv python3-dev virtualenv \
    libsctp1 iperf3
}

install_kernel_and_grub() {
  log "Installing Sage kernel deb packages"
  [[ -f "${LINUX_PATCH_DIR}/${KERNEL_IMAGE_DEB}" ]] || { echo "Missing ${KERNEL_IMAGE_DEB}" >&2; exit 1; }
  [[ -f "${LINUX_PATCH_DIR}/${KERNEL_HEADERS_DEB}" ]] || { echo "Missing ${KERNEL_HEADERS_DEB}" >&2; exit 1; }

  set +e
  sudo dpkg -i \
    "${LINUX_PATCH_DIR}/${KERNEL_IMAGE_DEB}" \
    "${LINUX_PATCH_DIR}/${KERNEL_HEADERS_DEB}"
  local dpkg_rc=$?
  set -e

  # On some CloudLab/Emulab images, emulab-ipod-dkms fails against this kernel.
  if [[ $dpkg_rc -ne 0 ]] || dpkg -l | grep -Eq '^iF\s+linux-(image|headers)-4\.19\.112-0062'; then
    if dpkg -l | grep -Eq '^ii\s+emulab-ipod-dkms'; then
      log "Removing emulab-ipod-dkms to unblock kernel configuration"
      sudo apt-get -y remove emulab-ipod-dkms || true
    fi
    sudo apt-get -y -f install
    sudo dpkg --configure -a
  fi

  # Ensure initrd exists for target kernel.
  if [[ ! -f "/boot/initrd.img-${KERNEL_VER}" ]]; then
    sudo update-initramfs -c -k "${KERNEL_VER}"
  fi

  log "Pinning GRUB default to ${KERNEL_VER}"
  sudo cp /etc/default/grub "/etc/default/grub.bak.$(date +%Y%m%d%H%M%S)"
  if grep -q '^GRUB_DEFAULT=' /etc/default/grub; then
    sudo sed -i "s|^GRUB_DEFAULT=.*|GRUB_DEFAULT=\"Advanced options for Ubuntu>Ubuntu, with Linux ${KERNEL_VER}\"|" /etc/default/grub
  else
    echo "GRUB_DEFAULT=\"Advanced options for Ubuntu>Ubuntu, with Linux ${KERNEL_VER}\"" | sudo tee -a /etc/default/grub >/dev/null
  fi
  sudo update-grub
}

install_mahimahi_patched() {
  log "Installing Mahimahi patched build from ccBench"
  sudo apt-get -y install \
    build-essential git debhelper autotools-dev dh-autoreconf iptables \
    protobuf-compiler libprotobuf-dev pkg-config libssl-dev dnsmasq-base \
    ssl-cert libxcb-present-dev libcairo2-dev libpango1.0-dev iproute2 \
    apache2-dev apache2-bin apache2-api-20120211 libwww-perl

  if [[ ! -d "${CCBENCH_DIR}" ]]; then
    git clone https://github.com/Soheil-ab/ccBench.git "${CCBENCH_DIR}"
  fi
  if [[ ! -d "${MAHIMAHI_DIR}" ]]; then
    git clone https://github.com/ravinet/mahimahi "${MAHIMAHI_DIR}"
  fi

  pushd "${MAHIMAHI_DIR}" >/dev/null
  patch --forward -p1 < "${CCBENCH_DIR}/patches/mahimahi.core.v2.2.patch" || true
  patch --forward -p1 < "${CCBENCH_DIR}/patches/mahimahi.extra.aqm.v1.5.patch" || true

  # Compatibility/text fixes observed on modern upstream/compiler.
  apply_text_fix_if_missing \
    "src/frontend/link_queue.cc" \
    '<< " " << departure_time - packet.arrival_time << endl;' \
    '<< " " << departure_time - packet.arrival_time << " " << packet.queue_num << endl;'

  apply_text_fix_if_missing \
    "src/packet/bode_packet_queue.cc" \
    'dodequeue_result r = std::move( dodequeue ( now ) );' \
    'dodequeue_result r = dodequeue( now );'

  ./autogen.sh
  ./configure
  make -j"$(nproc)"
  sudo make install
  sudo ldconfig
  popd >/dev/null

  sudo sysctl -w net.ipv4.ip_forward=1
}

setup_python_and_sage() {
  log "Setting up Python virtualenv and Sage package (best effort)"
  virtualenv --python=python3 "${VENV_DIR}"
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  python -m pip install -U pip
  python -m pip install -e "${SAGE_DIR}" --no-deps
  deactivate
}

build_and_sysctl_sage() {
  log "Building Sage binaries and applying sysctl tuning"
  pushd "${SAGE_DIR}/sage_rl" >/dev/null
  bash build.sh
  bash set_sysctl.sh
  popd >/dev/null
}

verify_summary() {
  log "Verification summary"
  dpkg -l | grep -E 'linux-(image|headers)-4.19.112-0062' || true
  grep -E '^(GRUB_DEFAULT|GRUB_TIMEOUT)=' /etc/default/grub || true
  ls -lh "/boot/vmlinuz-${KERNEL_VER}" "/boot/initrd.img-${KERNEL_VER}" 2>/dev/null || true
  command -v mm-delay mm-link mm-loss mm-adv-net || true
}

main() {
  require_cmd sudo
  require_cmd git
  require_cmd dpkg
  require_cmd apt-get

  if [[ $SKIP_PACKAGES -eq 0 ]]; then
    install_base_packages
  fi
  install_kernel_and_grub
  if [[ $SKIP_MAHIMAHI -eq 0 ]]; then
    install_mahimahi_patched
  fi
  build_and_sysctl_sage
  if [[ $SKIP_PYTHON -eq 0 ]]; then
    setup_python_and_sage || {
      log "Python env setup failed (likely version/pinned-wheel mismatch). Continuing."
    }
  fi
  verify_summary

  log "Done. Reboot and verify: uname -r (expect ${KERNEL_VER})"
}

main "$@"
