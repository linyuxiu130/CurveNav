#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
RUNTIME_ROOT="${CURVENAV_RUNTIME_ROOT:-${WORKSPACE_ROOT}/curvenav-runtime-self-contained/rootfs}"
HOST_DRIVER_ROOT="/usr/lib/x86_64-linux-gnu"

if [[ ! -x "${RUNTIME_ROOT}/bin/bash" ]]; then
    echo "CurveNav training runtime is missing: ${RUNTIME_ROOT}" >&2
    exit 1
fi
if (( $# == 0 )); then
    echo "usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi
if [[ ! -x "${RUNTIME_ROOT}/usr/bin/python3" \
    || ! -f "${RUNTIME_ROOT}/usr/include/python3.10/Python.h" \
    || ! -x "${RUNTIME_ROOT}/usr/bin/gcc" ]]; then
    echo "CurveNav self-contained Python/C toolchain is missing" >&2
    exit 1
fi
# bwrap creates the bind mount target itself, but its parents must already
# exist in a writable mount.  Overlay the deepest existing workspace ancestor
# in the immutable rootfs, then materialize only the missing descendants.
workspace_dirs=()
workspace_parent="$(dirname "${WORKSPACE_ROOT}")"
while [[ "${workspace_parent}" != "/" ]]; do
    workspace_dirs+=("${workspace_parent}")
    workspace_parent="$(dirname "${workspace_parent}")"
done
runtime_workspace_dirs=()
runtime_workspace_mount=""
for (( index=${#workspace_dirs[@]} - 1; index >= 0; index-- )); do
    workspace_dir="${workspace_dirs[index]}"
    if [[ -e "${RUNTIME_ROOT}${workspace_dir}" ]]; then
        runtime_workspace_mount="${workspace_dir}"
        runtime_workspace_dirs=()
    else
        runtime_workspace_dirs+=(--dir "${workspace_dir}")
    fi
done
if [[ -z "${runtime_workspace_mount}" ]]; then
    echo "runtime rootfs has no workspace mount ancestor" >&2
    exit 1
fi

driver_mounts=()
for soname in \
    libcuda.so.1 \
    libnvidia-ml.so.1 \
    libnvidia-nvvm.so.4 \
    libnvidia-ptxjitcompiler.so.1
do
    driver_path="$(readlink -f "${HOST_DRIVER_ROOT}/${soname}")"
    if [[ ! -f "${driver_path}" ]]; then
        echo "NVIDIA driver library is missing: ${HOST_DRIVER_ROOT}/${soname}" >&2
        exit 1
    fi
    driver_mounts+=(--ro-bind "${driver_path}" "/opt/nvidia/${soname}")
    runtime_driver_link="${RUNTIME_ROOT}/usr/lib/x86_64-linux-gnu/${soname}"
    if [[ -e "${runtime_driver_link}" || -L "${runtime_driver_link}" ]]; then
        runtime_driver_path="$(readlink -f "${runtime_driver_link}")"
        if [[ ! -f "${runtime_driver_path}" \
            || "${runtime_driver_path}" != "${RUNTIME_ROOT}"/* ]]; then
            echo "Invalid CurveNav runtime NVIDIA ABI target: ${soname}" >&2
            exit 1
        fi
        driver_mounts+=(
            --ro-bind "${driver_path}" "${runtime_driver_path#${RUNTIME_ROOT}}"
        )
    fi
    if [[ "${soname}" == "libcuda.so.1" ]]; then
        driver_mounts+=(--ro-bind "${driver_path}" /opt/nvidia/libcuda.so)
    fi
done

exec bwrap \
    --die-with-parent \
    --unshare-pid \
    --as-pid-1 \
    --ro-bind "${RUNTIME_ROOT}" / \
    --dev-bind /dev /dev \
    --proc /proc \
    --ro-bind /sys /sys \
    --tmpfs "${runtime_workspace_mount}" \
    "${runtime_workspace_dirs[@]}" \
    --bind "${WORKSPACE_ROOT}" "${WORKSPACE_ROOT}" \
    --ro-bind /etc/hosts /etc/hosts \
    --ro-bind /etc/resolv.conf /etc/resolv.conf \
    --tmpfs /tmp \
    --tmpfs /opt \
    --dir /opt/nvidia \
    "${driver_mounts[@]}" \
    --setenv HOME "${WORKSPACE_ROOT}" \
    --setenv LD_LIBRARY_PATH /opt/nvidia \
    --setenv LIBRARY_PATH /opt/nvidia \
    --setenv PYTHONPATH "${PROJECT_ROOT}/src" \
    --chdir "${PROJECT_ROOT}" \
    "$@"
