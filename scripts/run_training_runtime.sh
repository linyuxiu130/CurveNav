#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
RUNTIME_ROOT="${CURVENAV_RUNTIME_ROOT:-${WORKSPACE_ROOT}/curvenav-runtime/rootfs}"
HOST_DRIVER_ROOT="/usr/lib/x86_64-linux-gnu"

if [[ ! -x "${RUNTIME_ROOT}/bin/bash" ]]; then
    echo "CurveNav training runtime is missing: ${RUNTIME_ROOT}" >&2
    exit 1
fi
if (( $# == 0 )); then
    echo "usage: $0 COMMAND [ARG ...]" >&2
    exit 2
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
    if [[ "${soname}" == "libcuda.so.1" ]]; then
        driver_mounts+=(--ro-bind "${driver_path}" /opt/nvidia/libcuda.so)
    fi
done

exec bwrap \
    --die-with-parent \
    --ro-bind "${RUNTIME_ROOT}" / \
    --dev-bind /dev /dev \
    --proc /proc \
    --ro-bind /sys /sys \
    --bind "${WORKSPACE_ROOT}" "${WORKSPACE_ROOT}" \
    --ro-bind /etc/hosts /etc/hosts \
    --ro-bind /etc/resolv.conf /etc/resolv.conf \
    --tmpfs /tmp \
    --tmpfs /opt/nvidia \
    "${driver_mounts[@]}" \
    --setenv HOME "${WORKSPACE_ROOT}" \
    --setenv LD_LIBRARY_PATH /opt/nvidia \
    --chdir "${PROJECT_ROOT}" \
    "$@"
