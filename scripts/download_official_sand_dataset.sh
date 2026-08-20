#!/usr/bin/env bash
set -euo pipefail

readonly DATA_ROOT="/mnt/data/huangshibo/H/navigation_three_projects/datasets/sandplanner"
readonly ARCHIVE="${DATA_ROOT}/dataset_avoid.zip"
readonly PARTIAL="${ARCHIVE}.part"
readonly EXTRACT_ROOT="${DATA_ROOT}/dataset_avoid"
readonly STAGING_ROOT="${DATA_ROOT}/.extracting"
readonly REPO_REVISION="533fb49c1397a08ded3c7a201d1abc7d965cbd3f"
readonly ARCHIVE_SHA256="abd2ec134b0b707d5938e1584059338a3d0c42b2fb2b60583d1aa9455e500c4e"
readonly ARCHIVE_URL="https://huggingface.co/datasets/WJCUCL/sandplanner-dataset-avoid/resolve/${REPO_REVISION}/dataset_avoid.zip"

mkdir -p "${DATA_ROOT}"

if [[ ! -f "${ARCHIVE}" ]]; then
    wget \
        --continue \
        --progress=bar:force:noscroll \
        --output-document="${PARTIAL}" \
        "${ARCHIVE_URL}"
    mv "${PARTIAL}" "${ARCHIVE}"
fi

printf '%s  %s\n' "${ARCHIVE_SHA256}" "${ARCHIVE}" | sha256sum --check --strict

if [[ ! -d "${EXTRACT_ROOT}" ]]; then
    mkdir -p "${STAGING_ROOT}"
    unzip -q "${ARCHIVE}" -d "${STAGING_ROOT}"
    test -d "${STAGING_ROOT}/dataset_avoid"
    mv "${STAGING_ROOT}/dataset_avoid" "${EXTRACT_ROOT}"
    rmdir "${STAGING_ROOT}"
fi

run_count="$(find "${EXTRACT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'run_*' | wc -l)"
test "${run_count}" -eq 152

usable_count=0
for run_dir in "${EXTRACT_ROOT}"/run_*; do
    if [[ "${run_dir##*/}" == "run_0147" ]]; then
        test -d "${run_dir}/depth"
        test -z "$(find "${run_dir}" -type f -print -quit)"
        continue
    fi
    test -f "${run_dir}/traj_xyz.npy"
    test -f "${run_dir}/traj_yaw.npy"
    test -f "${run_dir}/traj_pitch.npy"
    test -d "${run_dir}/depth"
    find "${run_dir}/depth" -maxdepth 1 -type f -name 'depth_*.png' -print -quit | grep -q .
    usable_count=$((usable_count + 1))
done
test "${usable_count}" -eq 151

printf 'SanD official dataset ready: %s (%s usable runs; upstream run_0147 is empty)\n' \
    "${EXTRACT_ROOT}" "${usable_count}"
