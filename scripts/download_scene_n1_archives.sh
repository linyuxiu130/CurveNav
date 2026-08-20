#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: HF_TOKEN=... $0 <scene-asset-root>" >&2
    exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required after accepting the Scene-N1 access agreement." >&2
    echo "Access: https://huggingface.co/datasets/InternRobotics/Scene-N1" >&2
    exit 2
fi

scene_asset_root="$(realpath -m "$1")"
archive_root="${scene_asset_root}/archives"
repository="InternRobotics/Scene-N1"
revision="2195d46aaab0ff48673b275fdfdc0731075b5ff2"
mkdir -p "${archive_root}"

# Only the native InternScenes home/commercial dependencies are fetched.
# Cluttered-easy/hard benchmark archives are intentionally absent.
archives=(
    "n1_eval_scenes/Materials.tar.gz|863251449"
    "n1_eval_scenes/SkyTexture.tar.gz|102833350"
    "n1_eval_scenes/internscenes_commercial.tar.gz|23713050055"
    "n1_eval_scenes/internscenes_home/Materials.tar.gz|19205164812"
    "n1_eval_scenes/internscenes_home/layout.tar.gz|1707486835"
    "n1_eval_scenes/internscenes_home/object.tar.gz|29837778925"
    "n1_eval_scenes/internscenes_home/scenes_home.tar.gz|8232146"
)

for entry in "${archives[@]}"; do
    relative_path="${entry%%|*}"
    expected_bytes="${entry##*|}"
    target="${archive_root}/${relative_path#n1_eval_scenes/}"
    partial="${target}.part"
    mkdir -p "$(dirname "${target}")"

    if [[ -f "${target}" ]] && [[ "$(stat -c %s "${target}")" == "${expected_bytes}" ]]; then
        echo "cached ${relative_path}"
        continue
    fi

    url="https://huggingface.co/datasets/${repository}/resolve/${revision}/${relative_path}"
    curl \
        --fail \
        --location \
        --retry 5 \
        --retry-all-errors \
        --continue-at - \
        --header "Authorization: Bearer ${HF_TOKEN}" \
        --output "${partial}" \
        "${url}"

    actual_bytes="$(stat -c %s "${partial}")"
    if [[ "${actual_bytes}" != "${expected_bytes}" ]]; then
        echo "size mismatch for ${relative_path}: expected ${expected_bytes}, got ${actual_bytes}" >&2
        exit 1
    fi
    mv "${partial}" "${target}"
    echo "downloaded ${relative_path}"
done

echo "Scene-N1 archives ready under ${archive_root}"
