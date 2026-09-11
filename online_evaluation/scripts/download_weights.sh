#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/config/local.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/config/local.env"
    set +a
fi
WEIGHT_ROOT="${NAVBENCH_WEIGHT_ROOT:-${ROOT_DIR}/weights}"

usage() {
    cat <<'EOF'
Usage: scripts/download_weights.sh [all|iplanner|viplanner|navdp|sandplanner|x-navdp ...]

Weights are downloaded resumably and verified before installation.
EOF
}

download() {
    local url="$1" output="$2" expected_size="$3" expected_sha256="$4"
    mkdir -p "$(dirname "$output")"
    if [[ -f "$output" ]] && [[ "$(stat -c %s "$output")" == "$expected_size" ]] && \
       [[ "$(sha256sum "$output" | awk '{print $1}')" == "$expected_sha256" ]]; then
        echo "[ok] $output"
        return
    fi
    echo "[download] $output"
    curl -fL --retry 5 --retry-delay 2 -C - -o "${output}.part" "$url"
    [[ "$(stat -c %s "${output}.part")" == "$expected_size" ]] || {
        echo "Size mismatch: ${output}.part" >&2
        exit 1
    }
    [[ "$(sha256sum "${output}.part" | awk '{print $1}')" == "$expected_sha256" ]] || {
        echo "SHA-256 mismatch: ${output}.part" >&2
        exit 1
    }
    mv "${output}.part" "$output"
}

prepare_iplanner() {
    download "https://drive.usercontent.google.com/download?id=1UD11sSlOZlZhzij2gG_OmxbBN4WxVsO_&export=download&confirm=t" \
        "${WEIGHT_ROOT}/iplanner/plannernet.pt" 213343275 \
        685f16cde28d05249d50d24ed79ab4bdc94b3fbbcb99c8dbaed31039d11633b9
}

prepare_viplanner() {
    download "https://drive.usercontent.google.com/download?id=1PY7XBkyIGESjdh1cMSiJgwwaIT0WaxIc&export=download&confirm=t" \
        "${WEIGHT_ROOT}/viplanner/model.pt" 284201801 \
        2fd5219cfb160e5035d43319632b3d975637a0e770c4d455a26d3124a15ca87b
    download "https://drive.usercontent.google.com/download?id=1DZoaLbXA1qPtg-gUKRUWS2rOH2tvDOOl&export=download&confirm=t" \
        "${WEIGHT_ROOT}/viplanner/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth" \
        415550392 54df384aa7f293a7fb13aff779d687d8216545ebdcf8b38216b6927136406536
}

prepare_navdp() {
    download "https://huggingface.co/InternRobotics/X-NavDP/resolve/main/navdp_pretrain.ckpt" \
        "${WEIGHT_ROOT}/navdp/navdp_pretrain.ckpt" 543257151 \
        3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947
}

prepare_sandplanner() {
    download "https://github.com/WangJinCheng1998/sandplanner/releases/download/v1.0/NoMax.pth" \
        "${WEIGHT_ROOT}/sandplanner/NoMax.pth" 169194662 \
        a723791de6970999ae7b3843d263f955ed0e03471ef115e4223168f0963d035d
}

prepare_x_navdp() {
    download "https://huggingface.co/InternRobotics/X-NavDP/resolve/main/x-navdp_posttrain.ckpt" \
        "${WEIGHT_ROOT}/x-navdp/x-navdp_posttrain.ckpt" 873100357 \
        267089a81bbbe7a913debda6603f3f1b66a79520370ce953b2d888d793b89f24
}

[[ $# -gt 0 ]] || set -- all
for model in "$@"; do
    case "$model" in
        all) prepare_iplanner; prepare_viplanner; prepare_navdp; prepare_sandplanner; prepare_x_navdp ;;
        iplanner) prepare_iplanner ;;
        viplanner) prepare_viplanner ;;
        navdp) prepare_navdp ;;
        sand|sandplanner) prepare_sandplanner ;;
        x-navdp|x_navdp) prepare_x_navdp ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown model: $model" >&2; usage; exit 2 ;;
    esac
done
echo "[done] $WEIGHT_ROOT"
