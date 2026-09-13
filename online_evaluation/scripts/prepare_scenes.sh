#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/config/local.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/config/local.env"
    set +a
fi
ASSET_ROOT="${NAVBENCH_SCENE_ROOT:-${ROOT_DIR}/assets/scenes}"
ARCHIVE_ROOT="${NAVBENCH_ARCHIVE_ROOT:-${ASSET_ROOT}/.archives}"
SCENE_REVISION="2195d46aaab0ff48673b275fdfdc0731075b5ff2"
BASE_URL="https://huggingface.co/datasets/InternRobotics/Scene-N1/resolve/${SCENE_REVISION}"
HF_TOKEN="${HF_TOKEN:-}"

usage() {
    cat <<'EOF'
Usage: scripts/prepare_scenes.sh [download|check]

`download` installs the 40 official X-NavDP held-out Home/Commercial scenes and
their canonical X-NavDP navigation metadata. The 4,000 official episodes are
versioned in this repository and are verified byte-for-byte; they are never
regenerated or stratified locally. Downloads resume after interruption.

Scene-N1 is gated. Accept its license in a browser, then either run
`huggingface-cli login` or export HF_TOKEN before `download`.
EOF
}

if [[ -z "$HF_TOKEN" && -f "${HF_HOME:-${HOME}/.cache/huggingface}/token" ]]; then
    HF_TOKEN="$(<"${HF_HOME:-${HOME}/.cache/huggingface}/token")"
fi

download_archive() {
    local remote="$1" expected_size="$2" destination="$3"
    local filename archive
    filename="$(basename "$remote")"
    archive="${ARCHIVE_ROOT}/${remote}"
    mkdir -p "$(dirname "$archive")" "$destination"
    if [[ -f "$archive" && "$(stat -c %s "$archive")" == "$expected_size" ]]; then
        echo "[cached] $filename"
    else
        [[ -n "$HF_TOKEN" ]] || {
            echo "Scene-N1 authentication is required. Run: huggingface-cli login" >&2
            exit 1
        }
        echo "[download] $remote"
        HF_TOKEN="$HF_TOKEN" "${NAVBENCH_EVAL_PYTHON:-python}" \
            "$ROOT_DIR/scripts/download_hf_archive.py" \
            "${BASE_URL}/${remote}?download=true" "$archive" "$expected_size"
    fi
    echo "[extract] $filename -> $destination"
    tar -xzf "$archive" -C "$destination"
}

download_all() {
    mkdir -p "$ASSET_ROOT" "$ARCHIVE_ROOT"
    exec 9>"$ASSET_ROOT/.installation.lock"
    flock -n -x 9 || { echo "Scene assets are in use or being installed" >&2; exit 1; }
    touch "$ASSET_ROOT/.installing"

    if [[ ! -d "$ASSET_ROOT/Materials/Carpet" ]]; then
        download_archive "n1_eval_scenes/Materials.tar.gz" 863251449 "$ASSET_ROOT"
    fi
    if [[ ! -d "$ASSET_ROOT/SkyTexture" ]]; then
        download_archive "n1_eval_scenes/SkyTexture.tar.gz" 102833350 "$ASSET_ROOT"
    fi
    local home_root="$ASSET_ROOT/internscenes_home"
    mkdir -p "$home_root"
    if [[ ! -f "$home_root/.materials.complete" ]]; then
        download_archive "n1_eval_scenes/internscenes_home/Materials.tar.gz" 19205164812 "$home_root"
        touch "$home_root/.materials.complete"
    fi
    if [[ ! -f "$home_root/.layout.complete" ]]; then
        download_archive "n1_eval_scenes/internscenes_home/layout.tar.gz" 1707486835 "$home_root"
        touch "$home_root/.layout.complete"
    fi
    if [[ ! -f "$home_root/.object.complete" ]]; then
        download_archive "n1_eval_scenes/internscenes_home/object.tar.gz" 29837778925 "$home_root"
        touch "$home_root/.object.complete"
    fi
    if [[ ! -f "$home_root/.scenes.complete" ]]; then
        download_archive "n1_eval_scenes/internscenes_home/scenes_home.tar.gz" 8232146 "$home_root"
        touch "$home_root/.scenes.complete"
    fi

    if [[ ! -f "$ASSET_ROOT/internscenes_commercial/.complete" ]]; then
        download_archive "n1_eval_scenes/internscenes_commercial.tar.gz" 23713050055 "$ASSET_ROOT"
        touch "$ASSET_ROOT/internscenes_commercial/.complete"
    fi
    python "$ROOT_DIR/scripts/download_xnavdp_eval_metadata.py" \
        --staging-root "$ASSET_ROOT/.incoming" --install-root "$ASSET_ROOT"
    unlink "$ASSET_ROOT/.installing"
}

check_frozen() {
    "${NAVBENCH_EVAL_PYTHON:-python}" -c \
        'import json,sys; from pathlib import Path; status=json.loads(Path(sys.argv[1]).read_text())["status"]; assert status == "frozen", f"no frozen publication suite (status={status})"' \
        "$ROOT_DIR/suites/pointgoal-v2.json"
    "${NAVBENCH_EVAL_PYTHON:-python}" -m navbench --model navdp --check-assets
}

case "${1:-check}" in
    download) download_all ;;
    check) check_frozen ;;
    -h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
