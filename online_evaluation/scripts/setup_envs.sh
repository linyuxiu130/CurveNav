#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/config/local.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/config/local.env"
    set +a
fi
if [[ -n "${NAVBENCH_SERVER_ENV:-}" ]]; then
    SERVER_ENV="$NAVBENCH_SERVER_ENV"
elif [[ -n "${NAVBENCH_SERVER_PYTHON:-}" ]]; then
    SERVER_ENV="${NAVBENCH_SERVER_PYTHON%/bin/python}"
else
    SERVER_ENV="${ROOT_DIR}/.venv"
fi
EVAL_PYTHON="${NAVBENCH_EVAL_PYTHON:-python}"
CONDA_BIN="${NAVBENCH_CONDA:-conda}"
MODE="${1:-server}"

setup_server() {
    if [[ -d "$SERVER_ENV/conda-meta" ]]; then
        echo "Existing Conda environment: $SERVER_ENV. Use scripts/setup_unified_env.sh; refusing to create a venv over it." >&2
        return 1
    fi
    "$EVAL_PYTHON" -m venv --system-site-packages "$SERVER_ENV"
    "$SERVER_ENV/bin/python" -m pip install --upgrade pip
    "$SERVER_ENV/bin/python" -m pip install \
        'diffusers==0.33.1' 'timm==1.0.19' 'cupy-cuda12x==13.6.0' \
        'flask==3.1.2' 'opencv-python==4.11.0.86' 'scipy==1.15.3' \
        'pyyaml==6.0.3' 'imageio==2.37.0' 'imageio-ffmpeg==0.6.0'
    echo "Server environment: $SERVER_ENV"
}

setup_viplanner() {
    "$CONDA_BIN" create -y -n viplanner --override-channels \
        -c https://repo.anaconda.com/pkgs/main python=3.10 pip
    "$CONDA_BIN" run -n viplanner python -m pip install \
        --index-url https://download.pytorch.org/whl/cu118 \
        'torch==2.0.1' 'torchvision==0.15.2'
    "$CONDA_BIN" run -n viplanner python -m pip install \
        'mmcv==2.0.0' \
        -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
    "$CONDA_BIN" run -n viplanner python -m pip install \
        'numpy<2' 'mmengine==0.10.7' 'mmdet==3.1.0' 'flask==3.1.2' \
        'imageio==2.37.0' 'imageio-ffmpeg==0.6.0' 'opencv-python==4.11.0.86' \
        'scipy==1.15.3' 'pyyaml==6.0.3' \
        'git+https://github.com/cocodataset/panopticapi.git'
}

case "$MODE" in
    server) setup_server ;;
    viplanner) setup_viplanner ;;
    all) setup_server; setup_viplanner ;;
    *) echo "Usage: scripts/setup_envs.sh [server|viplanner|all]" >&2; exit 2 ;;
esac
