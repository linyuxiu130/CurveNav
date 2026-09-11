#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/config/evaluator-versions.env"
export PIP_CONSTRAINT="${ROOT_DIR}/config/unified-constraints.txt"

CONDA_BIN="${NAVBENCH_CONDA:-conda}"
RUNTIME_ROOT="${NAVBENCH_RUNTIME_ROOT:-${ROOT_DIR}/.runtime}"
ENV_PREFIX="${NAVBENCH_EVAL_ENV:-/opt/conda/envs/curvenav-unified}"
ISAACLAB_ROOT="${RUNTIME_ROOT}/IsaacLab-${ISAACLAB_REVISION:0:12}"
ACADOS_ROOT="${RUNTIME_ROOT}/acados-${ACADOS_REVISION:0:12}"
JOBS="${NAVBENCH_BUILD_JOBS:-$(nproc)}"

clone_at_revision() {
    local repository="$1" revision="$2" destination="$3" recursive="${4:-no}"
    if [[ ! -d "${destination}/.git" ]]; then
        local stage="${destination}.incoming.$$"
        [[ ! -e "$stage" ]] || { echo "staging path exists: $stage" >&2; exit 1; }
        git clone --quiet --filter=blob:none "$repository" "$stage"
        git -C "$stage" checkout --quiet --detach "$revision"
        if [[ "$recursive" == yes ]]; then
            git -C "$stage" submodule update --init --recursive
            git -C "$stage" clean -fdx
        fi
        mkdir -p "$(dirname "$destination")"
        mv "$stage" "$destination"
    fi
    [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]] || {
        echo "wrong source revision: $destination" >&2
        exit 1
    }
    [[ -z "$(git -C "$destination" status --short)" ]] || {
        echo "source checkout is dirty: $destination" >&2
        exit 1
    }
}

mkdir -p "$RUNTIME_ROOT"
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    "$CONDA_BIN" create -y -p "$ENV_PREFIX" "python=${PYTHON_VERSION}" pip make
fi
PYTHON="${ENV_PREFIX}/bin/python"

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install "setuptools==${SETUPTOOLS_VERSION}" "wheel==${WHEEL_VERSION}" "cmake==${CMAKE_VERSION}"
"$PYTHON" -m pip install "torch==${TORCH_VERSION}+${PYTORCH_CUDA_TAG}" "torchvision==${TORCHVISION_VERSION}+${PYTORCH_CUDA_TAG}" --index-url "$PYTORCH_INDEX_URL" --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
"$PYTHON" -m pip install "isaacsim[all,extscache]==${ISAACSIM_PIP_VERSION}" --index-url "$NVIDIA_PYPI_URL" --extra-index-url https://pypi.org/simple

clone_at_revision "$ISAACLAB_REPOSITORY" "$ISAACLAB_REVISION" "$ISAACLAB_ROOT"
# flatdict's build imports pkg_resources, retained by our pinned setuptools.
"$PYTHON" -m pip install --no-build-isolation "flatdict==4.0.1"
"$PYTHON" -m pip install \
    -e "${ISAACLAB_ROOT}/source/isaaclab" \
    -e "${ISAACLAB_ROOT}/source/isaaclab_assets" \
    -e "${ISAACLAB_ROOT}/source/isaaclab_tasks" \
    -e "${ISAACLAB_ROOT}/source/isaaclab_rl" \
    "rsl-rl-lib==${RSL_RL_VERSION}" \
    "tensordict==${TENSORDICT_VERSION}" \
    "warp-lang==${WARP_VERSION}"

if [[ -f "${ACADOS_ROOT}/interfaces/acados_template/setup.py" ]]; then
    if git -C "$ACADOS_ROOT" apply --reverse --check "$ROOT_DIR/config/acados-template-version.patch" 2>/dev/null; then
        git -C "$ACADOS_ROOT" apply --reverse "$ROOT_DIR/config/acados-template-version.patch"
    fi
fi
clone_at_revision "$ACADOS_REPOSITORY" "$ACADOS_REVISION" "$ACADOS_ROOT" yes
"${ENV_PREFIX}/bin/cmake" -S "$ACADOS_ROOT" -B "${ACADOS_ROOT}/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ACADOS_ROOT"
"${ENV_PREFIX}/bin/cmake" --build "${ACADOS_ROOT}/build" --target install --parallel "$JOBS"
renderer="${ACADOS_ROOT}/bin/t_renderer"
if ! echo "${ACADOS_TERA_RENDERER_SHA256}  ${renderer}" | sha256sum -c - >/dev/null 2>&1; then
    curl -fsSL "https://github.com/acados/tera_renderer/releases/download/v${ACADOS_TERA_RENDERER_VERSION}/t_renderer-v${ACADOS_TERA_RENDERER_VERSION}-linux-amd64" -o "$renderer"
fi
echo "${ACADOS_TERA_RENDERER_SHA256}  ${renderer}" | sha256sum -c -
chmod 0755 "$renderer"
git -C "$ACADOS_ROOT" apply "$ROOT_DIR/config/acados-template-version.patch"
"$PYTHON" -m pip install -e "${ACADOS_ROOT}/interfaces/acados_template"

XNAVDP_ROOT="$("${ROOT_DIR}/scripts/prepare_xnavdp_runtime.sh" | sed -n 's/^\[ready\] //p')"
"$PYTHON" -m pip install -r <(
    sed -E -e '/^acados-template==/d' -e 's/==[^[:space:]]+//' "${XNAVDP_ROOT}/requirements.txt"
)
"$PYTHON" -m pip install "numpy==${NUMPY_VERSION}" "scipy==${SCIPY_VERSION}" "open3d==${OPEN3D_VERSION}" "casadi==${CASADI_VERSION}" "pathfinding==${PATHFINDING_VERSION}" \
    "matplotlib==${MATPLOTLIB_VERSION}" "PyYAML==${PYYAML_VERSION}" "requests==${REQUESTS_VERSION}" "psutil==${PSUTIL_VERSION}" \
    "click==${ISAACSIM_CLICK_VERSION}" "typing_extensions==${ISAACSIM_TYPING_EXTENSIONS_VERSION}" "wheel==${WHEEL_VERSION}" "ipython==${IPYTHON_VERSION}" \
    "onnx==${ONNX_VERSION}" "huggingface-hub==${HUGGINGFACE_HUB_VERSION}"
"$PYTHON" -m pip install "torch==${TORCH_VERSION}+${PYTORCH_CUDA_TAG}" "torchvision==${TORCHVISION_VERSION}+${PYTORCH_CUDA_TAG}" --extra-index-url "$PYTORCH_INDEX_URL"

"$PYTHON" "${ROOT_DIR}/scripts/check_xnavdp_runtime.py" "$XNAVDP_ROOT" --python "$PYTHON"
"$PYTHON" -m pip check

cat <<EOF
[ready] evaluator environment
NAVBENCH_EVAL_PYTHON=${PYTHON}
NAVBENCH_XNAVDP_ROOT=${XNAVDP_ROOT}
ACADOS_SOURCE_DIR=${ACADOS_ROOT}
LD_LIBRARY_PATH=${ACADOS_ROOT}/lib
NAVBENCH_EVAL_KIT_ARGS=--/rtx/verifyDriverVersion/enabled=false

Set OMNI_KIT_ACCEPT_EULA=YES only after accepting NVIDIA's Omniverse EULA.
EOF
