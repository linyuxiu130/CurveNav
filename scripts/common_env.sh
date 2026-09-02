# Shared environment for the repository's training and evaluation entrypoints.
# Source this file from a script in scripts/; it intentionally contains no
# alternate runtime or error-recovery path.

CURVENAV_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURVENAV_WORKSPACE_ROOT="$(cd "${CURVENAV_PROJECT_ROOT}/.." && pwd)"
CURVENAV_ENV_ROOT="${CURVENAV_WORKSPACE_ROOT}/.venvs"
CURVENAV_PYTHON="${CURVENAV_ENV_ROOT}/curvenav/bin/python"

export PYTHONPATH="${CURVENAV_PROJECT_ROOT}/src"
CURVENAV_PYTHON_ABI="$("${CURVENAV_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
CURVENAV_PYTHON_INCLUDE_ROOT="${CURVENAV_ENV_ROOT}/curvenav/python-dev/usr/include"
export CPATH="${CURVENAV_PYTHON_INCLUDE_ROOT}/python${CURVENAV_PYTHON_ABI}:${CURVENAV_PYTHON_INCLUDE_ROOT}${CPATH:+:${CPATH}}"
export TORCHINDUCTOR_CACHE_DIR="${CURVENAV_ENV_ROOT}/curvenav/torchinductor"
export TORCHINDUCTOR_COMPILE_THREADS=2

cd "${CURVENAV_PROJECT_ROOT}"
