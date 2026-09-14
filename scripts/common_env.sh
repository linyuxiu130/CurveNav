# Shared environment for the repository's training and evaluation entrypoints.
# Source this file from a script in scripts/; it intentionally contains no
# alternate runtime or error-recovery path.

CURVENAV_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURVENAV_PYTHON="${CONDA_PREFIX:?run conda activate curvenav-unified first}/bin/python"

export PYTHONPATH="${CURVENAV_PROJECT_ROOT}/src"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/shibo_huang/data/curvenav/cache}"
export TORCHINDUCTOR_CACHE_DIR="${CONDA_PREFIX}/torchinductor"
export TORCHINDUCTOR_COMPILE_THREADS=8

cd "${CURVENAV_PROJECT_ROOT}"
