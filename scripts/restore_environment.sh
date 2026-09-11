#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$PROJECT/backups/recovery"
PREFIX=/opt/conda/envs/curvenav-unified
BENCH="$PROJECT/online_evaluation"
MODE="${1:-restore}"
case "$MODE" in restore|--check) ;; *) echo "Usage: bash scripts/restore_environment.sh [--check]" >&2; exit 2 ;; esac
cd "$ROOT/current"
sha256sum -c SHA256SUMS
if [[ "$MODE" == --check ]]; then exit 0; fi
# Editable packages require their persistent source directories.
for source in "$PROJECT/src" "$BENCH/navbench" \
    "$BENCH/.runtime/IsaacLab-661117d40bd2/source/isaaclab" \
    "$BENCH/.runtime/acados-48e223e85f04/lib" \
    "$BENCH/.runtime/x-navdp-878740a20118/baselines/x-navdp"; do
    [[ -d "$source" ]] || { echo "Missing persistent source: $source. See README.md." >&2; exit 1; }
done
if [[ ! -e "$PREFIX" && ! -L "$PREFIX" ]]; then
    mkdir -p /opt/conda/envs
    STAGE=$(mktemp -d /opt/conda/envs/.curvenav-restore.XXXXXX)
    trap 'echo "Interrupted staging directory retained: $STAGE" >&2' ERR
    tar --zstd -xf environment.tar.zst -C "$STAGE"
    mv "$STAGE/opt/conda/envs/curvenav-unified" "$PREFIX"
    rmdir "$STAGE/opt/conda/envs" "$STAGE/opt/conda" "$STAGE/opt" "$STAGE"
    trap - ERR
fi
# Never overwrite an existing environment. A damaged prefix fails visibly.
"$PREFIX/bin/python" -m pip check
CUDA_VISIBLE_DEVICES='' "$PREFIX/bin/python" -c 'import curvenav, navbench, torch; print("Training/evaluation imports OK:", torch.__version__)'
CUDA_VISIBLE_DEVICES='' "$PREFIX/bin/python" "$BENCH/scripts/check_xnavdp_runtime.py" \
    "$BENCH/.runtime/x-navdp-878740a20118/baselines/x-navdp" --python "$PREFIX/bin/python"
echo 'Ready: conda activate curvenav-unified'
