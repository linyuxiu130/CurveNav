#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/config/evaluator-versions.env"
REVISION="$XNAVDP_REVISION"
REPOSITORY="$XNAVDP_REPOSITORY"
CHECKOUT="${1:-${ROOT_DIR}/.runtime/x-navdp-${REVISION:0:12}}"
RUNTIME_ROOT="${CHECKOUT}/baselines/x-navdp"
RUNTIME_PATCH="${ROOT_DIR}/config/xnavdp-rootless-runtime.patch"

if [[ ! -d "${CHECKOUT}/.git" ]]; then
    stage="${CHECKOUT}.incoming.$$"
    [[ ! -e "$stage" ]] || { echo "staging path exists: $stage" >&2; exit 1; }
    git clone --filter=blob:none --no-checkout "$REPOSITORY" "$stage"
    git -C "$stage" sparse-checkout set baselines/x-navdp
    git -C "$stage" checkout --detach "$REVISION"
    mkdir -p "$(dirname "$CHECKOUT")"
    mv "$stage" "$CHECKOUT"
fi

actual="$(git -C "$CHECKOUT" rev-parse HEAD)"
[[ "$actual" == "$REVISION" ]] || {
    echo "X-NavDP checkout is $actual, expected $REVISION" >&2
    exit 1
}

mapfile -t patch_files < <(
    sed -n 's|^diff --git a/[^ ]* b/||p' "$RUNTIME_PATCH"
)
bridge_files=(
    baselines/x-navdp/eval/src/client_utils.py
    baselines/x-navdp/eval/src/policy_agent.py
    baselines/x-navdp/eval/src/policy_server.py
    baselines/x-navdp/eval/src/policy_backbone.py
    baselines/x-navdp/eval/src/policy_network_embodiment.py
)
git -C "$CHECKOUT" checkout -- "${patch_files[@]}" "${bridge_files[@]}"
git -C "$CHECKOUT" apply --unidiff-zero "$RUNTIME_PATCH"

# The simulator and every policy server share the sole raw float32/NPZ bridge.
for runtime_file in client_utils.py policy_agent.py policy_server.py policy_backbone.py policy_network_embodiment.py; do
    install -m 0644         "${ROOT_DIR}/baselines/x-navdp/eval/src/${runtime_file}"         "${RUNTIME_ROOT}/eval/src/${runtime_file}"
done

mapfile -t actual_files < <(
    git -C "$CHECKOUT" diff --name-only | LC_ALL=C sort
)
mapfile -t expected_files < <(
    printf '%s\n' "${patch_files[@]}" "${bridge_files[@]}" | LC_ALL=C sort -u
)
[[ "$(printf '%s\n' "${actual_files[@]}")" ==    "$(printf '%s\n' "${expected_files[@]}")" ]] || {
    echo "unexpected upstream runtime modifications:" >&2
    git -C "$CHECKOUT" status --short --untracked-files=no >&2
    exit 1
}

echo "[ready] ${RUNTIME_ROOT}"
echo "Set NAVBENCH_XNAVDP_ROOT=${RUNTIME_ROOT}"
