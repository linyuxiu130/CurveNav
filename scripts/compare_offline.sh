#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"
exec "${CURVENAV_PYTHON}" -m curvenav.evaluation.compare "$@"
