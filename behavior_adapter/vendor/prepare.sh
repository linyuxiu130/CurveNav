#!/usr/bin/env bash
# Reconstruct the two-file source fork without committing the simulator.
set -euo pipefail
vendor="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
behavior_root="${1:?Usage: prepare.sh /path/to/BEHAVIOR-1K-v3.9.2-baidu}"
[[ ! -e "$vendor/OmniGibson" ]] || { echo 'Local OmniGibson already exists; leaving it unchanged.' >&2; exit 1; }
staging="$(mktemp -d "$vendor/.prepare.XXXXXX")"
trap 'rm -rf "$staging"' EXIT
cp -a "$behavior_root/OmniGibson" "$staging/OmniGibson"
patch --batch --fuzz=0 -d "$staging/OmniGibson" -p1 < "$vendor/omnigibson-v3.9.2.patch"
mkdir -p "$vendor/docs/assets"
cp "$behavior_root/docs/assets/OmniGibson_logo.png" "$vendor/docs/assets/"
mv "$staging/OmniGibson" "$vendor/OmniGibson"
