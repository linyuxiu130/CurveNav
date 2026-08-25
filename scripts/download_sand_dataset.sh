#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

commit=533fb49c1397a08ded3c7a201d1abc7d965cbd3f
archive_sha256=abd2ec134b0b707d5938e1584059338a3d0c42b2fb2b60583d1aa9455e500c4e
output_root=$(realpath -m "$1")
dataset_root="$output_root/dataset_avoid"
work_root=$(mktemp -d)
trap 'rm -rf "$work_root"' EXIT

test ! -e "$dataset_root"
mkdir -p "$output_root" "$work_root/extracted"
curl --fail --location \
  --output "$work_root/dataset_avoid.zip" \
  "https://huggingface.co/datasets/WJCUCL/sandplanner-dataset-avoid/resolve/$commit/dataset_avoid.zip"
printf '%s  %s\n' "$archive_sha256" "$work_root/dataset_avoid.zip" \
  | sha256sum --check --strict
unzip -q "$work_root/dataset_avoid.zip" -d "$work_root/extracted"
mv "$work_root/extracted/dataset_avoid" "$dataset_root"
