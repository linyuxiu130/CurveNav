#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/data/huangshibo/H/navigation_three_projects/curvenav
python=/mnt/data/huangshibo/H/navigation_three_projects/.venvs/curvenav/bin/python

cd "$repo"
exec env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  "$python" -m curvenav.data_generation.hssd_policy_sidecar \
  outputs/hssd_curvenav_v2_pilot_200 \
  outputs/train_v2a_sand_official/checkpoint.pt \
  configs/train_sand_official.yaml \
  outputs/audits/hssd_v2_multitopology_design_20260819/critic_sidecar_schema_v1.json \
  outputs/hssd_curvenav_v2_pilot_200_policy_hardneg_v1 \
  --device cuda:1 \
  --batch-size 32 \
  --num-workers 8 \
  --seed 42
