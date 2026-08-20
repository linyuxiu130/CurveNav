#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/huangshibo/H/navigation_three_projects/curvenav

PYTHONPATH=src \
  /mnt/data/huangshibo/H/navigation_three_projects/.venvs/habitat-hssd/bin/python \
  -m curvenav.data_generation.hssd_v2 \
  configs/data_hssd_v2_pilot_200.json \
  --workers 2 \
  --gpu-devices 1,4 \
  2>&1 | tee outputs/hssd_curvenav_v2_pilot_200.tmux.log
