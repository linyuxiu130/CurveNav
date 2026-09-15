#!/usr/bin/env bash
# Usage: bash scripts/train_cloud_8gpu.sh DATASET OUTPUT [BATCH_PER_GPU] [EPOCHS]
set -euo pipefail
if (( $# < 2 || $# > 4 )); then
    echo "Usage: $0 DATASET OUTPUT [BATCH_PER_GPU=256] [EPOCHS=50]" >&2
    exit 2
fi
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export CURVENAV_DEPTH_CACHE_DIR="${CURVENAV_DEPTH_CACHE_DIR:-/tmp/curvenav-depth-cache}"
export PYTHONPATH="$project_root/src"
"${CONDA_PREFIX:?activate the training environment}/bin/python" - "$project_root" "$@" <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from curvenav.config_io import config_from_mapping
from curvenav.training.checkpoint import build_training_contract
from curvenav.precision import PRECISION_NAME

project, dataset, output = map(lambda p: Path(p).resolve(), sys.argv[1:4])
batch = int(sys.argv[4]) if len(sys.argv) > 4 else 256
epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 50
if torch.cuda.device_count() != 8:
    raise RuntimeError("This launcher requires exactly eight visible GPUs")
devices = []
for index in range(8):
    with torch.cuda.device(index):
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError(f"GPU {index} does not support native BF16")
    prop = torch.cuda.get_device_properties(index)
    devices.append(dict(name=prop.name, memory_bytes=prop.total_memory))
raw = yaml.safe_load((dataset / 'config.yaml').read_text())
samples = json.loads((dataset / 'train/manifest.json').read_text())['samples']
raw['data']['root'] = str(dataset)
global_batch = 8 * batch
if batch <= 0:
    raise ValueError('batch must be positive')
raw['training'].update(
    per_device_batch_size=batch, gradient_accumulation_steps=1,
    samples_per_epoch=((samples + global_batch - 1) // global_batch) * global_batch,
    epochs=epochs, num_workers=2, prefetch_factor=2,
    checkpoint_every_epochs=1, output_dir=str(output),
)
config = config_from_mapping(raw)
contract = build_training_contract(config, 8, PRECISION_NAME)
output.mkdir(parents=True, exist_ok=False)
config_path = output / 'config.yaml'
config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
plan = dict(devices=devices, **contract)
(output / 'training_plan.json').write_text(json.dumps(plan, indent=2))
print(json.dumps(plan, indent=2), flush=True)
with (output / 'train.log').open('w') as log:
    subprocess.run(['bash', str(project / 'scripts/train_policy.sh'), str(config_path)],
                   stdout=log, stderr=subprocess.STDOUT, check=True)
PY
