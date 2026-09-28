import json
from pathlib import Path

import yaml

from behavior_adapter.train_existing_h800_10ep import training_config
from curvenav.config_io import load_config


def test_short_benchmark_and_full_training_configs_are_valid(tmp_path):
    cache = tmp_path / "cache"
    (cache / "train").mkdir(parents=True)
    # The short benchmark must remain valid even when the source recipe has
    # a multi-epoch warmup; the full run should retain that warmup.
    recipe = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    recipe["training"]["warmup_epochs"] = 5
    (cache / "config.yaml").write_text(yaml.safe_dump(recipe))
    (cache / "train/manifest.json").write_text(json.dumps({"samples": 1024}))

    short_path, _ = training_config(cache, tmp_path / "short", 128, 1, steps=41)
    full_path, _ = training_config(cache, tmp_path / "full", 128, 10)
    assert load_config(short_path).training.warmup_epochs == 0
    assert load_config(full_path).training.warmup_epochs == 5
    five_path, _ = training_config(cache, tmp_path / "five", 448, 5)
    assert load_config(five_path).training.epochs == 5
    assert load_config(five_path).training.warmup_epochs == 2


def test_interrupted_benchmark_reaps_training_process(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time
    from types import SimpleNamespace
    import pytest
    import behavior_adapter.train_existing_h800_10ep as launcher

    children = []
    original_popen = subprocess.Popen

    def record_process(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        children.append(process)
        return process

    def interrupt(_):
        raise KeyboardInterrupt

    (tmp_path / 'benchmark_batch_1').mkdir()
    monkeypatch.setattr(launcher, 'BENCHMARK_CANDIDATES', (1,))
    monkeypatch.setattr(launcher, 'training_config', lambda *a, **k: (tmp_path / 'config.yaml', 1))
    monkeypatch.setattr(launcher, 'distributed_command', lambda _: [sys.executable, '-c', 'import time; time.sleep(60)'])
    monkeypatch.setattr(launcher.subprocess, 'Popen', record_process)
    monkeypatch.setattr(launcher, 'time', SimpleNamespace(monotonic=time.monotonic, sleep=interrupt))
    with pytest.raises(KeyboardInterrupt):
        launcher.benchmark(tmp_path, tmp_path, {})
    assert len(children) == 1
    assert children[0].poll() is not None
