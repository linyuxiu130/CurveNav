# Data directory

Datasets and simulator assets are deliberately not stored in Git. They are
large, have independent licenses, and must remain outside the source history.

Use `scripts/download_official_sand_dataset.sh` to fetch the pinned public SanD
training archive, then run `scripts/prepare_sand_depth_cache.py` with the
training configuration before starting CurveNav training. HSSD assets and
metadata are prepared with the dedicated scripts in `scripts/`.
