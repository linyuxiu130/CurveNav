# Maintained OmniGibson source fork

`OmniGibson/` is an ignored local source copy of the BEHAVIOR v3.9.2 package at
`/mnt/dataset/Benchmark/Behavior2026/BEHAVIOR-1K-v3.9.2-baidu/OmniGibson`.
The upstream mount is read-only. `run_scene.sh` imports this writable source fork
before the upstream package. The upstream license is retained.

Git tracks only the patch and reconstruction script. After cloning, run:

```bash
bash behavior_adapter/vendor/prepare.sh /path/to/BEHAVIOR-1K-v3.9.2-baidu
```

This requires the matching upstream source and its `docs/assets` directory.
The script refuses to replace an existing local fork. Set `CURVENAV_BEHAVIOR_ROOT`
and `CURVENAV_OMNIGIBSON_DATA_PATH` when launching on a different machine.

Source changes are limited to:

- `omnigibson/scenes/scene_base.py`: identify an embedded robot by its serialized
  class name `Robot` when `_include_robots` is false. The previous comparison
  against registered model names let the scene robot load beside the configured
  R1Pro.
- `omnigibson/utils/usd_utils.py`: ignore prim paths absent from the contact
  matrix column index, matching the existing row lookup. Some scene floor or
  carpet prims are not registered as contact columns.

`docs/assets/OmniGibson_logo.png` is the launch icon required by the package's
existing relative path lookup. No simulator physics or controller code changed.
