# Official X-NavDP assets and provenance

There is no episode rebuild step. The benchmark consumes the 40 official X-NavDP episode files
byte-for-byte. Their individual SHA-256 values and aggregate digest are frozen in
`suites/pointgoal-v2.json`; changing even one value invalidates the suite.

Large Scene-N1 assets are not stored in Git. To reproduce the data layout:

```bash
huggingface-cli login
scripts/prepare_scenes.sh download
scripts/prepare_scenes.sh check
```

The script downloads the Scene-N1 Home/Commercial archives at revision
`2195d46aaab0ff48673b275fdfdc0731075b5ff2`, then installs only the 40 official navigation
meshes from X-NavDP revision `7cee38a8d8308874d2b8488783c612f42060ac41`. It records repository,
revision, scene-split digest, file sizes, and file digests in `navigation_metadata/PROVENANCE.json`.
Interrupted archive downloads resume from `.part` files.

Expected external layout:

```text
<scene-root>/
  internscenes_home/scenes_home/<scene>/...
  internscenes_commercial/scenes_commercial/<scene>/...
  navigation_metadata/internscenes_home/esdf/<scene>/navigable.ply
  navigation_metadata/internscenes_commercial/esdf/<scene>/navigable.ply
```

At run time the launcher creates only symlinks that present these immutable files in the released
evaluator's `data/scenes` layout. It does not copy, rewrite, or regenerate navigation inputs.

The official episode arrays live in this repository at:

```text
assets/scenes/internscenes_home/<scene>/pointgoal_start_goal_pairs.npy
assets/scenes/internscenes_commercial/<scene>/pointgoal_start_goal_pairs.npy
```

Do not project USD geometry, sample navigation points, create Easy/Hard splits, change initial
yaw, or compute replacement paths. Such outputs define a different benchmark.
