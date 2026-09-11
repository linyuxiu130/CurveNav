Vendored from <https://github.com/WangJinCheng1998/sandplanner> at commit
`415589ef81970d0ec055c6eaeef5edcdf1cf6c46` (2026-08-18 checkout).

Local integration changes cover inference only: the shared raw-array/NPZ server
protocol, selectable CUDA device, quiet/no-video defaults, exact-batch warm-up,
CuPy ESDF support, Conv-BN/QKV fusion, and `torch.compile` on the hot modules.

The evaluation-only package keeps array depth input, the released model and
normalization (`trajectory_stats.json`), batch spline sampling, candidate scoring
and selected-candidate warm start. Standalone file inference, plotting, training,
duplicate samplers and silent dependency/checkpoint fallbacks have been removed.
Precision is owned by the benchmark inference context. The NoMax release omits
control-point count metadata; its eight points are declared in InferenceConfig.
