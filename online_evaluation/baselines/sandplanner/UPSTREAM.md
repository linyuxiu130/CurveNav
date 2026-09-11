Vendored from <https://github.com/WangJinCheng1998/sandplanner> at commit
`415589ef81970d0ec055c6eaeef5edcdf1cf6c46` (2026-08-18 checkout).

Local integration changes cover inference only: the shared raw-array/NPZ server
protocol, selectable CUDA device, quiet/no-video defaults, exact-batch warm-up,
CuPy ESDF support, Conv-BN/QKV fusion, and `torch.compile` on the hot modules.
