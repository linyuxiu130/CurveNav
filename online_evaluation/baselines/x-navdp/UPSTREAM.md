Vendored from `baselines/x-navdp` in
<https://github.com/InternRobotics/NavDP> at commit
`878740a2011856d0e3782dd6ccd880fd2eccd70f` (2026-08-18 checkout).

Local integration changes add the legacy benchmark's raw-array/NPZ protocol,
forward robot pose state for RTC guidance, and skip visualization work when the
server is started with `--no-visualization`. The released checkpoint is loaded
with the upstream server's intended `strict=False`; every parameter used by the
wheeled inference network is present, while unused pre-training heads remain as
extra checkpoint keys.
