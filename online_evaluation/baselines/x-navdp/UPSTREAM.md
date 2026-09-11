Vendored from `baselines/x-navdp` in
<https://github.com/InternRobotics/NavDP> at commit
`878740a2011856d0e3782dd6ccd880fd2eccd70f` (2026-08-18 checkout).

Local integration changes add the benchmark's raw-array/NPZ protocol,
forward robot pose state for RTC guidance, and skip visualization work when the
server is started with `--no-visualization`. Released pre-training-only heads
(`critic_head`, `image_encoder`, `log_alpha`, `pixel_encoder`) are explicitly
excluded; all remaining parameters load strictly, including missing-key checks.
Depth Anything / DINOv2 is shared with NavDP in `navbench/vision/`; inference
uses the native tensor attention path, without optional xFormers dispatch.
