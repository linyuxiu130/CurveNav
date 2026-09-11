# Depth request contract

Every PointGoal request carries contiguous raw tensors: RGB is `[B,H,W,3]` `uint8`; depth is
`[B,H,W,1]` `float32` in meters. `NaN` is the invalid-depth sentinel. Shape and dtype accompany
the multipart bytes. Policy responses are binary NPZ tensors.

The regression loop covers `0`, `NaN`, `6`, `7`, and `8` meters through the policy-service reader,
then applies each model's explicit invalid-depth convention.
