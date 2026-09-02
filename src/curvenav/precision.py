"""One numerical contract shared by CurveNav training and inference.

Tensor-Core neural operators use the capability-selected autocast dtype.  All
metric geometry, standardized Flow state, MeanFlow calculus, curve decoding,
and losses remain float32.  This is one mixed-precision path, not separate
model implementations.
"""

from dataclasses import dataclass

import torch


GEOMETRY_DTYPE = torch.float32


@dataclass(frozen=True)
class CudaPrecision:
    """The supported Tensor Core precision for one CUDA device."""

    accelerate_mode: str
    autocast_dtype: torch.dtype
    checkpoint_name: str


def precision_from_bf16_support(supports_bf16: bool) -> CudaPrecision:
    """Resolve the only two supported CurveNav CUDA precision modes."""
    if supports_bf16:
        return CudaPrecision(
            accelerate_mode="bf16",
            autocast_dtype=torch.bfloat16,
            checkpoint_name="bf16_neural_fp32_geometry_flow_jvp",
        )
    return CudaPrecision(
        accelerate_mode="fp16",
        autocast_dtype=torch.float16,
        checkpoint_name="fp16_neural_fp32_geometry_flow_jvp",
    )


def cuda_precision(device: torch.device | str) -> CudaPrecision:
    """Read the selected device capability without introducing a config branch."""
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CurveNav mixed precision requires one CUDA device")
    index = torch.cuda.current_device() if resolved.index is None else resolved.index
    with torch.cuda.device(index):
        supports_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    return precision_from_bf16_support(supports_bf16)
