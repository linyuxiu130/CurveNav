"""One CUDA mixed-precision contract shared by training and inference."""

from dataclasses import dataclass

import torch


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
            checkpoint_name="bf16_primal_fp32_detached_meanflow_jvp",
        )
    return CudaPrecision(
        accelerate_mode="fp16",
        autocast_dtype=torch.float16,
        checkpoint_name="fp16_primal_fp32_detached_meanflow_jvp",
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
