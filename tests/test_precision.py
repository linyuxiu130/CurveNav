import torch

from curvenav.precision import GEOMETRY_DTYPE, precision_from_bf16_support


def test_cuda_precision_selects_the_single_supported_capability_mode() -> None:
    assert precision_from_bf16_support(True).accelerate_mode == "bf16"
    assert precision_from_bf16_support(True).autocast_dtype is torch.bfloat16
    assert precision_from_bf16_support(False).accelerate_mode == "fp16"
    assert precision_from_bf16_support(False).autocast_dtype is torch.float16
    assert GEOMETRY_DTYPE is torch.float32
    assert (
        precision_from_bf16_support(True).checkpoint_name
        == "bf16_neural_fp32_geometry_flow_jvp"
    )
