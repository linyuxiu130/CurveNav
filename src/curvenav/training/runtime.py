"""Fixed CUDA runtime settings for CurveNav training."""

import torch
from torch import nn


def configure_cuda_training_backend() -> None:
    """Select the static-shape Tensor Core route without unbounded workspaces."""
    torch.set_float32_matmul_precision("highest")
    # All training shapes are fixed.  With the NCHW activation route, limiting
    # cuDNN's search to ten plans improves the complete ResNet step without the
    # transient workspace exhaustion seen on the discarded NHWC route.
    torch.backends.cudnn.benchmark_limit = 10
    torch.backends.cudnn.benchmark = True


def compile_static_training_functions(policy: nn.Module) -> None:
    """Fuse perception, scene preparation, and the single velocity field."""
    for module, name in (
        (policy.depth_encoder, "forward"),
        (policy.condition_encoder, "forward"),
        (policy.trajectory_decoder, "project_condition_memory"),
        (policy, "_predict_velocity"),
    ):
        setattr(module, name, torch.compile(getattr(module, name), fullgraph=True, dynamic=False))
