"""Fixed CUDA runtime settings for CurveNav training."""

import torch
from torch import nn


def configure_cuda_training_backend() -> None:
    """Select the static-shape Tensor Core route without unbounded workspaces."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # All training shapes are fixed.  With the NCHW activation route, limiting
    # cuDNN's search to ten plans improves the complete ResNet step without the
    # transient workspace exhaustion seen on the discarded NHWC route.
    torch.backends.cudnn.benchmark_limit = 10
    torch.backends.cudnn.benchmark = True


def compile_static_geometry_preprocessing(policy: nn.Module) -> None:
    """Fuse fixed-shape visual and metric preprocessing outside the MeanFlow JVP."""
    backbone = policy.depth_encoder.backbone
    backbone.forward = torch.compile(
        backbone.forward,
        fullgraph=True,
        mode="default",
    )
    projector = policy.depth_encoder.metric_projector
    projector.forward = torch.compile(
        projector.forward,
        fullgraph=True,
        mode="reduce-overhead",
    )
    configuration_encoder = policy.condition_encoder.configuration_encoder
    configuration_encoder.forward = torch.compile(
        configuration_encoder.forward,
        fullgraph=True,
        mode="default",
    )
