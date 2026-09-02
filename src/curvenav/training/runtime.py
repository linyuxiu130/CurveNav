"""Fixed CUDA runtime settings for CurveNav training."""

import torch
from torch import nn


def configure_cuda_training_backend() -> None:
    """Select the static-shape Tensor Core route without unbounded workspaces."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # CurveNav compiles only internal static functions.  DDPOptimizer is for
    # compiling an enclosing DDP model and cannot split a torch.func JVP graph.
    torch._dynamo.config.optimize_ddp = False
    # All training shapes are fixed.  With the NCHW activation route, limiting
    # cuDNN's search to ten plans improves the complete ResNet step without the
    # transient workspace exhaustion seen on the discarded NHWC route.
    torch.backends.cudnn.benchmark_limit = 10
    torch.backends.cudnn.benchmark = True


def compile_static_training_functions(policy: nn.Module) -> None:
    """Fuse fixed-shape perception, conditioning, primal, and stopped JVP."""
    policy.depth_encoder.forward = torch.compile(
        policy.depth_encoder.forward,
        fullgraph=True,
        dynamic=False,
    )
    policy.condition_encoder.forward = torch.compile(
        policy.condition_encoder.forward,
        fullgraph=True,
        dynamic=False,
    )
    policy._trainable_velocity_primal = torch.compile(
        policy._trainable_velocity_primal,
        fullgraph=True,
        dynamic=False,
    )
    policy._mean_flow_total_time_derivative = torch.compile(
        policy._mean_flow_total_time_derivative,
        fullgraph=True,
        dynamic=False,
    )
