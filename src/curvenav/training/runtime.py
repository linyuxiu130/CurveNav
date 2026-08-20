"""Fixed CUDA runtime settings for CurveNav training."""

import torch


def configure_cuda_training_backend() -> None:
    """Select the single profiled cuDNN training route."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 20
