"""Executed-motion context encoder for temporally consistent replanning."""

from torch import Tensor, nn

from curvenav.layers import RMSNorm


class MotionContextEncoder(nn.Module):
    """Encode ``[direction_x, direction_y, valid]`` into one token."""

    def __init__(self, model_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, model_dim),
            RMSNorm(model_dim),
        )

    def forward(self, motion_context: Tensor) -> Tensor:
        if motion_context.ndim != 2 or motion_context.shape[-1] != 3:
            raise ValueError("motion_context must have shape [B, 3]")
        return self.encoder(motion_context).unsqueeze(1)
