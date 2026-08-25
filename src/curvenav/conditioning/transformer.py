"""SanD visual tokens and NavDP learned-query multi-frame compression."""

import torch
from torch import Tensor, nn

from curvenav.layers import EncoderBlock, RMSNorm, SwiGLU
from curvenav.types import ConditionFeatures, DepthFeatures


COMPRESSED_TOKENS_PER_FRAME = 16
CONDITION_ENCODER_TYPE = "sand_geometry_aligned_navdp_query_memory"


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention followed by one SwiGLU residual update."""

    def __init__(self, model_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = RMSNorm(model_dim)
        self.memory_norm = RMSNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            model_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward_norm = RMSNorm(model_dim)
        self.feed_forward = SwiGLU(model_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor | None = None,
    ) -> Tensor:
        normalized_memory = self.memory_norm(memory)
        attended = self.attention(
            self.query_norm(query),
            normalized_memory,
            normalized_memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )[0]
        query = query + self.dropout(attended)
        return query + self.dropout(
            self.feed_forward(self.feed_forward_norm(query))
        )


class PolicyConditionEncoder(nn.Module):
    """Compress geometry-aligned depth tokens and fuse the PointGoal token."""

    def __init__(
        self,
        point_goal_encoder: nn.Module,
        *,
        observation_frames: int,
        spatial_tokens: int,
        model_dim: int = 256,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.point_goal_encoder = point_goal_encoder
        self.observation_frames = observation_frames
        self.spatial_tokens = spatial_tokens
        self.compressed_tokens = observation_frames * COMPRESSED_TOKENS_PER_FRAME

        self.frame_slot_embedding = nn.Parameter(
            torch.zeros(1, observation_frames, 1, model_dim)
        )
        self.compression_queries = nn.Parameter(
            torch.zeros(1, self.compressed_tokens, model_dim)
        )
        self.visual_compressor = CrossAttentionBlock(
            model_dim,
            transformer_heads,
            dropout,
        )
        self.condition_blocks = nn.ModuleList(
            EncoderBlock(model_dim, transformer_heads, dropout)
            for _ in range(transformer_layers)
        )
        self.output_norm = RMSNorm(model_dim)
        nn.init.trunc_normal_(self.frame_slot_embedding, std=0.02)
        nn.init.trunc_normal_(self.compression_queries, std=0.02)

    def forward(
        self,
        observation: DepthFeatures,
        point_goal: Tensor,
        observation_valid: Tensor,
    ) -> ConditionFeatures:
        batch = point_goal.shape[0]
        expected = (batch, self.observation_frames, self.spatial_tokens)
        if observation.tokens.ndim != 4 or observation.tokens.shape[:3] != expected:
            raise ValueError(
                "observation tokens must have shape "
                f"[B, {self.observation_frames}, {self.spatial_tokens}, D]"
            )
        if observation_valid.shape != (batch, self.observation_frames):
            raise ValueError("observation_valid must have shape [B, F]")
        if observation_valid.dtype != torch.bool:
            raise TypeError("observation_valid must be boolean")
        if not observation_valid[:, -1].all():
            raise ValueError("the current observation must always be valid")

        visual = observation.tokens + self.frame_slot_embedding
        visual_memory = visual.flatten(1, 2)
        visual_padding_mask = (
            (~observation_valid)
            .unsqueeze(-1)
            .expand(-1, -1, self.spatial_tokens)
            .flatten(1)
        )
        queries = self.compression_queries.expand(batch, -1, -1)
        compressed_visual = self.visual_compressor(
            queries,
            visual_memory,
            visual_padding_mask,
        )

        goal_token = self.point_goal_encoder(point_goal).unsqueeze(1)
        tokens = torch.cat((goal_token, compressed_visual), dim=1)
        for block in self.condition_blocks:
            tokens = block(tokens)
        return ConditionFeatures(tokens=self.output_norm(tokens))
