"""Metric geometry, route, and state queries for the CurveNav generator."""

import torch
from torch import Tensor, nn

from curvenav.layers import EncoderBlock, RMSNorm, SwiGLU
from curvenav.types import ConditionFeatures, DepthFeatures


CURRENT_GEOMETRY_QUERY_COUNT = 32
CONTEXT_GEOMETRY_QUERY_COUNT = 32
GEOMETRY_QUERY_COUNT = CURRENT_GEOMETRY_QUERY_COUNT + CONTEXT_GEOMETRY_QUERY_COUNT
ROUTE_QUERY_COUNT = 4
ROUTE_QUERY_LAYERS = 2
CONDITION_ENCODER_TYPE = (
    "current_context_metric_geometry_ordered_route_queries_masked_state"
)


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
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        normalized_memory = self.memory_norm(memory)
        attended = self.attention(
            self.query_norm(query),
            normalized_memory,
            normalized_memory,
            key_padding_mask=memory_padding_mask,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]
        query = query + self.dropout(attended)
        return query + self.dropout(self.feed_forward(self.feed_forward_norm(query)))


class PolicyConditionEncoder(nn.Module):
    """Build separate metric-geometry, ego-state, and local-route tokens."""

    def __init__(
        self,
        point_goal_encoder: nn.Module,
        *,
        observation_frames: int,
        spatial_tokens: int,
        history_scale_m: float,
        model_dim: int = 384,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if history_scale_m <= 0:
            raise ValueError("history_scale_m must be positive")
        self.point_goal_encoder = point_goal_encoder
        self.observation_frames = observation_frames
        self.spatial_tokens = spatial_tokens
        self.history_scale_m = float(history_scale_m)

        self.frame_slot_embedding = nn.Parameter(
            torch.zeros(1, observation_frames, 1, model_dim)
        )
        self.current_geometry_query_embedding = nn.Parameter(
            torch.zeros(1, CURRENT_GEOMETRY_QUERY_COUNT, model_dim)
        )
        self.context_geometry_query_embedding = nn.Parameter(
            torch.zeros(1, CONTEXT_GEOMETRY_QUERY_COUNT, model_dim)
        )
        geometry_attention_mask = torch.zeros(
            GEOMETRY_QUERY_COUNT,
            observation_frames * spatial_tokens,
            dtype=torch.bool,
        )
        geometry_attention_mask[
            :CURRENT_GEOMETRY_QUERY_COUNT,
            : (observation_frames - 1) * spatial_tokens,
        ] = True
        self.register_buffer(
            "geometry_attention_mask",
            geometry_attention_mask,
            persistent=True,
        )
        self.geometry_compressor = CrossAttentionBlock(
            model_dim,
            transformer_heads,
            dropout,
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(4, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.route_query_embedding = nn.Parameter(
            torch.zeros(1, ROUTE_QUERY_COUNT, model_dim)
        )
        self.route_blocks = nn.ModuleList(
            CrossAttentionBlock(model_dim, transformer_heads, dropout)
            for _ in range(ROUTE_QUERY_LAYERS)
        )
        self.condition_blocks = nn.ModuleList(
            EncoderBlock(model_dim, transformer_heads, dropout)
            for _ in range(transformer_layers)
        )
        self.output_norm = RMSNorm(model_dim)
        self.route_output_norm = RMSNorm(model_dim)
        self.route_summary_norm = RMSNorm(model_dim)
        nn.init.trunc_normal_(self.frame_slot_embedding, std=0.02)
        nn.init.trunc_normal_(self.current_geometry_query_embedding, std=0.02)
        nn.init.trunc_normal_(self.context_geometry_query_embedding, std=0.02)
        nn.init.trunc_normal_(self.route_query_embedding, std=0.02)

    def forward(
        self,
        observation: DepthFeatures,
        point_goal: Tensor,
        observation_to_current: Tensor,
        observation_valid: Tensor,
    ) -> ConditionFeatures:
        batch = point_goal.shape[0]
        expected = (batch, self.observation_frames, self.spatial_tokens)
        if observation.tokens.ndim != 4 or observation.tokens.shape[:3] != expected:
            raise ValueError(
                "observation tokens must have shape "
                f"[B, {self.observation_frames}, {self.spatial_tokens}, D]"
            )
        if observation_to_current.shape != (batch, self.observation_frames, 4):
            raise ValueError("observation_to_current must have shape [B, F, 4]")
        if observation_valid.shape != (batch, self.observation_frames):
            raise ValueError("observation_valid must have shape [B, F]")
        if observation_valid.dtype != torch.bool:
            raise TypeError("observation_valid must be boolean")

        visual = observation.tokens + self.frame_slot_embedding
        visual_memory = visual.flatten(1, 2)
        visual_padding_mask = (
            (~observation_valid)
            .unsqueeze(-1)
            .expand(-1, -1, self.spatial_tokens)
            .flatten(1)
        )
        geometry_queries = torch.cat(
            (
                self.current_geometry_query_embedding,
                self.context_geometry_query_embedding,
            ),
            dim=1,
        ).expand(batch, -1, -1)
        geometry = self.geometry_compressor(
            geometry_queries,
            visual_memory,
            visual_padding_mask,
            self.geometry_attention_mask,
        )

        state_input = torch.where(
            observation_valid[..., None],
            observation_to_current,
            torch.zeros_like(observation_to_current),
        )
        state_input[..., :2] = state_input[..., :2] / self.history_scale_m
        state = self.state_encoder(state_input)
        state = state.masked_fill(~observation_valid[..., None], 0.0)

        goal = self.point_goal_encoder(point_goal).unsqueeze(1)
        route = self.route_query_embedding.expand(batch, -1, -1) + goal
        route_memory = torch.cat((state, geometry), dim=1)
        route_memory_padding_mask = torch.cat(
            (
                ~observation_valid,
                torch.zeros(
                    batch,
                    GEOMETRY_QUERY_COUNT,
                    device=observation_valid.device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )
        for block in self.route_blocks:
            route = block(
                route,
                route_memory,
                memory_padding_mask=route_memory_padding_mask,
            )

        tokens = torch.cat((route, goal, state, geometry), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros(
                    batch,
                    ROUTE_QUERY_COUNT + 1,
                    device=observation_valid.device,
                    dtype=torch.bool,
                ),
                ~observation_valid,
                torch.zeros(
                    batch,
                    GEOMETRY_QUERY_COUNT,
                    device=observation_valid.device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )
        for block in self.condition_blocks:
            tokens = block(tokens, padding_mask)
        tokens = self.output_norm(tokens)
        route_latent = tokens[:, :ROUTE_QUERY_COUNT]
        route_tokens = self.route_output_norm(route_latent)
        route_token = self.route_summary_norm(route_tokens.mean(dim=1))
        tokens = torch.cat((route_tokens, tokens[:, ROUTE_QUERY_COUNT:]), dim=1)
        tokens = tokens.masked_fill(padding_mask[..., None], 0.0)
        return ConditionFeatures(
            tokens=tokens,
            route_token=route_token,
            padding_mask=padding_mask,
        )
