"""Metric geometry, route, and state queries for the CurveNav generator."""

import torch
from torch import Tensor, nn

from curvenav.layers import EncoderBlock, RMSNorm, SwiGLU
from curvenav.types import ConditionFeatures, DepthFeatures


GEOMETRY_QUERY_COUNT = 64
ROUTE_QUERY_LAYERS = 2
CONDITION_ENCODER_TYPE = (
    "metric_geometry_queries_supervised_route_bottleneck_explicit_state"
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
        planning_horizon_m: float,
        model_dim: int = 384,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if history_scale_m <= 0 or planning_horizon_m <= 0:
            raise ValueError("condition metric scales must be positive")
        self.point_goal_encoder = point_goal_encoder
        self.observation_frames = observation_frames
        self.spatial_tokens = spatial_tokens
        self.history_scale_m = float(history_scale_m)
        self.planning_horizon_m = float(planning_horizon_m)

        self.frame_slot_embedding = nn.Parameter(
            torch.zeros(1, observation_frames, 1, model_dim)
        )
        self.geometry_query_embedding = nn.Parameter(
            torch.zeros(1, GEOMETRY_QUERY_COUNT, model_dim)
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
        self.invalid_state_embedding = nn.Parameter(
            torch.zeros(1, observation_frames, model_dim)
        )
        self.route_query_embedding = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.route_blocks = nn.ModuleList(
            CrossAttentionBlock(model_dim, transformer_heads, dropout)
            for _ in range(ROUTE_QUERY_LAYERS)
        )
        self.condition_blocks = nn.ModuleList(
            EncoderBlock(model_dim, transformer_heads, dropout)
            for _ in range(transformer_layers)
        )
        self.output_norm = RMSNorm(model_dim)
        self.subgoal_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, 2),
        )
        self.subgoal_embedding = nn.Sequential(
            nn.Linear(2, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.route_output_norm = RMSNorm(model_dim)
        nn.init.trunc_normal_(self.frame_slot_embedding, std=0.02)
        nn.init.trunc_normal_(self.geometry_query_embedding, std=0.02)
        nn.init.trunc_normal_(self.invalid_state_embedding, std=0.02)
        nn.init.trunc_normal_(self.route_query_embedding, std=0.02)

    @staticmethod
    def _unit_disk(value: Tensor) -> Tensor:
        squared_radius = value.float().square().sum(dim=-1, keepdim=True)
        return value / torch.sqrt(1.0 + squared_radius).to(value.dtype)

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
        geometry = self.geometry_compressor(
            self.geometry_query_embedding.expand(batch, -1, -1),
            visual_memory,
            visual_padding_mask,
        )

        state_input = observation_to_current.clone()
        state_input[..., :2] = state_input[..., :2] / self.history_scale_m
        state = self.state_encoder(state_input)
        state = torch.where(
            observation_valid[..., None],
            state,
            self.invalid_state_embedding.expand(batch, -1, -1),
        )

        goal = self.point_goal_encoder(point_goal).unsqueeze(1)
        route = self.route_query_embedding.expand(batch, -1, -1) + goal
        route_memory = torch.cat((state, geometry), dim=1)
        for block in self.route_blocks:
            route = block(route, route_memory)

        tokens = torch.cat((route, goal, state, geometry), dim=1)
        for block in self.condition_blocks:
            tokens = block(tokens)
        tokens = self.output_norm(tokens)
        route_latent = tokens[:, 0]
        local_subgoal = self.planning_horizon_m * self._unit_disk(
            self.subgoal_head(route_latent)
        )
        route_token = self.route_output_norm(
            route_latent
            + self.subgoal_embedding(local_subgoal / self.planning_horizon_m)
        )
        tokens = torch.cat((route_token[:, None], tokens[:, 1:]), dim=1)
        return ConditionFeatures(
            tokens=tokens,
            route_token=route_token,
            local_subgoal=local_subgoal,
        )
