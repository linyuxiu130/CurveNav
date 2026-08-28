"""Goal-relative metric memory without an expert-motion route shortcut."""

import torch
from torch import Tensor, nn

from curvenav.layers import EncoderBlock, RMSNorm, SwiGLU, unit_rms
from curvenav.types import ConditionFeatures, DepthFeatures


HISTORY_GEOMETRY_QUERY_COUNT = 32
CONDITION_ENCODER_TYPE = (
    "contextual_goal_relative_current_geometry_plus_aligned_compressed_history"
)


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention followed by one SwiGLU residual update."""

    def __init__(self, model_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
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
        attention_mask = None
        if memory_padding_mask is not None:
            attention_mask = (
                memory_padding_mask[:, None, None, :]
                .expand(-1, self.heads, query.shape[1], -1)
                .reshape(-1, query.shape[1], memory.shape[1])
                .contiguous()
            )
        normalized_memory = self.memory_norm(memory)
        attended = self.attention(
            self.query_norm(query),
            normalized_memory,
            normalized_memory,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]
        query = query + self.dropout(attended)
        return query + self.dropout(self.feed_forward(self.feed_forward_norm(query)))


class PolicyConditionEncoder(nn.Module):
    """Fuse PointGoal with metric geometry before curve generation.

    Historical transforms are consumed only by the depth projector that aligns
    geometry into the current body frame.  They are deliberately not encoded as
    an independent token, so expert motion cannot act as a route label.
    """

    def __init__(
        self,
        point_goal_encoder: nn.Module,
        *,
        observation_frames: int,
        spatial_tokens: int,
        planning_horizon_m: float,
        max_depth_m: float,
        model_dim: int = 384,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if planning_horizon_m <= 0 or max_depth_m <= 0:
            raise ValueError("condition metric scales must be positive")
        self.point_goal_encoder = point_goal_encoder
        self.observation_frames = observation_frames
        self.spatial_tokens = spatial_tokens
        self.planning_horizon_m = float(planning_horizon_m)
        self.max_depth_m = float(max_depth_m)

        self.frame_slot_embedding = nn.Parameter(
            torch.zeros(1, observation_frames, 1, model_dim)
        )
        self.goal_geometry_projection = nn.Sequential(
            nn.Linear(6, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.history_query_embedding = nn.Parameter(
            torch.zeros(1, HISTORY_GEOMETRY_QUERY_COUNT, model_dim)
        )
        self.history_null_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.history_compressor = CrossAttentionBlock(
            model_dim,
            transformer_heads,
            dropout,
        )
        self.context_blocks = nn.ModuleList(
            EncoderBlock(model_dim, transformer_heads, dropout)
            for _ in range(transformer_layers)
        )
        self.memory_norm = RMSNorm(model_dim)
        nn.init.trunc_normal_(self.frame_slot_embedding, std=0.02)
        nn.init.trunc_normal_(self.history_query_embedding, std=0.02)
        nn.init.trunc_normal_(self.history_null_token, std=0.02)

    def _goal_relative_geometry(
        self,
        observation: DepthFeatures,
        point_goal: Tensor,
        observation_valid: Tensor,
    ) -> Tensor:
        planar = observation.points[..., :2]
        goal_distance = torch.linalg.vector_norm(point_goal, dim=-1, keepdim=True)
        goal_direction = point_goal / goal_distance.clamp_min(1e-6)
        goal_direction = torch.where(
            goal_distance > 1e-6,
            goal_direction,
            torch.zeros_like(goal_direction),
        )
        direction = goal_direction[:, None, None]
        longitudinal = (planar * direction).sum(dim=-1) / self.planning_horizon_m
        lateral = (
            direction[..., 0] * planar[..., 1]
            - direction[..., 1] * planar[..., 0]
        ) / self.planning_horizon_m
        radius = torch.linalg.vector_norm(planar, dim=-1) / self.planning_horizon_m
        goal_range = (
            torch.log1p(
                goal_distance.clamp_max(
                    self.point_goal_encoder.goal_clip_distance_m
                )
            )
            / self.point_goal_encoder.log_range_scale
        )[:, None].expand_as(longitudinal)
        surface_valid = observation.depth < self.max_depth_m
        obstacle_valid = observation.obstacle_valid & observation_valid[..., None]
        return torch.stack(
            (
                longitudinal,
                lateral,
                radius,
                goal_range,
                surface_valid.to(planar.dtype),
                obstacle_valid.to(planar.dtype),
            ),
            dim=-1,
        )

    def forward(
        self,
        observation: DepthFeatures,
        point_goal: Tensor,
        observation_valid: Tensor,
    ) -> ConditionFeatures:
        batch = point_goal.shape[0]
        expected = (batch, self.observation_frames, self.spatial_tokens)
        if observation.tokens.ndim != 4 or observation.tokens.shape[:3] != expected:
            raise ValueError("observation tokens do not match the condition contract")
        goal_geometry = self._goal_relative_geometry(
            observation,
            point_goal,
            observation_valid,
        )
        visual = (
            observation.tokens
            + self.frame_slot_embedding
            + self.goal_geometry_projection(goal_geometry.to(observation.tokens.dtype))
        )
        visual = torch.where(
            observation_valid[..., None, None],
            visual,
            torch.zeros_like(visual),
        )

        history_memory = visual[:, :-1].flatten(1, 2)
        history_padding_mask = (
            (~observation_valid[:, :-1])
            .unsqueeze(-1)
            .expand(-1, -1, self.spatial_tokens)
            .flatten(1)
        )
        history_memory = torch.cat(
            (history_memory, self.history_null_token.expand(batch, -1, -1)),
            dim=1,
        )
        history_padding_mask = torch.cat(
            (
                history_padding_mask,
                torch.zeros(batch, 1, dtype=torch.bool, device=point_goal.device),
            ),
            dim=1,
        )
        history_queries = unit_rms(self.history_query_embedding).expand(batch, -1, -1)
        history = self.history_compressor(
            history_queries,
            history_memory,
            history_padding_mask,
        )
        goal = self.point_goal_encoder(point_goal).unsqueeze(1)
        tokens = torch.cat((goal, visual[:, -1], history), dim=1).float()
        for block in self.context_blocks:
            tokens = block(tokens)
        tokens = self.memory_norm(tokens)
        current_end = 1 + self.spatial_tokens
        return ConditionFeatures(
            tokens=tokens,
            current_tokens=tokens[:, 1:current_end],
            current_points=observation.points[:, -1, :, :2].float(),
            current_obstacle_valid=observation.obstacle_valid[:, -1],
            point_goal=point_goal.float(),
        )
