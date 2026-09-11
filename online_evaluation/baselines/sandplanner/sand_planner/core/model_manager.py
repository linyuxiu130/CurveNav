#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SanD-planner 推理流程的模型管理器 / Model manager for the SanD-planner inference pipeline.

负责加载 checkpoint 并构建模型与归一化器。
Handles checkpoint loading and constructs the model and normalizer.
"""

import re

import torch
import torch.nn as nn

from sand_planner.config import InferenceConfig
from sand_planner.nn.condition_encoders import ConcatConditionEncoder
from sand_planner.nn.ddpm import BSplineDDPM
from sand_planner.utils.normalize import TrajectoryNormalizer


def _infer_num_transformer_layers(state_dict) -> int:
    """从 checkpoint 的 state_dict 推断 condition_encoder 中 transformer 的层数。 / Infer the number of transformer layers in the condition_encoder from the checkpoint state_dict."""
    pattern = re.compile(r'condition_encoder\.transformer\.layers\.(\d+)\.')
    layer_ids = {int(m.group(1)) for k in state_dict for m in [pattern.search(k)] if m}
    if not layer_ids:
        raise RuntimeError(
            "无法从 checkpoint 推断 transformer 层数：找不到 "
            "'condition_encoder.transformer.layers.<N>.' 键。"
        )
    return max(layer_ids) + 1


def _fuse_conv_bn_inplace(module: nn.Module):
    """递归融合所有 Conv+BN 对，消除推理时 BN 的额外计算和显存读写。 / Recursively fuse every Conv+BN pair to remove the extra BN compute and GPU memory traffic at inference time."""
    prev_name, prev_child = None, None
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d) and isinstance(prev_child, nn.Conv2d):
            fused = torch.nn.utils.fusion.fuse_conv_bn_eval(prev_child, child)
            setattr(module, prev_name, fused)
            setattr(module, name, nn.Identity())
        else:
            _fuse_conv_bn_inplace(child)
        prev_name, prev_child = name, child


class ModelManager:
    """模型管理器 / Model manager."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.normalizer = None

    def load_model(self):
        """加载模型和归一化器。 / Load the model and the trajectory normalizer."""

        # 加载 checkpoint / Load the checkpoint
        checkpoint = torch.load(
            self.config.checkpoint_path,
            map_location=self.config.device,
            weights_only=True,
        )
        state_dict_peek = checkpoint['model_state_dict']
        # Released NoMax weights omit sequence-length metadata. The evaluation
        # config explicitly specifies their eight control points.
        num_control_points = self.config.num_control_points

        # 从 checkpoint 自动推断 transformer 层数（num_heads 推不出，仍从 config 读取）
        # Infer the number of transformer layers from the checkpoint (num_heads cannot be inferred, still taken from config)
        inferred_layers = _infer_num_transformer_layers(state_dict_peek)
        if inferred_layers != self.config.num_transformer_layers:
            print(f"[ModelManager] num_transformer_layers: config={self.config.num_transformer_layers} "
                  f"→ 使用 checkpoint 实际值 {inferred_layers}")
        else:
            print(f"[ModelManager] num_transformer_layers={inferred_layers} (from checkpoint)")

        # 创建条件编码器 / Build the condition encoder
        condition_encoder = ConcatConditionEncoder(
            feature_dim=256,
            seq_len=self.config.sequence_length,
            num_transformer_layers=inferred_layers,
            num_heads=self.config.num_heads,
            cfg_drop_depth=False,  # 推理时不丢弃深度条件 / do not drop the depth condition at inference time
            cfg_drop_trajectory=True,  # 与训练时保持一致的默认值 / default kept consistent with training
            fusion_strategy=self.config.fusion_strategy,
            use_tea=False,  # checkpoint 含 motion_encoder / checkpoint contains a motion_encoder
            use_type_embed=False,  # checkpoint 没有 type_embed 层 / checkpoint has no type_embed layer
            use_initial_turn=True  # 启用初始转向输入（CP1+CP2 的 Y 均值）/ enable the initial-turn input (mean Y of CP1+CP2)
        )

        # 加载归一化器 / Load the trajectory normalizer
        self.normalizer = TrajectoryNormalizer(
            self.config.stats_path,
            method="percentile",
            margin=0.0,  # 备选/alt: 0.1 适用于其他数据集，0.0 适用于 mg2_mg50_2d / 0.1 for other datasets, 0.0 for mg2_mg50_2d
            clamp=True
        )

        # 创建完整模型 / Build the full model
        self.model = BSplineDDPM(
            condition_encoder=condition_encoder,
            num_train_timesteps=1000,
            fix_first_cp_zero=True,
            normalizer=self.normalizer,
            trajectory_interpolation=self.config.trajectory_interpolation,
            prediction_mode=self.config.prediction_mode,
            num_control_points=num_control_points
        )

        # 加载权重 / Load the weights
        self.model.load_state_dict(state_dict_peek, strict=True)

        self.model = self.model.to(self.config.device)
        self.model.eval()

        # --- 推理加速优化 / Inference speed optimizations ---

        # 1. Conv+BN 融合：消除 ResNet18 backbone 中 BN 层的额外开销
        #    Conv+BN fusion: remove the extra overhead of BN layers in the ResNet18 backbone
        n_fused = sum(1 for m in self.model.condition_encoder.depth_encoder.backbone.modules()
                      if isinstance(m, nn.BatchNorm2d))
        _fuse_conv_bn_inplace(self.model.condition_encoder.depth_encoder.backbone)
        print(f"[ModelManager] Conv+BN fusion: {n_fused} BN layers fused")

        # 2. UNet QKV 投影融合：将 3 次独立的 Q/K/V matmul 合并为 1 次
        #    UNet QKV projection fusion: merge 3 separate Q/K/V matmuls into a single one
        self.model.unet.fuse_qkv_projections()
        if torch.device(self.config.device).type == 'cuda':
            self.model.unet = torch.compile(self.model.unet, mode="reduce-overhead")
            self.model.condition_encoder = torch.compile(self.model.condition_encoder, mode="default")

        return self.model, self.normalizer
