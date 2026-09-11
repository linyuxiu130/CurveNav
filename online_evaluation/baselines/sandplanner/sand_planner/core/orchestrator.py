#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SanD-planner 推理流水线的编排器 / Orchestrator for the SanD-planner inference pipeline.

负责串联深度处理、模型推理、轨迹采样与评估等各组件。
Wires together depth processing, model inference, trajectory sampling, and evaluation.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sand_planner.config import InferenceConfig
from sand_planner.core.model_manager import ModelManager
from sand_planner.core.trajectory_inference import TrajectoryInference
from sand_planner.core.trajectory_evaluator import TrajectoryEvaluator
from sand_planner.trajectory.arc_length_sampling_vectorized import sample_predicted_trajectories_vectorized


class SandPlannerInference:
    """主推理类 / Main inference class."""

    def __init__(self, config: InferenceConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.model_manager = ModelManager(config)

        self.evaluator = TrajectoryEvaluator(config)

        # 初始化组件 / Initialize components
        self.model, self.normalizer = self.model_manager.load_model()
        self.inference_engine = TrajectoryInference(config, self.model, self.normalizer)

        # 由 Agent 传入的原始深度图（用于 ESDF 查询）
        # Original depth image passed in from the Agent (used for ESDF queries)
        self.agent_original_depth: Optional[np.ndarray] = None


    def reset_environment(self):
        """Reset the episode-scoped trajectory warm start."""
        self.inference_engine.reset_warm_start_cache()




    def process_depth_arrays(self, depth_sequences: torch.Tensor) -> Dict[str, Any]:
        """直接处理深度序列数组，避免文件 IO / Process depth-sequence arrays directly, avoiding file IO.

        Args:
            depth_sequences: (batch_size, seq_len, 1, H, W) 深度序列张量 / depth-sequence tensor.

        Returns:
            Dict[str, Any]: 推理结果字典 / inference results dictionary.
        """
        step_timing = {}

        if self.verbose:
            print(f"🔄 直接处理深度序列: {depth_sequences.shape}")

        # 确保在正确的设备上 / Make sure it is on the correct device
        depth_sequences = depth_sequences.to(self.config.device)

        # 取第一个 batch 进行推理（与原逻辑保持一致）
        # Take the first batch for inference (consistent with the original logic)
        depth_batch = depth_sequences[0:1]  # (1, seq_len, 1, H, W)

        original_depth_for_esdf = self.agent_original_depth
        self.agent_original_depth = None

        # 生成轨迹 / Generate trajectories
        inference_start = time.time()
        control_points, timing = self.inference_engine.generate_trajectories(depth_batch, self.config.target_position)
        step_timing['trajectory_generation'] = time.time() - inference_start

        # 轨迹采样 - 根据 prediction_mode 决定是否进行样条重建
        # waypoints 模式: 直接返回预测的 8 个 waypoints (0.2m 间隔)
        # control_points 模式: 样条拟合后等弧长采样
        # Trajectory sampling - prediction_mode decides whether spline reconstruction is performed:
        #   waypoints mode: directly return the 8 predicted waypoints (0.2 m spacing)
        #   control_points mode: fit a B-spline, then resample at equal arc length
        sampling_start = time.time()
        control_points_list = [
            control_points[i] for i in range(control_points.shape[0])
        ]
        sampled_trajectories = sample_predicted_trajectories_vectorized(
            control_points_list,
            arc_length=self.config.arc_length_step,
            method=self.config.trajectory_interpolation,
            prediction_mode=self.config.prediction_mode,
        )
        step_timing['arc_length_sampling'] = time.time() - sampling_start

        # 轨迹评估 / Trajectory evaluation
        eval_start = time.time()
        esdf_query_fn = self.evaluator.create_esdf_query(original_depth_for_esdf)
        best_index, evaluation_results = self.evaluator.evaluate_trajectories(
            sampled_trajectories,
            np.array(self.config.target_position),
            esdf_query_fn,
            clearance_max_points=self.config.clearance_max_points,
        )
        self.inference_engine.update_warm_start_cache(best_index)
        step_timing['trajectory_evaluation'] = time.time() - eval_start

        # 合并时间统计 / Merge timing statistics
        timing.update(step_timing)

        if self.verbose:
            print(f"\n✅ 直接深度处理完成:")
            print(f"\n📊 总体耗时统计:")
            total_time = (timing.get('trajectory_generation', 0) +
                         timing.get('arc_length_sampling', 0) +
                         timing.get('trajectory_evaluation', 0))

            print(f"   ├─ 轨迹生成:     {timing.get('trajectory_generation', 0)*1000:.2f}ms ({timing.get('trajectory_generation', 0)/max(total_time, 0.001)*100:.1f}%)")
            print(f"   │  ├─ 调度器初始化: {timing.get('scheduler_init', 0)*1000:.2f}ms")
            print(f"   │  ├─ 条件编码:     {timing.get('condition_encoding', 0)*1000:.2f}ms")
            print(f"   │  ├─ 噪声初始化:   {timing.get('noise_init', 0)*1000:.2f}ms")
            print(f"   │  │  ├─ 随机数生成:   {timing.get('randn_generation', 0)*1000:.2f}ms")
            print(f"   │  │  └─ 修正第一点:   {timing.get('fix_first_cp', 0)*1000:.2f}ms")
            print(f"   │  ├─ DDPM采样:     {timing.get('sampling', 0)*1000:.2f}ms")
            print(f"   │  └─ 后处理:       {timing.get('post_processing', 0)*1000:.2f}ms")
            print(f"   ├─ 等弧长采样:   {timing.get('arc_length_sampling', 0)*1000:.2f}ms ({timing.get('arc_length_sampling', 0)/max(total_time, 0.001)*100:.1f}%)")
            print(f"   ├─ 轨迹评估:     {timing.get('trajectory_evaluation', 0)*1000:.2f}ms ({timing.get('trajectory_evaluation', 0)/max(total_time, 0.001)*100:.1f}%)")
            print(f"   ├─ 可视化:       {timing.get('visualization', 0)*1000:.2f}ms ({timing.get('visualization', 0)/max(total_time, 0.001)*100:.1f}%)")
            print(f"   └─ 总计:         {total_time*1000:.2f}ms")
            print(f"\n🎯 性能瓶颈: DDPM采样 ({timing.get('sampling', 0)*1000:.2f}ms, {timing.get('sampling', 0)/max(total_time, 0.001)*100:.1f}%总时间)")
            print(f"\n💡 噪声初始化分析:")
            print(f"   - 随机数生成: {timing.get('randn_generation', 0)*1000:.2f}ms")
            print(f"   - 修正第一点: {timing.get('fix_first_cp', 0)*1000:.2f}ms")
            if timing.get('randn_generation', 0) > 0.01:  # 超过 10ms / over 10 ms
                print(f"   ⚠️  随机数生成较慢，可能是首次调用或CUDA同步问题")

        return {
            'control_points': control_points,
            'sampled_trajectories': sampled_trajectories,
            'best_index': best_index,
            'timing': timing,
            'evaluation_results': evaluation_results
        }
