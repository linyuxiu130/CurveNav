
import cv2
import numpy as np
import torch
from navbench.depth import replace_invalid_depth

from sand_planner.utils.image import downscale_to_target_size


class ArrayDepthProcessor:
    """深度图像处理器（基于数组），移入 agent 以提升性能，支持特征缓存 / Depth image processor (array-based), moved into the agent for performance, with feature caching."""

    def __init__(self, config):
        self.config = config
        # 缓存编码后的特征以提高效率 / Cache encoded features to improve efficiency
        self.encoded_features_cache = []  # 存储编码后的特征 List[torch.Tensor] / Stores encoded features, List[torch.Tensor]
        self.max_cache_size = config.depth_cache_size  # 最大缓存帧数 / Maximum number of cached frames

    def process_depth_array_for_model(self, depth_array) -> np.ndarray:
        """直接处理深度数组用于模型输入，避免文件 IO / Process a depth array directly for model input, avoiding file IO."""

        # 确保是浮点类型 / Ensure float32 dtype
        if depth_array.dtype != np.float32:
            depth_array = depth_array.astype(np.float32)
        depth_array = replace_invalid_depth(depth_array, 0.0)

        # 裁剪深度值 / Clip depth values
        depth_array = np.clip(depth_array, 0, self.config.max_depth)

        # 下采样（如果启用） / Downsampling (if enabled)
        if self.config.downscale_depth:
            depth_array = downscale_to_target_size(
                depth_array, self.config.image_height, self.config.image_width, is_depth=True
            )

        # 调整大小到目标尺寸 / Resize to the target size
        target_size = (self.config.image_height, self.config.image_width)
        if len(depth_array.shape) == 3:  # (H, W, 1)
            depth_array = depth_array[:, :, 0]  # 移除通道维度 / Remove the channel dimension

        if depth_array.shape != target_size:
            depth_array = cv2.resize(
                depth_array, (self.config.image_width, self.config.image_height),
                interpolation=cv2.INTER_NEAREST,
            )

        # 归一化到 [0, 1] / Normalize to [0, 1]
        return depth_array / self.config.max_depth

    def encode_single_depth_frame(self, depth_array: np.ndarray) -> torch.Tensor:
        """编码单帧深度图像为特征，用于缓存 / Encode a single depth frame into features for caching.

        Args:
            depth_array: (H, W, 1) 或 (H, W) 单帧深度数组 / a single-frame depth array of shape (H, W, 1) or (H, W).

        Returns:
            torch.Tensor: (1, H, W) 编码后的单帧特征 / (1, H, W) encoded single-frame features.
        """
        # 处理深度数组 / Process the depth array
        processed_depth = self.process_depth_array_for_model(depth_array)

        # 转换为张量并添加 batch 和 channel 维度 / Convert to a tensor and add batch and channel dimensions
        depth_tensor = torch.from_numpy(processed_depth).float().unsqueeze(0)  # (1, H, W)

        return depth_tensor



    def add_frame_to_cache(self, depth_array: np.ndarray) -> None:
        """添加新帧到特征缓存（支持跳帧） / Add a new frame to the feature cache (supports frame skipping).

        Args:
            depth_array: (H, W, 1) 或 (H, W) 新的深度帧 / a new depth frame of shape (H, W, 1) or (H, W).
        """
        # 编码单帧 / Encode the single frame
        encoded_frame = self.encode_single_depth_frame(depth_array)

        # 添加到缓存 / Add to the cache
        self.encoded_features_cache.append(encoded_frame)

        # 保持缓存大小限制 / Enforce the cache size limit
        if len(self.encoded_features_cache) > self.max_cache_size:
            self.encoded_features_cache.pop(0)  # 移除最旧的特征 / Remove the oldest feature

    def get_sequence_from_cache(self) -> torch.Tensor:
        """从缓存中构建深度序列 / Build a depth sequence from the cache.

        Returns:
            torch.Tensor: (1, seq_len, 1, H, W) 深度序列 / (1, seq_len, 1, H, W) depth sequence.
        """
        if not self.encoded_features_cache:
            raise ValueError("特征缓存为空，请先添加帧")

        # 如果缓存帧数不足，用最新帧重复填充 / If there are too few cached frames, pad by repeating the latest frame
        cached_features = self.encoded_features_cache.copy()
        while len(cached_features) < self.config.sequence_length:
            cached_features.append(cached_features[-1])  # 重复最新帧 / Repeat the latest frame

        # 取最近的 sequence_length 帧 / Take the most recent sequence_length frames
        recent_features = cached_features[-self.config.sequence_length:]

        # 堆叠成序列 (seq_len, 1, H, W) / Stack into a sequence of shape (seq_len, 1, H, W)
        sequence = torch.stack(recent_features, dim=0)

        # 添加 batch 维度 (1, seq_len, 1, H, W) / Add the batch dimension -> (1, seq_len, 1, H, W)
        return sequence.unsqueeze(0)

    def clear_cache(self):
        """清空特征缓存 / Clear the feature cache."""
        self.encoded_features_cache.clear()
