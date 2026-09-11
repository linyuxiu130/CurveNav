import torch.nn as nn

from diffusers import DDPMScheduler
from sand_planner.nn.models.unet.unet_1d_condition import UNet1DConditionModel
from sand_planner.nn.condition_encoders import ConcatConditionEncoder

class BSplineDDPM(nn.Module):
    """用于生成 B-spline 控制点的 DDPM 模型，基于官方 UNet1DConditionModel。 / DDPM model for generating B-spline control points, built on the official UNet1DConditionModel."""

    def __init__(self,
                 condition_encoder: ConcatConditionEncoder,
                 num_train_timesteps: int = 1000,
                 fix_first_cp_zero: bool = False,
                 normalizer = None,
                 trajectory_interpolation: str = 'bspline',
                 prediction_mode: str = 'control_points',
                 num_control_points: int = 8):
        super().__init__()

        self.condition_encoder = condition_encoder
        self.fix_first_cp_zero = fix_first_cp_zero
        # 若提供，则表示在"归一化空间"中训练；推理时自动归一化/反归一化
        # If provided, training happens in normalized space; inference normalizes/denormalizes automatically
        self.normalizer = normalizer
        # CFG 相关参数
        # CFG-related parameters
        # 轨迹插值方法
        # Trajectory interpolation method
        self.trajectory_interpolation = trajectory_interpolation
        # 预测模式
        # Prediction mode
        self.prediction_mode = prediction_mode
        # B-spline 控制点数量（= UNet 序列长度，训练/推理必须一致）
        # Number of B-spline control points (= UNet sequence length, must match between training and inference)
        self.num_control_points = int(num_control_points)
        # UNet 有 2 次下采样（因子 4），序列长度必须是 4 的倍数，否则 forward 时
        # skip 连接尺寸不匹配。提前 fail-fast，给出可读的报错。
        # UNet has 2 downsampling stages (factor 4), so the sequence length must be a multiple of 4;
        # otherwise skip-connection sizes mismatch in forward. Fail fast with a readable error.
        if self.num_control_points % 4 != 0:
            raise ValueError(
                f"num_control_points={self.num_control_points} 必须是 4 的倍数"
                f"(如 8/12/16/20)。当前 UNet 有 2 次下采样(因子4)，否则会在 forward 时"
                f"报 skip 连接尺寸不匹配。")

        # 根据预测模式初始化
        # Initialize according to the prediction mode
        if self.prediction_mode == 'waypoints':
            print(f"🔧 预测模式: Waypoints (固定0.2m间隔)")
        else:
            print(f"🔧 预测模式: Control Points")
            # 初始化轨迹插值器（当使用 cubic spline 时）
            # Initialize the trajectory interpolator (when using cubic spline)
            if self.trajectory_interpolation == 'cubic_spline':
                from sand_planner.utils.traj_opt import TrajOpt
                self.traj_opt = TrajOpt()
                print(f"🔧 使用 Cubic Spline 插值生成轨迹")
            else:
                print(f"🔧 使用 B-Spline 插值生成轨迹")

        self.unet = UNet1DConditionModel(
            sample_size=self.num_control_points,  # 目标信号长度（= 控制点数量） / target signal length (= number of control points)
            in_channels=3,
            out_channels=3,
            layers_per_block=2,  # 每个 UNet block 使用的 ResNet 层数 / number of ResNet layers per UNet block
            # 使用 3 个 stage，确保总下采样次数为 3 次（2**3=8），与长度 8 对齐。
            # Use 3 stages so the total number of downsamplings is 3 (2**3=8), aligned with length 8.
            # 优化：减小通道数（64->32）以防止过拟合和"绕远路"。
            # 在 Head Dim 修复为 32 后，模型感知力大增，不需要过大的通道数即可工作。
            # 较小的模型倾向于生成更平滑、更直接的轨迹（正则化效果）。
            # Optimization: reduce channel width (64->32) to prevent overfitting and detouring.
            # After fixing Head Dim to 32, the model's perception improves greatly, so large channel
            # widths are no longer needed. Smaller models tend to produce smoother, more direct
            # trajectories (a regularization effect).
            block_out_channels=(64, 128, 256),
            # 保持 Head Dim=32。对应 Head 数量为：32/32=1, 64/32=2, 128/32=4。
            # 这种配置既保证了每个 Head 的表达能力，又限制了总体的复杂度。
            # Keep Head Dim=32. The resulting head counts are: 32/32=1, 64/32=2, 128/32=4.
            # This configuration preserves per-head expressiveness while limiting overall complexity.
            attention_head_dim=(32, 32, 32),
            down_block_types=(
                "DownBlock1D",
                "CrossAttnDownBlock1D",
                "DownBlock1D",
            ),
            mid_block_type="UNetMidBlock1DCrossAttn",
            up_block_types=(
                "ResnetUpsampleBlock1D",
                "CrossAttnUpBlock1D",
                "UpBlock1D",
            ),
            num_class_embeds=None,
            class_embeddings_concat=False,
            # GroupNorm 配置：使用 32 个组以支持 64/128/256 通道数。
            # GroupNorm config: use 32 groups to support 64/128/256 channel widths.
            norm_num_groups=32,  # 各通道数均可被 32 整除 / each channel width is divisible by 32 (64/128/256)
            # 关键修复：显式设置 cross_attention_dim，避免默认值 1280。
            # Key fix: set cross_attention_dim explicitly to avoid the default of 1280.
            encoder_hid_dim=condition_encoder.feature_dim,  # 256
            cross_attention_dim=condition_encoder.feature_dim,  # 256，避免默认 1280 带来的巨大开销 / 256, avoids the huge cost of the default 1280
        )

        # DDPM 调度器
        # DDPM scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,   # 1000 或 2000 起步 / start from 1000 or 2000
            beta_schedule="squaredcos_cap_v2",         # 推荐 / recommended
            prediction_type="v_prediction",                 # 备选/alt: epsilon
            clip_sample=False,
            timestep_spacing="linspace",               # 训练期通常用线性 / training usually uses linear spacing
            rescale_betas_zero_snr=True,               # 推荐打开 / recommended to enable
        )
        self.num_train_timesteps = num_train_timesteps
