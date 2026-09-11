import torch
import numpy as np
import cv2
from navbench.depth import replace_invalid_depth
from policy_network import NavDP_Policy


def _resize_letterbox(image, size):
    scale = size / max(image.shape[:2])
    resized = cv2.resize(image, (-1, -1), fx=scale, fy=scale)
    pad_h = size - resized.shape[0]
    pad_w = size - resized.shape[1]
    return cv2.copyMakeBorder(
        resized,
        pad_h // 2,
        pad_h - pad_h // 2,
        pad_w // 2,
        pad_w - pad_w // 2,
        cv2.BORDER_CONSTANT,
        value=0,
    )

class NavDP_Agent:
    def __init__(self,
                 image_intrinsic,
                 image_size=224,
                 memory_size=8,
                 predict_size=24,
                 temporal_depth=16,
                 heads=8,
                 token_dim=384,
                 navi_model = "./100.ckpt",
                 device='cuda:0',
                 cache_rgb_tokens=True):
        self.image_intrinsic = image_intrinsic
        self.device = device
        self.predict_size = predict_size
        self.image_size = image_size
        self.memory_size = memory_size
        self.navi_former = NavDP_Policy(
            image_size, memory_size, predict_size, temporal_depth, heads,
            token_dim, device=device, cache_rgb_tokens=cache_rgb_tokens,
        )
        state_dict = torch.load(navi_model, map_location=self.device, weights_only=True)
        self.navi_former.load_state_dict(state_dict, strict=False)
        self.navi_former.to(self.device)
        self.navi_former.eval()

    def reset(self, batch_size, threshold, global_batch_size=None, batch_start=0):
        self.batch_size = int(batch_size)
        self.global_batch_size = int(
            self.batch_size if global_batch_size is None else global_batch_size
        )
        self.batch_start = int(batch_start)
        if not 0 <= self.batch_start < self.global_batch_size:
            raise ValueError("batch_start must index the global policy batch")
        if self.batch_start + self.batch_size > self.global_batch_size:
            raise ValueError("local policy shard exceeds the global policy batch")
        self.stop_threshold = threshold
        self.memory_queue = np.zeros(
            (self.batch_size, self.memory_size, self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )
        self.navi_former.rgbd_encoder.reset_cache()
    def reset_env(self,i):
        self.memory_queue[i].fill(0)
        self.navi_former.rgbd_encoder.reset_cache_env(i)

    def update_memory(self, images):
        """Append one frame to the preallocated zero-left-padded history."""
        self.memory_queue[:, :-1] = self.memory_queue[:, 1:]
        self.memory_queue[:, -1] = images
        return self.memory_queue

    def apply_stop_threshold(self, trajectories, values):
        """Apply the critic stop rule independently to each batched environment."""
        stop_mask = values.max(axis=1) < self.stop_threshold
        if not np.any(stop_mask):
            return trajectories
        stopped = trajectories[stop_mask]
        stopped[..., 0] = 0.0
        stopped[..., 1] = np.sign(
            stopped[..., 1].mean(axis=(1, 2), keepdims=True)
        )
        trajectories[stop_mask] = stopped
        return trajectories

    def process_image(self,images):
        return_images = []
        for img in images:
            return_images.append(_resize_letterbox(img, self.image_size))
        return np.asarray(return_images, dtype=np.uint8)

    def process_depth(self,depths):
        depths = replace_invalid_depth(depths, 0.0)
        return_depths = []
        for depth in depths:
            resize_depth = _resize_letterbox(depth, self.image_size)
            resize_depth[resize_depth>5.0] = 0
            resize_depth[resize_depth<0.1] = 0
            return_depths.append(resize_depth[:,:,np.newaxis])
        return np.array(return_depths)

    def process_pointgoal(self,goals):
        # Match the published NavDP wheeled adapter: the policy is trained
        # with forward-facing point goals and receives a non-negative x
        # component, while the evaluator still retains the complete goal for
        # metrics and termination.
        clip_goals = np.asarray(goals).clip(-10, 10)
        clip_goals[:, 0] = np.clip(clip_goals[:, 0], 0, 10)
        return clip_goals

    def step_pointgoal(self,goals,images,depths):
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        input_image = self.update_memory(process_images)
        input_depth = process_depths
        input_goals = self.process_pointgoal(goals)
        # cv2.imwrite("input_image.jpg",np.concatenate(self.memory_queue[0],axis=0)*255)
        all_trajectory, all_values, good_trajectory, _ = self.navi_former.predict_pointgoal_action(
            input_goals,
            input_image,
            input_depth,
            global_batch_size=self.global_batch_size,
            batch_start=self.batch_start,
        )
        good_trajectory = self.apply_stop_threshold(good_trajectory, all_values)

        return good_trajectory[:,0], all_trajectory, all_values, None
