import numpy as np
import torch
import torchvision.transforms as transforms
from configs.model_config import ModelConfig
from autoencoder import AutoEncoder,DualAutoEncoder
from m2f_inference import Mask2FormerInference
import traj_opt

class VIPlannerAgent():
    def __init__(
        self,
        image_intrinsic: torch.Tensor,
        m2f_path: str,
        m2f_config_path: str,
        model_path: str,
        model_config_path: str,
        device="cuda:0",
    ):
        self.image_intrinsic = image_intrinsic
        self.model_path = model_path
        self.model_config_path = model_config_path
        self.device = device
        self.traj_generate = traj_opt.TrajOpt()
        self.m2f_inference = Mask2FormerInference(
            config_file=m2f_config_path,
            checkpoint_file=m2f_path,
            device=device,
            debug=False,
        )
        self.model_config: ModelConfig | None = None
        self.load_model(self.model_path, self.model_config_path)
        self.transform = transforms.Resize(self.img_input_size, antialias=None)  # type: ignore
        self.traj_generate = traj_opt.TrajOpt()

    def load_model(self, model_path: str, model_config_path: str):
        self.model_config = ModelConfig.from_yaml(model_config_path)
        self.img_input_size = self.model_config.img_input_size
        if isinstance(self.model_config.data_cfg, list):
            self.max_goal_distance = self.model_config.data_cfg[0].max_goal_distance
            self.max_depth = self.model_config.data_cfg[0].max_depth
        else:
            self.max_goal_distance = self.model_config.data_cfg.max_goal_distance
            self.max_depth = self.model_config.data_cfg.max_depth
        if self.model_config.sem:
            self.net = DualAutoEncoder(self.model_config)
        else:
            self.net = AutoEncoder(self.model_config.in_channel, self.model_config.knodes)
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception:
            # Official ViPlanner releases may contain a serialized module rather
            # than a plain state dict. The download script pins the source URL.
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, (tuple, list)):
            checkpoint = checkpoint[0]
        if isinstance(checkpoint, torch.nn.Module):
            model_state_dict = checkpoint.state_dict()
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model_state_dict = checkpoint["state_dict"]
        else:
            model_state_dict = checkpoint
        self.net.load_state_dict(model_state_dict, strict=True)
        self.net.eval()
        self.net.to(self.device)

    def process_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth = self.transform(depth.unsqueeze(1)).expand(-1, 3, -1, -1).clone()
        depth.masked_fill_(depth > self.max_depth, 0.0)
        depth.masked_fill_(~torch.isfinite(depth), 0.0)
        return depth

    def plan(self, dep_image: torch.Tensor, sem_image: torch.Tensor, goal_robot_frame: torch.Tensor) -> tuple:
        # transform input
        sem_image = self.transform(sem_image) / 255
        with torch.no_grad():
            keypoints, fear = self.net(self.process_depth(dep_image), sem_image, goal_robot_frame)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)
        return keypoints, traj, fear

    def step_pointgoal(
        self,
        image: torch.Tensor,
        dep_image: torch.Tensor,
        goal_robot_frame: torch.Tensor,
    ):
        with torch.no_grad():
            tensor_dep_image = torch.as_tensor(dep_image[:,:,:,0], device=self.device, dtype=torch.float32)
            tensor_goal_robot_frame = torch.as_tensor(goal_robot_frame[:,0:3], device=self.device, dtype=torch.float32)
            semantic_image = self.m2f_inference.predict_batch(image)
            sem_image = torch.tensor(semantic_image).permute(0, 3, 1, 2).float().to(self.device)
            keypoints, traj, fear = self.plan(tensor_dep_image, sem_image, tensor_goal_robot_frame)
            #print(keypoints.shape,traj.shape)
            return keypoints, traj, fear
