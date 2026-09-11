import torch
import torch.nn as nn
import math
from navbench.vision.depth_anything_v2.dpt import DepthAnythingV2

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(max_len, dtype=torch.long), persistent=False
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        position_ids = self.position_ids[:seq_len].unsqueeze(0).expand(batch_size, -1)
        position_encoding = self.position_embedding(position_ids)  # (batch_size, seq_len, embed_dim)
        return position_encoding

class NavDP_RGBD_Backbone(nn.Module):
    def __init__(self,
                 image_size=224,
                 embed_size=512,
                 memory_size=8,
                 device='cuda:0',
                 cache_rgb_tokens=True):
        super().__init__()
        self.device = device
        self.memory_size = memory_size
        self.image_size = image_size
        self.embed_size = embed_size
        self.cache_rgb_tokens = cache_rgb_tokens
        self._rgb_token_cache = None
        self._zero_rgb_token = None
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.rgb_model = DepthAnythingV2(**model_configs['vits'])
        self.rgb_model = self.rgb_model.pretrained.float()
        self.rgb_model.eval()
        self.register_buffer(
            "preprocess_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "preprocess_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=False,
        )

        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained.float()
        self.depth_model.train()
        self.former_query = LearnablePositionalEncoding(384,self.memory_size*16)
        self.former_pe = LearnablePositionalEncoding(384,(self.memory_size+1)*256)
        self.former_net = nn.TransformerDecoder(nn.TransformerDecoderLayer(384,8,batch_first=True),2)
        self.project_layer = nn.Linear(384,embed_size)

    def reset_cache(self):
        self._rgb_token_cache = None
        self._zero_rgb_token = None

    def reset_cache_env(self, env_id):
        if self._rgb_token_cache is not None:
            with torch.inference_mode():
                self._rgb_token_cache[env_id].copy_(self._zero_rgb_token)

    def _encode_rgb(self, tensor_images):
        if tensor_images.dtype == torch.uint8:
            tensor_images = tensor_images.to(torch.float32).mul_(1.0 / 255.0)
        else:
            tensor_images = tensor_images.to(torch.float32)
        tensor_images = tensor_images.permute(0, 3, 1, 2)
        tensor_norm_images = (tensor_images - self.preprocess_mean) / self.preprocess_std
        return self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]

    def forward(self,images,depths):
        with torch.no_grad():
            if len(images.shape) == 5 and self.cache_rgb_tokens:
                B, T = images.shape[:2]
                current_images = torch.as_tensor(images[:, -1], device=self.device)
                current_token = self._encode_rgb(current_images)
                if (
                    self._rgb_token_cache is None
                    or self._rgb_token_cache.shape[:2] != (B, T)
                    or self._rgb_token_cache.dtype != current_token.dtype
                ):
                    zero_images = torch.zeros_like(current_images[:1])
                    zero_token = self._encode_rgb(zero_images)[0]
                    self._zero_rgb_token = zero_token.unsqueeze(0).expand(T, -1, -1).clone()
                    self._rgb_token_cache = self._zero_rgb_token.unsqueeze(0).expand(B, -1, -1, -1).clone()
                self._rgb_token_cache = torch.cat(
                    (self._rgb_token_cache[:, 1:], current_token[:, None]), dim=1
                )
                image_token = self._rgb_token_cache.reshape(B, T * 256, -1)
            else:
                tensor_images = torch.as_tensor(images, device=self.device)
                if tensor_images.dtype == torch.uint8:
                    tensor_images = tensor_images.to(torch.float32).mul_(1.0 / 255.0)
                else:
                    tensor_images = tensor_images.to(torch.float32)
                if len(images.shape) == 4:
                    tensor_images = tensor_images.permute(0,3,1,2)
                    tensor_images = tensor_images.reshape(-1,3,self.image_size,self.image_size)
                    tensor_norm_images = (tensor_images - self.preprocess_mean) / self.preprocess_std
                    image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]
                elif len(images.shape) == 5:
                    tensor_images = tensor_images.permute(0,1,4,2,3)
                    B,T,C,H,W = tensor_images.shape
                    tensor_images = tensor_images.reshape(-1,3,self.image_size,self.image_size)
                    tensor_norm_images = (tensor_images - self.preprocess_mean) / self.preprocess_std
                    image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0].reshape(B,T*256,-1)
            if len(depths.shape) == 4:
                tensor_depths = torch.as_tensor(depths,dtype=torch.float32,device=self.device).permute(0,3,1,2)
                tensor_depths = tensor_depths.reshape(-1,1,self.image_size,self.image_size)
                tensor_depths = torch.concat([tensor_depths,tensor_depths,tensor_depths],dim=1)
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
            elif len(depths.shape) == 5:
                tensor_depths = torch.as_tensor(depths,dtype=torch.float32,device=self.device).permute(0,1,4,2,3)
                B,T,C,H,W = tensor_depths.shape
                tensor_depths = tensor_depths.reshape(-1,1,self.image_size,self.image_size)
                tensor_depths = torch.concat([tensor_depths,tensor_depths,tensor_depths],dim=1)
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0].reshape(B,T*256,-1)
            former_token = torch.concat((image_token,depth_token),dim=1) + self.former_pe(torch.concat((image_token,depth_token),dim=1))
            former_query = self.former_query(torch.zeros((image_token.shape[0], self.memory_size * 16, 384),device=self.device))
            memory_token = self.former_net(former_query,former_token)
            memory_token = self.project_layer(memory_token)
            return memory_token
