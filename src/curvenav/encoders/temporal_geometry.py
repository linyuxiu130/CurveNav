"""One geometry decoder over the single six-frame-fused spatial memory."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from curvenav.precision import NEURAL_DTYPE
from curvenav.data.obstacle_memory import VOXEL_M
from curvenav.data.six_frame_memory import READ_RADIUS_M
from curvenav.encoders.configuration import metric_grid
from curvenav.types import ConfigurationFeatures


class FeatureGeometryDecoder(nn.Module):
    """Decode one body-aligned memory raster; no recent/older or executed-footprint branches."""
    def __init__(self,model_dim,horizon,radius,width=64):
        super().__init__();self.neural_dtype=NEURAL_DTYPE
        self.horizon,self.radius=horizon,radius
        resolution=2*horizon/63
        self.raster_size=32*math.ceil((64+2*math.ceil((READ_RADIUS_M-horizon)/resolution))/32)
        self.raster_padding=(self.raster_size-64)//2
        token_dim=model_dim-8
        position,encoding=metric_grid(token_dim,16,horizon*self.raster_size/64)
        self.register_buffer('metric_position',position)
        self.register_buffer('metric_encoding',encoding)
        def block(cin,cout,stride=1):
            return nn.Sequential(nn.Conv2d(cin,cout,3,stride=stride,padding=1),nn.GroupNorm(8,cout),nn.SiLU())
        self.stem=nn.Sequential(nn.Conv2d(model_dim+5,width,1),nn.GroupNorm(8,width),nn.SiLU())
        self.down=nn.ModuleList(block(width,width,2) for _ in range(5))
        self.up=nn.ModuleList(block(2*width,width) for _ in range(5))
        self.readout=nn.Conv2d(width,2,1)
        self.token_projection=nn.Conv2d(width,token_dim,1)

    def forward(self,memory,pose,mass):
        batch,_,size,_=memory.shape
        assert size==self.raster_size and memory.shape[-2]==size
        context=(size-64)//2
        axis=torch.arange(-context,64+context,device=memory.device,dtype=torch.float32)
        y,x=torch.meshgrid(axis,axis,indexing='ij')
        xy=torch.stack((x,y),-1)
        local=(xy-31.5)*(2*self.horizon/63)
        world=torch.einsum('hwj,bij->bhwi',local.double(),pose[:,:2,:2].double())+pose[:,None,None,:2,3]
        phase=torch.remainder(world,VOXEL_M).float()/VOXEL_M
        inputs=torch.cat((memory,torch.log1p(mass),
            (xy/31.5-1).permute(2,0,1)[None].expand(batch,-1,-1,-1),phase.permute(0,3,1,2)),1)
        with torch.autocast(device_type=memory.device.type,dtype=self.neural_dtype):
            value=self.stem(inputs);levels=[value]
            for layer in self.down:value=layer(value);levels.append(value)
            for layer,skip in zip(self.up,reversed(levels[:-1]),strict=True):
                value=layer(torch.cat((F.interpolate(value,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip),1))
            all_logits=self.readout(value).float()
            tokens=self.token_projection(F.adaptive_avg_pool2d(value,(16,16))).flatten(2).transpose(1,2)
        logits=all_logits[:,:,context:context+64,context:context+64]
        clearance=logits[:,:1]*self.horizon
        resolution=2*self.horizon/63
        padded=F.pad(clearance,(1,1,1,1),mode='replicate')
        dx=(padded[:,:,1:-1,2:]-padded[:,:,1:-1,:-2])/(2*resolution)
        dy=(padded[:,:,2:,1:-1]-padded[:,:,:-2,1:-1])/(2*resolution)
        overlap=torch.sigmoid(-clearance/resolution)
        free=logits[:,1:2].sigmoid()
        support=overlap+(1-overlap)*free
        configuration=ConfigurationFeatures(tokens=tokens+self.metric_encoding.to(tokens.dtype),
            metric_position=self.metric_position.expand(batch,-1,-1),
            observed_fraction=F.adaptive_avg_pool2d(-torch.expm1(-mass),(16,16)).flatten(1))
        return torch.cat((clearance,dx,dy,support,overlap,free),1),logits,configuration


def geometry_reconstruction_loss(field,logits,target):
    """Direct metric distance and observed-free supervision; support is derived.

    Distance targets refer to observed occupied sites, including in unknown
    regions. A positive distance never certifies free space without free evidence.
    Free evidence has its own Bernoulli output: its loss must not change an
    already correct distance merely to open or close a distance-dependent gate.
    """
    resolution=2*3.6/63
    distance=field[:,:1];truth=target[:,:1]
    spatial_gradient=sum(F.smooth_l1_loss(torch.diff(distance,dim=axis)/resolution,
        torch.diff(truth,dim=axis)/resolution) for axis in (-1,-2))/2
    return (F.smooth_l1_loss(distance/resolution,truth/resolution)+spatial_gradient
        +F.binary_cross_entropy_with_logits(logits[:,1:2],target[:,5:6]))
