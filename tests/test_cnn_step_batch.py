import copy
import numpy as np
import torch
from torch import nn

from types import SimpleNamespace
from curvenav.encoders.depth import DepthObservationEncoder

def test_full_image_batch_sample_loss_and_gradient():
    encoder=DepthObservationEncoder.__new__(DepthObservationEncoder)
    nn.Module.__init__(encoder);encoder.neural_dtype=torch.float32
    torch.manual_seed(51)
    encoder.backbone=nn.Sequential(nn.Conv2d(2,4,3,padding=1),nn.GroupNorm(2,4),nn.SiLU())
    encoder.spatial_projection=nn.Conv2d(4,3,1)
    other=copy.deepcopy(encoder)
    depth=torch.rand(3,1,16,16)
    pixels=torch.tensor([[[0,1,1,63,255]],[[2,14,14,79,250]],[[7,8,8,9,44]]])
    batch=encoder.sample_pixel_features(encoder.image_features(depth),pixels,16)
    serial=torch.cat([other.sample_pixel_features(other.image_features(depth[i:i+1]),pixels[i:i+1],16) for i in range(3)])
    torch.testing.assert_close(batch,serial,rtol=2e-6,atol=2e-7)
    batch.square().mean().backward();serial.square().mean().backward()
    for p,q in zip(encoder.parameters(),other.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-7)
    # Supplied maps follow the same sampled pixels, retaining duplicate sources.
    maps=encoder.image_features(depth)
    supplied=encoder.sample_pixel_features(maps,pixels,16)
    torch.testing.assert_close(supplied,batch,rtol=0,atol=0)



def test_step_image_batches_preserve_keys_shapes_duplicates_and_gradients():
    from curvenav.training.bspline_image_batch import step_feature_maps

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.network=nn.Sequential(nn.Conv2d(1,4,3,padding=1),nn.GroupNorm(2,4),nn.SiLU())
            self.batches=[]
        def image_features(self,depth):
            self.batches.append(tuple(depth.shape))
            return self.network(depth)

    torch.manual_seed(29)
    policy=nn.Module();policy.depth_encoder=Encoder()
    serial=copy.deepcopy(policy)
    raw=dict(depths=np.random.default_rng(29).random((5,12,16),dtype=np.float32))
    other=dict(depths=np.random.default_rng(30).random((2,8,12),dtype=np.float32))
    def record(sha,ids,data):
        return (dict(raw_sha256=sha,memory_plan=dict(frame_ids=np.array(ids))),data,None,None)
    records=[record('a',[0,1,1,2],raw),record('a',[2,3,4],raw),record('b',[0,1],other)]
    batched=step_feature_maps(SimpleNamespace(policy=policy),records,cnn_batch_size=3)
    reference={}
    for item,data,*_ in records:
        for i in np.unique(item['memory_plan']['frame_ids']):
            key=(item['raw_sha256'],int(i))
            if key not in reference:
                reference[key]=serial.depth_encoder.image_features(torch.from_numpy(data['depths'][i])[None,None])
    assert batched.keys()==reference.keys()
    assert [shape[0] for shape in policy.depth_encoder.batches]==[3,3,3]
    depths=raw['_device_depths'][torch.device('cpu')]
    assert depths.shape==(5,1,12,16) and depths.is_contiguous()
    torch.testing.assert_close(depths[:,0],torch.from_numpy(raw['depths']),rtol=0,atol=0)
    for key in reference:
        torch.testing.assert_close(batched[key],reference[key],rtol=2e-5,atol=2e-6)
    sum(batched[k].square().mean()*(i+1) for i,k in enumerate(reference)).backward()
    sum(reference[k].square().mean()*(i+1) for i,k in enumerate(reference)).backward()
    for actual,expected in zip(policy.parameters(),serial.parameters(),strict=True):
        torch.testing.assert_close(actual.grad,expected.grad,rtol=2e-5,atol=2e-6)
    with torch.no_grad():policy.depth_encoder.network[0].weight.add_(.1)
    updated=step_feature_maps(SimpleNamespace(policy=policy),records,cnn_batch_size=3)
    assert raw['_device_depths'][torch.device('cpu')] is depths
    assert not torch.allclose(updated['a',0],batched['a',0])


def test_frozen_image_cache_bypasses_cnn():
    from curvenav.training.bspline_image_batch import step_feature_maps
    policy=nn.Linear(1,1)
    policy.depth_encoder=SimpleNamespace(image_features=lambda _: (_ for _ in ()).throw(AssertionError('CNN called')))
    record=(dict(raw_sha256='a',memory_plan=dict(frame_ids=np.array([3,3,7]))),{},None,None)
    cache={('a',3):torch.randn(1,4,2,2),('a',7):torch.randn(1,4,2,2)}
    result=step_feature_maps(SimpleNamespace(policy=policy),[record],cache=cache)
    assert result.keys()==cache.keys()
    assert all(result[key] is cache[key] for key in cache)
