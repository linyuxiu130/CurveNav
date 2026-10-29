import torch
import pytest
import numpy as np
from scipy.spatial import cKDTree
from curvenav.encoders.temporal_geometry import FeatureGeometryDecoder,geometry_reconstruction_loss


@pytest.mark.parametrize('clearance,free_target',[(.01,1.),(.05,1.),(.05,0.),(-.05,0.)])
def test_free_evidence_cannot_move_an_already_correct_metric_distance(clearance,free_target):
    head=FeatureGeometryDecoder(40,3.6,.167584539,width=8)
    with torch.no_grad():
        head.readout.weight.zero_();head.readout.bias.copy_(torch.tensor([clearance/3.6,8.]))
    field,logits,_=head(torch.zeros(1,40,head.raster_size,head.raster_size),
        torch.eye(4,dtype=torch.float64)[None],torch.zeros(1,1,head.raster_size,head.raster_size))
    target=field.detach().clone();target[:,5]=free_target
    gradient=torch.autograd.grad(geometry_reconstruction_loss(field,logits,target),head.readout.bias)[0]
    torch.testing.assert_close(gradient[0],torch.zeros_like(gradient[0]),atol=1e-6,rtol=0)
    assert gradient[1]*(1-2*free_target)>0


def test_geometry_read_and_direct_supervision_reach_both_outputs_without_hard_gates():
    head=FeatureGeometryDecoder(40,3.6,.167584539,width=8)
    with torch.no_grad():
        head.readout.weight.zero_();head.readout.bias.copy_(torch.tensor([.1,-.5]))
    field,logits,configuration=head(torch.zeros(1,40,head.raster_size,head.raster_size),torch.eye(4,dtype=torch.float64)[None],torch.zeros(1,1,head.raster_size,head.raster_size))
    assert logits.shape==(1,2,64,64)
    grad=torch.autograd.grad(field[:,[0,3,5]].sum(),logits,retain_graph=True)[0]
    assert (grad.abs().sum((0,2,3))>0).all()
    torch.testing.assert_close(field[:,3],field[:,4]+(1-field[:,4])*field[:,5],rtol=0,atol=0)
    assert (field[:,3]<=1).all() and (field[:,5]>=0).all()
    target=field.detach().clone();target[:,0]-=.1;target[:,5]=0
    loss=geometry_reconstruction_loss(field,logits,target)
    gradient=torch.autograd.grad(loss,head.readout.bias)[0]
    assert torch.isfinite(gradient).all() and (gradient.abs()>0).all()
    changed=target.clone();changed[:,0]-=.1
    assert geometry_reconstruction_loss(field,logits,changed)>loss


def test_unknown_is_not_supervised_as_free():
    logits=torch.zeros(1,2,64,64,requires_grad=True)
    target=torch.zeros(1,6,64,64);target[:,0]=2.
    field=target.clone();field[:,0]=logits[:,0]*3.6
    loss=geometry_reconstruction_loss(field,logits,target)
    gradient=torch.autograd.grad(loss,logits)[0]
    assert (gradient[:,1]>0).all()  # Gradient descent reduces unknown-space free probability.
    assert (gradient[:,0]<0).all()  # Distance remains independently supervised.


def test_free_supervision_is_stationary_when_consumed_probability_matches_target():
    head=FeatureGeometryDecoder(40,3.6,.167584539,width=8)
    with torch.no_grad():
        head.readout.weight.zero_();head.readout.bias.zero_()
    field,logits,_=head(torch.zeros(1,40,head.raster_size,head.raster_size),
        torch.eye(4,dtype=torch.float64)[None],torch.zeros(1,1,head.raster_size,head.raster_size))
    target=field.detach().clone()
    torch.testing.assert_close(target[:,5],torch.full_like(target[:,5],.5))
    loss=geometry_reconstruction_loss(field,logits,target)
    gradient=torch.autograd.grad(loss,logits)[0]
    torch.testing.assert_close(gradient,torch.zeros_like(gradient),atol=1e-7,rtol=0)


def test_free_loss_keeps_finite_gradient_for_saturated_wrong_predictions():
    logits=torch.zeros(2,2,64,64,requires_grad=True)
    with torch.no_grad():logits[0,1]=1000.;logits[1,1]=-1000.
    target=torch.zeros(2,6,64,64);target[1,5]=1.
    field=target.clone();field[:,0]=logits[:,0]
    gradient=torch.autograd.grad(geometry_reconstruction_loss(field,logits,target),logits)[0]
    assert torch.isfinite(gradient).all()
    assert (gradient[0,1]>0).all() and (gradient[1,1]<0).all()
    assert not gradient[:,0].any()


def test_geometry_labels_match_gf_hits_error_and_radius_without_certifying_unknown():
    from curvenav.data.observed_geometry import field_from_observed_hits
    horizon=3.6;radius=.167584539;error=.013
    tree=cKDTree([[.137,.219],[5.,0.]])
    field=field_from_observed_hits(tree,error,np.zeros((64,64),bool),horizon,radius)
    axis=np.arange(64)*(2*horizon/63)-horizon
    x,y=np.meshgrid(axis,axis,indexing='xy');points=np.stack((x,y),-1)
    expected=np.linalg.norm(points[:,:,None]-tree.data,axis=-1).min(-1)-error-radius
    np.testing.assert_allclose(field[0],expected,atol=3e-7,rtol=0)
    assert (field[0]>0).any() and not field[5].any()
    torch.testing.assert_close(field[1,:,1:-1],(field[0,:,2:]-field[0,:,:-2])/(4*horizon/63))
    known=field_from_observed_hits(tree,error,np.ones((64,64),bool),horizon,radius)
    assert not known[5][known[0]<=0].any()
    assert (known[5][known[0]>0]==1).all()


def test_unified_tokens_read_pregeometry_features_and_keep_far_memory():
    from curvenav.encoders.configuration import splat_memory
    head=FeatureGeometryDecoder(40,3.6,.167584539,width=8)
    points=torch.tensor([[[11.9,0.,0.]]]);features=torch.randn(1,1,40,requires_grad=True)
    raster,mass=splat_memory(features,points,torch.ones(1,1,dtype=torch.bool),3.6,64,head.raster_padding)
    torch.testing.assert_close(mass.sum(),torch.tensor(1.))
    size=head.raster_size
    field,logits,configuration=head(raster.transpose(1,2).reshape(1,40,size,size),torch.eye(4,dtype=torch.float64)[None],mass.transpose(1,2).reshape(1,1,size,size))
    assert configuration.tokens.shape==(1,256,32)
    assert configuration.observed_fraction.max()>0
    assert configuration.metric_position[...,:2].abs().max()>=12.
    source_gradient,geometry_gradient=torch.autograd.grad(configuration.tokens.square().mean(),(features,head.readout.weight),allow_unused=True)
    assert source_gradient.abs().sum()>0 and geometry_gradient is None
