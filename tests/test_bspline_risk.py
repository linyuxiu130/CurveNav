"""Parallel exact nearest queries preserve the observed-risk value and gradient."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from curvenav.data.bspline_replay import observed_risk


def test_parallel_nearest_risk_matches_serial_values_and_gradients():
    rng=np.random.default_rng(51)
    points=rng.normal(size=(4096,2))*.3
    tree=cKDTree(np.concatenate((points,points[:16])))
    class SerialTree:
        data=tree.data
        def query(self,query,*,workers):return tree.query(query,workers=1)
    torch.manual_seed(15)
    original=(torch.randn(4,14)*.08).requires_grad_()
    parallel=original.detach().clone().requires_grad_()
    expected,details=observed_risk(original,SerialTree(),.0073,.167584539)
    actual,result=observed_risk(parallel,tree,.0073,.167584539)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for name in details:torch.testing.assert_close(result[name],details[name],rtol=0,atol=0)
    expected.backward();actual.backward()
    torch.testing.assert_close(parallel.grad,original.grad,rtol=0,atol=0)
def test_bounded_nearest_keeps_exact_gf_values_and_gradients():
    from types import SimpleNamespace
    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    from curvenav.training.bspline_objective import _nearest_sites,_risk_from_sites
    from curvenav.data.bspline_replay import SAMPLES

    torch.manual_seed(47)
    paths=torch.randn(1,SAMPLES,2,dtype=torch.float64)
    paths[:,::10]*=1e6
    paths.requires_grad_()
    tree=cKDTree(np.array([[0.,0.],[1.,1.],[2.,-1.]]))
    records=[(dict(point_error_m=.01),None,tree,None)]
    queries=paths.detach().numpy();bounds=np.array([.26])
    exact=np.stack([tree.data[tree.query(queries[0])[1]]])
    bounded=_nearest_sites(queries,records,bounds)
    actor=SimpleNamespace(head=SimpleNamespace(radius=.2))
    values=torch.zeros(1,1,14);pad=torch.tensor([.05],dtype=torch.float64)
    old=_risk_from_sites(actor,values,paths,pad,exact,records)
    new=_risk_from_sites(actor,values,paths,pad,bounded,records)
    torch.testing.assert_close(new,old,rtol=0,atol=0)
    a=torch.autograd.grad(old,paths)[0];b=torch.autograd.grad(new,paths)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)
