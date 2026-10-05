import torch
from prometheus.model import CollectivePolicy


def test_sparse_edge_budget():
    m=CollectivePolicy(obs_dim=12,hidden_dim=16)
    x=torch.randn(7,3,12)
    logits,edges=m(x,top_k=1)
    assert logits.shape==(7,3,2)
    assert edges.shape==(7,3,3)
    assert torch.all(edges.sum(-1)==1)


def test_dense_edge_budget():
    m=CollectivePolicy(obs_dim=12,hidden_dim=16)
    x=torch.randn(5,3,12)
    _,edges=m(x,top_k=2)
    assert torch.all(edges.sum(-1)==2)


def test_no_communication_has_no_edges():
    m=CollectivePolicy(obs_dim=12,hidden_dim=16)
    x=torch.randn(5,3,12)
    _,edges=m(x,top_k=0)
    assert int(edges.sum())==0


def test_ssl_projection_is_normalized():
    m=CollectivePolicy(obs_dim=12,hidden_dim=16)
    z=m.ssl_projection(torch.randn(20,12))
    assert torch.allclose(z.norm(dim=-1),torch.ones(20),atol=1e-5)
