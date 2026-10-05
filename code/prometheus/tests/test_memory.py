import torch
from prometheus.memory import BudgetedReplay


def test_memory_hard_capacity():
    m=BudgetedReplay(8,seed=0)
    for i in range(100):
        m.add(torch.zeros(3,5)+i,i%2,i%4,priority=float(i%3))
    assert len(m)==8


def test_memory_sample_shapes():
    m=BudgetedReplay(10,seed=0)
    for i in range(6):
        m.add(torch.randn(3,7),i%2,0)
    obs,y=m.sample(4)
    assert obs.shape==(4,3,7)
    assert y.shape==(4,)
