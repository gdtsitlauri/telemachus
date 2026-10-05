import torch
from prometheus.long_horizon import LongHorizonSpec, ContinualSignalGrid
from prometheus.policy_memory import PolicyReplay


def test_policy_replay_hard_capacity_and_shapes():
    m=PolicyReplay(7,seed=0)
    for i in range(30):
        obs=torch.randn(3,10)
        probs=torch.softmax(torch.randn(3,3),dim=-1)
        m.add(obs,probs,i%4)
    assert len(m)==7
    obs,probs=m.sample(5)
    assert obs.shape==(5,3,10)
    assert probs.shape==(5,3,3)


def test_expert_solves_no_slip_grid_by_horizon():
    s=LongHorizonSpec(n_tasks=6,horizon=10,slip_prob=0.0,cue_noise=0.0)
    e=ContinualSignalGrid(s,seed=3)
    for task in range(s.n_tasks):
        e.reset(task,64)
        info=None
        for _ in range(s.horizon):
            _,_,_,info=e.step(e.expert_actions())
        assert torch.all(info['success']==1)
