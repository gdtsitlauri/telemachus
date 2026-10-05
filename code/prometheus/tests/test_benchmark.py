import torch
from prometheus.benchmark import BenchmarkSpec, ContinualCoordinationBandit


def test_sample_shapes_and_labels():
    spec=BenchmarkSpec(n_tasks=4,task_context_dim=4)
    env=ContinualCoordinationBandit(spec,seed=1)
    obs,y=env.sample(2,11)
    assert obs.shape==(11,spec.n_agents,spec.obs_dim)
    assert y.shape==(11,)
    assert set(y.tolist()).issubset({0,1})


def test_team_reward_bounds_and_perfect_team():
    spec=BenchmarkSpec(n_tasks=2,task_context_dim=2)
    env=ContinualCoordinationBandit(spec)
    _,y=env.sample(0,10)
    perfect=y[:,None].expand(-1,spec.n_agents)
    r=env.team_reward(perfect,y)
    assert torch.allclose(r,torch.ones_like(r))
    wrong=1-perfect
    r2=env.team_reward(wrong,y)
    assert torch.all(r2==0)
