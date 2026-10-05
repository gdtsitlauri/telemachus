import torch
from prometheus.long_horizon import LongHorizonSpec, ContinualSignalGrid
from prometheus.long_model import LongHorizonActorCritic
from prometheus.long_train import LongMethodConfig, run_long_method


def test_long_env_shapes_and_expert_moves():
    s=LongHorizonSpec(n_tasks=3,horizon=5)
    e=ContinualSignalGrid(s,seed=0)
    obs,state=e.reset(1,12)
    assert obs.shape==(12,s.n_agents,s.obs_dim)
    assert state.shape==(12,s.state_dim)
    a=e.expert_actions()
    assert a.shape==(12,s.n_agents)
    _,_,r,info=e.step(a)
    assert r.shape==(12,)
    assert info['success'].shape==(12,)


def test_long_model_shapes_and_sparse_edges():
    s=LongHorizonSpec(n_tasks=3)
    m=LongHorizonActorCritic(s.obs_dim,s.state_dim,s.n_actions,32)
    e=ContinualSignalGrid(s,seed=1)
    obs,state=e.reset(0,7)
    logits,v,edges=m(obs,state,top_k=1)
    assert logits.shape==(7,s.n_agents,s.n_actions)
    assert v.shape==(7,)
    assert torch.all(edges.sum(-1)==1)


def test_quick_long_training_respects_budgets():
    s=LongHorizonSpec(n_tasks=2,horizon=5)
    c=LongMethodConfig(seed=0,hidden_dim=32,updates_per_task=2,batch_episodes=8,ppo_epochs=1,
                       ssl_steps=2,bc_steps=2,memory_budget=30,replay_batch=8,
                       query_budget_per_task=4)
    summary,detail=run_long_method(s,c)
    assert 0<=summary.final_mean_success<=1
    assert summary.queries_used<=s.n_tasks*c.query_budget_per_task
    assert summary.memory_items<=c.memory_budget
    assert len(detail['history'])==s.n_tasks
