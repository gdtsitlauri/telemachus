import math
import torch
from prometheus.benchmark import BenchmarkSpec
from prometheus.model import CollectivePolicy
from prometheus.train import MethodConfig, nt_xent, run_method


def test_nt_xent_finite():
    m=CollectivePolicy(10,16)
    x=torch.randn(16,10)
    z1=m.ssl_projection(x+0.01*torch.randn_like(x))
    z2=m.ssl_projection(x+0.01*torch.randn_like(x))
    loss=nt_xent(z1,z2)
    assert torch.isfinite(loss)


def test_quick_end_to_end_respects_query_budget():
    spec=BenchmarkSpec(n_tasks=3,task_context_dim=3)
    cfg=MethodConfig(
        name='test',seed=0,ssl_steps=2,warmstart_steps=2,
        online_steps_per_task=7,batch_size=16,replay_batch=8,
        memory_budget=20,query_budget_per_task=5,selective_query=True,
        query_entropy_threshold=0.65,
    )
    summary,detail=run_method(spec,cfg)
    assert 0.0<=summary.final_mean_accuracy<=1.0
    assert 0.0<=summary.final_mean_return<=1.0
    assert summary.queries_used<=spec.n_tasks*cfg.query_budget_per_task
    assert summary.memory_items<=cfg.memory_budget
    assert len(detail['eval_history'])==spec.n_tasks
