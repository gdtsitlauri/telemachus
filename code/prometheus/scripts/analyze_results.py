#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'results'/'runs.csv')
p=df.pivot(index='seed',columns='method')
rng=np.random.default_rng(20260927)
comparisons=[
    ('prometheus_budgeted','sparse_comm_active'),
    ('prometheus_budgeted','dense_comm_active'),
    ('sparse_comm_active','replay_comm_sparse'),
]
metrics=['final_mean_accuracy','final_mean_return','avg_forgetting','mean_first_window_accuracy']
out={}
for a,b in comparisons:
    key=f'{a}_vs_{b}'; out[key]={}
    for metric in metrics:
        d=(p[(metric,a)]-p[(metric,b)]).dropna().to_numpy(float)
        bs=np.empty(20000)
        for i in range(len(bs)):
            bs[i]=rng.choice(d,size=len(d),replace=True).mean()
        out[key][metric]={
            'mean_delta':float(d.mean()),
            'bootstrap_95_ci':[float(x) for x in np.quantile(bs,[0.025,0.975])],
            'paired_seed_deltas':[float(x) for x in d],
        }

q_full=df.loc[df.method=='prometheus_budgeted','queries_used'].mean()
q_sparse=df.loc[df.method=='sparse_comm_active','queries_used'].mean()
msg_full=df.loc[df.method=='prometheus_budgeted','messages_per_episode'].mean()
msg_dense=df.loc[df.method=='dense_comm_active','messages_per_episode'].mean()
out['resource_reductions']={
    'expert_query_reduction_vs_sparse_comm_active':float(1-q_full/q_sparse),
    'communication_reduction_vs_dense_comm_active':float(1-msg_full/msg_dense),
    'mean_queries_prometheus':float(q_full),
    'mean_queries_sparse_active':float(q_sparse),
    'messages_prometheus':float(msg_full),
    'messages_dense_active':float(msg_dense),
}
(ROOT/'results'/'paired_stats.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
