#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'results'/'runs.csv')
summary=df.groupby('method').agg(
    final_mean_accuracy=('final_mean_accuracy','mean'),
    accuracy_std=('final_mean_accuracy','std'),
    avg_forgetting=('avg_forgetting','mean'),
    forgetting_std=('avg_forgetting','std'),
    early_accuracy=('mean_first_window_accuracy','mean'),
    queries=('queries_used','mean'),
    messages=('messages_per_episode','mean'),
).reset_index()
order=['sequential_local','replay_local','replay_comm_sparse','sparse_comm_active','dense_comm_active','prometheus_budgeted']
summary['ord']=summary.method.map({m:i for i,m in enumerate(order)})
summary=summary.sort_values('ord')

plots=[
 ('final_mean_accuracy','accuracy_std','Final mean accuracy','final_mean_accuracy.png'),
 ('avg_forgetting','forgetting_std','Average forgetting (lower is better)','avg_forgetting.png'),
 ('early_accuracy',None,'Early task accuracy','early_task_accuracy.png'),
 ('queries',None,'Mean expert-label queries','expert_queries.png'),
 ('messages',None,'Messages per episode','messages_per_episode.png'),
]
for metric,err,ylabel,name in plots:
    fig,ax=plt.subplots(figsize=(9,4.8))
    yerr=summary[err] if err else None
    ax.bar(summary.method,summary[metric],yerr=yerr,capsize=3 if err else 0)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x',rotation=25)
    fig.tight_layout()
    fig.savefig(ROOT/'figures'/name,dpi=180)
    plt.close(fig)
