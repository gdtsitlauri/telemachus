#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'/'long_horizon';F=ROOT/'figures'/'long_horizon';F.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(R/'heldout_runs.csv')
order=['ppo_local_ft','ppo_sparse_ewc','ppo_sparse_replay','ppo_dense_replay_active','prometheus_v2_budgeted']
labels=['Local FT','Sparse EWC','Sparse Replay','Dense + Active','PROMETHEUS']
for metric,title,ylabel,name in [
 ('final_mean_success','Held-out final team success','Success rate','heldout_success.png'),
 ('final_mean_return','Held-out final return','Return','heldout_return.png'),
 ('avg_success_forgetting','Held-out continual forgetting','Success forgetting','heldout_forgetting.png'),
 ('mean_early_return','Held-out early adaptation','Early return','heldout_early_return.png')]:
 g=df.groupby('method')[metric].agg(['mean','std']).reindex(order)
 fig,ax=plt.subplots(figsize=(8,4.8));ax.bar(labels,g['mean'],yerr=g['std'],capsize=4);ax.set_title(title);ax.set_ylabel(ylabel);ax.tick_params(axis='x',rotation=25);fig.tight_layout();fig.savefig(F/name,dpi=160);plt.close(fig)
# resources
q=df.groupby('method')['queries_used'].mean().reindex(order);m=df.groupby('method')['messages_per_episode'].mean().reindex(order)
for series,title,ylabel,name in [(q,'Expert-query use','Queries across task stream','heldout_queries.png'),(m,'Communication use','Messages per episode','heldout_messages.png')]:
 fig,ax=plt.subplots(figsize=(8,4.8));ax.bar(labels,series);ax.set_title(title);ax.set_ylabel(ylabel);ax.tick_params(axis='x',rotation=25);fig.tight_layout();fig.savefig(F/name,dpi=160);plt.close(fig)
# query sweep
p=R/'query_sweep_runs.csv'
if p.exists():
 x=pd.read_csv(p);g=x.groupby('method').agg(success=('final_mean_success','mean'),ret=('final_mean_return','mean'),queries=('queries_used','mean')).sort_values('queries')
 fig,ax=plt.subplots(figsize=(6.5,4.8));ax.plot(g['queries'],g['success'],marker='o');ax.set_xlabel('Mean expert queries');ax.set_ylabel('Final success');ax.set_title('Query-budget trade-off');fig.tight_layout();fig.savefig(F/'query_budget_tradeoff.png',dpi=160);plt.close(fig)
# communication sweep
p=R/'comm_sweep_runs.csv'
if p.exists():
 x=pd.read_csv(p);g=x.groupby('method').agg(success=('final_mean_success','mean'),ret=('final_mean_return','mean'),messages=('messages_per_episode','mean')).sort_values('messages')
 fig,ax=plt.subplots(figsize=(6.5,4.8));ax.plot(g['messages'],g['success'],marker='o');ax.set_xlabel('Messages per episode');ax.set_ylabel('Final success');ax.set_title('Communication-budget trade-off');fig.tight_layout();fig.savefig(F/'communication_tradeoff.png',dpi=160);plt.close(fig)
