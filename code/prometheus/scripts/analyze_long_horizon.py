#!/usr/bin/env python3
from __future__ import annotations
import glob,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'/'long_horizon'
files=sorted(glob.glob(str(R/'heldout_seeds'/'seed_*.csv')))
if not files: raise SystemExit('no heldout seed files')
df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True).sort_values(['method','seed'])
df.to_csv(R/'heldout_runs.csv',index=False)
cols=['final_mean_success','final_mean_return','final_mean_distance','avg_success_forgetting','avg_return_forgetting','mean_early_return','queries_used','messages_per_episode','memory_items']
df.groupby('method')[cols].agg(['mean','std','count']).to_csv(R/'heldout_aggregate.csv')

rng=np.random.default_rng(20260927)
def ci(x,B=30000):
 x=np.asarray(x,float); n=len(x); idx=rng.integers(0,n,size=(B,n)); means=x[idx].mean(1); return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]
def paired(base,other):
 out={}
 A=df[df.method==base].sort_values('seed'); B=df[df.method==other].sort_values('seed')
 assert A.seed.tolist()==B.seed.tolist()
 for m in ['final_mean_success','final_mean_return','avg_success_forgetting','avg_return_forgetting','mean_early_return','queries_used','messages_per_episode']:
  d=A[m].to_numpy(float)-B[m].to_numpy(float)
  try: p=float(wilcoxon(d,alternative='two-sided').pvalue) if np.any(np.abs(d)>1e-12) else 1.0
  except Exception: p=float('nan')
  out[m]={'mean_delta':float(d.mean()),'bootstrap_95_ci':ci(d),'wilcoxon_two_sided_p':p,'paired_seed_deltas':[float(x) for x in d]}
 return out
base='prometheus_v2_budgeted'; stats={}
for o in ['ppo_dense_replay_active','ppo_sparse_replay','ppo_local_ft','ppo_sparse_ewc']:
 stats[f'{base}_vs_{o}']=paired(base,o)
P=df[df.method==base]; D=df[df.method=='ppo_dense_replay_active']
stats['resource_summary']={
 'heldout_seed_start':int(df.seed.min()),'heldout_seed_end':int(df.seed.max()),'n_heldout_seeds':int(df.seed.nunique()),
 'mean_queries_prometheus':float(P.queries_used.mean()),'mean_queries_dense_active':float(D.queries_used.mean()),
 'query_reduction_vs_dense_active':float(1-P.queries_used.mean()/D.queries_used.mean()),
 'messages_prometheus':float(P.messages_per_episode.mean()),'messages_dense_active':float(D.messages_per_episode.mean()),
 'communication_reduction_vs_dense_active':float(1-P.messages_per_episode.mean()/D.messages_per_episode.mean()),
}
json.dump(stats,open(R/'heldout_paired_stats.json','w'),indent=2)
open(R/'heldout_paired_stats.txt','w').write(json.dumps(stats,indent=2))

# ablation aggregation
ab=sorted(glob.glob(str(R/'ablation_seeds'/'seed_*.csv')))
if ab:
 adf=pd.concat([pd.read_csv(f) for f in ab],ignore_index=True);adf.to_csv(R/'ablation_runs.csv',index=False)
 adf.groupby('method')[cols].agg(['mean','std','count']).to_csv(R/'ablation_aggregate.csv')
# sweeps
for kind in ['query','comm']:
 fs=sorted(glob.glob(str(R/f'{kind}_sweep_seeds'/'seed_*.csv')))
 if fs:
  x=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True);x.to_csv(R/f'{kind}_sweep_runs.csv',index=False)
  x.groupby('method')[cols].agg(['mean','std','count']).to_csv(R/f'{kind}_sweep_aggregate.csv')
print(json.dumps(stats['resource_summary'],indent=2))
