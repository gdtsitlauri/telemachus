#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prometheus.long_horizon import LongHorizonSpec
from prometheus.long_train import LongMethodConfig,run_long_method
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
base=dict(seed=a.seed,use_communication=True,comm_top_k=1,use_replay=True,use_ewc=False,use_active_query=True,
          use_ssl=True,use_bc_warmstart=True,query_budget_per_task=16,query_entropy_threshold=1.0,query_coef=.5,memory_budget=384)
configs=[]
for q in [0,8,16,32]:
    configs.append(LongMethodConfig(name=f'query_budget_{q}',**{**base,'use_active_query':q>0,'query_budget_per_task':q}))
for m in [64,128,384,768]:
    configs.append(LongMethodConfig(name=f'memory_budget_{m}',**{**base,'memory_budget':m}))
for k in [0,1,2]:
    configs.append(LongMethodConfig(name=f'comm_topk_{k}',**{**base,'use_communication':k>0,'comm_top_k':k}))
# de-duplicate identical named configs is unnecessary; names preserve sweep families.
rows=[]
for c in configs:
    s,_=run_long_method(LongHorizonSpec(),c);r=s.__dict__;rows.append(r);print(a.seed,c.name,round(s.final_mean_success,3),round(s.final_mean_return,3),s.queries_used,s.messages_per_episode,flush=True)
out=ROOT/'results'/'long_horizon'/'budget_sweep_seeds';out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(out/f'seed_{a.seed:02d}.csv',index=False)
