#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prometheus.long_horizon import LongHorizonSpec
from prometheus.long_train import LongMethodConfig,run_long_method
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
base=dict(seed=a.seed,use_replay=True,use_ewc=False,use_active_query=True,use_ssl=True,use_bc_warmstart=True,
          query_budget_per_task=16,query_entropy_threshold=1.0,query_coef=.5,memory_budget=384)
rows=[]
for k in [0,1,2]:
 c=LongMethodConfig(name=f'comm_topk_{k}',use_communication=k>0,comm_top_k=k,**base)
 s,_=run_long_method(LongHorizonSpec(),c);rows.append(s.__dict__);print(a.seed,c.name,s.final_mean_success,s.final_mean_return,s.messages_per_episode,flush=True)
out=ROOT/'results'/'long_horizon'/'comm_sweep_seeds';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/f'seed_{a.seed:02d}.csv',index=False)
