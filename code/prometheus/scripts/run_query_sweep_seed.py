#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prometheus.long_horizon import LongHorizonSpec
from prometheus.long_train import LongMethodConfig,run_long_method
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
base=dict(seed=a.seed,use_communication=True,comm_top_k=1,use_replay=True,use_ewc=False,use_ssl=True,use_bc_warmstart=True,
          query_entropy_threshold=1.0,query_coef=.5,memory_budget=384)
rows=[]
for q in [0,8,16,32]:
 c=LongMethodConfig(name=f'query_budget_{q}',use_active_query=q>0,query_budget_per_task=q,**base)
 s,_=run_long_method(LongHorizonSpec(),c);rows.append(s.__dict__);print(a.seed,c.name,s.final_mean_success,s.final_mean_return,s.queries_used,flush=True)
out=ROOT/'results'/'long_horizon'/'query_sweep_seeds';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/f'seed_{a.seed:02d}.csv',index=False)
