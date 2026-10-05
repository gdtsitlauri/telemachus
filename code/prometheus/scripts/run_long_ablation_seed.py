#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from prometheus.long_horizon import LongHorizonSpec
from prometheus.long_train import LongMethodConfig,run_long_method
p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); a=p.parse_args()
base=dict(seed=a.seed,use_communication=True,comm_top_k=1,use_replay=True,use_ewc=False,use_active_query=True,
          use_ssl=True,use_bc_warmstart=True,query_budget_per_task=16,query_entropy_threshold=1.00,query_coef=0.5)
variants=[
 LongMethodConfig(name='full_budgeted',**base),
 LongMethodConfig(name='minus_ssl',**{**base,'use_ssl':False}),
 LongMethodConfig(name='minus_bc',**{**base,'use_bc_warmstart':False}),
 LongMethodConfig(name='minus_active_query',**{**base,'use_active_query':False,'query_budget_per_task':0}),
 LongMethodConfig(name='minus_replay',**{**base,'use_replay':False,'memory_budget':0}),
 LongMethodConfig(name='minus_communication',**{**base,'use_communication':False,'comm_top_k':0}),
]
rows=[]; details=[]
for c in variants:
 t=time.time(); s,d=run_long_method(LongHorizonSpec(),c); rows.append(s.__dict__);details.append(d)
 print(a.seed,c.name,round(time.time()-t,2),round(s.final_mean_success,4),round(s.final_mean_return,4),round(s.avg_success_forgetting,4),s.queries_used,flush=True)
out=ROOT/'results'/'long_horizon'/'ablation_seeds';out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(out/f'seed_{a.seed:02d}.csv',index=False)
json.dump(details,open(out/f'seed_{a.seed:02d}.json','w'),indent=2)
