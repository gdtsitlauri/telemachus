#!/usr/bin/env python3
from pathlib import Path
import sys, json, argparse
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prometheus.benchmark import BenchmarkSpec
from prometheus.train import MethodConfig, run_method

p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); args=p.parse_args()
base=dict(seed=args.seed,use_replay=True,use_communication=True,use_active_queries=True,comm_top_k=1,
          selective_query=True,query_entropy_threshold=0.65,selective_comm=False)
methods=[
 MethodConfig(name='budget_no_ssl_no_offline',use_ssl=False,use_offline_warmstart=False,**base),
 MethodConfig(name='budget_ssl_only',use_ssl=True,use_offline_warmstart=False,**base),
 MethodConfig(name='budget_offline_only',use_ssl=False,use_offline_warmstart=True,**base),
 MethodConfig(name='prometheus_budgeted',use_ssl=True,use_offline_warmstart=True,**base),
]
rows=[]; details=[]
for c in methods:
 s,d=run_method(BenchmarkSpec(),c); rows.append(s.__dict__); details.append(d); print(s,flush=True)
out=ROOT/'results'; pd.DataFrame(rows).to_csv(out/f'ablation_seed{args.seed}.csv',index=False)
(out/f'ablation_details_seed{args.seed}.json').write_text(json.dumps(details,indent=2))
