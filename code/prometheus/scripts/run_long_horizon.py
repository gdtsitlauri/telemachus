#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prometheus.long_horizon import LongHorizonSpec
from prometheus.long_train import default_long_methods, run_long_method

p=argparse.ArgumentParser()
p.add_argument('--seeds',type=int,default=5)
p.add_argument('--quick',action='store_true')
a=p.parse_args()

spec=LongHorizonSpec()
rows=[]; details=[]
for seed in range(a.seeds):
    methods=default_long_methods(seed)
    if a.quick:
        for c in methods:
            c.updates_per_task=3; c.batch_episodes=12; c.ppo_epochs=1; c.ssl_steps=3; c.bc_steps=4
            c.query_budget_per_task=min(c.query_budget_per_task,8); c.memory_budget=min(c.memory_budget,96)
    for cfg in methods:
        s,d=run_long_method(spec,cfg)
        rows.append(s.__dict__); details.append(d)
        print(seed,cfg.name,s.final_mean_success,s.final_mean_return,s.avg_success_forgetting)
out=ROOT/'results'/'long_horizon'; out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(out/'runs.csv',index=False)
with open(out/'details.json','w') as f: json.dump(details,f,indent=2)
agg=pd.DataFrame(rows).groupby('method').agg(['mean','std','count'])
agg.to_csv(out/'aggregate.csv')
print(agg)
