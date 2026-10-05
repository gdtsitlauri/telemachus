#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from prometheus.long_horizon import LongHorizonSpec
from prometheus.long_train import default_long_methods,run_long_method
p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); a=p.parse_args()
spec=LongHorizonSpec(); rows=[]; details=[]
for cfg in default_long_methods(a.seed):
    t=time.time(); s,d=run_long_method(spec,cfg); rows.append(s.__dict__); details.append(d)
    print(a.seed,cfg.name,round(time.time()-t,2),round(s.final_mean_success,4),round(s.final_mean_return,4),flush=True)
out=ROOT/'results'/'long_horizon'/'seeds'; out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(out/f'seed_{a.seed:02d}.csv',index=False)
with open(out/f'seed_{a.seed:02d}.json','w') as f: json.dump(details,f,indent=2)
