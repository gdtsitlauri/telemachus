#!/usr/bin/env python3
from pathlib import Path
import sys, json, argparse
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prometheus.benchmark import BenchmarkSpec
from prometheus.train import default_methods, run_method

p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--device',default='cpu'); args=p.parse_args()
rows=[]; details=[]
for cfg in default_methods(args.seed):
    print(f'[seed {args.seed}] {cfg.name}', flush=True)
    s,d=run_method(BenchmarkSpec(),cfg,args.device)
    rows.append(s.__dict__); details.append(d)
out=ROOT/'results'; out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out/f'runs_seed{args.seed}.csv',index=False)
with open(out/f'details_seed{args.seed}.json','w') as f: json.dump(details,f,indent=2)
