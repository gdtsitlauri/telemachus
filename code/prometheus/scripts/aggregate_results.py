#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
files=sorted((ROOT/'results').glob('runs_seed*.csv'))
if not files: raise SystemExit('no runs_seed*.csv found')
df=pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(ROOT/'results'/'runs.csv', index=False)
metrics=['final_mean_accuracy','final_mean_return','avg_forgetting','mean_first_window_accuracy','queries_used','messages_per_episode','memory_items']
agg=df.groupby('method')[metrics].agg(['mean','std','count'])
agg.to_csv(ROOT/'results'/'aggregate.csv')
# flat, presentation-friendly table
flat=df.groupby('method')[metrics].agg(['mean','std']).reset_index()
flat.columns=['method']+[f'{a}_{b}' for a,b in flat.columns.tolist()[1:]]
flat.to_csv(ROOT/'results'/'summary.csv', index=False)
# merge details
all_details=[]
for f in sorted((ROOT/'results').glob('details_seed*.json')):
    all_details.extend(json.loads(f.read_text()))
(ROOT/'results'/'details.json').write_text(json.dumps(all_details,indent=2))
print(agg)
