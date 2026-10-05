#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prometheus.benchmark import BenchmarkSpec
from prometheus.train import default_methods, run_method


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--quick", action="store_true", help="short smoke experiment")
    args = p.parse_args()

    spec = BenchmarkSpec()
    rows, details = [], []
    for seed in args.seeds:
        methods = default_methods(seed)
        if args.methods:
            methods = [m for m in methods if m.name in args.methods]
        for cfg in methods:
            if args.quick:
                cfg.ssl_steps = min(cfg.ssl_steps, 5)
                cfg.warmstart_steps = min(cfg.warmstart_steps, 5)
                cfg.online_steps_per_task = 8
                cfg.batch_size = 24
            print(f"running {cfg.name} seed={seed}", flush=True)
            summary, detail = run_method(spec, cfg, device=args.device)
            rows.append(summary.__dict__)
            details.append(detail)
            print(summary, flush=True)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
    with open(out / "details.json", "w") as f:
        json.dump(details, f, indent=2)
    if rows:
        df = pd.DataFrame(rows)
        agg = df.groupby("method").agg(["mean", "std"])
        agg.to_csv(out / "aggregate.csv")
        print("\n", agg)


if __name__ == "__main__":
    main()
