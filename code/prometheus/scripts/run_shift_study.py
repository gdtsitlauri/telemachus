#!/usr/bin/env python3
"""Shift study (configs/shift_study_frozen.yaml) on held-out seeds."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run_seed(args):
    seed, p = args
    import torch
    torch.set_num_threads(1)
    from prometheus.long_horizon import LongHorizonSpec
    from prometheus.long_train import default_long_methods, run_long_method
    from prometheus.shift import run_shift_study
    spec = LongHorizonSpec()
    cfg = [c for c in default_long_methods(seed) if c.name == p["method"]][0]
    _, _, model = run_long_method(spec, cfg, return_model=True)
    r = run_shift_study(model, spec, cfg.comm_top_k, p["guarantee"]["alpha"], p["guarantee"]["delta"], seed,
                        eta=p["online_update"]["eta"], n_chunks=p["deployment"]["chunks"],
                        eps_per_task=p["deployment"]["episodes_per_task_per_chunk"],
                        slip=p["shift"]["slip_prob"], cue_noise=p["shift"]["cue_noise"])
    r["seed"] = seed
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--output", default=str(ROOT / "results/shift_study"))
    a = ap.parse_args()
    p = yaml.safe_load((ROOT / "configs/shift_study_frozen.yaml").read_text())
    out = Path(a.output); (out / "seeds").mkdir(parents=True, exist_ok=True)
    todo = [(s, p) for s in p["heldout_seeds"] if not (out / "seeds" / f"seed_{s:02d}.json").exists()]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(run_seed, todo):
            (out / "seeds" / f"seed_{r['seed']:02d}.json").write_text(json.dumps(r, indent=2))
            print("seed", r["seed"], "done", flush=True)
    res = [json.loads((out / "seeds" / f"seed_{s:02d}.json").read_text()) for s in p["heldout_seeds"]]
    ok = [r for r in res if r.get("certified_nominal")]
    alpha = p["guarantee"]["alpha"]
    summary = {"n_seeds": len(res), "n_certified_nominal": len(ok)}
    for k in ("nominal_static", "shift_static", "shift_online", "shift_full_help", "shift_oracle"):
        f = np.array([r[k]["failure_rate"] for r in ok]); q = np.array([r[k]["queries_per_episode"] for r in ok])
        summary[k] = {"failure_mean": float(f.mean()), "failure_sd": float(f.std(ddof=1)),
                      "queries_mean": float(q.mean()), "seeds_failure_le_alpha": int((f <= alpha).sum())}
        if "failure_rate_second_half" in ok[0][k]:
            f2 = np.array([r[k]["failure_rate_second_half"] for r in ok])
            summary[k]["second_half_failure_mean"] = float(f2.mean())
            summary[k]["seeds_second_half_le_alpha"] = int((f2 <= alpha).sum())
    from scipy.stats import wilcoxon
    d = np.array([r["shift_static"]["failure_rate"] - r["shift_online"]["failure_rate"] for r in ok])
    summary["static_minus_online_failure"] = {"mean": float(d.mean()), "wilcoxon_p": float(wilcoxon(d).pvalue)}
    (out / "summary.json").write_text(json.dumps({"protocol": p, "summary": summary}, indent=2, default=str))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
