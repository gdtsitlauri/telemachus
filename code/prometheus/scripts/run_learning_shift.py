#!/usr/bin/env python3
"""Guaranteed online help control with a learning gate (configs/learning_shift_frozen.yaml)."""
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
    seed, p, K = args
    import torch
    torch.set_num_threads(1)
    from prometheus.long_horizon import LongHorizonSpec
    from prometheus.long_train import default_long_methods, run_long_method
    from prometheus.learning_online import run_learning_study
    spec = LongHorizonSpec()
    cfg = [c for c in default_long_methods(seed) if c.name == p["method"]][0]
    _, _, model = run_long_method(spec, cfg, return_model=True)
    return run_learning_study(model, spec, cfg.comm_top_k, p["guarantee"]["alpha"], p["guarantee"]["delta"], seed,
                              p["fallback"]["declared_p0"], p["online_update"]["eta"], K,
                              p["shift"]["slip_prob"], p["shift"]["cue_noise"], p["learning"]["retrain_every_chunks"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output", default=str(ROOT / "results/learning_shift"))
    a = ap.parse_args()
    p = yaml.safe_load((ROOT / "configs/learning_shift_frozen.yaml").read_text())
    if a.smoke:
        r = run_seed((0, p, p["deployment"]["chunks"]))
        print({k: v for k, v in r.items() if k != "shift_online_learning"},
              {k: v for k, v in r.get("shift_online_learning", {}).items() if k != "trace"})
        return
    out = Path(a.output); (out / "seeds").mkdir(parents=True, exist_ok=True)
    K = p["deployment"]["chunks"]
    todo = [(s, p, K) for s in p["heldout_seeds"] if not (out / "seeds" / f"seed_{s:02d}.json").exists()]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(run_seed, todo):
            (out / "seeds" / f"seed_{r['seed']:02d}.json").write_text(json.dumps(r, indent=2, default=str))
            print("seed", r["seed"], "done", flush=True)
    res = [json.loads((out / "seeds" / f"seed_{s:02d}.json").read_text()) for s in p["heldout_seeds"]]
    ok = [r for r in res if "shift_online_learning" in r]
    L = [r["shift_online_learning"] for r in ok]
    f = np.array([x["failure_rate"] for x in L])
    summary = {"n_run": len(ok), "failure_mean": float(f.mean()), "failure_max": float(f.max()),
               "seeds_failure_le_alpha": int((f <= p["guarantee"]["alpha"]).sum()),
               "queries_mean": float(np.mean([x["queries_per_episode"] for x in L])),
               "queries_second_half_mean": float(np.mean([x["queries_second_half"] for x in L])),
               "fallback_fraction_mean": float(np.mean([x["fallback_fraction"] for x in L])),
               "auc_shift_final_gate": {g: float(np.mean([x["auc_shift_final_gate"][g] for x in L])) for g in ("learned", "entropy")}}
    (out / "summary.json").write_text(json.dumps({"protocol": p, "summary": summary}, indent=2, default=str))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
