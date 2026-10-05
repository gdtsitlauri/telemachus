#!/usr/bin/env python3
"""High-probability online help control under shift (configs/hp_shift_frozen.yaml), held-out seeds."""
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
    from prometheus.hp_online import run_hp_study
    spec = LongHorizonSpec()
    cfg = [c for c in default_long_methods(seed) if c.name == p["method"]][0]
    _, _, model = run_long_method(spec, cfg, return_model=True)
    return run_hp_study(model, spec, cfg.comm_top_k, p["guarantee"]["alpha"], p["guarantee"]["delta"], seed,
                        p["fallback"]["declared_p0"], p["online_update"]["eta"], K,
                        p["shift"]["slip_prob"], p["shift"]["cue_noise"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="pilot seed 0, 20 chunks, not saved as a result")
    ap.add_argument("--output", default=str(ROOT / "results/hp_shift"))
    ap.add_argument("--eta", type=float, default=None, help="sensitivity analysis only: override the frozen eta")
    a = ap.parse_args()
    p = yaml.safe_load((ROOT / "configs/hp_shift_frozen.yaml").read_text())
    if a.eta is not None:
        p["online_update"]["eta"] = a.eta
        p["sensitivity_analysis"] = "eta overridden from the frozen value; not the primary result"
    if a.smoke:
        import time
        t = time.time(); r = run_seed((0, p, 20))
        r.pop("shift_online_hp", None) and None
        print(json.dumps({k: v for k, v in r.items() if k != "shift_online_hp"}, indent=1, default=str), time.time() - t)
        return
    out = Path(a.output); (out / "seeds").mkdir(parents=True, exist_ok=True)
    K = p["deployment"]["chunks"]
    todo = [(s, p, K) for s in p["heldout_seeds"] if not (out / "seeds" / f"seed_{s:02d}.json").exists()]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(run_seed, todo):
            (out / "seeds" / f"seed_{r['seed']:02d}.json").write_text(json.dumps(r, indent=2, default=str))
            print("seed", r["seed"], "done", flush=True)
    res = [json.loads((out / "seeds" / f"seed_{s:02d}.json").read_text()) for s in p["heldout_seeds"]]
    alpha = p["guarantee"]["alpha"]
    ok = [r for r in res if "shift_online_hp" in r]
    f = np.array([r["shift_online_hp"]["failure_rate"] for r in ok])
    summary = {"n_seeds": len(res), "n_run": len(ok),
               "alpha_star_mean": float(np.mean([r["alpha_star"] for r in ok])),
               "online_hp": {"failure_mean": float(f.mean()), "failure_max": float(f.max()),
                             "seeds_failure_le_alpha": int((f <= alpha).sum()),
                             "queries_mean": float(np.mean([r["shift_online_hp"]["queries_per_episode"] for r in ok])),
                             "fallback_fraction_mean": float(np.mean([r["shift_online_hp"]["fallback_fraction"] for r in ok]))},
               "full_help": {"failure_mean": float(np.mean([r["shift_full_help"]["failure_rate"] for r in ok])),
                             "failure_max": float(np.max([r["shift_full_help"]["failure_rate"] for r in ok])),
                             "seeds_le_p0": int(sum(r["shift_full_help"]["failure_rate"] <= p["fallback"]["declared_p0"] for r in ok)),
                             "queries_mean": float(np.mean([r["shift_full_help"]["queries_per_episode"] for r in ok]))},
               "auc": {w: {g: {"mean": float(np.mean([r[w][g] for r in res])), "sd": float(np.std([r[w][g] for r in res], ddof=1))}
                           for g in ("entropy", "margin", "learned")} for w in ("auc_nominal", "auc_shift")}}
    (out / "summary.json").write_text(json.dumps({"protocol": p, "summary": summary}, indent=2, default=str))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
