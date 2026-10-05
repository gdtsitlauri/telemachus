#!/usr/bin/env python3
"""Certified human-in-the-loop help experiment (configs/certified_help_frozen.yaml).

For every held-out seed and method: train through all continual tasks, train
the learned gate on its own stream, calibrate every gate with Learn-then-Test
and test it on a disjoint stream. Writes one JSON per seed plus an aggregate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run_seed(args):
    seed, protocol = args
    import torch
    from prometheus.long_horizon import LongHorizonSpec
    from prometheus.long_train import default_long_methods, run_long_method
    from prometheus.certified_help import evaluate_certified
    torch.set_num_threads(1)
    spec = LongHorizonSpec()
    g = protocol["guarantee"]; e = protocol["episodes"]
    out = {"seed": seed, "methods": {}}
    for cfg in default_long_methods(seed):
        if cfg.name not in protocol["methods"]:
            continue
        t = time.time()
        summary, _, model = run_long_method(spec, cfg, return_model=True)
        top_k = cfg.comm_top_k if cfg.use_communication else 0
        res = evaluate_certified(model, spec, top_k, g["alpha"], g["delta"],
                                 cal_stream=900000 + 10 * seed, test_stream=950000 + 10 * seed,
                                 gate_stream=700000 + 10 * seed,
                                 episodes_per_task_cal=e["calibration_per_task"],
                                 episodes_per_task_test=e["test_per_task"], gates=tuple(protocol["gates"]))
        res["training_summary"] = summary.__dict__
        res["seconds"] = time.time() - t
        out["methods"][cfg.name] = res
    return out


def aggregate(results, alpha):
    rows = []
    for r in results:
        for m, res in r["methods"].items():
            for gate in ("learned", "entropy", "margin", "random"):
                e = res.get(gate)
                if not e:
                    continue
                t = e.get("test", {})
                rows.append({"seed": r["seed"], "method": m, "gate": gate, "certified": e["certified"],
                             "theta": e["calibration"]["selected_theta"],
                             "test_failure": t.get("failure_rate"), "queries_per_episode": t.get("queries_per_episode"),
                             "within_alpha": t.get("within_alpha"),
                             "oracle_queries": res["oracle_help"]["queries_per_episode"],
                             "full_help_failure": res["full_help"]["failure_rate"],
                             "no_help_failure": res["no_help"]["failure_rate"]})
    summary = {}
    methods = sorted({x["method"] for x in rows})
    rng = np.random.default_rng(2026)
    for m in methods:
        summary[m] = {}
        for gate in ("learned", "entropy", "margin", "random"):
            sel = [x for x in rows if x["method"] == m and x["gate"] == gate and x["certified"]]
            if not sel:
                continue
            q = np.array([x["queries_per_episode"] for x in sel]); f = np.array([x["test_failure"] for x in sel])
            summary[m][gate] = {"n_seeds_certified": len(sel), "queries_mean": float(q.mean()),
                                "test_failure_mean": float(f.mean()),
                                "seeds_within_alpha": int(sum(x["within_alpha"] for x in sel))}
        # Paired learned vs entropy queries at the certified level.
        pairs = [(x["queries_per_episode"], y["queries_per_episode"]) for x in rows for y in rows
                 if x["method"] == m == y["method"] and x["seed"] == y["seed"] and x["gate"] == "learned"
                 and y["gate"] == "entropy" and x["certified"] and y["certified"]]
        if pairs:
            d = np.array([a - b for a, b in pairs])
            boot = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
            from scipy.stats import wilcoxon
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            summary[m]["learned_minus_entropy_queries"] = {"mean": float(d.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
                                                           "wilcoxon_p": p, "n_pairs": len(d)}
        oq = [x["oracle_queries"] for x in rows if x["method"] == m and x["gate"] == "learned"]
        summary[m]["oracle_queries_mean"] = float(np.mean(oq)) if oq else None
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=str(ROOT / "configs/certified_help_frozen.yaml"))
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--output", default=str(ROOT / "results/certified_help"))
    a = ap.parse_args()
    protocol = yaml.safe_load(Path(a.protocol).read_text())
    seeds = a.seeds if a.seeds else protocol["heldout_seeds"]
    out = Path(a.output); (out / "seeds").mkdir(parents=True, exist_ok=True)
    tasks = [(s, protocol) for s in seeds if not (out / "seeds" / f"seed_{s:02d}.json").exists()]
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(run_seed, tasks):
                (out / "seeds" / f"seed_{r['seed']:02d}.json").write_text(json.dumps(r, indent=2))
                print("seed", r["seed"], "done", flush=True)
    else:
        for t in tasks:
            r = run_seed(t)
            (out / "seeds" / f"seed_{r['seed']:02d}.json").write_text(json.dumps(r, indent=2))
            print("seed", r["seed"], "done", flush=True)
    results = [json.loads((out / "seeds" / f"seed_{s:02d}.json").read_text()) for s in seeds]
    rows, summary = aggregate(results, protocol["guarantee"]["alpha"])
    (out / "runs.json").write_text(json.dumps(rows, indent=2))
    (out / "summary.json").write_text(json.dumps({"protocol": protocol, "summary": summary}, indent=2, default=str))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
