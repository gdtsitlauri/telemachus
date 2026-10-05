#!/usr/bin/env python3
"""Robustness of the language-cascade results over random calibration/test splits.

The main analysis uses one fixed split (seed 2026, 600 calibration / 719 test).
Here the same certified pipeline is repeated over many random splits; for each
split and alpha it records test error and cost of every gate and of the
certified fixed stage. Learned-gate scores are cached as in llm_eval.py.
Usage: python analysis/llm_robustness.py <results subdir> [n_splits]
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import llm_eval  # noqa: E402  (patches cascade.score / train_learned_gate with a cache)
from llm_eval import cascade  # noqa: E402
from telemachus.calibration import binomial_pvalue  # noqa: E402

_SPLIT_SEED = [2026]


def seeded_split(table, n_cal, seed=2026):
    rows = sorted(table, key=lambda r: r["id"])
    random.Random(_SPLIT_SEED[0]).shuffle(rows)
    return rows[:n_cal], rows[n_cal:]


cascade.split_test_table = seeded_split


def fixed_certified(cal, test, alpha, delta=.1):
    st = cascade.stages_of(test[0])
    ok = []
    for s in st:
        w = sum(not r[s]["correct"] for r in cal)
        if binomial_pvalue(w, len(cal), alpha) <= delta / len(st):
            ok.append((np.mean([cascade.stage_cost(r, s) for r in cal]), s))
    if not ok:
        return {"stage": "EXPERT", "error": 0., "cost": cascade.DEFAULT_EXPERT_COST}
    s = min(ok)[1]
    f = cascade.fixed_stage(test, s)
    return {"stage": s, "error": f["error"], "cost": f["cost"]}


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "llm"
    n_splits = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    d = ROOT / "results" / sub
    train = cascade.load_table(d / "train_table.jsonl")
    test = cascade.load_table(d / "test_table.jsonl")
    llm_eval._TABLE.extend(test)
    runs = []
    for k in range(n_splits):
        _SPLIT_SEED[0] = 10_000 + k
        for alpha in (.3, .2):
            r = cascade.evaluate(train, test, alpha=alpha, delta=.1, n_cal=600)
            cal, te = seeded_split(test, 600)
            row = {"split": k, "alpha": alpha, "fixed_certified": fixed_certified(cal, te, alpha)}
            for g in ("learned", "confidence", "agreement"):
                t = r["gates"][g].get("test")
                row[g] = None if t is None else {"error": t["error"], "cost": t["cost"], "within_alpha": t["within_alpha"]}
            runs.append(row)
        print(f"split {k + 1}/{n_splits}", flush=True)
    summ = {}
    for alpha in (.3, .2):
        rs = [r for r in runs if r["alpha"] == alpha]
        e = {}
        for g in ("learned", "confidence", "agreement"):
            v = [r[g] for r in rs if r[g]]
            e[g] = {"certified_splits": len(v), "within_alpha_splits": sum(x["within_alpha"] for x in v),
                    "cost_mean": float(np.mean([x["cost"] for x in v])),
                    "cost_p5_p95": np.percentile([x["cost"] for x in v], [5, 95]).tolist(),
                    "error_mean": float(np.mean([x["error"] for x in v]))}
        f = [r["fixed_certified"] for r in rs]
        e["fixed_certified"] = {"stages": {s: sum(x["stage"] == s for x in f) for s in {x["stage"] for x in f}},
                                "cost_mean": float(np.mean([x["cost"] for x in f])),
                                "error_mean": float(np.mean([x["error"] for x in f])),
                                "within_alpha_splits": int(sum(x["error"] <= alpha for x in f))}
        both = [r for r in rs if r["learned"] and r["confidence"]]
        ratio = [r["confidence"]["cost"] / r["learned"]["cost"] for r in both]
        e["learned_cheaper_than_confidence"] = f"{sum(x > 1 for x in ratio)}/{len(both)}"
        e["confidence_over_learned_cost_ratio"] = {"mean": float(np.mean(ratio)), "p5_p95": np.percentile(ratio, [5, 95]).tolist()}
        lf = [r["fixed_certified"]["cost"] / r["learned"]["cost"] for r in rs if r["learned"]]
        e["learned_cheaper_than_fixed_certified"] = f"{sum(x > 1 for x in lf)}/{len(lf)}"
        summ[f"alpha_{alpha}"] = e
    (d / "robustness_splits.json").write_text(json.dumps({"n_splits": n_splits, "summary": summ, "runs": runs}, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
