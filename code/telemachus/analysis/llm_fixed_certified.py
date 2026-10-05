#!/usr/bin/env python3
"""Certified single-stage baseline and per-stage gate AUCs for the language cascade.

Certified fixed stage: every stage (and EXPERT) is one hypothesis 'error of
always answering with this stage <= alpha', tested with the exact binomial
p-value on the same calibration split as the cascade gates; among certified
stages the one with the lowest calibration cost is deployed (Bonferroni over
the stages, so the guarantee holds for the selection). Also reports, per
stage, the test AUC of model confidence and of the learned gate for
predicting whether that stage's answer is correct.
Usage: python analysis/llm_fixed_certified.py [results subdir]
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from telemachus.llm_agent import cascade  # noqa: E402
from telemachus.calibration import binomial_pvalue  # noqa: E402


def main():
    d = ROOT / "results" / (sys.argv[1] if len(sys.argv) > 1 else "llm")
    train = cascade.load_table(d / "train_table.jsonl")
    cal, test = cascade.split_test_table(cascade.load_table(d / "test_table.jsonl"), 600)
    st = cascade.stages_of(test[0])
    out = {"fixed_certified": {}, "stage_auc_test": {}}
    for alpha in (.3, .2):
        delta = .1 / len(st)
        cands = []
        for s in st:
            w = sum(not r[s]["correct"] for r in cal)
            p = binomial_pvalue(w, len(cal), alpha)
            cands.append({"stage": s, "cal_error": w / len(cal), "cal_cost": float(np.mean([cascade.stage_cost(r, s) for r in cal])),
                          "p_value": p, "certified": p <= delta})
        ok = [c for c in cands if c["certified"]]
        best = min(ok, key=lambda c: c["cal_cost"]) if ok else None
        res = {"candidates": cands, "selected": best["stage"] if best else "EXPERT"}
        if best:
            f = cascade.fixed_stage(test, best["stage"])
            res["test"] = {**f, "within_alpha": f["error"] <= alpha}
        else:
            res["test"] = {"error": 0., "cost": cascade.DEFAULT_EXPERT_COST, "n": len(test), "within_alpha": True}
        out["fixed_certified"][f"alpha_{alpha}"] = res
        print(f"alpha={alpha}: certified fixed stage = {res['selected']}  test error={res['test']['error']:.3f} cost={res['test']['cost']:.0f}")
    models = cascade.train_learned_gate(train)
    for j, s in enumerate(st):
        rows = [r for r in test if r[s]["value"] is not None]
        y = [int(r[s]["correct"]) for r in rows]
        if len(set(y)) < 2:
            continue
        conf = [cascade.score(r, j, "confidence") for r in rows]
        lp = models[j].predict_proba(np.array([cascade.stage_features(r, j) for r in rows]))[:, 1]
        out["stage_auc_test"][s] = {"n": len(rows), "accuracy": float(np.mean(y)),
                                    "confidence": float(roc_auc_score(y, conf)), "learned": float(roc_auc_score(y, lp))}
        print(f"  {s:7s} acc={np.mean(y):.3f} AUC confidence={out['stage_auc_test'][s]['confidence']:.3f} learned={out['stage_auc_test'][s]['learned']:.3f}")
    (d / "fixed_certified.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
