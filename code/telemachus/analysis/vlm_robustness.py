#!/usr/bin/env python3
"""VLM gates over random calibration/test splits, plus a paired AUC bootstrap.

Usage: python analysis/vlm_robustness.py <results subdir> [n_splits]
The learned gate is trained once on train_table.jsonl (as in vlm_run.py); the
600 evaluation scenes are re-split 300/300 at random n_splits times.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from vlm_run import features, certify  # noqa: E402


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "vlm_benchmark"
    n_splits = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    d = ROOT / "results" / sub
    train = [json.loads(l) for l in (d / "train_table.jsonl").read_text().splitlines()]
    ev = [json.loads(l) for l in (d / "eval_table.jsonl").read_text().splitlines()]
    clf = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=.05, random_state=0)
    clf.fit(np.array([features(r) for r in train]), np.array([int(r["correct"]) for r in train]))
    L = clf.predict_proba(np.array([features(r) for r in ev]))[:, 1]
    C = np.array([max(r["options"].values()) for r in ev])
    y = np.array([bool(r["correct"]) for r in ev])
    te = slice(len(ev) // 2, None)
    rng = np.random.default_rng(0); diffs = []
    yt, lt, ct = y[te], L[te], C[te]
    for _ in range(2000):
        i = rng.integers(0, len(yt), len(yt))
        if yt[i].min() != yt[i].max():
            diffs.append(roc_auc_score(yt[i], lt[i]) - roc_auc_score(yt[i], ct[i]))
    out = {"auc_diff_test": float(roc_auc_score(yt, lt) - roc_auc_score(yt, ct)),
           "auc_diff_ci95": np.percentile(diffs, [2.5, 97.5]).tolist(), "splits": {}}
    for alpha in (.05, .10):
        rows = []
        for k in range(n_splits):
            p = np.random.default_rng(1000 + k).permutation(len(ev))
            res = {g: certify(list(s[p]), list(y[p]), alpha, .1, len(ev) // 2) for g, s in (("learned", L), ("confidence", C))}
            rows.append(res)
        e = {}
        for g in ("learned", "confidence"):
            v = [r[g] for r in rows if r[g]]
            e[g] = {"certified": len(v), "within_alpha": sum(x["within_alpha"] for x in v),
                    "error_mean": float(np.mean([x["error"] for x in v])),
                    "expert_rate_mean": float(np.mean([x["expert_rate"] for x in v]))}
        both = [r for r in rows if r["learned"] and r["confidence"]]
        e["learned_fewer_expert_calls"] = f"{sum(r['learned']['expert_rate'] < r['confidence']['expert_rate'] for r in both)}/{len(both)}"
        out["splits"][f"alpha_{alpha}"] = e
    (d / "robustness.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
