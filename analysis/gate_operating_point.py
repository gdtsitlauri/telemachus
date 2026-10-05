#!/usr/bin/env python3
"""Proposition 1 (docs/when_gates_pay.md) checked on the single-decision settings.

For each setting and alpha: pi (base error), t* = 1 - alpha/pi, FPR of each gate at TPR >= t*
on the test half, help predicted by Proposition 1, and help needed empirically (smallest
expert rate whose test risk <= alpha when thresholding the score on the test half).
Settings: VLM 256M and 500M (one decision, escalate to expert); language agent, THINK stage
treated as one decision (accept THINK or escalate), 0.5B and 3B.
Writes results/gate_operating_point.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def fpr_at(err, score_err, t):
    """Smallest FPR with TPR >= t, escalating when score_err (higher = more likely error) is large."""
    order = np.argsort(-score_err)
    e = err[order]
    tpr = np.cumsum(e) / max(1, e.sum())
    fpr = np.cumsum(1 - e) / max(1, (1 - e).sum())
    ok = np.where(tpr >= t - 1e-12)[0]
    return float(fpr[ok[0]]) if len(ok) else 1.0


def empirical_help(err, score_err, alpha):
    order = np.argsort(-score_err)
    e = err[order]
    n = len(e)
    remaining_err = e.sum() - np.concatenate([[0], np.cumsum(e)])  # errors accepted after escalating first k
    k = np.where(remaining_err / n <= alpha + 1e-12)[0][0]
    return k / n


def analyse(err, scores, alphas):
    pi = float(err.mean())
    out = {"pi": pi, "n": int(len(err))}
    for a in alphas:
        if a >= pi:
            continue
        t = 1 - a / pi
        row = {"t_star": t, "oracle_help": pi - a, "random_help": 1 - a / pi}
        for g, s in scores.items():
            f = fpr_at(err, s, t)
            row[g] = {"fpr_at_t_star": f, "predicted_help": (pi - a) + (1 - pi) * f, "empirical_help": empirical_help(err, s, a)}
        out[f"alpha_{a}"] = row
    return out


def main():
    sys.path.insert(0, str(ROOT / "code/telemachus/src"))
    sys.path.insert(0, str(ROOT / "code/telemachus/analysis"))
    from sklearn.ensemble import HistGradientBoostingClassifier
    from vlm_run import features
    res = {}
    for sub in ("smolvlm2-256m", "smolvlm2-500m"):
        d = R / "6_vision_language" / sub
        train = [json.loads(l) for l in (d / "train_table.jsonl").read_text().splitlines()]
        ev = [json.loads(l) for l in (d / "eval_table.jsonl").read_text().splitlines()]
        te = ev[len(ev) // 2:]
        clf = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=.05, random_state=0)
        clf.fit(np.array([features(r) for r in train]), np.array([int(r["correct"]) for r in train]))
        err = np.array([1 - int(r["correct"]) for r in te])
        res[sub] = analyse(err, {"learned": 1 - clf.predict_proba(np.array([features(r) for r in te]))[:, 1],
                                 "confidence": -np.array([max(r["options"].values()) for r in te])}, (.05, .10))
    from telemachus.llm_agent import cascade
    for sub in ("qwen2.5-0.5b", "qwen2.5-3b"):
        d = R / "5_language_agent" / sub
        train = cascade.load_table(d / "train_table.jsonl")
        _, test = cascade.split_test_table(cascade.load_table(d / "test_table.jsonl"), 600)
        j = cascade.stages_of(test[0]).index("THINK")
        models = cascade.train_learned_gate(train)
        err = np.array([1 - int(r["THINK"]["correct"]) for r in test])
        lp = models[j].predict_proba(np.array([cascade.stage_features(r, j) for r in test]))[:, 1]
        conf = np.array([cascade.score(r, j, "confidence") for r in test])
        res[sub + "_THINK"] = analyse(err, {"learned": 1 - lp, "confidence": -conf}, (.05, .10, .20))
    (R / "gate_operating_point.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        for a, row in v.items():
            if a.startswith("alpha"):
                print(k, a, f"pi={v['pi']:.3f} oracle={row['oracle_help']:.3f} random={row['random_help']:.3f}",
                      {g: (round(row[g]["predicted_help"], 3), round(row[g]["empirical_help"], 3)) for g in ("learned", "confidence")})


if __name__ == "__main__":
    main()
