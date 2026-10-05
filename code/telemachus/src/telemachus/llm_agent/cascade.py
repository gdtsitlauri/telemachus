"""Gated escalation cascade with a Learn-then-Test guarantee on wrong answers.

After each stage a gate gives a score in [0,1] for "the current answer is
correct"; the agent stops when score >= theta, otherwise it escalates. The
final stage EXPERT is always correct (a human answer at a large fixed cost),
so a certified threshold always exists. theta is chosen by Learn-then-Test
(fixed sequence from most to least escalation) such that, with probability
>= 1 - delta, P(final answer wrong) <= alpha on exchangeable questions.

Gates:
  learned     per-stage classifier on observable features (trained on the
              GSM8K train split; never sees calibration/test questions)
  confidence  model confidence: exp(mean token log-prob); vote share for VOTE
  agreement   fraction of executed stages that agree with the current answer
  oracle      stops at the first correct stage (analysis only: uses the gold)
Fixed baselines answer with one stage only (no gate, no guarantee).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..calibration import learn_then_test

STAGES = ("ANSWER", "THINK", "TOOL", "VOTE")


def stages_of(rec: Dict) -> List[str]:
    """Stages present in a table record (VOTE is optional)."""
    return [s for s in STAGES if s in rec]
TOOL_EXECUTION_COST = 20.     # token-equivalents for running the sandbox
DEFAULT_EXPERT_COST = 1500.   # token-equivalents for a human answer


def load_table(path) -> List[Dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def stage_cost(rec: Dict, stage: str) -> float:
    return float(rec[stage]["tokens"]) + (TOOL_EXECUTION_COST if stage == "TOOL" else 0.)


def _same(a, b) -> bool:
    return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def stage_features(rec: Dict, j: int) -> List[float]:
    """Observable features after stages 0..j (no gold answer)."""
    st = stages_of(rec)
    s = rec[st[j]]
    value = s["value"]
    prev = [rec[st[i]]["value"] for i in range(j)]
    agree_prev = sum(_same(value, p) for p in prev)
    return [j, s["mean_logprob"], s.get("min_logprob", s["mean_logprob"]), math.log1p(s["tokens"]),
            float(value is not None), float(s.get("executed", 1.)), s.get("agreement", 0.),
            agree_prev, agree_prev / max(1, j), float(value is not None and float(value).is_integer()),
            math.log10(abs(value) + 1) if value is not None else -1.]


def score(rec: Dict, j: int, gate: str, models=None) -> float:
    st = stages_of(rec)
    s = rec[st[j]]
    if s["value"] is None:
        return 0.
    if gate == "learned":
        return float(models[j].predict_proba(np.array([stage_features(rec, j)]))[0, 1])
    if gate == "confidence":
        return s.get("agreement", 0.) if st[j] == "VOTE" else math.exp(s["mean_logprob"])
    if gate == "agreement":
        prev = [rec[st[i]]["value"] for i in range(j)]
        return sum(_same(s["value"], p) for p in prev) / j if j else 0.
    if gate == "oracle":
        return float(s["correct"])
    raise ValueError(gate)


def train_learned_gate(train_table: List[Dict]):
    from sklearn.ensemble import HistGradientBoostingClassifier
    models = []
    st = stages_of(train_table[0])
    for j in range(len(st)):
        X = np.array([stage_features(r, j) for r in train_table])
        y = np.array([int(r[st[j]]["correct"]) for r in train_table])
        models.append(HistGradientBoostingClassifier(max_iter=150, learning_rate=.05, max_depth=3,
                                                     random_state=0).fit(X, y))
    return models


def run_cascade(table: List[Dict], gate: str, theta: float, models=None, expert_cost=DEFAULT_EXPERT_COST):
    wrong, cost, stops = [], [], []
    for r in table:
        c = 0.
        stop: Optional[int] = None
        st = stages_of(r)
        for j, stage in enumerate(st):
            c += stage_cost(r, stage)
            if score(r, j, gate, models) >= theta:
                stop = j
                break
        if stop is None:
            c += expert_cost
            wrong.append(0)
            stops.append("EXPERT")
        else:
            wrong.append(int(not r[st[stop]]["correct"]))
            stops.append(st[stop])
        cost.append(c)
    return {"error": float(np.mean(wrong)), "cost": float(np.mean(cost)),
            "expert_rate": float(np.mean([s == "EXPERT" for s in stops])),
            "stops": {s: stops.count(s) for s in (*STAGES, "EXPERT")}, "n": len(table),
            "wrong": int(np.sum(wrong))}


def fixed_stage(table: List[Dict], stage: str):
    wrong = [int(not r[stage]["correct"]) for r in table]
    cost = [stage_cost(r, stage) for r in table]
    return {"error": float(np.mean(wrong)), "cost": float(np.mean(cost)), "n": len(table)}


THETA_GRID = [1.01] + [round(x, 3) for x in np.linspace(.99, 0., 100)]  # most -> least escalation


def certify_gate(cal: List[Dict], gate: str, alpha: float, delta: float, models=None, expert_cost=DEFAULT_EXPERT_COST):
    counts, rows = [], []
    for theta in THETA_GRID:
        r = run_cascade(cal, gate, theta, models, expert_cost)
        counts.append(({"theta": theta}, r["n"], r["wrong"]))
        sel, res = learn_then_test(counts, alpha, delta)
        rows.append({"theta": theta, "error": r["error"], "cost": r["cost"], "p_value": res[-1].p_value})
        if not res[-1].rejected:
            break
    sel, _ = learn_then_test(counts, alpha, delta)
    return {"selected_theta": None if sel is None else THETA_GRID[sel], "candidates": rows}


def split_test_table(table: List[Dict], n_cal: int, seed: int = 2026):
    rows = sorted(table, key=lambda r: r["id"])
    random.Random(seed).shuffle(rows)
    return rows[:n_cal], rows[n_cal:]


def evaluate(train_table, test_table, alpha=.1, delta=.1, n_cal=600, expert_cost=DEFAULT_EXPERT_COST):
    cal, test = split_test_table(test_table, n_cal)
    models = train_learned_gate(train_table)
    report = {"alpha": alpha, "delta": delta, "n_train": len(train_table), "n_calibration": len(cal),
              "n_test": len(test), "expert_cost": expert_cost, "gates": {}, "fixed": {}}
    for gate in ("learned", "confidence", "agreement", "oracle"):
        c = certify_gate(cal, gate, alpha, delta, models, expert_cost)
        entry = {"calibration": c}
        if c["selected_theta"] is not None:
            t = run_cascade(test, gate, c["selected_theta"], models, expert_cost)
            lo_hi = _clopper_pearson(t["wrong"], t["n"])
            entry["test"] = {**t, "error_ci95": lo_hi, "within_alpha": t["error"] <= alpha}
        report["gates"][gate] = entry
    for stage in stages_of(test[0]):
        report["fixed"][stage] = fixed_stage(test, stage)
    report["fixed"]["EXPERT"] = {"error": 0., "cost": expert_cost, "n": len(test)}
    return report


def _clopper_pearson(k, n, level=.95):
    from scipy.stats import beta
    a = (1 - level) / 2
    return [0. if k == 0 else float(beta.ppf(a, k, n - k + 1)), 1. if k == n else float(beta.ppf(1 - a, k + 1, n - k))]
