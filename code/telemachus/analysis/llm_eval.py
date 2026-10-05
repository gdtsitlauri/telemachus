#!/usr/bin/env python3
"""Certified evaluation of the language-agent cascade on the generated tables.

Reads results/<dir>/train_table.jsonl (gate training) and test_table.jsonl
(split into calibration and test with a fixed seed), runs every gate through
Learn-then-Test and writes results/<dir>/certified_summary.json.
Learned-gate scores are computed once per (question, stage) and cached; the
cascade logic itself is unchanged (telemachus.llm_agent.cascade).
Kept outside src/ so it does not change the frozen v6 navigation protocol.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from telemachus.llm_agent import cascade  # noqa: E402

_original_score = cascade.score
_cache = {}


def cached_score(rec, j, gate, models=None):
    if gate != "learned":
        return _original_score(rec, j, gate, models)
    key = (rec["id"], j, id(models))
    if key not in _cache:
        _cache[key] = _original_score(rec, j, gate, models)
    return _cache[key]


def prefill(models, table):
    """Batch-predict all learned scores (one predict_proba per stage)."""
    st = cascade.stages_of(table[0])
    for j in range(len(st)):
        rows = [r for r in table if r[st[j]]["value"] is not None]
        if rows:
            X = np.array([cascade.stage_features(r, j) for r in rows])
            p = models[j].predict_proba(X)[:, 1]
            for r, v in zip(rows, p):
                _cache[(r["id"], j, id(models))] = float(v)
        for r in table:
            if r[st[j]]["value"] is None:
                _cache[(r["id"], j, id(models))] = 0.


cascade.score = cached_score
_train_learned_gate = cascade.train_learned_gate
_TABLE = []


def train_and_prefill(train_table):
    models = _train_learned_gate(train_table)
    prefill(models, _TABLE)
    return models


cascade.train_learned_gate = train_and_prefill


def main():
    d = ROOT / "results" / (sys.argv[1] if len(sys.argv) > 1 else "llm")
    train = cascade.load_table(d / "train_table.jsonl")
    test = cascade.load_table(d / "test_table.jsonl")
    _TABLE.extend(test)
    out = {}
    for alpha in (.30, .20):
        r = cascade.evaluate(train, test, alpha=alpha, delta=.1, n_cal=600)
        out[f"alpha_{alpha}"] = r
        print(f"=== alpha={alpha}  (calibration {r['n_calibration']}, test {r['n_test']})")
        for g, e in r["gates"].items():
            t = e.get("test")
            if t:
                print(f"  {g:10s} theta={e['calibration']['selected_theta']:<6} error={t['error']:.3f} "
                      f"cost={t['cost']:.0f} expert={t['expert_rate']:.2f} stops={t['stops']} within_alpha={t['within_alpha']}")
            else:
                print(f"  {g:10s} no certificate")
        for s, f in r["fixed"].items():
            print(f"  fixed {s:7s} error={f['error']:.3f} cost={f['cost']:.0f}")
    (d / "certified_summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
