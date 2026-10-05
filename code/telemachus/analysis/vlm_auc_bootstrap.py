#!/usr/bin/env python3
"""Paired bootstrap of the AUC difference (learned - confidence) on the VLM test half."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from vlm_run import features  # noqa: E402

d = ROOT / "results/vlm_benchmark"
train = [json.loads(l) for l in (d / "train_table.jsonl").read_text().splitlines()]
ev = [json.loads(l) for l in (d / "eval_table.jsonl").read_text().splitlines()][len(train):] if False else \
     [json.loads(l) for l in (d / "eval_table.jsonl").read_text().splitlines()]
clf = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=.05, random_state=0)
clf.fit(np.array([features(r) for r in train]), np.array([int(r["correct"]) for r in train]))
te = ev[len(ev) // 2:]
y = np.array([int(r["correct"]) for r in te])
l = clf.predict_proba(np.array([features(r) for r in te]))[:, 1]
c = np.array([max(r["options"].values()) for r in te])
rng = np.random.default_rng(0); diffs = []
for _ in range(2000):
    i = rng.integers(0, len(y), len(y))
    if y[i].min() != y[i].max():
        diffs.append(roc_auc_score(y[i], l[i]) - roc_auc_score(y[i], c[i]))
diffs = np.array(diffs)
out = {"auc_diff": float(roc_auc_score(y, l) - roc_auc_score(y, c)), "ci95": np.percentile(diffs, [2.5, 97.5]).tolist(),
       "p_le_0": float((diffs <= 0).mean()), "n_options": sorted({len(r["options"]) for r in te}),
       "chance_accuracy": float(np.mean([1 / len(r["options"]) for r in te]))}
(d / "auc_bootstrap.json").write_text(json.dumps(out, indent=2)); print(out)
