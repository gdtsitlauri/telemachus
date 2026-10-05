#!/usr/bin/env python3
"""VLM grounding benchmark with learned vs confidence escalation gates.

Generates scenes with telemachus.vlm_benchmark.make_scene, queries the pinned
SmolVLM2 (ScoredResolver) and stores option probabilities plus image
statistics measurable at deployment (brightness, contrast, sharpness). A
learned gate is trained on a separate scene stream; both gates are certified
with Learn-then-Test (wrong mission goal <= alpha) on calibration scenes and
tested on held-out scenes. Kept outside src/ (frozen v6 protocol).

Usage: python analysis/vlm_run.py --device cuda
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from telemachus.vlm_benchmark import make_scene, ScoredResolver  # noqa: E402
from telemachus.calibration import learn_then_test  # noqa: E402


def image_stats(img):
    a = np.asarray(img.convert("L"), float) / 255.
    lap = np.abs(4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:])
    return {"brightness": float(a.mean()), "contrast": float(a.std()), "sharpness": float(lap.mean())}


def generate(resolver, n, seed, path):
    done = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    rng = random.Random(seed)
    t0 = time.time()
    with path.open("a") as f:
        for i in range(n):
            img, instr, expected, sides, meta = make_scene(rng)
            if i < len(done):
                continue
            r = resolver.resolve(img, instr, sides)
            rec = {"i": i, "instruction": instr, "expected": expected, "side": r["side"], "correct": r["side"] == expected,
                   "options": r["option_probabilities"], "image": image_stats(img), "meta": meta}
            f.write(json.dumps(rec) + "\n")
            if (i + 1) % 50 == 0:
                print(f"{path.name}: {i + 1}/{n} scenes, {time.time() - t0:.0f}s", flush=True)
    return [json.loads(l) for l in path.read_text().splitlines()]


def features(r):
    p = sorted(r["options"].values(), reverse=True)
    ent = -sum(x * np.log(max(x, 1e-12)) for x in p)
    return [p[0], p[0] - p[1], ent, len(p), float("triangle" in r["instruction"] or "circle" in r["instruction"]
            or "square" in r["instruction"]), r["image"]["brightness"], r["image"]["contrast"], r["image"]["sharpness"]]


def certify(scores, correct, alpha, delta, n_cal):
    grid = [1.01] + list(np.round(np.linspace(.99, 0., 100), 3))
    cal_s, cal_c, te_s, te_c = scores[:n_cal], correct[:n_cal], scores[n_cal:], correct[n_cal:]
    counts = []
    for th in grid:
        wrong = int(sum((s >= th) and not c for s, c in zip(cal_s, cal_c)))
        counts.append(({"theta": th}, len(cal_s), wrong))
        sel, res = learn_then_test(counts, alpha, delta)
        if not res[-1].rejected:
            break
    sel, _ = learn_then_test(counts, alpha, delta)
    if sel is None:
        return None
    th = grid[sel]
    accept = [s >= th for s in te_s]
    wrong = [a and not c for a, c in zip(accept, te_c)]
    return {"theta": float(th), "error": float(np.mean(wrong)), "expert_rate": float(1 - np.mean(accept)),
            "within_alpha": bool(np.mean(wrong) <= alpha), "n_test": len(te_s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=600)
    ap.add_argument("--output", default=str(ROOT / "results/vlm_benchmark"))
    ap.add_argument("--model", default=None, help="override the pinned model id (revision resolved and recorded)")
    a = ap.parse_args()
    if a.model:
        from huggingface_hub import HfApi
        from telemachus import vlm
        vlm.MODEL_ID, vlm.MODEL_REVISION = a.model, HfApi().model_info(a.model).sha
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    resolver = ScoredResolver(ROOT / ".cache/models", a.device)
    train = generate(resolver, a.n_train, 1, out / "train_table.jsonl")
    evalset = generate(resolver, a.n_eval, 2026, out / "eval_table.jsonl")
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=.05, random_state=0)
    clf.fit(np.array([features(r) for r in train]), np.array([int(r["correct"]) for r in train]))
    learned = clf.predict_proba(np.array([features(r) for r in evalset]))[:, 1].tolist()
    confidence = [max(r["options"].values()) for r in evalset]
    correct = [r["correct"] for r in evalset]
    n_cal = a.n_eval // 2
    from sklearn.metrics import roc_auc_score
    te = slice(n_cal, None)
    report = {"model": resolver.model_id, "revision": resolver.revision, "n_train": len(train), "n_calibration": n_cal, "n_test": a.n_eval - n_cal,
              "vlm_accuracy_test": float(np.mean(correct[te])),
              "auc_test": {"learned": float(roc_auc_score(correct[te], learned[te])),
                           "confidence": float(roc_auc_score(correct[te], confidence[te]))},
              "gates": {}}
    for alpha in (.05, .10):
        report["gates"][f"alpha_{alpha}"] = {"learned": certify(learned, correct, alpha, .1, n_cal),
                                             "confidence": certify(confidence, correct, alpha, .1, n_cal)}
    (out / "certified_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
