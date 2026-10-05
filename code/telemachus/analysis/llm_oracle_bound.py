#!/usr/bin/env python3
"""Cost lower bound for the language cascade at error <= alpha.

The 'oracle' gate in cascade.py stops at the first correct stage (zero
error); it does not spend the alpha error budget, so it is not a lower bound
on cost at a given alpha. The alpha-oracle here knows correctness and may
accept a wrong first-stage answer on at most floor(alpha * n) questions,
chosen where that saves the most cost: the minimum cost of any cascade
policy with empirical error <= alpha on the same test split.
Usage: python analysis/llm_oracle_bound.py [results subdir]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from telemachus.llm_agent import cascade  # noqa: E402


def bound(test, alpha, expert_cost=cascade.DEFAULT_EXPERT_COST):
    st = cascade.stages_of(test[0])
    zero, save = [], []
    for r in test:
        c, done = 0., None
        for s in st:
            c += cascade.stage_cost(r, s)
            if r[s]["correct"]:
                done = c
                break
        z = done if done is not None else c + expert_cost
        cheap = cascade.stage_cost(r, st[0])
        zero.append(z)
        save.append(z - cheap if not r[st[0]]["correct"] else 0.)
    k = int(alpha * len(test))
    gain = sum(sorted(save, reverse=True)[:k])
    return {"alpha": alpha, "n": len(test), "zero_error_oracle_cost": sum(zero) / len(test),
            "alpha_oracle_cost": (sum(zero) - gain) / len(test), "alpha_oracle_error": k / len(test)}


if __name__ == "__main__":
    d = ROOT / "results" / (sys.argv[1] if len(sys.argv) > 1 else "llm")
    _, test = cascade.split_test_table(cascade.load_table(d / "test_table.jsonl"), 600)
    out = {f"alpha_{a}": bound(test, a) for a in (.3, .2)}
    (d / "oracle_bound.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
