#!/usr/bin/env python3
"""Online safety control with a zero-risk fallback (navigation; Theorem A in
TELEMACHUS docs/online_guarantee.md).

Control s is updated after every episode, s <- max(0, s + eta * (collision - alpha*)),
with no clipping on the safe side. For s < s_f the agent drives with safety
margin s; for s >= s_f the episode is handed to a human before any motion
(deferred, no collision possible). With alpha* = alpha - (s_f - s_1 + eta) / (eta * K)
the realised collision rate of every K-episode deployment is <= alpha, for any
shift, deterministically. Same severe shift and seeds as analysis/online_margin.py.

  python analysis/online_margin_hp.py        (protocol configs/online_margin_hp_frozen.json)
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from telemachus.config import ControllerConfig  # noqa: E402
from telemachus import experiments as ex  # noqa: E402
from online_margin import SEVERE  # noqa: E402


def run_stream(args):
    base, p = args
    cfg = ControllerConfig.load(ROOT / "configs/v6_main.json")
    ex._worker_init(str(ROOT), cfg.to_dict())
    K, eta, s_f = p["episodes_per_stream"], p["eta"], p["s_f"]
    a_star = p["alpha"] - (s_f - p["s_1"] + eta) / (eta * K)
    tasks = [(s, sc) for s in range(base, base + K // len(SEVERE)) for sc in SEVERE]
    s, rows = p["s_1"], []
    for seed, sc in tasks:
        if s >= s_f:
            r = {"seed": seed, "scenario": sc.name, "handover": 1, "collision": 0, "success": 0, "deferred": 1,
                 "cost": 0., "margin": None}
        else:
            c = replace(cfg, safety_margin=round(float(s), 4)).to_dict()
            f = ex.flatten_result(ex._worker_episode((str(ROOT), c, seed, sc, "telemachus")))
            r = {"seed": seed, "scenario": sc.name, "handover": 0, "collision": f["collision"], "success": f["success"],
                 "deferred": f["deferred"], "cost": f["cost"], "margin": round(float(s), 4)}
        r["s"] = s
        rows.append(r)
        s = max(0., s + eta * (r["collision"] - a_star))
    col = np.array([r["collision"] for r in rows])
    bound = a_star + (s_f - p["s_1"] + eta * (1 - a_star)) / (eta * K)
    return {"base": base, "alpha_star": a_star, "theorem_bound": bound, "n": len(rows),
            "collision": float(col.mean()), "max_running_collision_after_100": float(np.max(np.cumsum(col)[99:] / np.arange(100, K + 1))),
            "success": float(np.mean([r["success"] for r in rows])), "deferred": float(np.mean([r["deferred"] for r in rows])),
            "handover": float(np.mean([r["handover"] for r in rows])), "cost": float(np.mean([r["cost"] for r in rows])),
            "episodes": rows}


def main():
    p = json.loads((ROOT / "configs/online_margin_hp_frozen.json").read_text())
    out = ROOT / "results/online_margin"
    with ProcessPoolExecutor(len(p["stream_bases"])) as pool:
        res = list(pool.map(run_stream, [(b, p) for b in p["stream_bases"]]))
    rep = {"protocol": p, "streams": res}
    (out / "final_hp.json").write_text(json.dumps(rep, indent=2))
    for r in res:
        print({k: v for k, v in r.items() if k != "episodes"})


if __name__ == "__main__":
    main()
