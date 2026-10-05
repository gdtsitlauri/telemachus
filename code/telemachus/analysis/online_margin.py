#!/usr/bin/env python3
"""Online safety-margin adaptation under deployment shift (navigation domain).

The v6 certificate selects a static safety margin on calibration episodes
from the holdout generator. Here deployment runs on a harsher generator
("severe" scenarios below, outside the certificate's exchangeability
assumption). After every chunk of episodes the margin is updated with the
online risk-control rule

    margin <- clip(margin + eta * (chunk_collision_rate - alpha), lo, hi)

which targets a long-run collision rate of alpha without exchangeability.
Compared: static certified margin, online margin (same start), and a fixed
conservative margin. Kept outside src/ so the frozen v6 protocol is intact.

  pilot : python analysis/online_margin.py pilot   (development seeds only)
  final : python analysis/online_margin.py final   (fresh seeds 50000+)
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from telemachus.config import ControllerConfig  # noqa: E402
from telemachus.simulation import Scenario  # noqa: E402
from telemachus import experiments as ex  # noqa: E402

ALPHA, START, LO, HI = .03, .04, 0., .16
COMMON = {"layout_family": 3, "camera_style": 3}
SEVERE = [Scenario("visual", camera_noise=.40, camera_brightness=.18, camera_occlusion=.5, **COMMON),
          Scenario("localization", localization_sigma=.55, **COMMON),
          Scenario("dynamics", speed_gain=.28, onset=8, **COMMON),
          Scenario("compound", camera_noise=.34, camera_brightness=.28, localization_sigma=.32,
                   speed_gain=.42, channel_snr_db=-6., onset=8, **COMMON)]


def run_batch(pool, cfg, margin, seeds):
    c = replace(cfg, safety_margin=round(float(margin), 4)).to_dict()
    tasks = [(str(ROOT), c, s, sc, "telemachus") for s in seeds for sc in SEVERE]
    return [ex.flatten_result(r) for r in pool.map(ex._worker_episode, tasks)]


def stats(rows):
    return {"n": len(rows), "collision": float(np.mean([r["collision"] for r in rows])),
            "success": float(np.mean([r["success"] for r in rows])),
            "deferred": float(np.mean([r["deferred"] for r in rows])),
            "cost": float(np.mean([r["cost"] for r in rows]))}


def stream(pool, cfg, mode, seeds, chunk, eta):
    margin, out, trace = START if mode != "conservative" else .08, [], []
    for k in range(0, len(seeds), chunk):
        rows = run_batch(pool, cfg, margin, seeds[k:k + chunk])
        c = float(np.mean([r["collision"] for r in rows]))
        trace.append({"chunk": k // chunk, "margin": margin, "collision": c, "n": len(rows)})
        out += rows
        if mode == "online":
            margin = float(np.clip(margin + eta * (c - ALPHA), LO, HI))
        print(f"  {mode:12s} chunk {k // chunk:3d} margin={trace[-1]['margin']:.3f} coll={c:.3f}", flush=True)
    return {"summary": stats(out), "second_half": stats(out[len(out) // 2:]), "trace": trace}


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    cfg = ControllerConfig.load(ROOT / "configs/v6_main.json")
    out = ROOT / "results/online_margin"
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(10, initializer=ex._worker_init, initargs=(str(ROOT), cfg.to_dict())) as pool:
        if phase == "pilot":
            seeds = list(range(2400, 2420)) + list(range(12000, 12040))
            rep = {str(m): stats(run_batch(pool, cfg, m, seeds)) for m in (.04, .08, .12)}
            for m, s in rep.items():
                print(m, s, flush=True)
            (out / "pilot.json").write_text(json.dumps({"seeds": seeds, "alpha": ALPHA, "margins": rep}, indent=2))
            return
        frozen = json.loads((ROOT / "configs/online_margin_frozen.json").read_text())
        rep = {"protocol": frozen}
        for i, base in enumerate(frozen["stream_bases"]):
            seeds = list(range(base, base + frozen["chunks"] * frozen["seeds_per_chunk"]))
            rep[f"stream_{i}"] = {m: stream(pool, cfg, m, seeds, frozen["seeds_per_chunk"], frozen["eta"])
                                  for m in ("static", "online", "conservative")}
            (out / "final.json").write_text(json.dumps(rep, indent=2))
        print(json.dumps({k: {m: v[m]["summary"] for m in v} for k, v in rep.items() if k.startswith("stream")}, indent=1))


if __name__ == "__main__":
    main()
