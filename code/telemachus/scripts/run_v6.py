#!/usr/bin/env python3
"""TELEMACHUS v6 certified evaluation, end to end (resumable).

  1. freeze   the certified protocol (methods, seeds, alpha, delta, code hashes)
  2. calibrate every method with Learn-then-Test on calibration episodes
  3. test     every method at its certified setting on untouched seeds
  4. stress   the same certified settings on the stress generator (outside
              the guarantee's exchangeability assumption)
  5. budget   tight-budget study (reserve vs no reserve) at certified settings

Usage:
  python scripts/run_v6.py --config configs/v6_main.json --workers 10 --output results/v6
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_METHODS = ["telemachus", "no_expert", "no_learning", "budget_voi", "confidence_gate"]
BUDGET_METHODS = ["telemachus", "no_reserve"]
# Coarse fixed-sequence grid, most -> least conservative. Development pilots
# (seeds 2400-2419) showed no collisions at >= 8 cm, where episodes become
# long; an earlier run with an 8-step grid from 16 cm was aborted for runtime
# before any test episode (results/v6_aborted_fine_grid).
GRID = [{"safety_margin": m} for m in (0.08, 0.04, 0.02, 0.0)]


def main():
    from telemachus.config import ControllerConfig
    from telemachus.certified import freeze, calibrate, test
    from telemachus.experiments import DEVELOPMENT_SEEDS

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v6_main.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=.03)
    ap.add_argument("--delta", type=float, default=.1)
    ap.add_argument("--budget-max-cost", type=float, default=40.)
    ap.add_argument("--output", default="results/v6")
    ap.add_argument("--skip-budget", action="store_true")
    a = ap.parse_args()
    out = ROOT / a.output
    out.mkdir(parents=True, exist_ok=True)
    cfg = ControllerConfig.load(ROOT / a.config)

    studies = [("main", cfg, MAIN_METHODS, range(30000, 30060), range(40000, 40080))]
    if not a.skip_budget:
        tight = replace(cfg, budget=replace(cfg.budget, max_cost=a.budget_max_cost))
        studies.append(("budget", tight, BUDGET_METHODS, range(31000, 31060), range(41000, 41080)))

    report = {}
    for name, c, methods, cal_seeds, test_seeds in studies:
        d = out / name
        protocol = d / "protocol.json"
        if not protocol.exists():
            freeze(ROOT, protocol, c, methods, cal_seeds, test_seeds, a.alpha, a.delta, grid=GRID,
                   development_seeds=DEVELOPMENT_SEEDS)
            print(f"[{name}] protocol frozen", flush=True)
        cert = d / "calibration" / "calibration_certificate.json"
        if not cert.exists() or "certificate_hash" not in json.loads(cert.read_text()) or \
                "protocol_hash" not in json.loads(cert.read_text()):
            calibrate(ROOT, protocol, d / "calibration", a.workers)
        print(f"[{name}] calibrated", flush=True)
        res = test(ROOT, protocol, cert, d / "test", a.workers, split="certified")
        report[name] = res["certified_methods"]
        if name == "main":
            stress = test(ROOT, protocol, cert, d / "stress", a.workers, split="certified_stress")
            report["main_stress"] = stress["certified_methods"]
        print(f"[{name}] tested", flush=True)
    (out / "v6_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
