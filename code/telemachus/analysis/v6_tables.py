#!/usr/bin/env python3
"""Readable tables for the v6 certified navigation study (results/v6)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def show(title, block):
    print(f"\n=== {title}")
    print(f"{'method':16s} {'setting':22s} {'n':>4s} {'coll':>6s} {'coll_ci95':>16s} {'succ':>6s} {'defer':>6s} {'cost':>6s}")
    for m, v in block.items():
        if not v.get("certified"):
            print(f"{m:16s} no certificate")
            continue
        ci = "[%.3f,%.3f]" % tuple(v["collision_ci95"])
        print(f"{m:16s} {str(v['setting']):22s} {v['n']:4d} {v['collision_rate']:6.3f} {ci:>16s} "
              f"{v['success']:6.3f} {v['deferred']:6.3f} {v['cost_mean']:6.1f}")


def main():
    rep = json.loads((ROOT / "results/v6/v6_report.json").read_text())
    for name in ("main", "main_stress", "budget"):
        if name in rep:
            show(name, rep[name])
    for study in ("main", "budget"):
        cert = ROOT / f"results/v6/{study}/calibration/calibration_certificate.json"
        if cert.exists():
            c = json.loads(cert.read_text())
            print(f"\n--- {study} calibration (alpha={c['alpha']}, delta={c['delta']})")
            for m, e in c["methods"].items():
                path = " ".join(f"{x['setting']['safety_margin']}:{x['collisions']}/{x['n']}{'*' if x['rejected'] else ''}"
                                for x in e["candidates"])
                print(f"  {m:16s} selected={e['selected_setting']}  {path}")
    test = json.loads((ROOT / "results/v6/main/test/summary.json").read_text())
    print("\n--- paired comparisons vs telemachus (main test; seed-cluster bootstrap)")
    for b, c in test.get("paired_comparisons", {}).items():
        print(f"  {b:16s} " + "  ".join(f"{k}: {v['difference']:+.3f} {['%.3f' % x for x in v['ci95']]}"
                                        for k, v in c.items() if k in ("success", "collision", "cost")))


if __name__ == "__main__":
    main()
