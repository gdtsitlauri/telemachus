"""Certified evaluation protocol (v6).

1. freeze      -- methods, calibration seeds, test seeds, alpha, delta, grid,
                  base config and source hashes are fixed before any episode.
2. calibrate   -- each method gets its own Learn-then-Test safety setting on
                  calibration episodes (calibration.py).
3. test        -- each method runs at its certified setting on untouched seeds.
4. report      -- methods are compared at the SAME certified safety level, and
                  the guarantee itself is checked on the test episodes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from scipy.stats import beta

from .calibration import certify, calibrated_config, default_grid, load_certificate
from .config import ControllerConfig
from .experiments import run_suite, summarize, flatten_result
from .provenance import canonical_hash, code_manifest, environment_info, file_hash, utc_now, write_json


def clopper_pearson(k: int, n: int, level: float = .95):
    a = (1 - level) / 2
    lo = 0. if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1. if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return [lo, hi]


def freeze(root, path, cfg: ControllerConfig, methods: List[str], calibration_seeds, test_seeds,
           alpha: float, delta: float, grid=None, development_seeds=()):
    root = Path(root)
    calibration_seeds, test_seeds = list(calibration_seeds), list(test_seeds)
    if set(calibration_seeds) & set(test_seeds):
        raise ValueError("calibration and test seeds must be disjoint")
    if (set(calibration_seeds) | set(test_seeds)) & set(development_seeds):
        raise ValueError("certified evaluation seeds must be unseen during development")
    manifest = {
        "version": 6, "created": utc_now(), "methods": list(methods),
        "calibration_seeds": calibration_seeds, "test_seeds": test_seeds,
        "alpha": alpha, "delta": delta, "grid": default_grid() if grid is None else grid,
        "config": cfg.to_dict(), "source_hashes": code_manifest(root),
        "vision_sha256": file_hash(root / cfg.vision_artifact),
        "resource_policy_sha256": None if cfg.resource_policy_mode == "heuristic" else file_hash(root / cfg.learned_resource_policy),
        "environment": environment_info(),
        "estimand": "per-episode collision probability (safety); success, deferral and cost at the certified setting",
        "statistical_unit": "episode = (seed, scenario); scenarios equally weighted",
    }
    manifest["protocol_hash"] = canonical_hash({k: v for k, v in manifest.items() if k != "created"})
    write_json(path, manifest)
    return manifest


def verify(root, protocol):
    if code_manifest(root) != protocol["source_hashes"]:
        raise ValueError("source changed after the certified protocol was frozen")
    body = {k: v for k, v in protocol.items() if k not in ("created", "protocol_hash")}
    if canonical_hash(body) != protocol["protocol_hash"]:
        raise ValueError("protocol was modified")


def calibrate(root, protocol_path, output, workers=1):
    root, output = Path(root), Path(output)
    protocol = json.loads(Path(protocol_path).read_text())
    verify(root, protocol)
    cfg = ControllerConfig.from_dict(protocol["config"])
    report = certify(root, cfg, protocol["methods"], protocol["calibration_seeds"], protocol["alpha"],
                     protocol["delta"], output, protocol["grid"], workers)
    report["protocol_hash"] = protocol["protocol_hash"]
    report["certificate_hash"] = canonical_hash({k: v for k, v in report.items() if k not in ("created", "certificate_hash")})
    write_json(output / "calibration_certificate.json", report)
    return report


def test(root, protocol_path, certificate_path, output, workers=1, split="certified"):
    root, output = Path(root), Path(output)
    protocol = json.loads(Path(protocol_path).read_text())
    verify(root, protocol)
    cert = load_certificate(certificate_path)
    if cert.get("protocol_hash") != protocol["protocol_hash"]:
        raise ValueError("certificate belongs to a different protocol")
    base = ControllerConfig.from_dict(protocol["config"])
    rows: List[Dict] = []
    per_method = {}
    for method in protocol["methods"]:
        entry = cert["methods"][method]
        if entry["selected_setting"] is None:
            per_method[method] = {"certified": False}
            continue
        cfg = calibrated_config(base, cert, method)
        rep = run_suite(root, output / method, cfg, protocol["test_seeds"], [method], split=split, workers=workers)
        for f in sorted((output / method / "episodes").glob("*.json")):
            rows.append(flatten_result(json.loads(f.read_text())))
        m = rep["methods"][method]
        k = int(round(m["collision"] * m["n"]))
        per_method[method] = {
            "certified": True, "setting": entry["selected_setting"], "n": m["n"],
            "collisions": k, "collision_rate": m["collision"], "collision_ci95": clopper_pearson(k, m["n"]),
            "guarantee_alpha": protocol["alpha"],
            "empirical_rate_within_alpha": m["collision"] <= protocol["alpha"],
            "success": m["success"], "deferred": m["deferred"], "cost_mean": m["cost_mean"],
            "cost_per_success": m["cost_per_success"],
        }
    report = summarize(rows) if rows else {}
    report.update({"split": split, "protocol_hash": protocol["protocol_hash"],
                   "certificate_hash": cert["certificate_hash"], "created": utc_now(),
                   "certified_methods": per_method,
                   "interpretation": ("Each method runs at its own Learn-then-Test safety setting, so success, "
                                      "deferral and cost are compared at the same certified collision level alpha.")})
    write_json(output / "summary.json", report)
    return report
