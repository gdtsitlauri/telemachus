"""Episode-level safety calibration with a finite-sample guarantee.

The per-decision Monte Carlo certificate (certificate.py) bounds risk under the
agent's *internal* belief. It says nothing about real collisions caused by
model mismatch. This module adds an outer, distribution-free guarantee on the
quantity that matters: the probability that an episode ends in a collision.

Method: Learn-then-Test (Angelopoulos, Bates, Candes, Jordan, Lei 2021) with
fixed-sequence testing. Candidate safety settings lambda_1, ..., lambda_K are
ordered from most to least conservative. For each, the null hypothesis
H_k: P(collision | lambda_k) > alpha is tested with the exact binomial p-value
p_k = P(Bin(n, alpha) <= c_k) from c_k collisions in n calibration episodes.
Testing stops at the first non-rejection; the least conservative rejected
setting is selected. Then, for exchangeable calibration and deployment
episodes,  P( P(collision | lambda_hat) <= alpha ) >= 1 - delta.

Every method is calibrated separately, so methods are compared at the same
certified safety level.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from scipy.stats import binom

from .config import ControllerConfig
from .provenance import canonical_hash, write_json, utc_now


def binomial_pvalue(losses: int, n: int, alpha: float) -> float:
    """Exact p-value for H0: risk > alpha given `losses` Bernoulli losses in n trials."""
    if n <= 0 or not 0 <= losses <= n or not 0 < alpha < 1:
        raise ValueError("invalid binomial test input")
    return float(binom.cdf(losses, n, alpha))


@dataclass(frozen=True)
class CandidateResult:
    setting: Dict[str, Any]
    n: int
    losses: int
    p_value: float
    rejected: bool


def learn_then_test(counts: Sequence[tuple], alpha: float, delta: float) -> tuple:
    """Fixed-sequence LTT.

    counts: [(setting, n, losses), ...] ordered most -> least conservative.
    Returns (index of selected setting or None, [CandidateResult]).
    """
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("alpha and delta must be in (0, 1)")
    results: List[CandidateResult] = []
    selected: Optional[int] = None
    testing = True
    for i, (setting, n, losses) in enumerate(counts):
        p = binomial_pvalue(losses, n, alpha)
        rejected = testing and p <= delta
        if rejected:
            selected = i
        else:
            testing = False
        results.append(CandidateResult(dict(setting), int(n), int(losses), p, rejected))
    return selected, results


def apply_setting(cfg: ControllerConfig, setting: Dict[str, Any]) -> ControllerConfig:
    allowed = {"safety_margin", "risk_threshold"}
    if not set(setting) <= allowed:
        raise ValueError(f"calibration may only change {sorted(allowed)}")
    return replace(cfg, **setting)


def default_grid() -> List[Dict[str, Any]]:
    """Most -> least conservative clearance margins (metres)."""
    return [{"safety_margin": m} for m in (0.16, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0.0)]


def calibrate_method(root, cfg: ControllerConfig, method: str, seeds, grid, alpha: float, delta: float,
                     output: Path, workers: int = 1, stop_early: bool = True) -> Dict[str, Any]:
    """Run calibration episodes for one method and select its safety setting."""
    from .experiments import run_suite

    output = Path(output)
    counts = []
    per_setting = []
    selected_so_far = True
    for k, setting in enumerate(grid):
        if stop_early and not selected_so_far:
            break  # fixed-sequence testing cannot reject later hypotheses
        cfg_k = apply_setting(cfg, setting)
        rep = run_suite(root, output / f"{method}" / f"setting_{k}", cfg_k, seeds, [method],
                        split="calibration", workers=workers)
        summary = rep["methods"][method]
        n = int(summary["n"])
        losses = int(round(summary["collision"] * n))
        counts.append((setting, n, losses))
        _, res = learn_then_test(counts, alpha, delta)
        selected_so_far = res[-1].rejected
        per_setting.append({"setting": setting, "n": n, "collisions": losses,
                            "success": summary["success"], "deferred": summary["deferred"],
                            "cost_mean": summary["cost_mean"], "p_value": res[-1].p_value,
                            "rejected": res[-1].rejected})
    selected, _ = learn_then_test(counts, alpha, delta)
    return {
        "method": method,
        "alpha": alpha,
        "delta": delta,
        "selected_index": selected,
        "selected_setting": None if selected is None else grid[selected],
        "candidates": per_setting,
        "guarantee": (
            f"With probability >= {1 - delta:.2f} over the calibration episodes, the per-episode "
            f"collision probability of '{method}' at the selected setting is <= {alpha} for episodes "
            "exchangeable with the calibration episodes (same scenario generator and split)."
        ),
        "no_certificate": selected is None,
    }


def certify(root, cfg: ControllerConfig, methods, calibration_seeds, alpha: float, delta: float,
            output: Path, grid=None, workers: int = 1) -> Dict[str, Any]:
    root, output = Path(root), Path(output)
    grid = default_grid() if grid is None else [dict(g) for g in grid]
    report = {"created": utc_now(), "alpha": alpha, "delta": delta, "grid": grid,
              "calibration_seeds": list(calibration_seeds), "base_config_hash": canonical_hash(cfg.to_dict()),
              "method": "Learn-then-Test, fixed-sequence, exact binomial p-values", "methods": {}}
    for method in methods:
        report["methods"][method] = calibrate_method(root, cfg, method, list(calibration_seeds), grid,
                                                     alpha, delta, output, workers)
        write_json(output / "calibration_certificate.json", report)
    report["certificate_hash"] = canonical_hash({k: v for k, v in report.items() if k != "created"})
    write_json(output / "calibration_certificate.json", report)
    return report


def calibrated_config(cfg: ControllerConfig, certificate: Dict[str, Any], method: str) -> ControllerConfig:
    entry = certificate["methods"][method]
    if entry["selected_setting"] is None:
        raise ValueError(f"{method} has no certified setting at alpha={certificate['alpha']}")
    return apply_setting(cfg, entry["selected_setting"])


def load_certificate(path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text())
    body = {k: v for k, v in data.items() if k not in ("created", "certificate_hash")}
    if data.get("certificate_hash") != canonical_hash(body):
        raise ValueError("calibration certificate was modified")
    return data
