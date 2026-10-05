"""Finite-horizon, high-probability online help control under deployment shift.

The online rule of shift.py (theta <- clip(theta - eta * (failure - alpha), 0, 1.01))
controls the long-run average failure rate only. Here (TELEMACHUS docs/online_guarantee.md,
Theorem B) the threshold is not clipped on its safe side, so for theta <= 0 every agent asks
the expert at every step (full help, the fallback), and the target alpha* < alpha is computed
before deployment so that, with probability >= 1 - delta, the realised failure rate of the
whole deployment is <= alpha under arbitrary shift, provided the fallback's per-episode
failure probability is <= p0 (declared in advance).

Also measures, on held-out seeds, how well each gate score predicts a wrong own action (AUC).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch

from .certified_help import deploy, train_help_gate, calibrate_gate
from .shift import shifted_spec


def slack(alpha_star: float, p0: float, n: int, K: int, eta: float, s_gap: float, delta: float) -> Optional[float]:
    """Right-hand side of Theorem B minus alpha*; None if alpha* <= p0."""
    g = alpha_star - p0
    if g <= 0:
        return None
    v = p0 * (1 - p0)
    u = 2 * g / (v / n + 2 * g / 3)
    return s_gap / (eta * K) + 1. / K + math.log(K / delta) / (u * K)


def solve_alpha_star(alpha, delta, p0, n, K, eta, s_gap, step=1e-5) -> Optional[float]:
    """Largest alpha* on a 1e-5 grid with alpha* + slack(alpha*) <= alpha; None if infeasible.

    alpha* + slack(alpha*) is not monotone in alpha* (the slack falls as alpha* moves away
    from p0), so the feasible set is an interval that bisection can miss. The first
    implementation used bisection; every value it returned is feasible (so the guarantee
    holds) but possibly below the largest feasible alpha*. See solve_alpha_star_bisection.
    """
    found = None
    for a in np.arange(alpha, p0, -step):
        s = slack(float(a), p0, n, K, eta, s_gap, delta)
        if s is not None and a + s <= alpha:
            found = float(a)
            break
    # bisection can land between grid points; keep the larger feasible value so that
    # every stored run (primary run: bisection) is reproduced exactly
    b = solve_alpha_star_bisection(alpha, delta, p0, n, K, eta, s_gap)
    cands = [x for x in (found, b) if x is not None]
    return max(cands) if cands else None


def solve_alpha_star_bisection(alpha, delta, p0, n, K, eta, s_gap, tol=1e-6) -> Optional[float]:
    """Solver used for the primary run (results/hp_shift); kept for exact reproduction."""
    lo, hi = p0, alpha
    ok = None
    for _ in range(100):
        mid = (lo + hi) / 2
        s = slack(mid, p0, n, K, eta, s_gap, delta)
        if s is not None and mid + s <= alpha:
            ok, lo = mid, mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return ok


def online_hp(model, spec, top_k, gate_model, theta0, alpha_star, eta, K, stream_seed):
    """K chunks of one episode per task (n = n_tasks). theta is never clipped below."""
    theta, trace = float(theta0), []
    for c in range(K):
        r = deploy(model, spec, top_k, "learned", theta, 1, stream_seed + c, gate_model=gate_model)
        fail, q = float(r["fail"].mean()), float(r["queries"].mean())
        trace.append({"chunk": c, "theta": theta, "failure": fail, "queries": q, "n": len(r["fail"])})
        theta = min(1.01, theta - eta * (fail - alpha_star))
    f = np.array([t["failure"] for t in trace]); q = np.array([t["queries"] for t in trace])
    return {"trace": trace, "failure_rate": float(f.mean()), "queries_per_episode": float(q.mean()),
            "fallback_fraction": float(np.mean([t["theta"] <= 0 for t in trace])),
            "max_running_excess": float(np.max(np.cumsum(f - alpha_star) / np.arange(1, K + 1)))}


def gate_auc(model, spec, top_k, gate_model, stream_seed, episodes_per_task=100) -> Dict[str, float]:
    """AUC of each gate score for 'own action differs from the expert' on the policy's own states."""
    from sklearn.metrics import roc_auc_score
    r = deploy(model, spec, top_k, "random", 0., episodes_per_task, stream_seed, collect=True)
    X, y = r["X"], r["Y"]
    A = spec.n_actions  # gate_features layout: [probs (A), entropy, margin, obs...]
    return {"n": int(len(y)), "error_rate": float(y.mean()),
            "entropy": float(roc_auc_score(y, X[:, A])), "margin": float(roc_auc_score(y, -X[:, A + 1])),
            "learned": float(roc_auc_score(y, gate_model.predict_proba(X)[:, 1]))}


@torch.no_grad()
def run_hp_study(model, spec, top_k, alpha, delta, seed, p0, eta, K, slip, cue_noise):
    gate = train_help_gate(model, spec, top_k, 700000 + 10 * seed)
    cal = calibrate_gate(model, spec, top_k, "learned", alpha, delta, 100, 900000 + 10 * seed, gate)
    theta0 = cal["selected_theta"]
    hard = shifted_spec(spec, slip, cue_noise)
    out = {"seed": seed, "certified_nominal": theta0 is not None, "theta_nominal": theta0,
           "auc_nominal": gate_auc(model, spec, top_k, gate, 1_500_000 + 1000 * seed),
           "auc_shift": gate_auc(model, hard, top_k, gate, 1_600_000 + 1000 * seed)}
    if theta0 is None:
        return out
    n = spec.n_tasks
    a_star = solve_alpha_star(alpha, delta, p0, n, K, eta, theta0)
    out["alpha_star"] = a_star
    if a_star is None:
        return out
    stream = 1_700_000 + 1000 * seed
    out["shift_online_hp"] = online_hp(model, hard, top_k, gate, theta0, a_star, eta, K, stream)
    full = [deploy(model, hard, top_k, "random", 1., 1, stream + c) for c in range(K)]
    out["shift_full_help"] = {"failure_rate": float(np.mean([r["fail"].mean() for r in full])),
                              "queries_per_episode": float(np.mean([r["queries"].mean() for r in full]))}
    return out
