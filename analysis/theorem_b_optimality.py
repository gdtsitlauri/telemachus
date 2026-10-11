"""Proposition 2 (lower bound): how far is Theorem B from the best possible finite-horizon guarantee?

Lower bound. In a world where every non-fallback action fails, the best any online rule can do is to use the fallback
on every episode, so the realised failure count is Binomial(nK, p0). By Slud's inequality, any rule whose realised
failure rate is <= alpha with probability >= 1 - delta in every world needs
    alpha >= p0 + z_{1-delta} * sqrt(p0 (1 - p0) / (n K))      (up to integer rounding).
Upper bound. Theorem B is feasible for every alpha >= alpha_B, where alpha_B = min over alpha* of its bound.

This script computes both for the multi-agent study (alpha 0.10, delta 0.10, p0 0.08, n 6, K 1200, eta 0.5,
s_f - s_1 = theta_nominal = 0.45), checks the lower bound by simulation, and sweeps K.
Output: results/theorem_b_optimality.json
"""
import json
import math
import os

import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def theorem_b_bound(alpha_star, p0, n, K, eta, gap, delta):
    v = p0 * (1 - p0)
    g = alpha_star - p0
    u = 2 * g / (v / n + 2 * g / 3)
    return alpha_star + gap / (eta * K) + 1 / K + math.log(K / delta) / (u * K)


def alpha_b(p0, n, K, eta, gap, delta):
    grid = p0 + np.geomspace(1e-6, 0.5, 20000)
    vals = [theorem_b_bound(a, p0, n, K, eta, gap, delta) for a in grid]
    i = int(np.argmin(vals))
    return float(vals[i]), float(grid[i])


def alpha_lower(p0, n, K, delta):
    return p0 + stats.norm.ppf(1 - delta) * math.sqrt(p0 * (1 - p0) / (n * K))


def simulate_lower(p0, n, K, alpha, trials=200_000, seed=0):
    """Probability that the always-fallback rule (the best possible in the adversarial world) exceeds alpha."""
    rng = np.random.default_rng(seed)
    x = rng.binomial(n * K, p0, size=trials) / (n * K)
    return float((x > alpha).mean())


def main():
    p0, n, K, eta, gap, delta, alpha = 0.08, 6, 1200, 0.5, 0.45, 0.10, 0.10
    aB, astar = alpha_b(p0, n, K, eta, gap, delta)
    aL = alpha_lower(p0, n, K, delta)
    out = {
        "setting": {"p0": p0, "n": n, "K": K, "eta": eta, "s_f_minus_s_1": gap, "delta": delta, "alpha_study": alpha},
        "lower_bound_alpha": aL,
        "theorem_b_min_alpha": aB,
        "theorem_b_argmin_alpha_star": astar,
        "gap_ratio_(alphaB-p0)/(alphaL-p0)": (aB - p0) / (aL - p0),
        "simulated_P(rate>alpha_L)_always_fallback": simulate_lower(p0, n, K, aL),
        "simulated_P(rate>alpha_L-0.001)_always_fallback": simulate_lower(p0, n, K, aL - 0.001),
        "sweep_K": [],
    }
    for k in (100, 300, 1000, 3000, 10000, 30000, 100000):
        b, _ = alpha_b(p0, n, k, eta, gap, delta)
        lo = alpha_lower(p0, n, k, delta)
        out["sweep_K"].append({"K": k, "alpha_lower": lo, "alpha_theorem_b": b, "ratio_of_gaps": (b - p0) / (lo - p0)})
    path = os.path.join(ROOT, "results", "theorem_b_optimality.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
