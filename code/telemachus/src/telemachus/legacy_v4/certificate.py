from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import beta


def clopper_pearson_bounds(counts, n, alpha):
    counts = np.asarray(counts, dtype=int)
    if n <= 0 or np.any(counts < 0) or np.any(counts > n) or not 0 < alpha < 1:
        raise ValueError("Invalid binomial count or error probability")
    lo = np.zeros(counts.shape, float)
    hi = np.ones(counts.shape, float)
    m = counts > 0
    lo[m] = beta.ppf(alpha, counts[m], n - counts[m] + 1)
    m = counts < n
    hi[m] = beta.ppf(1 - alpha, counts[m] + 1, n - counts[m])
    return lo, hi


@dataclass(frozen=True)
class DecisionCertificate:
    support_lcb: np.ndarray
    risk_ucb: np.ndarray
    empirical_support: np.ndarray
    empirical_risk: np.ndarray
    n_samples: int
    simultaneous_delta: float
    n_actions: int
    max_looks: int

    def acceptable(self, support_threshold, risk_threshold):
        return (self.support_lcb >= support_threshold) & (self.risk_ucb <= risk_threshold)


def certify_all_actions(support_counts, risk_counts, n, delta, max_looks):
    """Union bound over TWO Bernoulli endpoints, ALL actions, ALL declared looks.

    This covers data-dependent action selection and stopping at declared looks.
    It is conditional on independent samples from the frozen internal generator
    for this decision. It provides no model-correctness or trajectory guarantee.
    """
    s = np.asarray(support_counts, int); r = np.asarray(risk_counts, int)
    if s.shape != r.shape or s.ndim != 1 or len(s) == 0 or max_looks <= 0:
        raise ValueError("Invalid action family")
    alpha = delta / (2 * len(s) * max_looks)
    lo, _ = clopper_pearson_bounds(s, n, alpha)
    _, hi = clopper_pearson_bounds(r, n, alpha)
    return DecisionCertificate(lo, hi, s / n, r / n, n, delta, len(s), max_looks)
