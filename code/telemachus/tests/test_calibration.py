import numpy as np
import pytest

from telemachus.calibration import binomial_pvalue, learn_then_test, apply_setting, default_grid
from telemachus.config import ControllerConfig
from telemachus.simulation import scenarios


def test_binomial_pvalue_is_monotone_and_valid():
    assert binomial_pvalue(0, 300, .03) < 1e-3
    assert binomial_pvalue(9, 300, .03) > .5
    ps = [binomial_pvalue(k, 200, .05) for k in range(20)]
    assert all(a <= b for a, b in zip(ps, ps[1:]))
    with pytest.raises(ValueError):
        binomial_pvalue(5, 3, .1)


def test_fixed_sequence_stops_at_first_non_rejection():
    counts = [({"safety_margin": .16}, 300, 0), ({"safety_margin": .08}, 300, 2),
              ({"safety_margin": .04}, 300, 20), ({"safety_margin": 0.}, 300, 0)]
    selected, res = learn_then_test(counts, alpha=.03, delta=.1)
    assert selected == 1  # the later 0-collision setting must NOT be selected
    assert [r.rejected for r in res] == [True, True, False, False]


def test_no_certificate_when_safest_setting_fails():
    selected, _ = learn_then_test([({"safety_margin": .16}, 50, 10)], alpha=.03, delta=.1)
    assert selected is None


def test_ltt_guarantee_holds_in_simulation():
    """Empirical check of P(true risk(lambda_hat) > alpha) <= delta."""
    rng = np.random.default_rng(0)
    true_risk = np.array([.005, .012, .022, .029, .034, .05, .09])  # most -> least conservative
    alpha, delta, n = .03, .1, 400
    violations = 0
    trials = 4000
    for _ in range(trials):
        losses = rng.binomial(n, true_risk)
        sel, _ = learn_then_test([({"i": i}, n, int(l)) for i, l in enumerate(losses)], alpha, delta)
        if sel is not None and true_risk[sel] > alpha:
            violations += 1
    assert violations / trials <= delta


def test_settings_are_restricted_and_calibration_split_matches_holdout():
    cfg = ControllerConfig()
    assert apply_setting(cfg, {"safety_margin": .08}).safety_margin == .08
    with pytest.raises(ValueError):
        apply_setting(cfg, {"max_steps": 10})
    grid = [g["safety_margin"] for g in default_grid()]
    assert grid == sorted(grid, reverse=True)
    assert [s.__dict__ for s in scenarios("calibration")] == [s.__dict__ for s in scenarios("holdout")]
    with pytest.raises(ValueError):
        ControllerConfig(safety_margin=-.01)
