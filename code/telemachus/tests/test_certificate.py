import numpy as np
import pytest
from scipy.stats import beta
from telemachus.certificate import certify_all_actions,clopper_pearson_bounds


def test_action_selection_and_look_family_are_both_charged():
    c=certify_all_actions(np.array([40,38,35,42]),np.array([0,2,1,0]),48,.1,3)
    alpha=.1/(2*4*3)
    assert c.risk_ucb[0]==pytest.approx(beta.ppf(1-alpha,1,48))
    assert c.support_lcb[3]==pytest.approx(beta.ppf(alpha,42,7))
    assert c.risk_ucb[0] > beta.ppf(1-.1,1,48)


def test_simultaneous_coverage_under_selected_actions_and_adaptive_looks():
    rng=np.random.default_rng(51);a=8;trials=1200
    p=np.linspace(.05,.95,a);q=p[::-1]*.3
    sc=np.zeros((trials,a),int);rc=np.zeros_like(sc);miss=np.zeros(trials,bool);previous=0
    for n in (48,144,432):
        sc+=rng.binomial(n-previous,p,(trials,a));rc+=rng.binomial(n-previous,q,(trials,a))
        lo,_=clopper_pearson_bounds(sc,n,.1/(2*a*3))
        _,hi=clopper_pearson_bounds(rc,n,.1/(2*a*3))
        miss |= ((lo>p)|(hi<q)).any(1)
        previous=n
    assert miss.mean()<.13


@pytest.mark.parametrize("counts,n,alpha",[([-1],10,.1),([11],10,.1),([1],0,.1),([1],10,0.)])
def test_invalid_binomial_problem_rejected(counts,n,alpha):
    with pytest.raises(ValueError):clopper_pearson_bounds(counts,n,alpha)
