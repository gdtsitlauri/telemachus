import numpy as np
import pytest
from telemachus.physics import unicycle_step,rk4_step,wrap_angle,PoseFilter,DynamicsIdentifier
from telemachus.planning import sampled_collision


@pytest.mark.parametrize("control",[(.8,0.),(.8,1.8),(.25,-1.2),(0.,1.5)])
def test_exact_motion_matches_numerical_integration(control):
    start=np.array([3.,2.,2.9]);z=start.copy()
    for _ in range(20):z=rk4_step(z,control,.4/20,speed_gain=.62)
    expected=unicycle_step(start,control,.4,speed_gain=.62)
    np.testing.assert_allclose(z,expected,atol=1e-8)


def test_collision_inflates_obstacle_for_robot_body():
    grid=np.zeros((1,20,20),bool);grid[0,4,4]=True
    points=np.array([[[1.90,2.25,0.],[1.85,2.25,0.],[2.25,2.25,0.]]])
    assert sampled_collision(points,grid).tolist()==[[True,False,True]]


def test_burst_filter_estimates_sensor_noise_without_true_sigma():
    rng=np.random.default_rng(7);f=PoseFilter([1.,1.,0.])
    for _ in range(30):f.update(rng.normal([1.,1.,0.],[.2,.2,.1],(8,3)))
    assert .015 < f.noise_evidence < .08
    assert np.linalg.eigvalsh(f.cov).min()>0


def test_small_passive_burst_does_not_erase_recent_noise_evidence():
    f=PoseFilter([1.,1.,0.]);f.measurement_var=np.array([.09,.09,.03]);f.cov=np.diag([.04,.04,.03])
    f.update(np.repeat([[1.,1.,0.]],3,axis=0))
    assert f.cov[0,0]>.007
    f.update(np.repeat([[1.,1.,0.]],6,axis=0),focused=True)
    assert f.cov[0,0]<.0001


def test_dynamics_identifier_learns_from_measured_motion():
    d=DynamicsIdentifier();rng=np.random.default_rng(17)
    for _ in range(30):
        d.update([1,1],[1+.5*.8*.4+rng.normal(0,.006),1],0,.8,.4,weight=4)
    assert abs(d.mean-.5)<.04
