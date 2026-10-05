from dataclasses import replace
from pathlib import Path
import copy
import json
import numpy as np
import pytest
from telemachus.config import ControllerConfig
from telemachus.controller import Controller
from telemachus.experiments import run_episode,load_resource_policy,freeze_protocol,verify_protocol
from telemachus.simulation import NavigationWorld,SensorActuatorClient,Scenario
from telemachus.vision import NeuralVision
from telemachus.resource_policy import ResourcePolicy,POLICY_FEATURES,policy_features,counterfactual_outcomes,export_forest
from telemachus.physics import TrajectoryIdentifier
from telemachus.planning import sample_worlds,candidate_controls

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def agent_world():
    cfg=ControllerConfig()
    vision=NeuralVision(ROOT/cfg.vision_artifact)
    world=NavigationWorld(2300,Scenario(),cfg)
    return Controller(SensorActuatorClient(world,vision),vision,cfg,seed=2300),world


def test_offline_interventions_do_not_mutate_live_world_or_agent(agent_world):
    agent,world=agent_world
    agent.begin_tick();d=agent.initial_compute(agent.prepare_decision())
    before=(copy.deepcopy(world.truth_record()),world.sense_index,world.packet_index,
            agent.ledger.summary(),agent.map.log_odds.copy(),agent.pose.mean.copy())
    outcomes=counterfactual_outcomes(agent,d)
    assert {"ACT","OBSERVE","ASK","SIMULATE"}<=set(outcomes)
    assert world.truth_record()==before[0]
    assert (world.sense_index,world.packet_index)==before[1:3]
    assert agent.ledger.summary()==before[3]
    np.testing.assert_array_equal(agent.map.log_odds,before[4])
    np.testing.assert_array_equal(agent.pose.mean,before[5])


def test_runtime_features_and_predictions_exclude_fault_names(agent_world):
    agent,world=agent_world
    agent.begin_tick();d=agent.initial_compute(agent.prepare_decision())
    x=policy_features(agent,d)
    forbidden={"scenario","speed_gain","camera_noise","localization_sigma","true_pose","grid","layout_family"}
    assert not forbidden&set(POLICY_FEATURES)
    policy=ResourcePolicy(ROOT/"artifacts/resource_policy.json")
    p=policy.predict(x)
    original=world.scenario
    world.scenario=replace(original,speed_gain=.4,camera_noise=.8)
    np.testing.assert_array_equal(x,policy_features(agent,d))
    assert policy.predict(x)==p
    assert all(0<=v["resolution_probability"]<=1 for v in p.values())


def test_source_blind_policy_does_not_use_source_feature_values(agent_world):
    agent,_=agent_world
    agent.begin_tick();d=agent.initial_compute(agent.prepare_decision())
    x=policy_features(agent,d);y=x.copy();y[:14]+=100
    policy=ResourcePolicy(ROOT/"artifacts/resource_policy.json")
    assert policy.predict(x,blind=True)==policy.predict(y,blind=True)


def test_safe_json_export_preserves_forest_predictions():
    from sklearn.ensemble import ExtraTreesRegressor
    rng=np.random.default_rng(81);x=rng.normal(size=(64,4));y=rng.normal(size=(64,3))
    model=ExtraTreesRegressor(n_estimators=3,max_depth=4,random_state=83).fit(x,y)
    exported=export_forest(model)
    for row in x[:8]:
        np.testing.assert_allclose(ResourcePolicy.predict_trees(exported,row),model.predict(row[None])[0],atol=1e-12)


def test_learned_policy_cannot_authorize_uncertified_motion(agent_world):
    agent,_=agent_world
    agent.begin_tick();d=agent.initial_compute(agent.prepare_decision())
    d["action"]=None
    with pytest.raises(ValueError,match="Uncertified"):agent._move(d)


def test_missing_declared_learned_policy_is_an_error(agent_world):
    agent,_=agent_world
    with pytest.raises(ValueError,match="checkpoint"):
        Controller(agent.client,agent.vision,replace(agent.cfg,resource_policy_mode="learned"))


def test_frozen_v4_reproduces_a_delivered_reference_episode():
    gold=json.loads((ROOT/"artifacts/v4_reference.json").read_text())
    cfg=ControllerConfig.from_dict(gold["config"])
    vision=NeuralVision(ROOT/cfg.vision_artifact)
    result=run_episode(gold["seed"],Scenario(**gold["scenario"]),cfg,"legacy_v4",vision)
    for key in ("outcome","reason","steps","resources","world_hash"):
        assert result[key]==gold[key]


def test_regression_learns_gain_without_differencing_high_noise():
    rng=np.random.default_rng(281);d=TrajectoryIdentifier(window=10)
    measured=np.array([2.,1.]);true=measured.copy()
    estimates=[]
    for i in range(80):
        previous=measured.copy();true[0]+=.52*.8*.4
        measured=true+rng.normal(0,.10,2)
        d.update(previous,measured,0,.8,.4,measurement_variance=.01)
        if i>20:estimates.append(d.mean)
    assert np.mean(np.abs(np.array(estimates)-.52))<.08


def test_executed_interval_risk_is_distinct_from_longer_planning_risk():
    p=np.zeros((20,20));p[:,5]=1.
    args=(p,np.array([1.5,3.,0.]),np.zeros((3,3)),np.array([[.8,0.]]),np.array([4.,3.]),1.,0.)
    first=sample_worlds(*args,np.random.default_rng(4),100,risk_horizon=1)
    full=sample_worlds(*args,np.random.default_rng(4),100,risk_horizon=3)
    assert first.risk_counts[0]==0 and full.risk_counts[0]==100
    np.testing.assert_allclose(first.mean_scores,full.mean_scores)


def test_large_heading_error_constrains_controls_before_sampling():
    controls=candidate_controls([1.,1.,np.pi],[2.,1.])
    assert np.all(controls[:,0]==0) and len(controls)==8


def test_seed_groups_are_disjoint_and_policy_file_is_verified(tmp_path):
    data=json.loads((ROOT/"artifacts/resource_policy.json").read_text())
    splits=[set(s) for s in data["splits"].values()]
    assert all(not splits[i]&splits[j] for i in range(3) for j in range(i+1,3))
    (tmp_path/"artifacts").mkdir()
    (tmp_path/"artifacts/vision_ensemble.pt").write_bytes(b"vision")
    target=tmp_path/"artifacts/resource_policy.json";target.write_text("policy1")
    cfg=ControllerConfig(resource_policy_mode="learned")
    protocol=freeze_protocol(tmp_path,tmp_path/"p.json",cfg,[25000],["telemachus"])
    verify_protocol(tmp_path,protocol)
    target.write_text("policy2")
    with pytest.raises(ValueError,match="Resource policy changed"):verify_protocol(tmp_path,protocol)
    with pytest.raises(ValueError,match="unseen"):
        freeze_protocol(tmp_path,tmp_path/"p2.json",cfg,[2000],["telemachus"])
