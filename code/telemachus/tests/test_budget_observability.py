from dataclasses import fields,replace
from pathlib import Path
import numpy as np
import pytest
from telemachus.config import BudgetConfig,ControllerConfig
from telemachus.budget import ResourceLedger,Charge,BudgetExceeded
from telemachus.simulation import NavigationWorld,SensorActuatorClient,SensorReading,Scenario
from telemachus.vision import NeuralVision
from telemachus.controller import Controller


ROOT=Path(__file__).resolve().parents[1]


def test_budget_reservation_rejects_before_side_effects():
    ledger=ResourceLedger(BudgetConfig(max_cost=1.))
    with pytest.raises(BudgetExceeded):ledger.spend("OBSERVE",ledger.charge_for("OBSERVE"))
    assert ledger.summary()["cost"]==0 and ledger.entries==[]
    assert not ledger.can_afford(Charge(float("nan")))


def test_nonfinite_budget_rejected():
    with pytest.raises(ValueError):BudgetConfig(max_cost=float("nan"))


def test_radio_reserves_before_operating_then_records_actual_traffic():
    ledger=ResourceLedger(BudgetConfig())
    ledger.spend("ASK",ledger.charge_for("ASK",n_bytes=1000,radio_latency_ms=500))
    ledger.reconcile_radio(600,300)
    assert ledger.bytes_sent==600 and ledger.logical_latency_ms==300
    assert ledger.entries[-1]["reserved_bytes"]==1000
    with pytest.raises(ValueError):ledger.reconcile_radio(1001,500)


def test_sensor_contract_excludes_truth_and_fault_flags():
    assert {f.name for f in fields(SensorReading)}=={"gps_burst","tiles","tile_centers","tick","focus"}
    vision=NeuralVision(ROOT/"artifacts/vision_ensemble.pt")
    w=NavigationWorld(19,Scenario(speed_gain=.5),ControllerConfig())
    client=SensorActuatorClient(w,vision)
    assert not hasattr(client,"scenario") and not hasattr(client,"pose") and not hasattr(client,"grid")
    assert not hasattr(client.sense(),"speed_gain")


def test_world_rng_is_independent_of_policy_compute():
    c=ControllerConfig();w1=NavigationWorld(19,Scenario(),c);w2=NavigationWorld(19,Scenario(),c)
    np.random.default_rng(19).normal(size=10000)
    r1,r2=w1.sense(),w2.sense()
    np.testing.assert_array_equal(r1.tiles,r2.tiles)
    np.testing.assert_array_equal(r1.gps_burst,r2.gps_burst)
    assert w1.world_hash==w2.world_hash


def test_real_episode_never_overspends_and_can_finish():
    cfg=ControllerConfig();v=NeuralVision(ROOT/cfg.vision_artifact)
    w=NavigationWorld(100,Scenario(),cfg)
    result=Controller(SensorActuatorClient(w,v),v,cfg,seed=100).run()
    assert result["outcome"]=="success"
    assert result["resources"]["cost"]<=cfg.budget.max_cost
    assert result["resources"]["mc_samples"]<=cfg.budget.max_mc_samples
    assert all(r.get("risk_ucb",0)<=cfg.risk_threshold for r in result["trace"] if r["route"] in ("ACT","PROBE"))


def test_tiny_budget_stops_without_unpaid_operation():
    cfg=replace(ControllerConfig(),budget=BudgetConfig(max_cost=.1))
    v=NeuralVision(ROOT/cfg.vision_artifact);w=NavigationWorld(100,Scenario(),cfg)
    result=Controller(SensorActuatorClient(w,v),v,cfg).run()
    assert result["outcome"]=="deferred" and result["steps"]==0
    assert result["resources"]["cost"]==0
