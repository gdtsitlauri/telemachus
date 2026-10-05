import json
from dataclasses import replace
from pathlib import Path

import pytest

from telemachus.budget import ResourceLedger, BudgetExceeded
from telemachus.certified import freeze, verify, clopper_pearson
from telemachus.config import BudgetConfig, ControllerConfig
from telemachus.controller import Controller, EXPERT_METHODS
from telemachus.experiments import run_episode, DEVELOPMENT_SEEDS
from telemachus.simulation import NavigationWorld, SensorActuatorClient, scenarios
from telemachus.vision import NeuralVision

ROOT = Path(__file__).resolve().parents[1]


def test_expert_is_disabled_by_default_and_budgeted():
    assert BudgetConfig().max_expert_queries == 0
    ledger = ResourceLedger(BudgetConfig(max_expert_queries=1))
    ledger.spend("EXPERT", ledger.charge_for("EXPERT"))
    assert ledger.expert_queries == 1 and ledger.cost == BudgetConfig().expert_cost
    with pytest.raises(BudgetExceeded):
        ledger.spend("EXPERT", ledger.charge_for("EXPERT"))
    with pytest.raises(ValueError):
        BudgetConfig(max_expert_queries=-1)


def test_expert_report_is_local_and_hedged():
    cfg = ControllerConfig.load(ROOT / "configs/cpu_core.json")
    world = NavigationWorld(31, scenarios("development")[0], cfg)
    burst, centers, p, var = world.expert_report()
    assert burst.shape == (6, 3) and len(centers) == len(p) == len(var)
    assert set(p.tolist()) <= {.03, .97}


def test_expert_used_only_by_expert_methods_and_within_limit():
    base = ControllerConfig.load(ROOT / "configs/cpu_core.json")
    cfg = replace(base, safety_margin=.08, budget=replace(base.budget, max_expert_queries=2))
    vision = NeuralVision(ROOT / cfg.vision_artifact)
    hard = next(s for s in scenarios("development") if s.name == "compound")
    for method in ("no_learning", "budget_voi"):
        r = run_episode(2401, hard, cfg, method, vision, root=ROOT)
        assert r["resources"]["expert_queries"] <= 2
        if method not in EXPERT_METHODS:
            assert r["resources"]["expert_queries"] == 0


def test_certified_protocol_rejects_overlap_and_detects_tampering(tmp_path):
    cfg = ControllerConfig.load(ROOT / "configs/cpu_core.json")
    with pytest.raises(ValueError):
        freeze(ROOT, tmp_path / "p.json", cfg, ["telemachus"], range(30000, 30010), range(30005, 30015), .03, .1)
    with pytest.raises(ValueError):
        freeze(ROOT, tmp_path / "p.json", cfg, ["telemachus"], range(16000, 16010), range(40000, 40010), .03, .1,
               development_seeds=DEVELOPMENT_SEEDS)
    m = freeze(ROOT, tmp_path / "p.json", cfg, ["telemachus"], range(30000, 30010), range(40000, 40010), .03, .1,
               development_seeds=DEVELOPMENT_SEEDS)
    verify(ROOT, m)
    m["alpha"] = .5
    with pytest.raises(ValueError):
        verify(ROOT, m)


def test_clopper_pearson_interval():
    lo, hi = clopper_pearson(0, 480)
    assert lo == 0 and 0 < hi < .01
    lo, hi = clopper_pearson(12, 480)
    assert lo < 12 / 480 < hi
