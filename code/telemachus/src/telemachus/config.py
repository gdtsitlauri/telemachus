from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class BudgetConfig:
    max_cost: float = 120.0
    max_logical_latency_ms: float = 6500.0
    max_mc_samples: int = 20000
    max_observations: int = 18
    max_queries: int = 12
    max_probes: int = 5
    passive_cost: float = 0.18
    sample_cost: float = 0.004
    observe_cost: float = 1.5
    query_cost: float = 1.1
    probe_cost: float = 2.0
    passive_latency_ms: float = 2.0
    sample_latency_ms: float = 0.012
    observe_latency_ms: float = 12.0
    probe_latency_ms: float = 30.0
    # Human/remote expert: near-exact local map and pose, expensive and rare.
    # max_expert_queries=0 disables it (reproduces v5).
    max_expert_queries: int = 0
    expert_cost: float = 6.0
    expert_latency_ms: float = 400.0

    def __post_init__(self):
        for k, v in asdict(self).items():
            if k == "max_expert_queries":
                if not isinstance(v, int) or v < 0:
                    raise ValueError(f"Invalid expert query limit: {v}")
                continue
            if not math.isfinite(v) or v < 0 or (k.startswith("max_") and v == 0):
                raise ValueError(f"Invalid non-positive budget: {k}={v}")


@dataclass(frozen=True)
class ControllerConfig:
    cell_size: float = 0.5
    world_size: float = 10.0
    dt: float = 0.4
    max_steps: int = 160
    goal_tolerance: float = 0.32
    risk_threshold: float = 0.18
    # Extra clearance (m) added to the robot radius inside the internal risk
    # model only. 0 reproduces v5. Calibrated by telemachus.calibration.
    safety_margin: float = 0.0
    # When no action meets risk AND near-optimality, execute the best action
    # meeting the risk bound alone instead of stopping (False reproduces v5).
    safe_recovery: bool = False
    stability_threshold: float = 0.36
    mc_looks: tuple[int, ...] = (48, 144, 432)
    mc_delta: float = 0.10
    horizon: int = 3
    risk_horizon: int = 3
    collision_substeps: int = 8
    stability_regret: float = .07
    max_epistemic_streak: int = 4
    source_threshold: float = 0.45
    channel_min_delivery: float = 0.45
    channel_backend: Literal["analytic", "sionna"] = "analytic"
    adapt_dynamics: bool = True
    learned_resource_policy: str | None = "artifacts/resource_policy.json"
    resource_policy_mode: str = "heuristic"
    reserve_fraction: float = .55
    pose_georeferencing: bool = True
    trajectory_identification: bool = True
    turn_before_move: float = .9
    vision_artifact: str = "artifacts/vision_ensemble.pt"
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    def __post_init__(self):
        if not 0 < self.risk_threshold < 1 or not 0 < self.stability_threshold <= 1:
            raise ValueError("Invalid risk/support threshold")
        if not math.isfinite(self.safety_margin) or not 0 <= self.safety_margin <= .5:
            raise ValueError("safety_margin must be in [0, 0.5]")
        if not 0 < self.mc_delta < 1:
            raise ValueError("mc_delta must be in (0,1)")
        if not self.mc_looks or any(n <= 0 for n in self.mc_looks):
            raise ValueError("Monte Carlo looks must be positive")
        if tuple(sorted(set(self.mc_looks))) != tuple(self.mc_looks):
            raise ValueError("Monte Carlo looks must be distinct and increasing")
        if self.dt <= 0 or self.max_steps <= 0:
            raise ValueError("Invalid control duration")
        if not all(isinstance(v,int) for v in (self.horizon,self.risk_horizon,self.collision_substeps,self.max_steps)) or not 1<=self.risk_horizon<=self.horizon or self.collision_substeps<2:
            raise ValueError("Invalid action duration or collision integration")
        if self.resource_policy_mode not in ("heuristic","learned","blind"):
            raise ValueError("Unknown resource policy mode")
        if not 0<=self.reserve_fraction<=1 or not math.isfinite(self.stability_regret) or self.stability_regret<0:
            raise ValueError("Invalid mission reserve or regret")
        if not math.isfinite(self.turn_before_move) or not 0<self.turn_before_move<=math.pi:
            raise ValueError("Invalid heading threshold")
        if not all(math.isfinite(x) for x in (self.dt,self.cell_size,self.world_size,self.goal_tolerance)):
            raise ValueError("Non-finite geometry")
        if self.cell_size <= 0 or self.world_size <= 0 or self.goal_tolerance <= 0:
            raise ValueError("Invalid geometry")
        if not math.isclose(self.world_size/self.cell_size,round(self.world_size/self.cell_size)):
            raise ValueError("World must contain an integer number of cells")

    def to_dict(self):
        values=asdict(self)
        values["mc_looks"]=list(self.mc_looks)
        return values

    @classmethod
    def from_dict(cls, d):
        values = dict(d)
        values["budget"] = BudgetConfig(**values.get("budget", {}))
        if "mc_looks" in values:
            values["mc_looks"] = tuple(values["mc_looks"])
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text()))
