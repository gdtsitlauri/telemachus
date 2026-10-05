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

    def __post_init__(self):
        for k, v in asdict(self).items():
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
    stability_threshold: float = 0.36
    mc_looks: tuple[int, ...] = (48, 144, 432)
    mc_delta: float = 0.10
    horizon: int = 3
    max_epistemic_streak: int = 4
    source_threshold: float = 0.45
    channel_min_delivery: float = 0.45
    channel_backend: Literal["analytic", "sionna"] = "analytic"
    adapt_dynamics: bool = True
    vision_artifact: str = "artifacts/vision_ensemble.pt"
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    def __post_init__(self):
        if not 0 < self.risk_threshold < 1 or not 0 < self.stability_threshold <= 1:
            raise ValueError("Invalid risk/support threshold")
        if not 0 < self.mc_delta < 1:
            raise ValueError("mc_delta must be in (0,1)")
        if not self.mc_looks or any(n <= 0 for n in self.mc_looks):
            raise ValueError("Monte Carlo looks must be positive")
        if tuple(sorted(set(self.mc_looks))) != tuple(self.mc_looks):
            raise ValueError("Monte Carlo looks must be distinct and increasing")
        if self.dt <= 0 or self.max_steps <= 0:
            raise ValueError("Invalid control duration")
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
