from __future__ import annotations

from dataclasses import dataclass, field
from .config import BudgetConfig


@dataclass(frozen=True)
class Charge:
    cost: float = 0.0
    latency_ms: float = 0.0
    mc_samples: int = 0
    observations: int = 0
    queries: int = 0
    probes: int = 0
    bytes_sent: int = 0
    expert_queries: int = 0


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class ResourceLedger:
    limits: BudgetConfig
    cost: float = 0.0
    logical_latency_ms: float = 0.0
    mc_samples: int = 0
    observations: int = 0
    queries: int = 0
    probes: int = 0
    bytes_sent: int = 0
    expert_queries: int = 0
    entries: list[dict] = field(default_factory=list)

    def can_afford(self, c: Charge):
        return all([
            c.cost >= 0, c.latency_ms >= 0,
            self.cost + c.cost <= self.limits.max_cost + 1e-9,
            self.logical_latency_ms + c.latency_ms <= self.limits.max_logical_latency_ms + 1e-9,
            self.mc_samples + c.mc_samples <= self.limits.max_mc_samples,
            self.observations + c.observations <= self.limits.max_observations,
            self.queries + c.queries <= self.limits.max_queries,
            self.probes + c.probes <= self.limits.max_probes,
            self.expert_queries + c.expert_queries <= self.limits.max_expert_queries,
            min(c.mc_samples, c.observations, c.queries, c.probes, c.bytes_sent, c.expert_queries) >= 0,
        ])

    def spend(self, action: str, c: Charge):
        # Pre-authorize before operating sensors, radio, or Monte Carlo.
        if not self.can_afford(c):
            raise BudgetExceeded(action)
        self.cost += c.cost
        self.logical_latency_ms += c.latency_ms
        self.mc_samples += c.mc_samples
        self.observations += c.observations
        self.queries += c.queries
        self.probes += c.probes
        self.bytes_sent += c.bytes_sent
        self.expert_queries += c.expert_queries
        self.entries.append({"action": action, "cost": c.cost, "latency_ms": c.latency_ms,
                             "mc_samples": c.mc_samples, "bytes": c.bytes_sent})

    def charge_for(self, action: str, n_samples: int = 0, radio_latency_ms: float = 0, n_bytes: int = 0):
        b = self.limits
        if action == "PASSIVE":
            return Charge(b.passive_cost, b.passive_latency_ms)
        if action == "SIMULATE":
            return Charge(n_samples * b.sample_cost, n_samples * b.sample_latency_ms, mc_samples=n_samples)
        if action == "OBSERVE":
            return Charge(b.observe_cost, b.observe_latency_ms, observations=1)
        if action == "ASK":
            return Charge(b.query_cost, radio_latency_ms, queries=1, bytes_sent=n_bytes)
        if action == "PROBE":
            return Charge(b.probe_cost, b.probe_latency_ms, probes=1)
        if action == "EXPERT":
            return Charge(b.expert_cost, b.expert_latency_ms, expert_queries=1)
        raise ValueError(action)

    def reconcile_radio(self,actual_bytes,actual_latency_ms):
        if not self.entries or self.entries[-1]["action"]!="ASK":
            raise ValueError("No reserved radio operation")
        row=self.entries[-1]
        if not 0<=actual_bytes<=row["bytes"] or not 0<=actual_latency_ms<=row["latency_ms"]:
            raise ValueError("Actual radio operation exceeds reservation")
        row["reserved_bytes"]=row["bytes"]
        row["reserved_latency_ms"]=row["latency_ms"]
        self.bytes_sent+=actual_bytes-row["bytes"]
        self.logical_latency_ms+=actual_latency_ms-row["latency_ms"]
        row["bytes"],row["latency_ms"]=actual_bytes,actual_latency_ms

    def summary(self):
        return {k: getattr(self, k) for k in ["cost", "logical_latency_ms", "mc_samples", "observations", "queries", "probes", "bytes_sent", "expert_queries"]}
