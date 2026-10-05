"""Synthetic continual cooperative contextual-bandit benchmark for PROMETHEUS.

The benchmark is intentionally small enough to run on CPU while exposing the
research ingredients we care about: partial observability, multi-agent
coordination, sequential task shift, limited labels, memory and communication.
It is a proof-of-concept benchmark, not a replacement for MEAL/SMAC/MPE.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch


@dataclass(frozen=True)
class BenchmarkSpec:
    n_agents: int = 3
    latent_dim: int = 6
    local_signal_dim: int = 2
    nuisance_dim: int = 6
    n_tasks: int = 8
    task_context_dim: int = 8
    obs_noise: float = 0.35
    nuisance_scale: float = 1.0
    task_rotation: float = 0.55

    @property
    def obs_dim(self) -> int:
        return self.local_signal_dim + self.nuisance_dim + self.task_context_dim


class ContinualCoordinationBandit:
    """A sequence of cooperative, partially observed binary decision tasks.

    Each episode samples a latent global state z. Each agent observes a disjoint
    slice of z plus nuisance variables and a task cue. The shared correct action
    depends on all latent dimensions, so communication can be useful. Task
    weights change over time, so sequential training can forget older mappings.
    """

    def __init__(self, spec: BenchmarkSpec, seed: int = 0, device: str = "cpu"):
        self.spec = spec
        self.device = torch.device(device)
        g = torch.Generator(device="cpu").manual_seed(seed)

        # Construct related but non-identical task separators. Gradual rotation
        # plus orthogonal perturbation yields transfer and interference.
        w0 = torch.randn(spec.latent_dim, generator=g)
        w0 = w0 / w0.norm()
        weights = [w0]
        for _ in range(1, spec.n_tasks):
            v = torch.randn(spec.latent_dim, generator=g)
            prev = weights[-1]
            v = v - torch.dot(v, prev) * prev
            v = v / (v.norm() + 1e-8)
            theta = spec.task_rotation
            w = math.cos(theta) * prev + math.sin(theta) * v
            weights.append(w / w.norm())
        self.task_weights = torch.stack(weights).to(self.device)

        # Fixed nuisance projection creates structured distractors that SSL can
        # denoise rather than merely memorizing independent Gaussian noise.
        proj = torch.randn(spec.latent_dim, spec.nuisance_dim, generator=g)
        proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-8)
        self.nuisance_proj = proj.to(self.device)

    def sample(self, task_id: int, batch_size: int, generator: Optional[torch.Generator] = None):
        s = self.spec
        z = torch.randn(batch_size, s.latent_dim, generator=generator, device=self.device)
        margins = z @ self.task_weights[task_id]
        y = (margins > 0).long()

        nuisance_base = z @ self.nuisance_proj
        obs = []
        context = torch.zeros(batch_size, s.task_context_dim, device=self.device)
        context[:, task_id % s.task_context_dim] = 1.0
        for agent in range(s.n_agents):
            start = (agent * s.local_signal_dim) % s.latent_dim
            idx = torch.arange(start, start + s.local_signal_dim, device=self.device) % s.latent_dim
            local = z[:, idx] + s.obs_noise * torch.randn(
                batch_size, s.local_signal_dim, generator=generator, device=self.device
            )
            nuisance = s.nuisance_scale * nuisance_base + 0.6 * torch.randn(
                batch_size, s.nuisance_dim, generator=generator, device=self.device
            )
            # Agent-specific cyclic nuisance shift prevents all observations from
            # being identical while preserving common latent structure.
            nuisance = torch.roll(nuisance, shifts=agent, dims=1)
            obs.append(torch.cat([local, nuisance, context], dim=1))
        return torch.stack(obs, dim=1), y

    @staticmethod
    def team_reward(actions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Dense cooperative reward in [0,1], one scalar per joint episode."""
        correct = (actions == labels[:, None]).float()
        # Reward coordination: partial credit for individual correctness plus a
        # bonus when the full team is correct.
        return 0.5 * correct.mean(dim=1) + 0.5 * correct.min(dim=1).values
