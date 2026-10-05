"""Long-horizon continual cooperative environment for PROMETHEUS v2.0.

This is a small CPU-friendly Dec-POMDP-like benchmark.  Three agents move on a
1-D grid toward an episode-specific goal.  Exactly one randomly chosen scout
receives the goal cue; the others must infer it through learned communication.
Each continual task changes the motor action permutation and goal pair.  The
shared reward is delayed over a multi-step trajectory.

It is intentionally an internal validation environment, not a substitute for
MEAL/JaxMARL/SMAC-style external benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import itertools
import torch


@dataclass(frozen=True)
class LongHorizonSpec:
    n_agents: int = 3
    grid_size: int = 7
    horizon: int = 10
    n_tasks: int = 6
    cue_noise: float = 0.06
    slip_prob: float = 0.04

    @property
    def obs_dim(self) -> int:
        # own position, private goal cue, scout flag, progress + task one-hot + agent one-hot
        return 4 + self.n_tasks + self.n_agents

    @property
    def state_dim(self) -> int:
        # all positions + goal + scout one-hot + task one-hot + progress
        return self.n_agents + 1 + self.n_agents + self.n_tasks + 1

    @property
    def n_actions(self) -> int:
        return 3


class ContinualSignalGrid:
    """Vectorized cooperative long-horizon environment."""

    def __init__(self, spec: LongHorizonSpec, seed: int = 2026, device: str = "cpu"):
        self.spec = spec
        self.device = torch.device(device)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        # Six permutations of physical deltas; task context identifies the current dynamics.
        perms = list(itertools.permutations([-1, 0, 1]))
        self.action_deltas = torch.tensor(perms[: spec.n_tasks], dtype=torch.long, device=self.device)
        pairs = [(1, 5), (0, 6), (2, 4), (1, 6), (0, 5), (2, 6)]
        self.goal_pairs = torch.tensor(pairs[: spec.n_tasks], dtype=torch.long, device=self.device)
        self.batch_size = 0
        self.task_id = 0
        self.positions = None
        self.goals = None
        self.scouts = None
        self.t = 0

    def reset(self, task_id: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.task_id = int(task_id)
        self.batch_size = int(batch_size)
        s = self.spec
        self.positions = torch.randint(0, s.grid_size, (batch_size, s.n_agents), generator=self.gen, device=self.device)
        choice = torch.randint(0, 2, (batch_size,), generator=self.gen, device=self.device)
        self.goals = self.goal_pairs[self.task_id, choice]
        self.scouts = torch.randint(0, s.n_agents, (batch_size,), generator=self.gen, device=self.device)
        self.t = 0
        return self.observe(), self.global_state()

    def observe(self) -> torch.Tensor:
        s = self.spec
        b = self.batch_size
        pos = 2.0 * self.positions.float() / (s.grid_size - 1) - 1.0
        pos = pos[..., None]
        goal_norm = 2.0 * self.goals.float() / (s.grid_size - 1) - 1.0
        scout_mask = torch.nn.functional.one_hot(self.scouts, s.n_agents).float()
        cue = scout_mask * goal_norm[:, None]
        if s.cue_noise > 0:
            noise = s.cue_noise * torch.randn(b, s.n_agents, generator=self.gen, device=self.device)
            cue = cue + scout_mask * noise
        progress = torch.full((b, s.n_agents, 1), self.t / max(s.horizon - 1, 1), device=self.device)
        task = torch.zeros(b, s.n_agents, s.n_tasks, device=self.device)
        task[:, :, self.task_id] = 1.0
        agent = torch.eye(s.n_agents, device=self.device)[None].expand(b, -1, -1)
        return torch.cat([pos, cue[..., None], scout_mask[..., None], progress, task, agent], dim=-1)

    def global_state(self) -> torch.Tensor:
        s = self.spec
        b = self.batch_size
        pos = 2.0 * self.positions.float() / (s.grid_size - 1) - 1.0
        goal = (2.0 * self.goals.float() / (s.grid_size - 1) - 1.0)[:, None]
        scout = torch.nn.functional.one_hot(self.scouts, s.n_agents).float()
        task = torch.zeros(b, s.n_tasks, device=self.device)
        task[:, self.task_id] = 1.0
        progress = torch.full((b, 1), self.t / max(s.horizon - 1, 1), device=self.device)
        return torch.cat([pos, goal, scout, task, progress], dim=-1)

    def expert_actions(self) -> torch.Tensor:
        """Shortest-path expert under the task-specific action permutation."""
        desired = torch.sign(self.goals[:, None] - self.positions).long()  # -1,0,+1
        deltas = self.action_deltas[self.task_id]  # [A]
        eq = desired[..., None] == deltas[None, None, :]
        return eq.float().argmax(dim=-1).long()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        s = self.spec
        actions = actions.long().clamp(0, s.n_actions - 1)
        deltas = self.action_deltas[self.task_id][actions]
        if s.slip_prob > 0:
            slip = torch.rand(self.batch_size, s.n_agents, generator=self.gen, device=self.device) < s.slip_prob
            random_actions = torch.randint(0, s.n_actions, (self.batch_size, s.n_agents), generator=self.gen, device=self.device)
            random_delta = self.action_deltas[self.task_id][random_actions]
            deltas = torch.where(slip, random_delta, deltas)
        self.positions = (self.positions + deltas).clamp(0, s.grid_size - 1)
        self.t += 1

        dist = (self.positions - self.goals[:, None]).abs().float()
        norm_dist = dist.mean(dim=1) / (s.grid_size - 1)
        frac_at_goal = (dist == 0).float().mean(dim=1)
        all_at_goal = (dist == 0).all(dim=1).float()
        # Dense shaping + true team coordination bonus.  Maximum per-step reward is 1.25.
        reward = 0.25 * frac_at_goal + 1.0 * all_at_goal - 0.35 * norm_dist - 0.01
        done = torch.full((self.batch_size,), self.t >= s.horizon, dtype=torch.bool, device=self.device)
        info = {
            "success": all_at_goal,
            "mean_distance": dist.mean(dim=1),
            "fraction_at_goal": frac_at_goal,
        }
        return self.observe(), self.global_state(), reward, info
