"""Actor/critic modules for the PROMETHEUS v2 long-horizon benchmark."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .model import GraphCommunication


class LongHorizonActorCritic(nn.Module):
    def __init__(self, obs_dim: int, state_dim: int, n_actions: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
        )
        self.comm = GraphCommunication(hidden_dim)
        self.actor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, n_actions))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 32))

    def policy(self, obs: torch.Tensor, top_k: int = 0, selective_receiver: torch.Tensor | None = None):
        b, n, d = obs.shape
        h = self.encoder(obs.reshape(b * n, d)).reshape(b, n, -1)
        h2, edges = self.comm(h, top_k=top_k, receiver_mask=selective_receiver)
        return self.actor(h2), edges

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)

    def forward(self, obs: torch.Tensor, state: torch.Tensor, top_k: int = 0):
        logits, edges = self.policy(obs, top_k=top_k)
        return logits, self.value(state), edges

    def ssl_projection(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return F.normalize(self.projection(h), dim=-1)
