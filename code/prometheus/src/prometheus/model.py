"""Neural components for PROMETHEUS proof-of-concept."""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class LocalEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphCommunication(nn.Module):
    """Single-head learned message passing with an explicit top-k edge budget."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, h: torch.Tensor, top_k: int, receiver_mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: [B,N,H]
        b, n, d = h.shape
        if top_k <= 0 or n <= 1:
            return h, torch.zeros(b, n, n, dtype=torch.bool, device=h.device)
        top_k = min(top_k, n - 1)
        q, k, v = self.q(h), self.k(h), self.v(h)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
        eye = torch.eye(n, dtype=torch.bool, device=h.device)[None]
        scores = scores.masked_fill(eye, float("-inf"))
        vals, idx = torch.topk(scores, k=top_k, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        sparse_scores = scores.masked_fill(~mask, float("-inf"))
        if receiver_mask is not None:
            mask = mask & receiver_mask[..., None]
            sparse_scores = scores.masked_fill(~mask, float("-inf"))
            # Rows with no allowed edges would softmax to NaN. Replace them by
            # zeros after computing only over active rows.
            active = mask.any(dim=-1, keepdim=True)
            safe_scores = torch.where(active, sparse_scores, torch.zeros_like(sparse_scores))
            attn = torch.softmax(safe_scores, dim=-1) * mask.float()
        else:
            attn = torch.softmax(sparse_scores, dim=-1)
        msg = torch.matmul(attn, v)
        return self.out(torch.cat([h, msg], dim=-1)), mask


class CollectivePolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 48):
        super().__init__()
        self.encoder = LocalEncoder(obs_dim, hidden_dim)
        self.comm = GraphCommunication(hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, 2)
        self.projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 32))

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        b, n, d = obs.shape
        return self.encoder(obs.reshape(b * n, d)).reshape(b, n, -1)

    def forward(self, obs: torch.Tensor, top_k: int = 0, selective_comm_threshold: float | None = None):
        h = self.encode(obs)
        receiver_mask = None
        if selective_comm_threshold is not None and top_k > 0:
            local_logits = self.policy_head(h)
            local_p = torch.softmax(local_logits, dim=-1)
            local_entropy = -(local_p * torch.log(local_p.clamp_min(1e-8))).sum(dim=-1)
            receiver_mask = local_entropy > selective_comm_threshold
        h2, edges = self.comm(h, top_k=top_k, receiver_mask=receiver_mask)
        logits = self.policy_head(h2)
        return logits, edges

    def ssl_projection(self, obs_flat: torch.Tensor) -> torch.Tensor:
        h = self.encoder(obs_flat)
        return F.normalize(self.projection(h), dim=-1)
