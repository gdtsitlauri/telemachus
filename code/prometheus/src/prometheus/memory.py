"""Budgeted replay memory."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Tuple

import torch


@dataclass
class MemoryItem:
    obs: torch.Tensor
    label: int
    task_id: int
    priority: float


class BudgetedReplay:
    """Reservoir-like memory with priority replacement under a hard item budget."""
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.items: List[MemoryItem] = []
        self.rng = random.Random(seed)
        self.seen = 0

    def __len__(self):
        return len(self.items)

    def add(self, obs: torch.Tensor, label: int, task_id: int, priority: float = 1.0):
        if self.capacity <= 0:
            return
        self.seen += 1
        item = MemoryItem(obs.detach().cpu(), int(label), int(task_id), float(priority))
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        # Blend reservoir sampling with priority: low-priority items are easier
        # to replace, preserving diversity while rewarding informative examples.
        j = self.rng.randrange(self.seen)
        if j < self.capacity:
            low = min(range(len(self.items)), key=lambda i: self.items[i].priority + 0.05 * self.rng.random())
            self.items[low] = item

    def sample(self, batch_size: int, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.items:
            raise ValueError("cannot sample empty memory")
        k = min(batch_size, len(self.items))
        chosen = self.rng.sample(self.items, k)
        obs = torch.stack([x.obs for x in chosen]).to(device)
        labels = torch.tensor([x.label for x in chosen], dtype=torch.long, device=device)
        return obs, labels
