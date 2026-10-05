"""Hard-budget reservoir memory for policy distillation replay."""
from __future__ import annotations

from dataclasses import dataclass
import random
import torch


@dataclass
class PolicyMemoryItem:
    obs: torch.Tensor
    probs: torch.Tensor
    task_id: int


class PolicyReplay:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.items: list[PolicyMemoryItem] = []
        self.rng = random.Random(seed)
        self.seen = 0

    def __len__(self):
        return len(self.items)

    def add(self, obs: torch.Tensor, probs: torch.Tensor, task_id: int):
        if self.capacity <= 0:
            return
        self.seen += 1
        item = PolicyMemoryItem(obs.detach().cpu(), probs.detach().cpu(), int(task_id))
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.capacity:
                self.items[j] = item

    def sample(self, batch_size: int, device: str = "cpu"):
        if not self.items:
            raise ValueError("empty policy replay")
        k = min(int(batch_size), len(self.items))
        xs = self.rng.sample(self.items, k)
        obs = torch.stack([x.obs for x in xs]).to(device)
        probs = torch.stack([x.probs for x in xs]).to(device)
        return obs, probs
