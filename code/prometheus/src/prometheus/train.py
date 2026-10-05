"""Training and evaluation utilities for PROMETHEUS."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import copy
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .benchmark import BenchmarkSpec, ContinualCoordinationBandit
from .memory import BudgetedReplay
from .model import CollectivePolicy


@dataclass
class MethodConfig:
    name: str = "prometheus_full"
    use_ssl: bool = True
    use_offline_warmstart: bool = True
    use_replay: bool = True
    use_communication: bool = True
    use_active_queries: bool = True
    comm_top_k: int = 1
    selective_comm: bool = False
    selective_query: bool = False
    comm_entropy_threshold: float = 0.66
    query_entropy_threshold: float = 0.66
    memory_budget: int = 256
    query_budget_per_task: int = 24
    ssl_steps: int = 40
    warmstart_steps: int = 50
    online_steps_per_task: int = 60
    batch_size: int = 48
    replay_batch: int = 32
    hidden_dim: int = 48
    lr: float = 2e-3
    entropy_coef: float = 0.01
    query_weight: float = 2.0
    replay_weight: float = 1.0
    seed: int = 0


@dataclass
class RunSummary:
    method: str
    seed: int
    final_mean_accuracy: float
    final_mean_return: float
    avg_forgetting: float
    mean_first_window_accuracy: float
    queries_used: int
    messages_per_episode: float
    memory_items: int


def set_seed(seed: int):
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.15) -> torch.Tensor:
    n = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.T / temp
    sim.fill_diagonal_(float("-inf"))
    positives = torch.arange(2 * n, device=z.device)
    positives = (positives + n) % (2 * n)
    return F.cross_entropy(sim, positives)


def pretrain_ssl(model: CollectivePolicy, env: ContinualCoordinationBandit, cfg: MethodConfig, opt):
    if not cfg.use_ssl or cfg.ssl_steps <= 0:
        return
    model.train()
    for step in range(cfg.ssl_steps):
        task = step % env.spec.n_tasks
        obs, _ = env.sample(task, cfg.batch_size)
        x = obs.reshape(-1, env.spec.obs_dim)
        aug1 = x + 0.12 * torch.randn_like(x)
        aug2 = x + 0.12 * torch.randn_like(x)
        # Feature dropout, except the one-hot task cue, forces robust local reps.
        signal_end = env.spec.local_signal_dim + env.spec.nuisance_dim
        drop1 = (torch.rand_like(aug1[:, :signal_end]) > 0.12).float()
        drop2 = (torch.rand_like(aug2[:, :signal_end]) > 0.12).float()
        aug1[:, :signal_end] *= drop1
        aug2[:, :signal_end] *= drop2
        z1 = model.ssl_projection(aug1)
        z2 = model.ssl_projection(aug2)
        loss = nt_xent(z1, z2)
        opt.zero_grad()
        loss.backward()
        opt.step()


def warmstart_offline(model: CollectivePolicy, env: ContinualCoordinationBandit, cfg: MethodConfig, opt):
    """Behavior-cloning + conservative one-step value warm start on task 0 only.

    This provides genuine demonstration/offline-learning coverage without giving
    the method future-task labels before the continual sequence begins.
    """
    if not cfg.use_offline_warmstart or cfg.warmstart_steps <= 0:
        return
    model.train()
    top_k = cfg.comm_top_k if cfg.use_communication else 0
    for _ in range(cfg.warmstart_steps):
        obs, y = env.sample(0, cfg.batch_size)
        logits, _ = model(obs, top_k=top_k, selective_comm_threshold=(cfg.comm_entropy_threshold if cfg.selective_comm else None))
        # Expert behavior cloning.
        bc = F.cross_entropy(logits.reshape(-1, 2), y[:, None].expand(-1, env.spec.n_agents).reshape(-1))

        # One-step CQL-style conservative objective on a mixed behavior dataset:
        # data actions are expert 70% of the time and random otherwise.
        random_a = torch.randint(0, 2, (cfg.batch_size, env.spec.n_agents), device=obs.device)
        expert_a = y[:, None].expand(-1, env.spec.n_agents)
        expert_mask = (torch.rand(cfg.batch_size, env.spec.n_agents, device=obs.device) < 0.7)
        data_a = torch.where(expert_mask, expert_a, random_a)
        rewards = (data_a == expert_a).float()
        q_data = logits.gather(-1, data_a[..., None]).squeeze(-1)
        td = F.mse_loss(q_data, rewards)
        conservative = (torch.logsumexp(logits, dim=-1) - q_data).mean()
        loss = bc + 0.25 * td + 0.05 * conservative
        opt.zero_grad()
        loss.backward()
        opt.step()


def evaluate(model: CollectivePolicy, env: ContinualCoordinationBandit, task_ids: List[int], top_k: int,
             selective_comm_threshold: float | None = None, batches: int = 6, batch_size: int = 128):
    model.eval()
    acc, ret = {}, {}
    with torch.no_grad():
        for t in task_ids:
            accs, rets = [], []
            for _ in range(batches):
                obs, y = env.sample(t, batch_size)
                logits, _ = model(obs, top_k=top_k, selective_comm_threshold=selective_comm_threshold)
                a = logits.argmax(dim=-1)
                accs.append((a == y[:, None]).float().mean().item())
                rets.append(env.team_reward(a, y).mean().item())
            acc[t] = float(np.mean(accs))
            ret[t] = float(np.mean(rets))
    return acc, ret


def run_method(spec: BenchmarkSpec, cfg: MethodConfig, device: str = "cpu") -> Tuple[RunSummary, Dict]:
    set_seed(cfg.seed)
    env = ContinualCoordinationBandit(spec, seed=2026, device=device)
    model = CollectivePolicy(spec.obs_dim, cfg.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    pretrain_ssl(model, env, cfg, opt)
    warmstart_offline(model, env, cfg, opt)

    memory = BudgetedReplay(cfg.memory_budget if cfg.use_replay else 0, seed=cfg.seed)
    top_k = cfg.comm_top_k if cfg.use_communication else 0
    eval_history = []
    first_window_accs = []
    total_queries = 0
    total_messages = 0
    total_episodes = 0

    # Evaluate task 0 immediately after warmstart to define pre-sequence baseline.
    for task in range(spec.n_tasks):
        queries_left = cfg.query_budget_per_task if cfg.use_active_queries else 0
        running_baseline = 0.5

        for step in range(cfg.online_steps_per_task):
            obs, y = env.sample(task, cfg.batch_size)
            selective_threshold = cfg.comm_entropy_threshold if cfg.selective_comm else None
            logits, edges = model(obs, top_k=top_k, selective_comm_threshold=selective_threshold)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            actions = dist.sample()
            team_r = env.team_reward(actions, y)
            running_baseline = 0.97 * running_baseline + 0.03 * team_r.mean().item()
            advantage = team_r - running_baseline
            pg_loss = -(dist.log_prob(actions).sum(dim=1) * advantage.detach()).mean()
            entropy = dist.entropy().mean()
            loss = pg_loss - cfg.entropy_coef * entropy

            # Active query: choose the highest mean predictive entropy episodes,
            # but only from the remaining per-task label budget.
            queried_idx = torch.empty(0, dtype=torch.long, device=device)
            if queries_left > 0:
                episode_entropy = dist.entropy().mean(dim=1)
                q = min(queries_left, max(1, cfg.batch_size // 16))
                if cfg.selective_query:
                    candidates = (episode_entropy > cfg.query_entropy_threshold).nonzero(as_tuple=False).flatten()
                    if len(candidates) > 0:
                        kq = min(q, len(candidates))
                        rel = torch.topk(episode_entropy[candidates], k=kq).indices
                        queried_idx = candidates[rel]
                    else:
                        queried_idx = torch.empty(0, dtype=torch.long, device=device)
                else:
                    queried_idx = torch.topk(episode_entropy, k=q).indices
                q = int(len(queried_idx))
                if q > 0:
                    q_logits = logits[queried_idx]
                    q_y = y[queried_idx]
                    q_loss = F.cross_entropy(
                        q_logits.reshape(-1, 2),
                        q_y[:, None].expand(-1, spec.n_agents).reshape(-1),
                    )
                    loss = loss + cfg.query_weight * q_loss
                    queries_left -= q
                    total_queries += q

            # Budgeted replay of labeled/successful knowledge.
            if cfg.use_replay and len(memory) > 0:
                m_obs, m_y = memory.sample(cfg.replay_batch, device=device)
                m_logits, _ = model(m_obs, top_k=top_k, selective_comm_threshold=selective_threshold)
                m_loss = F.cross_entropy(
                    m_logits.reshape(-1, 2),
                    m_y[:, None].expand(-1, spec.n_agents).reshape(-1),
                )
                loss = loss + cfg.replay_weight * m_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            # Store queried examples with high priority.
            for i in queried_idx.tolist():
                confidence = probs[i].max(dim=-1).values.mean().item()
                memory.add(obs[i], int(y[i].item()), task, priority=2.0 - confidence)

            # Store fully successful joint episodes without asking the oracle;
            # success makes the chosen team action a valid pseudo-label.
            success = (actions == y[:, None]).all(dim=1)
            succ_idx = success.nonzero(as_tuple=False).flatten()
            if cfg.use_replay and len(succ_idx) > 0:
                # Keep only a few each step to avoid flooding memory.
                for i in succ_idx[:2].tolist():
                    memory.add(obs[i], int(actions[i, 0].item()), task, priority=0.8)

            total_messages += int(edges.sum().item())
            total_episodes += cfg.batch_size

            if step == 5:
                # Early adaptation measured before most of the task's training.
                a, _ = evaluate(model, env, [task], top_k, selective_comm_threshold=selective_threshold, batches=3, batch_size=128)
                first_window_accs.append(a[task])

        seen = list(range(task + 1))
        acc, ret = evaluate(model, env, seen, top_k, selective_comm_threshold=selective_threshold)
        eval_history.append({"after_task": task, "accuracy": acc, "return": ret})

    final_acc = eval_history[-1]["accuracy"]
    final_ret = eval_history[-1]["return"]
    # Forgetting: best historical accuracy for task minus final accuracy.
    forgetting = []
    for t in range(spec.n_tasks - 1):
        hist = [h["accuracy"][t] for h in eval_history if t in h["accuracy"]]
        forgetting.append(max(hist) - final_acc[t])

    summary = RunSummary(
        method=cfg.name,
        seed=cfg.seed,
        final_mean_accuracy=float(np.mean(list(final_acc.values()))),
        final_mean_return=float(np.mean(list(final_ret.values()))),
        avg_forgetting=float(np.mean(forgetting)) if forgetting else 0.0,
        mean_first_window_accuracy=float(np.mean(first_window_accs)),
        queries_used=int(total_queries),
        messages_per_episode=float(total_messages / max(total_episodes, 1)),
        memory_items=len(memory),
    )
    detail = {
        "config": asdict(cfg),
        "summary": asdict(summary),
        "eval_history": eval_history,
    }
    return summary, detail


def default_methods(seed: int) -> List[MethodConfig]:
    base = dict(seed=seed)
    return [
        MethodConfig(name="sequential_local", use_ssl=False, use_offline_warmstart=False, use_replay=False,
                     use_communication=False, use_active_queries=False, memory_budget=0, query_budget_per_task=0, **base),
        MethodConfig(name="replay_local", use_ssl=False, use_offline_warmstart=False, use_replay=True,
                     use_communication=False, use_active_queries=False, **base),
        MethodConfig(name="replay_comm_sparse", use_ssl=False, use_offline_warmstart=False, use_replay=True,
                     use_communication=True, use_active_queries=False, comm_top_k=1, **base),
        MethodConfig(name="sparse_comm_active", use_ssl=False, use_offline_warmstart=False, use_replay=True,
                     use_communication=True, use_active_queries=True, comm_top_k=1, **base),
        MethodConfig(name="dense_comm_active", use_ssl=False, use_offline_warmstart=False, use_replay=True,
                     use_communication=True, use_active_queries=True, comm_top_k=2, **base),
        MethodConfig(name="prometheus_budgeted", use_ssl=True, use_offline_warmstart=True, use_replay=True,
                     use_communication=True, use_active_queries=True, comm_top_k=1, selective_comm=False,
                     selective_query=True, query_entropy_threshold=0.65, **base),
    ]
