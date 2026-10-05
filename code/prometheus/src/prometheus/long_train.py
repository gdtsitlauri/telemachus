"""PPO-style continual MARL training for PROMETHEUS v2.0.

The implementation is deliberately compact and auditable.  It uses a shared
multi-agent actor, a centralized value function (CTDE), clipped PPO updates,
optional graph communication, policy-distillation replay, online-EWC-style
regularization, task-0 contrastive pretraining / behavior cloning, and a hard
budget on expert queries.

It is MAPPO-style rather than a drop-in reproduction of the canonical MAPPO
codebase; that distinction is maintained in the documentation and claims.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import copy
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .long_horizon import LongHorizonSpec, ContinualSignalGrid
from .long_model import LongHorizonActorCritic
from .policy_memory import PolicyReplay
from .train import nt_xent


@dataclass
class LongMethodConfig:
    name: str = "prometheus_v2_budgeted"
    seed: int = 0
    hidden_dim: int = 64
    lr: float = 2.5e-3
    gamma: float = 0.96
    gae_lambda: float = 0.92
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.015
    updates_per_task: int = 14
    batch_episodes: int = 24
    ppo_epochs: int = 2
    use_communication: bool = True
    comm_top_k: int = 1
    use_replay: bool = True
    memory_budget: int = 384
    replay_batch: int = 64
    replay_coef: float = 0.7
    use_ewc: bool = False
    ewc_lambda: float = 8.0
    ewc_decay: float = 0.85
    use_active_query: bool = True
    query_budget_per_task: int = 32
    query_coef: float = 1.2
    selective_query: bool = True
    query_entropy_threshold: float = 0.85
    use_ssl: bool = True
    ssl_steps: int = 30
    use_bc_warmstart: bool = True
    bc_steps: int = 35


@dataclass
class LongRunSummary:
    method: str
    seed: int
    final_mean_success: float
    final_mean_return: float
    final_mean_distance: float
    avg_success_forgetting: float
    avg_return_forgetting: float
    mean_early_return: float
    queries_used: int
    messages_per_episode: float
    memory_items: int


def set_seed(seed: int):
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _random_obs(env: ContinualSignalGrid, task: int, batch: int):
    obs, state = env.reset(task, batch)
    return obs, state


def ssl_pretrain(model, env, cfg, opt):
    if not cfg.use_ssl or cfg.ssl_steps <= 0:
        return
    model.train()
    for _ in range(cfg.ssl_steps):
        obs, _ = _random_obs(env, 0, cfg.batch_episodes * 2)
        x = obs.reshape(-1, env.spec.obs_dim)
        a = x + 0.06 * torch.randn_like(x)
        b = x + 0.06 * torch.randn_like(x)
        # preserve task/agent identity but mask local continuous cues stochastically
        a[:, :4] *= (torch.rand_like(a[:, :4]) > 0.08).float()
        b[:, :4] *= (torch.rand_like(b[:, :4]) > 0.08).float()
        loss = nt_xent(model.ssl_projection(a), model.ssl_projection(b), temp=0.2)
        opt.zero_grad(); loss.backward(); opt.step()


def bc_warmstart(model, env, cfg, opt):
    if not cfg.use_bc_warmstart or cfg.bc_steps <= 0:
        return
    model.train()
    for _ in range(cfg.bc_steps):
        obs, state = env.reset(0, cfg.batch_episodes * 2)
        # sample states at varying trajectory depths under the expert
        k = int(torch.randint(0, env.spec.horizon, (1,)).item())
        for _j in range(k):
            expert = env.expert_actions()
            obs, state, _, _ = env.step(expert)
        expert = env.expert_actions()
        logits, _ = model.policy(obs, top_k=(cfg.comm_top_k if cfg.use_communication else 0))
        loss = F.cross_entropy(logits.reshape(-1, env.spec.n_actions), expert.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()


class EWCState:
    def __init__(self):
        self.fisher: Dict[str, torch.Tensor] = {}
        self.anchor: Dict[str, torch.Tensor] = {}

    def penalty(self, model) -> torch.Tensor:
        if not self.fisher:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        p = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.fisher:
                p = p + (self.fisher[name] * (param - self.anchor[name]).pow(2)).sum()
        return p

    def consolidate(self, model, obs: torch.Tensor, actions: torch.Tensor, top_k: int, decay: float = 0.85):
        # Empirical-Fisher approximation from policy log-likelihood on recent on-policy states.
        names = [n for n, p in model.named_parameters() if p.requires_grad and not n.startswith("critic")]
        accum = {n: torch.zeros_like(dict(model.named_parameters())[n]) for n in names}
        total = 0
        chunks = min(4, max(1, obs.shape[0] // 32))
        for x, a in zip(torch.chunk(obs, chunks), torch.chunk(actions, chunks)):
            model.zero_grad(set_to_none=True)
            logits, _ = model.policy(x, top_k=top_k)
            dist = torch.distributions.Categorical(logits=logits)
            loss = -dist.log_prob(a).mean()
            loss.backward()
            weight = x.shape[0]
            total += weight
            for n, p in model.named_parameters():
                if n in accum and p.grad is not None:
                    accum[n] += weight * p.grad.detach().pow(2)
        for n in names:
            f = accum[n] / max(total, 1)
            if n in self.fisher:
                f = decay * self.fisher[n] + (1.0 - decay) * f
            self.fisher[n] = f.detach().clone()
            self.anchor[n] = dict(model.named_parameters())[n].detach().clone()
        model.zero_grad(set_to_none=True)


def collect_rollout(model, env, task: int, cfg: LongMethodConfig):
    obs, state = env.reset(task, cfg.batch_episodes)
    top_k = cfg.comm_top_k if cfg.use_communication else 0
    data = {k: [] for k in ["obs","state","actions","old_logp","rewards","values","entropy","expert"]}
    messages = 0
    final_info = None
    for _ in range(env.spec.horizon):
        with torch.no_grad():
            logits, edges = model.policy(obs, top_k=top_k)
            value = model.value(state)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            logp = dist.log_prob(actions).mean(dim=1)
            ent = dist.entropy().mean(dim=1)
            expert = env.expert_actions()
        data["obs"].append(obs)
        data["state"].append(state)
        data["actions"].append(actions)
        data["old_logp"].append(logp)
        data["values"].append(value)
        data["entropy"].append(ent)
        data["expert"].append(expert)
        messages += int(edges.sum().item())
        obs, state, reward, final_info = env.step(actions)
        data["rewards"].append(reward)
    for k in data:
        data[k] = torch.stack(data[k], dim=0)
    data["messages"] = messages
    data["final_info"] = final_info
    return data


def compute_gae(rewards, values, gamma: float, lam: float):
    tmax, b = rewards.shape
    adv = torch.zeros_like(rewards)
    last = torch.zeros(b, device=rewards.device)
    next_value = torch.zeros(b, device=rewards.device)
    for t in reversed(range(tmax)):
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
        next_value = values[t]
    returns = adv + values
    return adv, returns


def ppo_update(model, opt, rollout, cfg, memory: PolicyReplay, ewc: EWCState, queries_left: int, top_k: int):
    T, B, N, D = rollout["obs"].shape
    adv, returns = compute_gae(rollout["rewards"], rollout["values"], cfg.gamma, cfg.gae_lambda)
    flat_obs = rollout["obs"].reshape(T * B, N, D)
    flat_state = rollout["state"].reshape(T * B, -1)
    flat_actions = rollout["actions"].reshape(T * B, N)
    flat_old_logp = rollout["old_logp"].reshape(T * B)
    flat_adv = adv.reshape(T * B)
    flat_returns = returns.reshape(T * B)
    flat_entropy = rollout["entropy"].reshape(T * B)
    flat_expert = rollout["expert"].reshape(T * B, N)
    flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-6)

    # Query the most uncertain joint states once per rollout, under a hard task budget.
    queried = torch.empty(0, dtype=torch.long, device=flat_obs.device)
    if cfg.use_active_query and queries_left > 0:
        qmax = min(queries_left, max(1, cfg.query_budget_per_task // max(cfg.updates_per_task, 1) + 1))
        if cfg.selective_query:
            cand = (flat_entropy > cfg.query_entropy_threshold).nonzero(as_tuple=False).flatten()
        else:
            cand = torch.arange(T * B, device=flat_obs.device)
        if len(cand) > 0:
            k = min(qmax, len(cand))
            rel = torch.topk(flat_entropy[cand], k=k).indices
            queried = cand[rel]

    for _ in range(cfg.ppo_epochs):
        logits, _ = model.policy(flat_obs, top_k=top_k)
        dist = torch.distributions.Categorical(logits=logits)
        new_logp = dist.log_prob(flat_actions).mean(dim=1)
        ratio = torch.exp((new_logp - flat_old_logp).clamp(-8, 8))
        s1 = ratio * flat_adv
        s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * flat_adv
        actor_loss = -torch.min(s1, s2).mean()
        value = model.value(flat_state)
        value_loss = F.mse_loss(value, flat_returns)
        entropy = dist.entropy().mean()
        loss = actor_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

        if len(queried) > 0:
            qlogits = logits[queried]
            qexpert = flat_expert[queried]
            qloss = F.cross_entropy(qlogits.reshape(-1, qlogits.shape[-1]), qexpert.reshape(-1))
            loss = loss + cfg.query_coef * qloss

        if cfg.use_replay and len(memory) > 0:
            m_obs, m_probs = memory.sample(cfg.replay_batch, device=flat_obs.device)
            m_logits, _ = model.policy(m_obs, top_k=top_k)
            # KL(old || current), distillation to retain previous-task behavior.
            replay_loss = F.kl_div(F.log_softmax(m_logits, dim=-1), m_probs, reduction="batchmean")
            loss = loss + cfg.replay_coef * replay_loss

        if cfg.use_ewc and ewc.fisher:
            loss = loss + cfg.ewc_lambda * ewc.penalty(model)

        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()

    # Store a bounded sample of current policy states after the update.
    if cfg.use_replay and cfg.memory_budget > 0:
        with torch.no_grad():
            logits, _ = model.policy(flat_obs, top_k=top_k)
            probs = torch.softmax(logits, dim=-1)
        stride = max(1, (T * B) // 24)
        for i in range(0, T * B, stride):
            memory.add(flat_obs[i], probs[i], task_id=-1)
    return int(len(queried)), flat_obs.detach(), flat_actions.detach()


def evaluate(model, env, tasks: List[int], cfg: LongMethodConfig, episodes: int = 192):
    model.eval()
    top_k = cfg.comm_top_k if cfg.use_communication else 0
    out = {}
    with torch.no_grad():
        for task in tasks:
            obs, state = env.reset(task, episodes)
            total_return = torch.zeros(episodes, device=obs.device)
            messages = 0
            info = None
            for _ in range(env.spec.horizon):
                logits, edges = model.policy(obs, top_k=top_k)
                actions = logits.argmax(dim=-1)
                messages += int(edges.sum().item())
                obs, state, reward, info = env.step(actions)
                total_return += reward
            out[task] = {
                "success": float(info["success"].mean().item()),
                "return": float(total_return.mean().item()),
                "distance": float(info["mean_distance"].mean().item()),
                "messages_per_episode": messages / episodes,
            }
    model.train()
    return out


def run_long_method(spec: LongHorizonSpec, cfg: LongMethodConfig, device: str = "cpu", return_model: bool = False):
    """Train one method through all continual tasks.

    Returns (summary, details), or (summary, details, model) with
    return_model=True for deployment-time evaluation (certified_help.py).
    """
    set_seed(cfg.seed)
    env = ContinualSignalGrid(spec, seed=2026 + cfg.seed, device=device)
    model = LongHorizonActorCritic(spec.obs_dim, spec.state_dim, spec.n_actions, cfg.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ssl_pretrain(model, env, cfg, opt)
    bc_warmstart(model, env, cfg, opt)

    memory = PolicyReplay(cfg.memory_budget if cfg.use_replay else 0, seed=cfg.seed)
    ewc = EWCState()
    history = []
    early_returns = []
    total_queries = 0
    total_messages = 0
    total_episodes = 0
    last_rollout = None
    top_k = cfg.comm_top_k if cfg.use_communication else 0

    for task in range(spec.n_tasks):
        queries_left = cfg.query_budget_per_task if cfg.use_active_query else 0
        for update in range(cfg.updates_per_task):
            rollout = collect_rollout(model, env, task, cfg)
            q, flat_obs, flat_actions = ppo_update(model, opt, rollout, cfg, memory, ewc, queries_left, top_k)
            queries_left -= q
            total_queries += q
            total_messages += rollout["messages"]
            total_episodes += cfg.batch_episodes
            last_rollout = (flat_obs, flat_actions)
            if update == 1:
                early = evaluate(model, env, [task], cfg, episodes=96)[task]["return"]
                early_returns.append(early)
        if cfg.use_ewc and last_rollout is not None:
            ewc.consolidate(model, last_rollout[0], last_rollout[1], top_k=top_k, decay=cfg.ewc_decay)
        history.append({"after_task": task, "metrics": evaluate(model, env, list(range(task + 1)), cfg)})

    final = history[-1]["metrics"]
    succ_f, ret_f = [], []
    for t in range(spec.n_tasks - 1):
        succ_hist = [h["metrics"][t]["success"] for h in history if t in h["metrics"]]
        ret_hist = [h["metrics"][t]["return"] for h in history if t in h["metrics"]]
        succ_f.append(max(succ_hist) - final[t]["success"])
        ret_f.append(max(ret_hist) - final[t]["return"])

    summary = LongRunSummary(
        method=cfg.name,
        seed=cfg.seed,
        final_mean_success=float(np.mean([x["success"] for x in final.values()])),
        final_mean_return=float(np.mean([x["return"] for x in final.values()])),
        final_mean_distance=float(np.mean([x["distance"] for x in final.values()])),
        avg_success_forgetting=float(np.mean(succ_f)) if succ_f else 0.0,
        avg_return_forgetting=float(np.mean(ret_f)) if ret_f else 0.0,
        mean_early_return=float(np.mean(early_returns)),
        queries_used=int(total_queries),
        messages_per_episode=float(total_messages / max(total_episodes, 1)),
        memory_items=len(memory),
    )
    details = {"config": asdict(cfg), "summary": asdict(summary), "history": history}
    if return_model:
        return summary, details, model
    return summary, details


def default_long_methods(seed: int) -> List[LongMethodConfig]:
    b = dict(seed=seed)
    return [
        LongMethodConfig(name="ppo_local_ft", use_communication=False, use_replay=False, use_ewc=False,
                         use_active_query=False, use_ssl=False, use_bc_warmstart=False, memory_budget=0,
                         query_budget_per_task=0, **b),
        LongMethodConfig(name="ppo_sparse_replay", use_communication=True, comm_top_k=1, use_replay=True,
                         use_ewc=False, use_active_query=False, use_ssl=False, use_bc_warmstart=False, **b),
        LongMethodConfig(name="ppo_sparse_ewc", use_communication=True, comm_top_k=1, use_replay=False,
                         use_ewc=True, use_active_query=False, use_ssl=False, use_bc_warmstart=False,
                         memory_budget=0, **b),
        LongMethodConfig(name="ppo_dense_replay_active", use_communication=True, comm_top_k=2, use_replay=True,
                         use_ewc=False, use_active_query=True, use_ssl=False, use_bc_warmstart=True, **b),
        # Frozen after pilot seeds 0-4: sparse communication + bounded replay +
        # SSL/BC warm-start + conservative uncertainty-triggered expert queries.
        # The main method intentionally omits EWC because the pilot showed a
        # poor stability/plasticity trade-off for this environment.
        LongMethodConfig(name="prometheus_v2_budgeted", use_communication=True, comm_top_k=1, use_replay=True,
                         use_ewc=False, use_active_query=True, use_ssl=True, use_bc_warmstart=True,
                         query_budget_per_task=16, query_entropy_threshold=1.00, query_coef=0.5, **b),
    ]
