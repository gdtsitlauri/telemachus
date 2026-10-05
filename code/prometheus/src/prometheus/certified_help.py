"""Certified human-in-the-loop help for a continually trained agent team.

After continual training, the team is deployed on all learned tasks. At every
step each agent may ask the expert (teleoperator) for its action. A gate with
threshold theta decides when to ask:

  learned  ask if P(own action is wrong) >= theta, from a gate classifier
           trained DAgger-style on its own episode stream
  entropy  ask if the policy entropy  >= theta   (uncertainty)
  margin   ask if top1 - top2 prob     <= theta   (decision margin)
  random   ask with probability theta             (uninformed control)
  oracle   ask iff the own action differs from the expert (lower bound on
           help; analysis only, it uses the expert's answer)

theta is chosen by Learn-then-Test (fixed sequence, exact binomial p-values)
on calibration episodes so that, with probability >= 1 - delta,
P(team failure) <= alpha on exchangeable deployment episodes. Gates and
trained models are then compared by the expert help they need at the SAME
certified failure level. Calibration and test episodes use disjoint
environment random streams.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy.stats import binom

from .long_horizon import ContinualSignalGrid, LongHorizonSpec


GATES = ("learned", "entropy", "margin", "random")


def gate_features(obs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Inputs of the learned gate: policy probabilities, entropy, margin, observation."""
    p = torch.softmax(logits, -1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1, keepdim=True)
    top2 = p.topk(2, -1).values
    return torch.cat([p, ent, top2[..., :1] - top2[..., 1:2], obs], -1)


def binomial_pvalue(losses: int, n: int, alpha: float) -> float:
    return float(binom.cdf(losses, n, alpha))


def learn_then_test(counts: Sequence[tuple], alpha: float, delta: float):
    """counts ordered most -> least help; returns selected index or None."""
    selected: Optional[int] = None
    for i, (_, n, losses) in enumerate(counts):
        if binomial_pvalue(losses, n, alpha) <= delta:
            selected = i
        else:
            break
    return selected


def gate_grid(gate: str) -> List[float]:
    """Thresholds ordered from most to least help (fixed-sequence order)."""
    if gate == "learned":   # P(wrong) >= theta; theta > 1 never asks
        return [round(x, 3) for x in np.linspace(0., 1., 21)] + [1.01]
    if gate == "entropy":   # ln(3) ~ 1.0986: theta above it never asks
        return [round(x, 3) for x in np.linspace(0., 1.1, 23)]
    if gate == "margin":    # margin <= theta asks; theta < 0 never asks
        return [round(x, 3) for x in np.linspace(1., 0., 21)] + [-1.]
    if gate == "random":    # probability of asking
        return [round(x, 3) for x in np.linspace(1., 0., 21)]
    raise ValueError(gate)


@torch.no_grad()
def deploy(model, spec: LongHorizonSpec, top_k: int, gate: str, theta: float, episodes_per_task: int,
           stream_seed: int, gate_model=None, collect: bool = False) -> Dict[str, np.ndarray]:
    """Run deployment episodes on every learned task with the help gate.

    collect=True also returns gate features and 'own action is wrong' labels
    for every agent-step (used to train the learned gate on its own stream).
    """
    env = ContinualSignalGrid(spec, seed=stream_seed)
    gen = torch.Generator().manual_seed(stream_seed + 7919)
    model.eval()
    fails, queries, X, Y = [], [], [], []
    for task in range(spec.n_tasks):
        obs, _ = env.reset(task, episodes_per_task)
        asked = torch.zeros(episodes_per_task)
        info = None
        for _ in range(spec.horizon):
            logits, _ = model.policy(obs, top_k=top_k)
            probs = torch.softmax(logits, dim=-1)
            expert = env.expert_actions()
            own = logits.argmax(-1)
            feats = gate_features(obs, logits) if (gate == "learned" or collect) else None
            if collect:
                X.append(feats.reshape(-1, feats.shape[-1]).numpy())
                Y.append((own != expert).reshape(-1).numpy())
            if gate == "learned":
                pr = gate_model.predict_proba(feats.reshape(-1, feats.shape[-1]).numpy())[:, 1]
                ask = torch.from_numpy(pr.reshape(own.shape) >= theta)
            elif gate == "entropy":
                score = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)
                ask = score >= theta
            elif gate == "margin":
                top2 = probs.topk(2, dim=-1).values
                ask = (top2[..., 0] - top2[..., 1]) <= theta
            elif gate == "oracle":
                ask = own != expert
            elif gate == "random":
                ask = torch.rand(probs.shape[:2], generator=gen) < theta
            else:
                raise ValueError(gate)
            actions = torch.where(ask, expert, own)
            asked += ask.float().sum(dim=1)
            obs, _, _, info = env.step(actions)
        fails.append((1. - info["success"]).numpy())
        queries.append(asked.numpy())
    out = {"fail": np.concatenate(fails), "queries": np.concatenate(queries)}
    if collect:
        out["X"], out["Y"] = np.concatenate(X), np.concatenate(Y).astype(int)
    return out


def train_help_gate(model, spec, top_k, stream_seed: int, episodes_per_task: int = 200,
                    dagger_iters: int = 3, dagger_theta: float = .3):
    """Gate classifier for P(own action is wrong), DAgger-style.

    Iteration 0 uses states visited by the policy alone; later iterations add
    states visited when help follows the current gate, matching deployment.
    Uses only its own environment stream (disjoint from calibration/test).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    r = deploy(model, spec, top_k, "random", 0., episodes_per_task, stream_seed, collect=True)
    X, Y = r["X"], r["Y"]
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X, Y)
    for it in range(dagger_iters):
        r = deploy(model, spec, top_k, "learned", dagger_theta, episodes_per_task, stream_seed + 1 + it,
                   gate_model=clf, collect=True)
        X, Y = np.concatenate([X, r["X"]]), np.concatenate([Y, r["Y"]])
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X, Y)
    return clf


def calibrate_gate(model, spec, top_k, gate, alpha, delta, episodes_per_task, stream_seed, gate_model=None):
    grid = gate_grid(gate)
    counts, rows = [], []
    for theta in grid:
        r = deploy(model, spec, top_k, gate, theta, episodes_per_task, stream_seed, gate_model=gate_model)
        n, k = len(r["fail"]), int(r["fail"].sum())
        counts.append((theta, n, k))
        rows.append({"theta": theta, "n": n, "failures": k, "failure_rate": k / n,
                     "queries_per_episode": float(r["queries"].mean()),
                     "p_value": binomial_pvalue(k, n, alpha)})
        if learn_then_test(counts, alpha, delta) != len(counts) - 1:
            break  # fixed sequence stops at the first non-rejection
    sel = learn_then_test(counts, alpha, delta)
    return {"gate": gate, "alpha": alpha, "delta": delta, "candidates": rows,
            "selected_theta": None if sel is None else grid[sel]}


def evaluate_certified(model, spec, top_k, alpha, delta, cal_stream, test_stream, gate_stream=None,
                       episodes_per_task_cal=100, episodes_per_task_test=200, gates=GATES):
    """Train the learned gate on gate_stream, calibrate every gate on
    cal_stream, then test it on test_stream (three disjoint streams)."""
    out = {}
    gate_model = None
    if "learned" in gates:
        if gate_stream is None or len({gate_stream, cal_stream, test_stream}) < 3:
            raise ValueError("learned gate needs its own stream, disjoint from calibration and test")
        gate_model = train_help_gate(model, spec, top_k, gate_stream)
    for gate in gates:
        cal = calibrate_gate(model, spec, top_k, gate, alpha, delta, episodes_per_task_cal, cal_stream, gate_model)
        entry = {"calibration": cal, "certified": cal["selected_theta"] is not None}
        if entry["certified"]:
            r = deploy(model, spec, top_k, gate, cal["selected_theta"], episodes_per_task_test, test_stream,
                       gate_model=gate_model)
            entry["test"] = {"n": int(len(r["fail"])), "failure_rate": float(r["fail"].mean()),
                             "queries_per_episode": float(r["queries"].mean()),
                             "queries_per_agent_step": float(r["queries"].mean() / (spec.horizon * spec.n_agents)),
                             "within_alpha": bool(r["fail"].mean() <= alpha)}
        out[gate] = entry
    # Reference points without a gate.
    never = deploy(model, spec, top_k, "random", 0., episodes_per_task_test, test_stream)
    always = deploy(model, spec, top_k, "random", 1., episodes_per_task_test, test_stream)
    out["no_help"] = {"failure_rate": float(never["fail"].mean())}
    out["full_help"] = {"failure_rate": float(always["fail"].mean()),
                        "queries_per_episode": float(always["queries"].mean())}
    oracle = deploy(model, spec, top_k, "oracle", 0., episodes_per_task_test, test_stream)
    out["oracle_help"] = {"failure_rate": float(oracle["fail"].mean()),
                          "queries_per_episode": float(oracle["queries"].mean()),
                          "note": "lower bound on help; uses the expert answer to decide, analysis only"}
    return out
