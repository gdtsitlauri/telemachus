"""Guaranteed online help control while the agent keeps learning (Corollary of Theorem B).

Theorem B (TELEMACHUS docs/online_guarantee.md) places no condition on the policy used while
the control is outside the fallback region: it may be any predictable (past-dependent) policy.
So the agent may keep learning during deployment and the finite-horizon guarantee still holds,
provided only the fallback (full help, theta <= 0) has per-episode failure <= p0.

Here the help gate is retrained during the shifted deployment. It learns only from what it
can see: on steps where an agent asked for help, the expert's action is revealed, which labels
whether the agent's own action would have been wrong. Every `retrain_every` chunks the gate is
refit on nominal data plus all labels gathered so far. The threshold follows the Theorem B rule.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .certified_help import deploy, gate_features, train_help_gate, calibrate_gate
from .long_horizon import ContinualSignalGrid
from .shift import shifted_spec
from .hp_online import solve_alpha_star, gate_auc


@torch.no_grad()
def deploy_observed(model, spec, top_k, theta, stream_seed, gate_model):
    """One episode per task with the learned gate; returns failures, queries and the
    (features, own-action-wrong) labels of the steps on which help was asked."""
    env = ContinualSignalGrid(spec, seed=stream_seed)
    model.eval()
    fails, queries, X, Y = [], [], [], []
    for task in range(spec.n_tasks):
        obs, _ = env.reset(task, 1)
        asked = torch.zeros(1)
        info = None
        for _ in range(spec.horizon):
            logits, _ = model.policy(obs, top_k=top_k)
            expert = env.expert_actions()
            own = logits.argmax(-1)
            feats = gate_features(obs, logits)
            f2 = feats.reshape(-1, feats.shape[-1]).numpy()
            pr = gate_model.predict_proba(f2)[:, 1]
            ask = torch.from_numpy(pr.reshape(own.shape) >= theta)
            m = ask.reshape(-1).numpy()
            if m.any():
                X.append(f2[m]); Y.append((own != expert).reshape(-1).numpy()[m])
            actions = torch.where(ask, expert, own)
            asked += ask.float().sum(dim=1)
            obs, _, _, info = env.step(actions)
        fails.append((1. - info["success"]).numpy())
        queries.append(asked.numpy())
    X = np.concatenate(X) if X else np.zeros((0, f2.shape[1]))
    Y = np.concatenate(Y).astype(int) if Y else np.zeros(0, int)
    return np.concatenate(fails), np.concatenate(queries), X, Y


def online_learning_hp(model, spec, top_k, gate0, X0, Y0, theta0, alpha_star, eta, K, stream_seed,
                       retrain_every, auc_stream):
    from sklearn.ensemble import HistGradientBoostingClassifier
    gate, theta, trace = gate0, float(theta0), []
    Xs, Ys, refits = [], [], []
    for c in range(K):
        if c and c % retrain_every == 0 and Xs:
            Xa, Ya = np.concatenate([X0] + Xs), np.concatenate([Y0] + Ys)
            gate = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xa, Ya)
            refits.append({"chunk": c, "n_deploy_labels": int(sum(len(y) for y in Ys))})
        f, q, X, Y = deploy_observed(model, spec, top_k, theta, stream_seed + c, gate)
        if len(Y):
            Xs.append(X); Ys.append(Y)
        fail = float(f.mean())
        trace.append({"chunk": c, "theta": theta, "failure": fail, "queries": float(q.mean()), "n": len(f)})
        theta = min(1.01, theta - eta * (fail - alpha_star))
    fr = np.array([t["failure"] for t in trace]); qs = np.array([t["queries"] for t in trace])
    h = K // 2
    return {"trace": trace, "failure_rate": float(fr.mean()), "queries_per_episode": float(qs.mean()),
            "queries_second_half": float(qs[h:].mean()), "fallback_fraction": float(np.mean([t["theta"] <= 0 for t in trace])),
            "refits": refits, "auc_shift_final_gate": gate_auc(model, spec, top_k, gate, auc_stream)}


@torch.no_grad()
def run_learning_study(model, spec, top_k, alpha, delta, seed, p0, eta, K, slip, cue_noise, retrain_every):
    gate = train_help_gate(model, spec, top_k, 700000 + 10 * seed)
    cal = calibrate_gate(model, spec, top_k, "learned", alpha, delta, 100, 900000 + 10 * seed, gate)
    theta0 = cal["selected_theta"]
    out = {"seed": seed, "certified_nominal": theta0 is not None, "theta_nominal": theta0}
    if theta0 is None:
        return out
    a_star = solve_alpha_star(alpha, delta, p0, spec.n_tasks, K, eta, theta0)
    out["alpha_star"] = a_star
    if a_star is None:
        return out
    # nominal labelled data for refits (own stream, same kind as the gate's first DAgger round)
    r0 = deploy(model, spec, top_k, "random", 0., 50, 800000 + 10 * seed, collect=True)
    hard = shifted_spec(spec, slip, cue_noise)
    out["shift_online_learning"] = online_learning_hp(model, hard, top_k, gate, r0["X"], r0["Y"], theta0, a_star, eta, K,
                                                      1_700_000 + 1000 * seed, retrain_every, 1_600_000 + 1000 * seed)
    return out
