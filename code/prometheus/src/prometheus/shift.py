"""Keeping a certified help guarantee under deployment shift.

Learn-then-Test certifies the help gate for episodes exchangeable with the
calibration episodes. Under shift (here: a harder world with more actuator
slips and noisier goal cues) that assumption fails. This module compares:

  static    theta certified by LTT on the nominal world, then frozen
  online    theta updated after every chunk of episodes from the observed
            team outcome only (success is visible; no expert labels):
                theta <- clip(theta - eta * (failure_rate_chunk - alpha))
            which keeps the long-run failure rate near alpha without any
            exchangeability assumption (adaptive / conformal-decision style)
  full      ask the expert at every step (upper bound on help)
  oracle    ask exactly when the own action is wrong (lower bound on help)

The learned gate classifier is trained on the nominal world only.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

import numpy as np
import torch

from .certified_help import deploy, train_help_gate, calibrate_gate
from .long_horizon import LongHorizonSpec


def shifted_spec(spec: LongHorizonSpec, slip: float = .04, cue_noise: float = .5) -> LongHorizonSpec:
    """Default shift: much noisier goal cues (hurts the learned team, not the
    expert, who acts on the true goal). Higher slip also hurts the expert and
    can make alpha unattainable even with full help."""
    return replace(spec, slip_prob=slip, cue_noise=cue_noise)


def online_deployment(model, spec, top_k, gate_model, theta0, alpha, eta, n_chunks, eps_per_task, stream_seed,
                      static=False):
    """Sequential deployment in chunks of 6*eps_per_task episodes."""
    theta = float(theta0)
    trace = []
    for c in range(n_chunks):
        r = deploy(model, spec, top_k, "learned", theta, eps_per_task, stream_seed + c, gate_model=gate_model)
        fail, q = float(r["fail"].mean()), float(r["queries"].mean())
        trace.append({"chunk": c, "theta": theta, "failure": fail, "queries": q, "n": len(r["fail"])})
        if not static:
            theta = float(np.clip(theta - eta * (fail - alpha), 0., 1.01))
    fails = np.array([t["failure"] for t in trace]); qs = np.array([t["queries"] for t in trace])
    half = n_chunks // 2
    return {"trace": trace, "failure_rate": float(fails.mean()), "queries_per_episode": float(qs.mean()),
            "failure_rate_second_half": float(fails[half:].mean()), "queries_second_half": float(qs[half:].mean())}


@torch.no_grad()
def run_shift_study(model, spec, top_k, alpha, delta, seed, eta=.5, n_chunks=40, eps_per_task=10,
                    slip=.04, cue_noise=.5):
    gate = train_help_gate(model, spec, top_k, 700000 + 10 * seed)
    cal = calibrate_gate(model, spec, top_k, "learned", alpha, delta, 100, 900000 + 10 * seed, gate)
    theta0 = cal["selected_theta"]
    if theta0 is None:
        return {"certified_nominal": False}
    hard = shifted_spec(spec, slip, cue_noise)
    stream = 1_300_000 + 1000 * seed
    out = {"certified_nominal": True, "theta_nominal": theta0}
    out["nominal_static"] = online_deployment(model, spec, top_k, gate, theta0, alpha, eta, n_chunks, eps_per_task,
                                              1_200_000 + 1000 * seed, static=True)
    out["shift_static"] = online_deployment(model, hard, top_k, gate, theta0, alpha, eta, n_chunks, eps_per_task,
                                            stream, static=True)
    out["shift_online"] = online_deployment(model, hard, top_k, gate, theta0, alpha, eta, n_chunks, eps_per_task,
                                            stream, static=False)
    full = [deploy(model, hard, top_k, "random", 1., eps_per_task, stream + c) for c in range(n_chunks)]
    orac = [deploy(model, hard, top_k, "oracle", 0., eps_per_task, stream + c) for c in range(n_chunks)]
    out["shift_full_help"] = {"failure_rate": float(np.mean([r["fail"].mean() for r in full])),
                              "queries_per_episode": float(np.mean([r["queries"].mean() for r in full]))}
    out["shift_oracle"] = {"failure_rate": float(np.mean([r["fail"].mean() for r in orac])),
                           "queries_per_episode": float(np.mean([r["queries"].mean() for r in orac]))}
    return out
