# TELEMACHUS

**When should an AI agent look, think, ask or act — with a guarantee on its error rate?**

TELEMACHUS is one framework for budgeted information acquisition and escalation in AI agents. At each decision the
agent may act now or first buy information: observe again, sample more, reason longer, call a tool, ask a team member
or escalate to a human expert. A *gate* decides when to stop and act. Its single threshold is calibrated with
Learn-then-Test (exact binomial tests, fixed-sequence testing), so that the deployed error rate is at most `alpha`
with probability at least `1 - delta` when deployment is exchangeable with calibration. When it is not (deployment
shift), the threshold is updated online from observed failures.

The same recipe is evaluated in four domains:

| # | domain | decision | information / escalation actions | risk controlled |
|---|---|---|---|---|
| 1-2 | robot navigation (simulated, Sionna radio) | move | observe, Monte-Carlo samples, dynamics probe, query over the radio link, human expert | collision per episode |
| 3-4 | multi-agent team (continual MARL, PPO/CTDE) | act | ask for human help | team episode failure |
| 5 | language agent, GSM8K (Qwen2.5 0.5B / 3B) | answer | think longer, run a tool, majority vote, expert | wrong final answer |
| 6 | vision-language grounding (SmolVLM2 256M / 500M) | point to a side | expert | wrong goal |

## Main findings

All numbers are on seeds / questions / scenes never used for design or calibration. Full tables: `tables/RESULTS.md`.

1. **The certified guarantee held in every domain, as far as test data can show.** Language and vision: every
   certified gate within `alpha` on its test set. Navigation (alpha 3%): TELEMACHUS 1.9% [0.9, 3.5]; one baseline
   at 3.1% [1.8, 5.1], which is consistent with a `1 - delta` guarantee. Multi-agent: the learned gate within alpha
   on 18-19 of 19 held-out seeds per agent (the guarantee allows a delta = 10% chance per seed).
2. **A learned gate pays off exactly when the model's own confidence is a poor or non-comparable signal of
   correctness** (`figures/fig6_when_learned_gate_pays.png`).
   - Multi-agent help: on 19 held-out seeds, policy entropy points the wrong way (AUC 0.36 +- 0.06; 0.36 under
     shift) while the learned gate reaches 0.98 +- 0.01 (0.93 under shift). It needs 19.5-21.7 help queries per
     episode against 27-30 for entropy, margin and random gates (Wilcoxon p = 3.8e-6 for every agent), within 0-2.1
     queries of the oracle.
   - Language agent: the learned gate puts heterogeneous stages on one probability scale. With Qwen2.5-3B it is
     3.5-4.4x cheaper than the confidence gate (199 vs 867 tokens at alpha 0.3) and closes 81-86% of the gap to
     the alpha-oracle cost bound; with 0.5B it is cheaper than confidence but closes only 12-17%. Over 50 random
     calibration/test splits it is cheaper than confidence in 50/50 splits for both models (3B: 3.3-5.1x, p5-p95).
   - Navigation: the learned resource policy (TELEMACHUS and its no-expert ablation) is the cheapest certified
     method (36.0-36.6 vs 37.9-44.0 for heuristic, value-of-information and confidence gates) with the lowest
     collision rate, and keeps the highest success under stress (0.90-0.91 vs 0.80-0.84).
   - Vision-language: confidence is already informative (AUC 0.80-0.90); the learned gate gives no significant gain
     (256M: +0.028 AUC, 95% CI [-0.007, 0.063]; 500M: -0.016, CI [-0.049, 0.017]), also over 50 random splits.
   - Why (`docs/when_gates_pay.md`, Proposition 1). The help a gate needs at risk alpha is exactly
     `(pi - alpha) + (1 - pi) FPR(1 - alpha/pi)`: it depends on one, high-recall point of its error-detection ROC,
     not on AUC. A score whose ROC is below the diagonal there needs more help than random asking (multi-agent
     entropy). In cascades the gain also comes from making scores comparable across stages: for the 3B THINK stage
     alone, learned and confidence gates need the same help (0.36), yet the cascade is 3.3-5.1x cheaper. This is
     standard selective-classification mathematics, used here to explain and check the cross-domain pattern.
3. **Under deployment shift the static certificate breaks; online adaptation restores the target rate on average,
   and with a safe fallback it guarantees it for every deployment** (`figures/fig2_navigation_online_shift.png`, `figures/fig4_multiagent_online_shift.png`,
   `docs/online_guarantee.md`).
   - Navigation (alpha 3%): static margin 8.1% collisions, long-run online rule 3.5%, a hand-picked conservative
     margin 4.3%. With the human-handover fallback (Theorem A), all 3 independent 480-episode streams finished at
     2.7-2.9%, below the theorem's bound of 2.995%.
   - Multi-agent (alpha 10%): static 12.9% (7/19 seeds <= alpha), long-run online rule 10.1% (4/19), and the new
     guaranteed rule (Theorem B, stochastic fallback = full help) 9.2% with every one of 19/19 held-out seeds
     <= alpha (max 9.3%), for 22.3 help queries per episode against 30 for full help.
4. **Theorem B (new as far as our search found, `docs/online_guarantee.md`, `docs/related_work.md`).** If the
   fallback's risk is bounded only in probability (p0 > 0), a pre-computed target alpha* and an online threshold
   that is never clipped on its safe side give realised failure <= alpha over a finite horizon with probability
   >= 1 - delta, under arbitrary shift. Theorem A, the zero-risk-fallback case, is a special case of Conformal
   Decision Theory (Lekeufack et al. 2024).
   **Proposition 2 (lower bound).** No online rule can guarantee a realised failure rate below
   p0 + z_{1-delta} sqrt(p0(1-p0)/(nK)) in every world allowed by the fallback assumption (Slud's inequality; needs
   p0 <= 1/4). In the multi-agent study this bound is 8.41%, while Theorem B certifies down to 9.80%. Measured from
   p0 = 8%, Theorem B's gap is about 4.4 times the lower bound's gap (`tables/RESULTS.md`, section 10).
5. **Learning while guaranteed (Corollary C).** The guarantee places no condition on the agent's policy outside
   the fallback, so the agent may keep learning during deployment. In the multi-agent shift study, the help gate
   was retrained 11 times per deployment, only from labels it could actually see (the expert's answer when it
   asked). It kept 19/19 held-out seeds <= alpha (max 9.2%) and used 1.1 fewer help queries per episode than the
   frozen gate (fewer on 18/19 seeds, Wilcoxon p = 7.6e-6). The saving comes from falling back to full help less
   often, not from a better gate AUC. The design pilot had shown no saving.

## Negative and limiting results (reported as such)

- The plain online rule controls only the **long-run average** (multi-agent: 4/19 seeds <= alpha). The guaranteed
  rules fix this, but they need a safe fallback. Theorem B relies on a declared bound for the fallback's risk
  (p0 = 0.08; measured full-help failure was <= 0.067 on every seed). The guarantee holds at the horizon K; running
  rates earlier in a deployment can be higher.
- Safety under shift costs task success: navigation success falls from 0.52 (static) to 0.38 (long-run online) and
  to 0.23-0.31 with the guaranteed rule, because 15-22% of episodes are handed to a human.
- A test error above alpha on one finite test set happens in some random splits (e.g. 3B learned gate, alpha
  0.2: 38/50 splits <= alpha). One-sided binomial tests find the risk significantly above alpha in at most 2/50
  splits per setting, which is within delta = 0.10. The learned gate spends its error budget close to alpha.
- With Qwen2.5-3B a **certified fixed stage** ("always THINK", error 0.163, cost 189) is cheaper than every
  certified gate; the gate's value there is the guarantee and robustness, not cost. With 0.5B no fixed stage certifies
  and the only certified fixed policy is the expert (cost 1500).
- The human expert action in navigation had no measurable effect (`no_expert` equal to `telemachus`), and the budget
  reserve had no effect in the tight-budget study (identical results).
- Learned or calibrated cascades and certified ask-for-help already exist (KnowNo, Trust or Escalate, Conformal
  Cascade, UCCI; see `docs/related_work.md`). The contribution is the cross-domain recipe and finding, plus
  Theorem B, not certified cascades as such.
- Not done, left as future work: models larger than 3B (a 7B 4-bit run would fit a free Colab T4), and a real
  robot or real-world navigation data.
- All environments are simulated or benchmark-based; models are small (<= 3B parameters); cost is a declared
  resource proxy (token-equivalents, simulation units), not energy or money.

## Folder map

```
TELEMACHUS/
  README.md, LICENSE (MIT), CITATION.cff
  SUMMARY_FOR_SUPERVISOR_GR.txt   plain-language summary of the whole study (Greek)
  docs/online_guarantee.md        Theorems A and B with proofs, Proposition 2 (lower bound), Corollary C
  docs/when_gates_pay.md          Proposition 1: help needed at risk alpha from one ROC point
  docs/related_work.md            comparison with prior work: what is and is not new
  analysis/make_figures.py        builds figures/ and tables/ from results/ only
  analysis/repro_check.py         reruns parts of every study from code/ and compares with results/
  analysis/theorem_b_optimality.py  Proposition 2: lower bound versus Theorem B
  figures/                        fig0 (framework diagram), fig1..fig6 (png + pdf)
  tables/RESULTS.md               every number quoted above
  environment/                    exact package versions used (CPU and GPU environments)
  results/
    1_navigation_certified/       v6 frozen protocol, calibration certificates, test/stress/budget summaries,
                                  per-episode CSVs (480 test episodes per method)
    2_navigation_online_shift/    frozen protocols, pilots (development seeds), final runs (3 streams x 480
                                  episodes): long-run rule (final.json), guaranteed rule (final_hp.json)
    3_multiagent_certified_help/  frozen protocol, per-seed results, summary
    4_multiagent_online_shift/    frozen protocol, per-seed traces, summary (long-run rule)
      high_probability/           guaranteed rule (Theorem B) + held-out gate AUCs, 19 seeds x 7200 episodes
      high_probability_eta*/      sensitivity to eta (not primary)
      learning/                   guaranteed rule while the gate keeps learning (Corollary C)
    5_language_agent/{qwen2.5-0.5b,qwen2.5-3b}/
                                  generated tables, certified summary, alpha-oracle bound, certified fixed stage,
                                  50 random-split robustness
    6_vision_language/{smolvlm2-256m,smolvlm2-500m}/
                                  scene tables, certified summary, AUC bootstrap, 50 random-split robustness
    gate_operating_point.json     Proposition 1 check (VLM, language THINK stage)
    repro_check.json              result of analysis/repro_check.py
  code/
    telemachus/                   navigation, language agent, vision-language, calibration (src, tests, configs,
                                  scripts, analysis, trained artifacts, Colab notebook)
    prometheus/                   multi-agent team, certified help, shift study
```

Per-episode JSON traces of the navigation runs (2.9 GB) were not kept (deleted with the working folders on
2026-10-04); the CSVs here hold every per-episode endpoint, and `analysis/repro_check.py` re-simulates episodes.

## Reproducing

Environments: Python 3.12.15. CPU environment `environment/requirements_cpu_navigation_multiagent.txt`; GPU
environment `environment/requirements_gpu_llm_vlm.txt` (torch 2.11.0+cu128, run on a GTX 1650 4 GB). The 3B
language tables were generated on a free Colab T4 with `code/telemachus/notebooks/TELEMACHUS_LLM_Colab.ipynb`.

From `code/telemachus` (set `PYTHONPATH=src`):

| result | command | runtime (12-core CPU) |
|---|---|---|
| 1 | `python scripts/run_v6.py --config configs/v6_main.json --workers 10 --output results/v6` | ~3.4 h |
| 2 | `python analysis/online_margin.py pilot` then `python analysis/online_margin.py final` (protocol `configs/online_margin_frozen.json`); guaranteed rule: `python analysis/online_margin_hp.py` (protocol `configs/online_margin_hp_frozen.json`) | ~1.2 h + ~0.9 h |
| 5 | `python -m telemachus.llm_agent.generate --split train --limit 400 --vote-k 0 --output results/llm/train_table.jsonl` and `--split test` (0.5B); 3B: Colab notebook with `--model Qwen/Qwen2.5-3B-Instruct --vote-k 3`; then `python analysis/llm_eval.py llm`, `llm_oracle_bound.py llm`, `llm_fixed_certified.py llm`, `llm_robustness.py llm 50` | GPU hours |
| 6 | `python analysis/vlm_run.py --device cuda` (256M, pinned revision) and `--model HuggingFaceTB/SmolVLM2-500M-Video-Instruct --output results/vlm_benchmark_500m`; `python analysis/vlm_robustness.py <dir> 50` | ~1.6 h / ~1.8 h |

From `code/prometheus`:

| result | command |
|---|---|
| 3 | `python scripts/run_certified_help.py` (protocol `configs/certified_help_frozen.yaml`) |
| 4 | `python scripts/run_shift_study.py --workers 3` (protocol `configs/shift_study_frozen.yaml`); guaranteed rule: `python scripts/run_hp_shift.py --workers 9` (protocol `configs/hp_shift_frozen.yaml`; set `OMP_NUM_THREADS=1`, otherwise the gate classifier oversubscribes the CPU and runs ~18x slower); learning gate: `python scripts/run_learning_shift.py --workers 10` (protocol `configs/learning_shift_frozen.yaml`) |

Then, from this folder: `python analysis/gate_operating_point.py`, `python analysis/make_figures.py` and `python analysis/repro_check.py`.

The MSc thesis (in Greek) and the defence slides built from these results are not part of this public
repository.

## Status and what remains

The study is complete: every result above is final and was produced under frozen protocols. Left for future
work:
- models larger than 3B (a 7B 4-bit run fits a free Colab T4);
- a real robot or real-world navigation data;
- energy or money costs instead of the declared resource proxy.

## Protocol discipline

Every study was frozen before its held-out data were run: methods, seeds, alpha, delta, candidate grids and (for
navigation) SHA-256 hashes of source, configs and trained artifacts (`results/1_navigation_certified/*/protocol.json`).
Design choices (grid, shift strength, eta, chunking) used only development or pilot seeds, as stated in each frozen
protocol. One navigation run with a finer grid was aborted for runtime before any test episode; its log is kept in
`results/1_navigation_certified/v6_aborted_fine_grid.log`. The navigation shift severity was raised once on
development seeds because the first pilot did not break the static certificate; both pilots are kept (first pilot:
`online_margin_pilot.log`; second: `online_margin_pilot2.log` and `pilot_harsh.json`, which is the file the frozen
protocol calls `results/online_margin/pilot.json`). The first
launch of the multi-agent guaranteed study was stopped for runtime (CPU oversubscription) before any seed had
finished. It was relaunched unchanged except for `OMP_NUM_THREADS=1`, which does not change any result
(`results/repro_check.json` reruns seed 5). The eta-sensitivity runs override the frozen eta and are labelled
non-primary in their stored protocol. They held 19/19 seeds <= alpha at eta = 0.25, 0.5 and 1.0. The
alpha* solver was first a bisection. alpha* + slack is not monotone, so bisection can miss the feasible interval;
it did so once, for seed 7 at eta = 0.25 (kept as `seed_07_bisection_infeasible.json`). That seed was rerun with
the grid solver now in `hp_online.py`. Any value bisection returned is feasible, so no guarantee was affected. The current solver returns the larger of the grid and bisection values. It reproduces every stored alpha* of the
primary run and of eta = 0.25 exactly. For eta = 1.0, which was run with bisection only, it differs on 5 seeds by
less than 2e-5; use `solve_alpha_star_bisection` to reproduce that run exactly.

## Citation and license

George David Tsitlauri, University of Thessaly, 2026. MIT License (`LICENSE`); citation metadata in
`CITATION.cff`.
