# Related work: what exists and what TELEMACHUS adds

Checked 2026-10-04 by targeted web search. Entries marked (abstract) were read from the abstract or a summary
only; read the full papers before citing details.

## Comparison

| work | domain | guarantee | gate / deferral score | several information actions | several domains | shift: online + finite-horizon guarantee |
|---|---|---|---|---|---|---|
| Learn-then-Test (Angelopoulos, Bates, Candes, Jordan, Lei 2021) | general | risk <= alpha w.p. 1-delta | any | - | - | no (exchangeable) |
| KnowNo, Robots That Ask for Help (Ren et al., CoRL 2023) | LLM robot planning | conformal task-success coverage | LLM option scores (self-confidence) | ask / act | 1 | no |
| Trust or Escalate (Jung, Brahman, Choi, ICLR 2025) | LLM-as-judge cascade | human agreement >= 1-alpha w.p. 1-delta (fixed-sequence testing) | improved self-confidence ("simulated annotators") | cascade of judges | 1 | no |
| Conformal Cascade (Dou, Lian, Li, Jul 2026) (abstract) | multiple-choice QA, 2-tier LLM cascade | finite-sample coverage | conformal set size (self-confidence) | 2 tiers | 1 (18 benchmarks) | no |
| UCCI (Kotte, May 2026) (abstract) | NER cascade routing | optimality under assumptions; no finite-sample risk bound | isotonic-calibrated token margin | 2 tiers | 1 | no |
| Is Escalation Worth It? (Bouchard, May 2026) (abstract) | LLM cascades | cost-quality frontier theory | confidence thresholds | k-model cascade | 1 | no |
| FrugalGPT (Chen, Zaharia, Zou 2023) | LLM API cascade | none | learned scorer | k-model cascade | 1 | no |
| Learning to defer (Mozannar & Sontag 2020, and follow-ups) | classification | consistency of surrogate losses | learned rejector | defer / predict | - | no |
| ACI (Gibbs & Candes 2021); rolling risk control (Feldman et al. 2023); conformal PID (Angelopoulos et al. 2023) | sequential prediction | long-run average coverage/risk, any sequence | - | - | - | online, long-run only |
| Conformal Decision Theory (Lekeufack et al., ICRA 2024) | robot decisions | realised average risk, any sequence, given a deterministically safe parameter region | - | nominal vs safe backup | robot examples | online; finite-T deterministic bound |
| Confidence-Gated Robot Autonomy (Gaus, Charaja, Haeufle, May 2026) (abstract) | activity recognition for robots | none | entropy, MC dropout, ensembles | act / defer | 3 datasets, 1 task type | no |
| **TELEMACHUS (this work)** | navigation, multi-agent help, language agent, vision-language | LTT risk <= alpha w.p. 1-delta at episode / team / answer level | learned gate vs self-confidence, with an alpha-oracle bound | observe, sample, probe, query, think, tool, vote, expert | 4 | online; Theorem B: finite-horizon, high probability with a **stochastic** fallback |

## What is new here, and what is not

**Not new:**
- certified cascades and ask-for-help with LTT or conformal guarantees (KnowNo, Trust or Escalate, Conformal
  Cascade);
- learned or calibrated deferral scores (FrugalGPT, UCCI, learning to defer);
- online long-run risk control (ACI, rolling RC);
- deterministic realised-risk control with a safe backup (CDT, of which our Theorem A is a special case).

**New, as far as this search found:**
1. **One certified recipe across four structurally different agent settings.** The guarantee is on the agent's
   real unit of risk: collision per episode, team failure per episode, wrong final answer, wrong grounding. Most
   prior work covers one setting.
2. **A cross-domain answer to "when does a learned gate pay off".** On held-out data, the gain tracks how poor
   the model's own confidence is as an error signal:
   - multi-agent: entropy AUC 0.36 vs learned 0.98, about 10 fewer expert calls per episode;
   - LLM 3B: 3.3-5.1x cheaper than confidence in 50/50 random splits;
   - VLM: no gain where confidence is already good.

   This is consistent with the competence-regime finding of Gaus et al. and the calibration finding of UCCI, but
   here it comes under formal guarantees and with an alpha-oracle bound. The honest counterpart: a certified fixed
   stage can match the gate (LLM 3B).
3. **Theorem B: a finite-horizon, high-probability guarantee for online risk control whose fallback is itself
   risky** (p0 > 0). CDT needs a deterministically safe region. Empirically, in the multi-agent shift study, the
   realised failure was <= alpha in 19/19 held-out seeds, against 4/19 for the long-run rule and 7/19 for the
   static certificate. The navigation instance (Theorem A, human handover) held in 3/3 streams.

## Sources
- https://arxiv.org/pdf/2310.05921 (Conformal Decision Theory)
- https://arxiv.org/html/2605.18045 (Confidence-Gated Robot Autonomy)
- https://arxiv.org/abs/2307.01928 (KnowNo)
- https://arxiv.org/pdf/2407.18370 (Trust or Escalate)
- https://arxiv.org/abs/2607.25018 (Conformal Cascade)
- https://arxiv.org/html/2605.18796 (UCCI)
- https://arxiv.org/abs/2605.06350 (Is Escalation Worth It?)
