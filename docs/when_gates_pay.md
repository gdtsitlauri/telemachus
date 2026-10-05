# When does a learned gate pay off? An exact operating-point formula

This note gives the mathematical basis of finding 2. The mathematics is elementary and standard: it is the
risk-coverage view of selective classification (Chow 1970; El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017) and
the Neyman-Pearson lemma. What we add is its use to explain, and quantitatively check, the cross-domain results.

## Setting

A single decision either errs (E = 1) or not, with base error rate `pi = P(E = 1)`. A gate with score S escalates
(defers to the expert, who does not err) when S indicates a likely error. It accepts the agent's own decision
otherwise. The deployed risk is `R = P(accept, E = 1)` and the help rate is `h = P(escalate)`.

View S as a detector of errors with ROC curve `(FPR, TPR)`, where `TPR = P(escalate | E = 1)` and
`FPR = P(escalate | E = 0)`. Then `R = pi (1 - TPR)` and `h = pi TPR + (1 - pi) FPR`.

## Proposition 1 (help needed at risk alpha)

For alpha < pi, the smallest help rate at which a threshold gate on S reaches risk <= alpha is

    h_S(alpha) = (pi - alpha) + (1 - pi) * FPR_S(t*),   with   t* = 1 - alpha / pi,

where `FPR_S(t)` is the smallest false-positive rate at which S reaches true-positive rate t.

*Proof.* `R <= alpha` is equivalent to `TPR >= t*`. h increases in both TPR and FPR, and along the ROC curve FPR
does not decrease with TPR. So the minimum is attained at `TPR = t*` with the smallest feasible FPR. QED.

## Consequences

1. **Oracle:** `FPR = 0`, so `h* = pi - alpha`. This is the unavoidable help, set by how often the agent is
   wrong.
2. **Uninformed (random) gate:** `FPR = TPR`, so `h = 1 - alpha / pi`.
3. **A score whose ROC lies below the diagonal near t* needs more help than a random gate.** This explains the
   multi-agent entropy and margin gates (AUC 0.36 on held-out seeds). They needed about 27-30 of 30 possible help
   queries, i.e. as much as full help or random asking, while the learned gate (AUC 0.98) needed about 20, near
   the oracle.
4. **Only one point of the ROC matters, and for strict alpha it is the high-recall end.** The extra help a score C
   needs over a score L is `(1 - pi) [FPR_C(t*) - FPR_L(t*)]`. AUC averages over all operating points, so two
   gates with different AUCs can need the same help, and vice versa. When `alpha << pi`, `t*` is close to 1 and only
   the top-right corner of the ROC counts.
5. **A learned gate pays off when the self-confidence ROC is poor at t*.** Training a calibrated classifier on
   observable features can only help where it improves the ROC at t*. Where the model's own confidence is already
   near the best achievable ROC at t* (VLM 500M), it cannot pay off.
6. **Cascades of several actions also need comparable scores.** With several stages, one global threshold
   compares scores from different stages. A confidence score that ranks errors well within each stage but is on a
   different scale per stage makes the shared threshold sub-optimal. A learned gate outputs `P(correct)` for every
   stage, which is comparable by construction. This is the 3B language-agent case: within-stage confidence AUC
   (THINK 0.79) about equals the learned AUC (0.78), yet the learned gate is 3.3-5.1x cheaper. Calibration across
   stages, not discrimination, drives the gain, in line with UCCI (Kotte 2026).

## Quantitative check

`analysis/gate_operating_point.py` estimates `FPR_S(t*)` from the test ROC and compares the help rate predicted by
Proposition 1 with the help measured at the same test risk. Both are reported in `tables/RESULTS.md`, section 8.
The prediction is an identity on the test set itself. The check is that the cross-model pattern (who needs more
help, and by how much) is read off a single ROC point, with no simulation.
