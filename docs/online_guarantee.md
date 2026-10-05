# A finite-horizon guarantee for online risk control with a safe fallback

Learn-then-Test certifies a gate threshold when deployment is exchangeable with calibration. Under deployment
shift we update the threshold online from observed failures (adaptive / conformal-decision style). That update
controls the *long-run average* failure rate, but with a finite deployment the realised rate overshoots alpha in
many runs (multi-agent study: 4/19 seeds <= alpha). Here we make the online rule **finite-horizon and
high-probability** (or deterministic) by targeting a slightly lower level alpha* that is computed in advance,
and by never clipping the control on its safe side, which lets it run into a **safe fallback**.

## Setting

Deployment proceeds in chunks k = 1..K of n episodes. The gate has a scalar control `s_k`. Larger `s` is safer
(more help, larger safety margin, more deferral). `c_k` in [0, 1] is the observed failure rate of chunk k. There
is no assumption on how the world changes between chunks: arbitrary, even adaptive, shift.

Online rule, with clipping only on the unsafe side:

    s_{k+1} = max(s_min, s_k + eta * (c_k - alpha*))

**Fallback assumption F(p0).** There is a level `s_f` such that whenever `s_k >= s_f`, the deployed policy is a
fallback whose per-episode failure probability, conditional on the past, is at most `p0 < alpha*`. Episodes
within a chunk are conditionally independent given the past.

- Multi-agent team: s = -theta for the learned help gate. For s >= 0 (theta <= 0) every agent asks the expert at
  every step ("full help"). Expert help still fails through actuator slips, so p0 > 0.
- Navigation: s >= s_f hands the whole episode to a human (deferral) before any motion, so p0 = 0 exactly.

## Lemma 1 (accounting identity)

For every K, `sum_{k<=K} (c_k - alpha*) <= (s_{K+1} - s_1) / eta`.

*Proof.* `s_{k+1} = s_k + eta (c_k - alpha*) + r_k`, where `r_k >= 0` when the max is active and 0 otherwise.
Summing gives `eta * sum (c_k - alpha*) = s_{K+1} - s_1 - sum r_k <= s_{K+1} - s_1`. QED.

So the realised failure rate can exceed alpha* only through an excursion of `s` to high values. It remains to
bound how far `s` can climb.

## Theorem A (zero-risk fallback, deterministic)

Under F(0), for every sequence of worlds and every K:

    (1/K) sum_{k<=K} c_k  <=  alpha* + (s_f - s_1 + eta (1 - alpha*)) / (eta K).

*Proof.* While `s_k < s_f`, a single step raises `s` by at most `eta (1 - alpha*)`, so `s` can enter the fallback
region at a value of at most `s_f + eta (1 - alpha*)`. Inside the region `c_k = 0`, so `s` decreases. Hence
`s_{K+1} <= s_f + eta (1 - alpha*)`, and Lemma 1 gives the bound. QED.

Choosing `alpha* = alpha - (s_f - s_1 + eta) / (eta K)` makes the realised failure rate at most alpha in
**every** deployment of length K, with probability one, for any shift.

## Theorem B (bounded-risk fallback, high probability)

Under F(p0), let `v = p0 (1 - p0)`, `g = alpha* - p0 > 0` and `u = 2g / (v/n + 2g/3)`. With probability at least
1 - delta, simultaneously for all K' <= K:

    (1/K') sum_{k<=K'} c_k  <=  alpha* + (s_f - s_1) / (eta K') + 1/K' + log(K/delta) / (u K').

*Proof.* An excursion is a maximal run of chunks with `s_k >= s_f`. It begins at a value of at most
`s_f + eta (1 - alpha*) <= s_f + eta`. During an excursion the increment is
`X_k = eta (c_k - alpha*) = -eta g_k + eta Y_k`, where `g_k = alpha* - E[c_k | past] >= g`,
`Y_k = c_k - E[c_k | past]`, `|Y_k| <= 1` and `Var(Y_k | past) <= v / n`. The variance bound uses conditional
independence and `p(1 - p) <= p0(1 - p0)` for `p <= p0 <= 1/2`.

Bernstein's moment bound gives, for `0 < lambda eta < 3`:

    E[exp(lambda X_k) | past] <= exp(-lambda eta g + (lambda eta)^2 (v/n) / (2 (1 - lambda eta / 3))).

With `lambda eta = u` the exponent is exactly zero. So `exp(lambda (s_k - s_start))`, stopped at the end of the
excursion, is a non-negative supermartingale. By Ville's inequality the excursion rises more than x above its
start with probability at most `exp(-u x / eta)`.

There are at most K excursions. A union bound with `x = (eta / u) log(K / delta)` shows that, with probability at
least 1 - delta, `s_k <= s_f + eta + (eta / u) log(K / delta)` for all k <= K + 1. Lemma 1 then gives the bound.
QED.

Theorem A is the special case p0 = 0, because with `c_k = 0` no rise above the entry point is possible. In
Theorem B, alpha* is computed **before** deployment as the largest value with
`alpha* + (s_f - s_1)/(eta K) + 1/K + log(K/delta)/(u K) <= alpha` (u depends on alpha*). This needs only alpha,
delta, n, K, eta, the starting control s_1 (the Learn-then-Test certificate) and the declared fallback risk p0.

## Corollary C (learning while guaranteed)

Theorems A and B place no condition on the policy used while `s_k < s_f`. It may be any predictable policy, that
is, any function of the past, including an agent that keeps learning during deployment: retraining its gate or
its policy on what it has observed. So the finite-horizon guarantee holds unchanged for a continually learning
agent, provided only that the fallback satisfies F(p0) and that alpha* is fixed before deployment from s_1. Learning
can change cost (help used), never the guarantee. The multi-agent check is in `results/4_multiagent_online_shift/learning`
and `tables/RESULTS.md`, section 9.

## What is and is not claimed

- The guarantee is about the **realised** failure rate of the deployment that actually happens, under arbitrary
  shift. The price is the slack terms: the target alpha* is below alpha, and more help or deferral is used.
- It relies on the fallback assumption. For deferral (navigation) it holds by construction. For expert help
  (multi-agent) it is an assumption about the expert's own failure rate under the shift; we declare p0 in
  advance and report the measured full-help failure per seed as a check.
- **Prior work (checked 2026-10-04).** Theorem A is essentially Theorem 1 of Conformal Decision Theory
  (Lekeufack, Angelopoulos, Bajcsy, Jordan, Malik, ICRA 2024, arXiv:2310.05921). There the update
  `lambda_{t+1} = lambda_t + eta (eps - l_t)` gives `R_t <= eps + (lambda_1 - lambda_safe)/(eta t) + K/t` for any
  data sequence, when a "safe" parameter region **deterministically** bounds the loss. Our Theorem A is that result
  with a K = 1 fallback (human handover) and eps set to alpha* in advance. Related: ACI (Gibbs & Candes 2021),
  rolling risk control (Feldman et al. 2023), conformal PID (Angelopoulos et al. 2023).
- **What Theorem B adds.** Here the fallback's risk is bounded only in conditional probability (`p0 > 0`, e.g.
  expert help that still fails through actuator noise). CDT's deterministic safety assumption does not cover this.
  We replace it with a Bernstein/Ville excursion bound and obtain a finite-horizon guarantee that holds with
  probability `1 - delta`. We found no prior statement of this, but the search was not exhaustive. Present it as
  a modest extension of CDT, not as a new framework.
- A close empirical neighbour of finding 2 ("when does a learned gate pay off") is Gaus, Charaja, Haeufle,
  "Confidence-Gated Robot Autonomy: When Does Uncertainty Actually Help?" (arXiv:2605.18045, May 2026). On three
  activity-recognition benchmarks, they report that uncertainty ranks errors reliably only above a
  dataset-dependent competence threshold. They have no formal guarantees, no learned gate and no shift
  adaptation. Our VLM 256M vs 500M result agrees with their competence-regime finding and must cite it.
