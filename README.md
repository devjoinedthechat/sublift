<h1 align="center">sublift</h1>

<p align="center">
  <b>Measure what a retention experiment was actually worth.</b><br>
  Incremental LTV for subscription A/B tests — censored outcomes, discrete billing periods,
  anytime-valid inference.
</p>

<p align="center">
  <a href="https://github.com/devjoinedthechat/sublift/actions/workflows/ci.yml"><img src="https://github.com/devjoinedthechat/sublift/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python 3.10–3.13">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Status: alpha">
  <img src="https://img.shields.io/badge/dependencies-numpy%20%C2%B7%20scipy%20%C2%B7%20pandas-lightgrey" alt="Dependencies">
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#what-you-get">What you get</a> ·
  <a href="#why-not-just">Why not just…</a> ·
  <a href="#does-it-actually-work">Validation</a> ·
  <a href="#api">API</a> ·
  <a href="#assumptions-and-scope">Scope</a> ·
  <a href="docs/">Docs</a>
</p>

---

You ran a save offer against a holdout. Retention went up. Was it worth it?

```python
result = sublift.incremental_ltv(panel, horizon=12, strata=["plan", "tenure_bucket"])
print(result)
```

```
Incremental LTV over 12 billing periods
=======================================
  -12.1845 revenue per subscriber
  95% CI [-13.2493, -11.1197]   se 0.5433   p = 0.0000
  relative to control: -14.09% [-15.23%, -12.95%]

  control      n=  20,140   value      86.4767
  treatment    n=  19,860   value      74.2922

  estimator: stratified   inference: influence
  strata: plan, tenure_bucket

  if you have been monitoring this test, read this line instead:
  anytime-valid 95% CS at n=40,000: [-13.8343, -10.5347] (excludes 0); 1.55x the fixed-sample width
```

The offer bought **+0.32 billing periods** per subscriber and **destroyed $12.18 of lifetime
value**, because the discount that bought the retention cost more than the retention was worth.
A 30-day retention test calls that a win.

## Install

```bash
pip install sublift
```

Requires Python 3.10+. Depends on numpy, scipy and pandas — nothing else.

## Quickstart

Start from subscription dates, which is how the data actually leaves your warehouse:

```python
import sublift as sl

panel = sl.SubscriberPanel.from_spans(
    df,
    subject="user_id",
    arm="variant",
    assigned_at="experiment_entered_at",   # start of period 1, not signup date
    ended_at="subscription_ended_at",      # null = still active
    observed_through="2026-09-01",         # the data cut
    billing_interval="month",
    price="mrr",
    cause="churn_reason",                  # optional: voluntary vs involuntary
    covariates=["plan", "tenure_bucket", "engagement"],
)

sl.check_randomization(panel)                                    # before anything else
sl.incremental_ltv(panel, horizon=12, strata=["plan"])           # the money question
sl.retained_periods_lift(panel, horizon=12, strata=["plan"])     # the retention question
sl.churn_decomposition(panel, horizon=12)                        # which churn moved
sl.qini(panel, horizon=12, covariates=["engagement", "plan"])    # who to target
```

## Why this is hard

Four problems at once, and the usual answer — a two-proportion z-test on 30-day retention —
solves none of them.

**The outcome is censored.** Most subscribers in your test are still active. Their lifetime
value hasn't happened yet. Averaging realized revenue is biased downward, and biased
*differently in each arm* whenever the treatment moved the churn curve — which is the entire
thing you're trying to measure.

**Churn is discrete.** Subscriptions don't decay continuously; they end at renewal. The hazard
is a spike train on billing boundaries, with a cliff at the annual renewal. Continuous-time
machinery smooths over exactly the structure that matters.

**The effect is slow and the base is noisy.** A 1.5-point retention lift on a 6% monthly churn
base needs either enormous samples or real variance reduction. So teams substitute a 30-day
proxy — a different question with a more convenient answer.

**Everybody peeks.** A fixed-sample 95% interval is only a 95% interval if you look once.
Checked daily, its false-positive rate climbs past 5%, because a random walk eventually crosses
any fixed boundary. [Measured below](#does-it-actually-work): **26.4%** across sixteen looks.

## What you get

### One estimand, stated explicitly

Fix a horizon `H` of billing periods. With `T` periods paid and `S(t) = P(T > t)`:

```
LTV(H)  =  sum_{t=1..H}  w_t · S(t-1)
```

`w_t` is period-`t` revenue — a known price schedule, or revenue observed in the panel. With
`w_t = 1` it collapses to restricted mean survival time, the expected billing periods retained.
One primitive, both readouts, and the contrast between arms is the answer.

**The horizon is mandatory.** A "lifetime" value with no horizon is an extrapolation wearing a
measurement's clothes. sublift makes you name it and refuses to estimate past your data unless
you pass `allow_extrapolation=True` and label the number a projection.

### Three estimators, same estimand

| | assumes | inference | monitor sequentially? | good below ~5k subscribers? |
|---|---|---|---|---|
| `unadjusted` | randomization, independent censoring | influence function | ✅ | ✅ |
| **`stratified`** (default) | + strata are pre-assignment | influence function | ✅ | ✅ |
| `adjusted` | + a hazard model in the covariates | efficient influence function | ✅ | ❌ |

`stratified` is the default because it reduces variance, stays valid under monitoring, and its
influence function is **exact at any sample size**. A handful of prognostic strata — plan,
tenure bucket, pre-period engagement quantile — recovers most of what full covariate adjustment
gets you.

`adjusted` fits a discrete-time logistic hazard **per arm with a saturated time baseline**,
standardizes over the covariate distribution, then corrects with its efficient influence
function — a one-step (AIPW) estimator. That buys three things at once: standard errors without
200 bootstrap refits, **anytime-valid monitoring for the estimator that reduces variance the
most**, and double robustness, since the augmentation holds the estimate up where the hazard
model is wrong. It gets the most from continuous covariates, which stratification can only
bucket.

Its limitation is asymptotic: coverage is ~93% at 4,000 subscribers, nominal by ~16,000.
sublift warns below 5,000 rather than quietly running narrow. See
[choosing an estimator](docs/choosing-an-estimator.md).

### Censoring you know rather than estimate

Under administrative censoring, a subscriber's potential follow-up is fixed the day they enter
the experiment, by the distance from their assignment date to the data cut. That is known for
**everyone** — including subscribers who churned in period one, long before the cut.

`from_spans` records it, so the censoring distribution used by `adjusted` is computed exactly
instead of inferred from whoever happened to survive. Panels built from period counts fall back
on reverse Kaplan–Meier, and sublift tells you which one it used rather than leaving it implicit.

### Anytime-valid monitoring

```python
result.confidence_sequence(n_target=50_000)
# anytime-valid 95% CS at n=40,000: [-13.8343, -10.5347] (excludes 0); 1.55x the fixed-sample width
```

A confidence sequence is valid *simultaneously at every sample size*: across unlimited looks,
the probability it ever excludes the truth is at most α. Stop whenever you like, including
because of what you just saw.

All three estimators support it. This is why the influence functions are derived by hand rather
than bootstrapped: the bootstrap gives standard errors, but only an influence function gives the
i.i.d. per-subject sequence a confidence sequence needs. The price is ~1.5–1.7× the fixed-sample
width — the honest cost of looking.

### Voluntary vs involuntary churn

A large share of subscription churn is involuntary — a card expires, dunning runs out of
retries. The subscriber never decided anything. A save offer acts on voluntary churn; measured
against all-cause churn its effect is diluted by a baseline it cannot move, while a card-updater
shows up as "retention improved" and the retention team takes the credit.

```python
print(sl.churn_decomposition(panel, horizon=12))
```

```
Retention effect by cause of churn, over 12 billing periods
===========================================================
  total  +0.3216 periods per subscriber [+0.2297, +0.4134]

  involuntary  -0.0422  [-0.0908, +0.0063]     -13% of the effect
  voluntary    +0.3638  [+0.2720, +0.4556]     113% of the effect

  Causes sum to the total exactly; the split is an identity, not an attribution.
```

Note the involuntary number is *negative*: keeping subscribers alive longer gives their card
more chances to fail. That is a real competing-risks trade-off, it reproduces the simulator's
closed-form truth, and a single all-cause hazard cannot express it.

Writing `L_j` for periods lost to cause `j`, the decomposition is exact:
`RMST(H) = H - Σ_j L_j`, so the causes sum to the headline effect with no residual.

What sublift deliberately does *not* report is a cause-specific curve with the other cause
censored out ("what if nobody ever had a failed payment?"). That isn't identified without
assuming the causes are independent, which for subscriptions they plainly aren't — the
subscriber halfway out the door is the one who doesn't bother updating their card.

### Randomization checks that run whether you ask or not

```python
print(sl.check_randomization(panel, expected_ratio=0.5))
```

Sample ratio mismatch is tested automatically on every estimate, at the standard `p < 0.001`
threshold, and warns loudly. A broken assignment invalidates everything downstream and the
failure is silent — a targeting rule that excluded a segment, a flag that defaulted on for iOS,
an ETL job that dropped a partition. The result still prints a tight interval.

Baseline covariate balance is reported as standardized mean differences, not t-tests: with a
large experiment a trivial imbalance is "significant", and with a small one a serious imbalance
isn't. The standardized difference measures *how big* it is, which is the question.

### Planning, before you start

```python
plan = sl.duration_to_detect(arrivals_per_period=8_000, horizon=12,
                             baseline_hazard=0.06, treatment_odds_ratio=0.90)
```

Enrollment duration needed under both fixed-sample and always-valid analysis. Computed by
simulating your actual enrollment schedule, because the variance of a censored survival contrast
depends on the enrollment pattern in a way no closed form captures — subscribers who joined last
month contribute one period of follow-up each to a 12-period estimand.

### Targeting

```python
curve = sl.qini(panel, horizon=12, covariates=["engagement", "plan", "tenure_bucket"])
curve.best_fraction()
```

Per-subscriber effects on the same restricted-mean scale as the headline number,
**cross-fitted** so nobody is ranked by a model that saw them. Each point of the curve is a real
censoring-aware estimate re-run inside the targeted subset — not a sum of predicted scores.

The default learner is deliberately **not** a T-learner. Fitting a model per arm and differencing
them makes predicted heterogeneity depend on `(γ̂₁ − γ̂₀)′x`, the gap between two independently
estimated coefficient vectors. With no real effect modification that gap is pure noise, and it
does not average out. sublift fits one pooled model with **ridge-penalized treatment×covariate
interactions**, with the penalty chosen by held-out likelihood. Real heterogeneity survives it;
noise doesn't. [It wins in both regimes.](#does-it-actually-work)

## Why not just…

| | has | missing |
|---|---|---|
| `lifelines`, `scikit-survival` | survival, RMST | no causal contrast, no experiment readout |
| `EconML`, `DoWhy`, `CausalPy` | causal contrasts | no censoring, no subscription semantics |
| `lifetimes`, `PyMC-Marketing` | LTV | built for *non-contractual* retail; wrong model class |
| Eppo, Statsig, Optimizely | sequential testing | closed source, generic metrics, no censored-LTV estimand |
| a t-test on 30-day retention | simplicity | answers a different question |

sublift is the intersection: **contractual discrete-time survival + a causal contrast + censored
LTV + anytime-valid inference**, in one estimand.

## Does it actually work?

A library that reports confidence intervals is worth nothing if the intervals don't cover, and
you can't check coverage against real data, because real data doesn't come with a true effect
attached. So sublift ships its ground-truth simulator as **part of the public API**, and the test
suite is built on it. Everything below is reproducible:

```bash
python examples/validation_report.py     # prints this section
pytest -m slow                           # asserts it
```

**The intervals are real.** 400 replications, 4,000 subscribers each, nominal 95%:

| estimator | coverage | bias | reported se | actual spread |
|---|---|---|---|---|
| `unadjusted` | 94.8% | +0.00007 | 0.0919 | 0.0929 |
| `stratified` | 93.8% | −0.00097 | 0.0910 | 0.0927 |
| `adjusted` (n=16k) | 94.5% | −0.00018 | — | se/sd = 1.009 |

`adjusted` is shown at 16,000 subscribers because its influence function is first-order; at
4,000 it covers at 93.0%, which is why it warns below 5,000.

**Censoring is handled where the obvious alternative fails.** "Naive" differences mean observed
tenure — treating still-active subscribers as though they ended on the day you pulled the data.
True effect: +0.2587 periods.

| follow-up | censored | sublift bias | naive bias |
|---|---|---|---|
| 9 periods | 65% | +0.021 | **−0.121** |
| 12 periods | 59% | +0.018 | −0.085 |
| 18 periods | 51% | −0.014 | −0.074 |

At 65% censoring the naive estimator is wrong by 47% of the effect it's trying to measure.

**Peeking really is that bad, and the sequence really does fix it.** A true null, monitored at
16 interim looks, 250 replications:

| | false positives |
|---|---|
| fixed-sample 95% interval, checked at every look | **26.4%** |
| anytime-valid 95% confidence sequence | **0.8%** |

Nominal rate: 5%. Sixteen looks turn a 5% test into a 26% one.

**The uplift learner earns its default.** Correlation between predicted and true individual
effect:

| regime | T-learner | pooled + shrunk |
|---|---|---|
| constant odds ratio | 0.26 | **0.62** |
| strong effect modification | 0.94 | **0.97** |

**The competing-risks influence functions are calibrated.** 400 replications, per cause: bias
within Monte Carlo error, and the reported standard error within 1.5% and 4% of the *actual*
sampling spread.

**The new influence functions agree with the bootstrap.** The `adjusted` estimator's efficient
influence function and a 300-draw bootstrap of the same estimator land within a few percent of
each other — two independent routes to the same standard error.

## API

| | |
|---|---|
| `SubscriberPanel.from_spans` | dates in, billing periods out — **start here** |
| `.from_periods` / `.from_subjects` | already have period counts |
| `check_randomization` | SRM + baseline balance |
| `incremental_ltv` | the money question |
| `retained_periods_lift` | the retention question |
| `churn_decomposition` | voluntary vs involuntary |
| `LiftResult.confidence_sequence` | anytime-valid interval |
| `LiftResult.relative_ci` | interval for the % lift (delta method, not `ci / control`) |
| `LiftResult.curves` | per-arm survival and cumulative value |
| `duration_to_detect` | how long until this test can answer |
| `qini` / `uplift_scores` | targeting |
| `simulate_experiment` | ground-truth data for planning and validation |

Results render as HTML in Jupyter, and `SubliftError` / `PanelError` / `NotIdentifiedError`
separate "your data is malformed" from "the data cannot answer that question". Both still
subclass `ValueError`, so existing handlers keep working.

## Assumptions and scope

What sublift assumes, stated rather than buried:

- **Assignment is randomized.** No observational identification.
- **Censoring is administrative** — you cut the data on a date. Dropout that depends on
  subscriber state violates this.
- **Baseline covariates only.** Adjusting for anything measured after assignment reintroduces
  the bias randomization removed; the panel constructors reject it.
- **Two arms** at a time.

Roadmap:

- [x] Efficient influence function for `adjusted`, giving it a confidence sequence
- [ ] Informative censoring via inverse-probability-of-censoring weights *(largest known gap)*
- [ ] More than two arms, with multiple-comparison control across segments
- [ ] Stratified and covariate-adjusted versions of `churn_decomposition`
- [ ] Pauses, plan switches and win-backs in the panel and the simulator

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | From a warehouse table to a defensible number |
| [Choosing an estimator](docs/choosing-an-estimator.md) | Which of the three, and why |
| [Monitoring a running test](docs/monitoring.md) | Why peeking breaks a p-value |
| [Voluntary vs involuntary churn](docs/competing-risks.md) | Competing risks, and why the split is exact |
| [Assumptions](docs/assumptions.md) | When sublift is wrong — read this one |
| [Method](docs/method.md) | The estimand, the influence functions, the references |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). One rule: **anything that reports an interval ships with
a coverage test.** An estimator that is fast, elegant and miscalibrated is worse than no
estimator, because someone will ship a decision on it.

Good first contributions, and issues where help is genuinely wanted, are labelled on the
[issue tracker](https://github.com/devjoinedthechat/sublift/issues). Method questions are
welcome in [Discussions](https://github.com/devjoinedthechat/sublift/discussions) — there are no
stupid ones, statistics is large, and "you should already know this" is never a useful answer.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## References

- Waudby-Smith, Arbour, Sinha, Kennedy & Ramdas — *Time-uniform central limit theory and
  asymptotic confidence sequences*. The confidence sequence.
- Moore & van der Laan (2009) — *Covariate adjustment in randomized trials with binary outcomes*.
  Why arm-specific models with a saturated time baseline survive misspecification.
- Andersen, Borgan, Gill & Keiding — *Statistical Models Based on Counting Processes*. The
  influence function for the product-limit estimator.
- Fine & Gray (1999) — *A proportional hazards model for the subdistribution of a competing
  risk*. Background for the cumulative-incidence decomposition.

## Citing

If sublift contributes to something you publish, there is a [CITATION.cff](CITATION.cff);
GitHub's "Cite this repository" button will format it for you.

## License

[Apache-2.0](LICENSE)
