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

One call runs the checks in the order that matters, then the estimate, then says what to make of
both:

```python
import sublift as sl

print(sl.review(panel, horizon=12, price=schedule, strata=["plan"], monitoring=True))
```

```
Experiment review
=================

  VERDICT: no problems found in the checks sublift can run.

  [note] Retention went up and lifetime value went down
      The intervention bought +0.316 billing periods per subscriber and -12.184 in
      value. Whatever it spent to hold those subscribers cost more than holding them
      was worth. A retention-only readout would have called this a win.
  ...
```

A **blocker** — a sample ratio mismatch, say — means the number below it is not measuring what
it says, and the verdict reads *do not act on this*. The estimate is still computed, because
hiding it only invites someone to go and compute it a worse way. A **warning** means the result
stands with a caveat. A **note** is a finding in its own right.

Nothing in `review` is new statistics. It is the existing estimators in the order an experienced
analyst would run them, with the verdict written down instead of assumed. Everything it calls is
available separately.

## Building a panel

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

sl.review(panel, horizon=12, strata=["plan"])                    # all of the below, in order
sl.check_randomization(panel)                                    # before anything else
sl.check_censoring(panel)                                        # is censoring really benign?
sl.incremental_ltv(panel, horizon=12, strata=["plan"])           # the money question
sl.retained_periods_lift(panel, horizon=12, strata=["plan"])     # the retention question
sl.churn_decomposition(panel, horizon=12)                        # which churn moved
sl.segment_scan(panel, by=["plan"], horizon=12)                  # who it worked for
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

### And then the questions that follow

Each of these has a page in [docs/](docs/) that explains the method; here is what they are for
and the number that makes the case.

**[Several arms, one error rate](docs/multi-arm.md)** — `multi_arm_lift`. Testing three save
offers and reporting the best is one experiment with three chances to be wrong: with four null
arms, declaring a winner happens **13.0%** of the time uncorrected against a nominal 5%. The
default correction is max-t, which beats Bonferroni because contrasts sharing a control arm
correlate at 0.5. `comparisons="all-pairs"` when the arms are alternatives rather than variations
on a holdout.

**[Subscribers who come back](docs/win-backs.md)** — `from_spells`, `occupancy_lift`. People
cancel and resubscribe, or pause over the summer. Time-to-first-cancellation cannot see any of
it, and the error runs the wrong way: the subscribers written off as lost are disproportionately
in the *control* arm, so at a 20% win-back rate it overstates the win by **44%**.
`occupancy_decomposition` splits those periods by why each lapse happened.

**[Slicing the base](docs/segments.md)** — `segment_scan`. "It worked great for annual
subscribers on iOS" is the most common false finding. Slice a **null** experiment eight ways and
per-comparison tests report a segment that differs **26.4%** of the time; this reports one 5.2%
of the time. It also catches the trap that a *uniform* odds ratio produces genuinely different
retained periods per segment, and labels that a scale artefact rather than a mechanism.

**[Monitoring a running test](docs/monitoring.md)** — `result.confidence_sequence()`. A
fixed-sample interval is only a 95% interval if you look once; checked at sixteen interim points
its false-positive rate is **26.4%**. A confidence sequence is valid at every sample size
simultaneously, so you may stop whenever you like, including because of what you just saw. All
three estimators support it, which is why the influence functions are derived by hand.

**[Voluntary vs involuntary churn](docs/competing-risks.md)** — `churn_decomposition`. A fifth to
two fifths of subscription churn is a failed card, not a decision, and a save offer cannot act on
it. The split is an exact identity, not an attribution, so the causes sum to the headline with no
residual — and it surfaces effects a single hazard cannot express, such as retention *increasing*
exposure to payment failure.

**Diagnostics that run whether you ask or not** — `check_randomization` tests sample ratio
mismatch on every estimate, because a broken assignment invalidates everything downstream and
fails silently. `check_censoring` tests whether censoring is really administrative, and
[`censoring_sensitivity`](docs/assumptions.md#2-censoring-is-administrative) bounds the part no
test can reach: *how far* independent censoring would have to fail before the conclusion changes.

**Planning and targeting** — `duration_to_detect` answers how long until the test can answer the
question, by simulating your actual enrollment schedule. `qini` gives cross-fitted per-subscriber
effects, with a pooled shrunk-interaction learner that beats a T-learner whether or not effect
modification is real (0.26 → 0.62 when it isn't, 0.94 → 0.97 when it is).

## Does it run on a real subscriber base?

A retention team at a media company has millions of subscribers, so this is not a rhetorical
question. One million subscribers, twelve billing periods, on a laptop:

| call | time | peak memory |
|---|---|---|
| `retained_periods_lift` (unadjusted) | 0.30s | 76 MB |
| `retained_periods_lift` (stratified) | 1.54s | 138 MB |
| `incremental_ltv` | 0.41s | 168 MB |
| `churn_decomposition` | 0.18s | 97 MB |
| `segment_scan` (8 segments) | 0.22s | 53 MB |
| `retained_periods_lift` (adjusted) | 1.89s | 803 MB |
| **`review`** (checks + both metrics + causes) | **1.89s** | **160 MB** |

Ten times that — **ten million subscribers** — and everything is still under twenty seconds:
`retained_periods_lift` 3.3s/0.5 GB, `segment_scan` 6.7s/0.8 GB, `review` 11.4s/0.7 GB. The
algorithms are linear in subscribers, so a hundred million extrapolates to roughly half a minute
and 5–8 GB — a server, not a laptop.

Two properties are worth knowing because they are not the obvious ones. **Memory does not grow
with the horizon**: the influence function forms no subscriber-by-period array at all, because
both of its terms collapse to lookups into arrays of length `horizon`. And **`segment_scan`
memory does not grow with the number of segments**, only with the number of dimensions scanned —
cross-producting three dimensions into 18 segments costs less than scanning them as 8.

`tests/test_scale.py` asserts these budgets, because reintroducing an `(n, horizon)` temporary in
a hot path is easy and no correctness test would catch it. The optimisation history, including
what turned out to be dead code, is in [CHANGELOG.md](CHANGELOG.md).

## Families the library can't see

`multi_arm_lift` corrects across arms and `segment_scan` across segments, but only you know
what you actually looked at. `correct_family` takes any set of results that expose an influence
function and corrects across them:

```python
sl.correct_family({
    "retained periods": sl.retained_periods_lift(panel, horizon=12, estimator="unadjusted"),
    "LTV":              sl.incremental_ltv(panel, horizon=12, price=12.0, estimator="unadjusted"),
})
```

This is where max-t earns the most. With a flat price, LTV and retained periods are the same
statistic scaled — they correlate at **1.00** — and max-t prices them as the one comparison they
are, 16% narrower than Bonferroni. Three arms over four segments is twelve comparisons, not
three plus four; assemble the family and pass it.

`multi_arm_lift(..., comparisons="all-pairs")` tests every pair rather than every arm against
the control, for when the arms are alternatives rather than variations on a holdout. Six
comparisons instead of three, correlation dropping from 0.51 to 0.14 as the shared control
disappears, and a correspondingly wider critical value.

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

**The validation is not circular.** Everything above generates from `sublift.datasets`, which
encodes one author's assumptions about how subscriptions behave — and if those were the same
assumptions baked into the estimators, none of it would prove anything. So one test generates
from somewhere else entirely: continuous-time Weibull lifetimes discretised onto billing periods,
gamma frailty so hazards are *not* logistic in anything observed, and a treatment that acts by
accelerating time rather than shifting odds. All three estimators recover the true effect within
1.5 standard errors, including the covariate-adjusted one with a now badly misspecified model.

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
| **`review`** | **the checks, the estimate and a verdict — start here** |
| `SubscriberPanel.from_spans` | dates in, billing periods out |
| `.from_spells` | several paying spells per subscriber (win-backs, pauses) |
| `.from_periods` / `.from_subjects` | already have period counts |
| `check_randomization` | SRM + baseline balance |
| `check_censoring` | is censoring really administrative? |
| `censoring_sensitivity` | how wrong that would have to be to change the answer |
| `incremental_ltv` | the money question |
| `retained_periods_lift` | the retention question |
| `occupancy_lift` | periods paid for, when subscribers return |
| `occupancy_decomposition` | those periods split by why they lapsed |
| `correct_family` | correct across any family you assembled yourself |
| `churn_decomposition` | voluntary vs involuntary |
| `multi_arm_lift` | many arms vs one control, family-wise error control |
| `segment_scan` | effects by segment, without manufacturing findings |
| `SubscriberPanel.contrast` | pull one comparison out of a multi-arm panel |
| `LiftResult.confidence_sequence` | anytime-valid interval |
| `LiftResult.relative_ci` | interval for the % lift (delta method, not `ci / control`) |
| `LiftResult.curves` | per-arm survival and cumulative value |
| `duration_to_detect` | how long until this test can answer |
| `qini` / `uplift_scores` | targeting |
| `simulate_experiment` / `simulate_multi_arm` | ground-truth data for planning and validation |
| `simulate_experiment` | ground-truth data for planning and validation |

Results render as HTML in Jupyter, and `SubliftError` / `PanelError` / `NotIdentifiedError`
separate "your data is malformed" from "the data cannot answer that question". Both still
subclass `ValueError`, so existing handlers keep working.

## Assumptions and scope

What sublift assumes, stated rather than buried:

- **Assignment is randomized.** No observational identification.
- **Censoring is administrative** — you cut the data on a date. Dropout that depends on
  subscriber state violates this; `check_censoring` tests it.
- **Baseline covariates only.** Adjusting for anything measured after assignment reintroduces
  the bias randomization removed; the panel constructors reject it.
- **Two arms** for the ordinary estimators; use `multi_arm_lift` for more.

### What is not here

- **Stratified or covariate-adjusted competing risks.** `churn_decomposition` is nonparametric.
- **A hazard model that is not logistic.** Flexible enough in practice with a saturated time
  baseline, but it is a parametric choice and it is made for you.
- **Multiplicity across horizons.** Fix the horizon in advance; `correct_family` will correct
  across several if you insist, but choosing one after seeing the data is not something any
  correction repairs.
- **Informative censoring, solved.** It is detected (`check_censoring`), partly corrected
  (covariate adjustment more than halves the bias) and bounded (`censoring_sensitivity`). It is
  not solved, here or anywhere, and the docs say so with the numbers.

Everything else on the original roadmap is done: an efficient influence function for the adjusted
estimator, arms and segments and metrics as families, all-pairs comparisons, win-backs and pauses,
covariate-adjusted occupancy, competing risks beyond the first spell, and the scale work. See
[CHANGELOG.md](CHANGELOG.md).

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | From a warehouse table to a defensible number |
| [Choosing an estimator](docs/choosing-an-estimator.md) | Which of the three, and why |
| [Monitoring a running test](docs/monitoring.md) | Why peeking breaks a p-value |
| [Testing several arms](docs/multi-arm.md) | Many offers, one holdout, one error rate |
| [Slicing the base](docs/segments.md) | Segment scans that don't manufacture findings |
| [Subscriptions that come back](docs/win-backs.md) | Win-backs, pauses, and why first-cancellation overstates |
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
