# sublift

**Incremental LTV measurement for subscription retention experiments.**

You ran a save offer against a holdout. How much lifetime value did it actually create?

```python
import sublift as sl

panel = sl.SubscriberPanel.from_periods(
    df, subject="user_id", period="billing_cycle", churned="churned",
    arm="variant", revenue="amount",
    covariates=["plan", "tenure_bucket"],
)

result = sl.incremental_ltv(panel, horizon=12, strata=["plan", "tenure_bucket"])
print(result)
```

```
Incremental LTV over 12 billing periods
=======================================
  -6.4624 revenue per subscriber (-7.2% vs control)
  95% CI [-7.9780, -4.9468]   se 0.7733   p = 0.0000

  control      n=   9,952   value      89.7944
  treatment    n=  10,048   value      83.3320

  estimator: stratified   inference: influence
  strata: plan, tenure_bucket

  if you have been monitoring this test, read this line instead:
  anytime-valid 95% CS at n=20,000: [-8.8107, -4.1141] (excludes 0); 1.55x the fixed-sample width
```

That save offer lifted retention by half a billing period and **destroyed** $6.46 of LTV per
subscriber, because the discount that bought the retention cost more than the retention was
worth. A 30-day retention test would have called it a win.

---

## Why this exists

Measuring the LTV impact of a retention intervention has four problems at once, and the usual
answer — a two-proportion z-test on 30-day retention — solves none of them.

**The outcome is censored.** Most subscribers in your test are still active. Their lifetime
value hasn't happened yet. Averaging realized revenue is biased downward, and biased
*differently in each arm* whenever the treatment moved the churn curve — which is the entire
thing you're trying to measure.

**Churn is discrete.** Subscriptions don't decay continuously; they end at renewal. The hazard
is a spike train on billing boundaries, and annual plans put a cliff at period 12. Continuous-time
machinery smooths over exactly the structure that matters.

**The effect is slow and the base is noisy.** A 1.5-point retention lift on a 6% monthly churn
base needs either enormous samples or real variance reduction. So teams substitute a 30-day
proxy, which is a different question with a more convenient answer.

**Everybody peeks.** Growth teams check the dashboard daily. A fixed-sample 95% interval is only
a 95% interval if you look once; under daily monitoring its false-positive rate climbs far past
5%, because a random walk eventually crosses any fixed boundary. Every "we called it early and
it didn't replicate" retrospective is this effect.

Existing tools each cover part of this. `lifelines` and `scikit-survival` do survival but not
causal contrasts. `EconML`, `DoWhy` and `CausalPy` do causal contrasts but not censoring.
`lifetimes` and `PyMC-Marketing` do LTV, for *non-contractual* businesses — the wrong model
class for subscriptions. Eppo and Statsig do sequential testing, on generic metrics, and are
closed. sublift is the intersection: **contractual discrete-time survival, a causal contrast,
censored LTV, and anytime-valid inference, in one estimand.**

## Install

```bash
pip install sublift          # numpy, scipy, pandas. Nothing else.
```

## The estimand

Fix a horizon `H` of billing periods. Let `T` be periods paid for, `S(t) = P(T > t)`:

```
LTV(H)  =  sum_{t=1..H}  w_t * S(t-1)
```

`w_t` is period-`t` revenue: a known price schedule, or revenue observed in the panel.
With `w_t = 1` it collapses to restricted mean survival time — expected billing periods
retained. One primitive, both readouts, and the contrast between arms is the answer.

**The horizon is mandatory.** A "lifetime" value with no horizon is an extrapolation wearing a
measurement's clothes. sublift makes you name the horizon and refuses to estimate past the data
you have unless you explicitly pass `allow_extrapolation=True` and label the result a projection.

## The three estimators

All three target the same estimand and differ only in what they assume in exchange for variance.

| | assumes | inference | monitor sequentially? |
|---|---|---|---|
| `unadjusted` | randomization, independent censoring | influence function | yes |
| `stratified` **(default)** | + strata are pre-assignment | influence function | yes |
| `adjusted` | + a hazard model in the covariates | bootstrap | no |

`stratified` is the default because it is the one that both reduces variance and stays valid
under monitoring. On real subscriber data a handful of prognostic strata — plan, tenure bucket,
pre-period engagement quantile — recovers most of what full covariate adjustment gets you.

`adjusted` fits a discrete-time logistic hazard **separately per arm with a saturated time
baseline**, then standardizes over the covariate distribution (g-computation). That specific
configuration is what keeps it consistent under randomization even when the covariate model is
misspecified (Moore & van der Laan, 2009). It gets the most out of continuous covariates, and
costs you the confidence sequence.

## Anytime-valid monitoring

```python
result = sl.incremental_ltv(panel, horizon=12, strata=["plan"])
print(result.confidence_sequence(n_target=50_000))
```

```
anytime-valid 95% CS at n=20,000: [-8.8107, -4.1141] (excludes 0); 1.55x the fixed-sample width
```

A confidence sequence is valid *simultaneously at every sample size*: across unlimited looks,
the probability it ever excludes the truth is at most α. You can stop whenever you want,
including because of what you just saw. sublift uses the asymptotic confidence sequence of
Waudby-Smith et al., applied to the estimator's influence function — which is why the influence
functions are derived by hand rather than bootstrapped. The bootstrap would give standard errors;
only the influence function gives the i.i.d. sequence a confidence sequence needs.

The price is about 1.5–1.7× the fixed-sample width at the tuning point. That is the honest cost
of being allowed to look.

## Planning, before you start

```python
plan = sl.duration_to_detect(
    arrivals_per_period=8_000, horizon=12,
    baseline_hazard=0.06, treatment_odds_ratio=0.90,
)
print(plan)
```

Returns the enrollment duration needed under both fixed-sample and always-valid analysis. The
variance of a censored survival contrast depends on the *enrollment pattern* in a way no closed
form captures — subscribers who joined last month contribute one period of follow-up each to a
12-period estimand — so this is computed by simulating your actual schedule.

## Targeting

```python
curve = sl.qini(panel, horizon=12, covariates=["engagement", "plan", "tenure_bucket"])
print(curve.best_fraction())
```

Per-subscriber effects on the same restricted-mean scale as the headline number,
**cross-fitted** so nobody is ranked by a model that saw them. Each point of the curve is a real
censoring-aware estimate re-run inside the targeted subset — not a sum of predicted scores. A
Qini curve built from a model's own predictions measures the model's confidence; this one
measures the effect.

The default learner is deliberately **not** a T-learner. Fitting a model per arm and
differencing them makes predicted heterogeneity depend on `(γ̂₁ − γ̂₀)′x` — the gap between two
independently estimated coefficient vectors. When the treatment doesn't genuinely modify a
covariate's effect, that gap is pure noise, and it does not average out. sublift instead fits
one pooled model with shared effects and **ridge-penalized treatment×covariate interactions**,
with the penalty chosen by held-out likelihood. Real effect modification survives it; noise
doesn't.

## Does it actually work?

A library that reports confidence intervals is worth nothing if the intervals don't cover, and
you can't check coverage against real data, because real data doesn't come with a true effect
attached. So sublift ships its ground-truth simulator as part of the public API, and the test
suite is built on it. Everything below is measured, reproducible with
`python examples/validation_report.py`, and asserted in `pytest -m slow`.

**The intervals are real.** 400 replications, 4,000 subscribers each, nominal 95%:

| estimator | coverage | bias | reported se | actual spread |
|---|---|---|---|---|
| `unadjusted` | 94.8% | +0.00007 | 0.0919 | 0.0929 |
| `stratified` | 93.8% | −0.00097 | 0.0910 | 0.0927 |

**Censoring is handled where the obvious alternative fails.** "Naive" is the difference in mean
observed tenure, capped at the horizon — what you get by treating still-active subscribers as
though their subscription ended on the day you pulled the data. True effect: +0.2587 periods.

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

Nominal rate: 5%. Checking a fixed-sample interval sixteen times turns a 5% test into a 26% one.
That is the entire argument for the sequential layer, and it's a test rather than a paragraph.

**The uplift learner earns its default.** Correlation between predicted and true individual
effect, 10,000 subscribers:

| regime | T-learner | pooled + shrunk |
|---|---|---|
| constant odds ratio | 0.26 | **0.62** |
| strong effect modification | 0.94 | **0.97** |

The pooled model wins in both regimes — it isn't a bet on heterogeneity being small.

You can also point the simulator at your own numbers before committing to a test:

```python
sim = sl.simulate_experiment(n=40_000, baseline_hazard=0.06, treatment_odds_ratio=0.9)
sl.retained_periods_lift(sim.panel, horizon=12, estimator="unadjusted")
```

## Not in v0.1

Stated rather than silently absent:

- **Informative censoring.** Censoring is assumed administrative — you cut the data on a date.
  Dropout that depends on subscriber state (a failed card that also predicts cancellation)
  violates this.
- **Competing risks.** Voluntary and involuntary churn are modelled as one cause. They are
  different decisions and deserve separate hazards.
- **An efficient influence function for `adjusted`**, which is why that estimator has no
  confidence sequence. Shipping an unvalidated EIF would be worse than shipping this sentence.
- **Multi-arm tests.** Two arms at a time.
- **Observational identification.** Assignment is assumed randomized.

## References

- Waudby-Smith, Arbour, Sinha, Kennedy & Ramdas, *Time-uniform central limit theory and
  asymptotic confidence sequences* — the confidence sequence.
- Moore & van der Laan (2009), *Covariate adjustment in randomized trials with binary outcomes* —
  why arm-specific models with a saturated time baseline stay consistent under misspecification.
- Andersen, Borgan, Gill & Keiding, *Statistical Models Based on Counting Processes* — the
  influence function for the product-limit estimator.

## License

Apache-2.0
