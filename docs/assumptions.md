# Assumptions

Read this before you ship a decision on a sublift number. It is short on purpose.

## 1. Assignment is randomized

sublift is an experiment analysis tool. There is no observational identification — no
propensity scores, no instrument, no difference-in-differences. If subscribers chose their arm,
or a targeting rule chose it for them, none of these estimates mean what they say.

**Checked:** partially. `check_randomization` runs on every estimate and tests for sample ratio
mismatch at `p < 0.001`, plus baseline covariate balance. It catches broken assignment; it
cannot catch assignment that was never random but is balanced by luck.

## 2. Censoring is administrative

Subscribers are censored because you cut the data on a date, not because of anything correlated
with their propensity to churn.

This holds for the usual case: a retention experiment analysed at a fixed date, where follow-up
length is set by enrollment date. It **fails** when subscribers are *lost* rather than merely
not-yet-observed — an account deletion flow that correlates with dissatisfaction, a migration
that dropped a cohort, a plan the extract stopped covering.

**Checked:** yes.

```python
print(sl.check_censoring(panel, covariates=[...]))
```

If the panel records `potential_followup` (which `from_spans` does automatically), censoring is
administrative by construction and the question does not arise. Otherwise sublift fits the
censoring hazard with and without your covariates and compares the fits. If censoring depends on
who the subscriber is, it says so and tells you what to do.

### What to do when it fires

**Use `estimator="adjusted"` with the covariates that drive censoring.** Censoring that depends
only on `X` is independent *given* `X`, so a hazard model containing `X` removes most of the
bias. Measured on sublift's simulator, with dropout driven hard by engagement and a true effect
of +0.2587:

| | bias | coverage |
|---|---|---|
| `unadjusted` | +0.0179 | 94.0% |
| `adjusted`, with engagement | **+0.0077** | 94.7% |
| `adjusted`, engagement omitted | +0.0176 | 93.3% |
| …plus `censoring_covariates=` | +0.0136 | 92.7% |

Two honest readings of that table:

- Adjustment does most of the work — it more than halves the bias, and only when the offending
  covariate is actually in the model.
- `censoring_covariates=` (inverse-probability-of-censoring weighting) is a **partial hedge, not
  a cure**. It helps when the outcome model is incomplete, and it buys nothing when the outcome
  model is already right. It is available for that reason and no stronger one.

### The thing to take away

Informative censoring is **detectable and partly correctable, not solvable**. No estimator here
eliminates it. If `check_censoring` fires, the number is directional: treat a 3% effect as
"probably positive", not as "+3.0%". The value of the check is that you know which kind of
number you are holding.

A contrast is also far more forgiving than a level, because the bias largely cancels between
arms. On the same simulation the control arm's *level* is off by −0.15 periods while the
*contrast* is off by +0.02. Quote the contrast; be much more careful quoting a survival curve.

## 3. Covariates and strata are pre-assignment

Anything measured after randomization can be a consequence of the treatment. Adjusting for it
reintroduces exactly the bias randomization removed, and can flip a sign.

**Checked:** yes. The panel constructors reject covariates that vary within a subscriber. This
catches the common case; it cannot catch a column that is constant per subscriber but was
*computed* after assignment. Only you know that.

## 4. The horizon is within your follow-up

Past the point where an arm has nobody at risk, its survival curve is being carried forward on
no data, and the contrast is an extrapolation.

**Checked:** yes, and refused by default. `allow_extrapolation=True` overrides it, and then the
number is a projection and should be labelled one.

## 5. Churn has one cause — unless you say otherwise

By default voluntary and involuntary churn are pooled. See [competing risks](competing-risks.md).

**Checked:** no, but opt-in and cheap: pass `cause=` and get the split.

## 6. One comparison, or a family you declared

The ordinary estimators compare two arms. Running them once per arm and reporting the best
inflates the error rate by roughly the number of arms — 13% against a nominal 5% with four arms,
measured. Use [`multi_arm_lift`](multi-arm.md), which controls the family-wise rate.

**Checked:** yes. The two-arm estimators refuse to run on a multi-arm panel and point at
`multi_arm_lift` or `panel.contrast()`.

Slicing into segments is the same problem again, and [`segment_scan`](segments.md) handles it:
per-comparison tests on a null experiment sliced eight ways report a finding 26.4% of the time.

What is **not** checked is multiplicity across *metrics and horizons*, or arms and segments
together. Testing one arm on twelve metrics has the same problem and sublift does not track it.
Fix the primary metric and horizon before you look.

## 7. For `adjusted` specifically

- Its influence function is **first-order**: coverage is a large-sample promise. ~93% at 4,000
  subscribers, nominal by ~16,000. **Checked:** warns below 5,000.
- It carries inverse-censoring weights that destabilize when the horizon is long relative to
  follow-up. **Checked:** warns when few subscribers could have been observed to the horizon.

## What "checked" means

Where sublift says it checks something, there is a test in `tests/` asserting the check fires.
Where it says it does not, that is a genuine gap and the burden is on you. The point of listing
them is that an unstated assumption is indistinguishable from a bug when the number turns out
to be wrong.
