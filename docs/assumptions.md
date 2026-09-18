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

This holds for the usual case: a retention experiment analysed at a fixed date, where
follow-up length is set by enrollment date. It **fails** if subscribers leave your dataset for
reasons related to their state — an account deletion flow that correlates with dissatisfaction,
a migration that dropped a cohort, a failed card that also predicts cancellation and removes
the row entirely rather than marking it churned.

**Checked:** no. This is the assumption most likely to be silently violated, and sublift cannot
test it. Inverse-probability-of-censoring weighting for informative censoring is on the roadmap.

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

## 6. Two arms

`unadjusted` and `stratified` assume nothing about the number of arms beyond there being two;
`adjusted` fits a model per arm. Multi-arm tests with proper multiple-comparison control are
not supported. Filtering to one pair at a time and comparing several pairs inflates your error
rate in a way sublift does not track.

**Checked:** yes, in that a third arm is rejected rather than silently collapsed.

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
