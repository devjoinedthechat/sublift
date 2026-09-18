# Getting a tighter interval

Four ways, in rough order of how much they usually buy. They are not additive —
all four use overlapping information — so pick the one that fits your data rather
than stacking them.

## 1. Pre-period behaviour: `cuped`

Usually the largest single win, and the one people reach for last.

```python
base = sl.retained_periods_lift(panel, horizon=12, estimator="unadjusted")
print(sl.cuped(base, panel, ["pre_period_tenure", "pre_period_engagement"]))
```

Someone who was on their way out before the experiment started is still on their way
out afterwards. Subtracting that predictable part leaves less noise to see through.
Pre-period tenure alone removes about **11%** of the variance in simulation.

The usual CUPED derivation is for a difference in means and this estimand is not one.
It applies anyway through the influence function: any estimator that exposes one is, to
first order, the mean of it, and a covariate's imbalance between arms is another such
mean whose expectation is zero under randomisation. Subtracting a multiple of something
with expectation zero changes nothing in expectation and a great deal in variance.

That generality is the point — it works on occupancy, competing risks or a segment
exactly as it does on survival, because it never touches the outcome model. Those are
precisely the estimators where outcome modelling is awkward.

**The covariate must be pre-assignment.** With a post-assignment column the imbalance no
longer has expectation zero, and CUPED does not merely bias the estimate, it *moves* it.

## 2. Stratification: `estimator="stratified"`

The default. Two or three prognostic, pre-assignment strata — plan, tenure bucket, an
engagement quantile.

Its influence function is exact at any sample size, and it supports anytime-valid
monitoring. On real subscriber data a handful of good strata recovers most of what full
covariate adjustment would.

Strata must be pre-assignment. Stratifying on something the treatment could have changed
is not variance reduction; it is conditioning on a collider.

## 3. Covariate adjustment: `estimator="adjusted"`

A one-step (AIPW) estimator: a discrete-time hazard model per arm, standardised over the
covariate distribution, then corrected by its efficient influence function.

Gets the most out of **continuous** covariates, which stratification can only bucket. The
augmentation makes it doubly robust, so a poor model costs variance and not correctness.

Its variance is first-order, so its coverage is a large-sample promise: about 93% at
4,000 subscribers against a nominal 95%, nominal by around 16,000. It warns below 5,000.

## 4. A flexible model: `learner=`

```python
from sklearn.ensemble import HistGradientBoostingClassifier

sl.retained_periods_lift(
    panel, horizon=12, estimator="adjusted",
    covariates=[...], learner=HistGradientBoostingClassifier(), n_folds=5,
)
```

Any classifier with `fit` and `predict_proba`. It is **cross-fitted automatically**, and
that is not a convenience: a flexible model fitted on the same rows the estimate is read
from carries a bias of the same order as the effect, and the augmentation does not remove
it. Cross-fitting is the difference between double machine learning and using machine
learning.

Tested against a deliberately memorising learner — one that overfits badly in-sample —
which cross-fitted lands on the parametric answer.

No learner is selected for you, and scikit-learn is not a dependency.

## What not to do

Run all four and report the tightest interval. That is a garden of forking paths and the
reported interval stops having its nominal coverage. Choose from the design — sample
size, covariate types, whether you have pre-period data — before looking at the result.
