# Choosing an estimator

All three target the same estimand. They differ in what they assume in exchange for variance.

## Short version

**Use `stratified`.** Give it two or three prognostic, pre-assignment strata — plan, tenure
bucket, an engagement quantile. It reduces variance, its influence function is exact at any
sample size, and it supports anytime-valid monitoring.

Reach for the others when:

| situation | use |
|---|---|
| You have strong *continuous* covariates and ≥ ~10k subscribers | `adjusted` |
| Fewer than a few thousand subscribers | `stratified` or `unadjusted` |
| You want the fewest possible assumptions, for a headline number | `unadjusted` |
| A reviewer wants no asymptotics at all | `adjusted` with `inference="bootstrap"` |

## The three

### `unadjusted`

Nonparametric product-limit per arm. Assumes randomization and independent censoring, and
nothing else. Its influence function is exact — no model, no asymptotic approximation beyond
the central limit theorem. This is the number to quote when someone asks "what does the data
say without any modelling".

### `stratified` — the default

The same estimator, computed within pre-assignment strata and recombined on stratum shares.

The variance reduction comes from homogeneity within strata. The influence function picks up
two terms — the within-stratum estimation error and the error in the stratum shares themselves.
Dropping the second is a common and quiet mistake that understates variance exactly when the
effect differs across strata, which is when people reach for stratification in the first place.

Strata must be **pre-assignment**. Stratifying on something the treatment could have changed is
not variance reduction, it is conditioning on a collider.

Choose 2–3 variables with a handful of levels each. Too many strata and cells get thin; sublift
drops cells with fewer than two subscribers per arm and tells you in `result.notes`.

### `adjusted`

A discrete-time logistic hazard model, fit **separately per arm with a saturated time
baseline**, standardized over the covariate distribution, then corrected by its efficient
influence function (a one-step / AIPW estimator).

- The arm-specific, time-saturated configuration is what keeps g-computation consistent under
  randomization even when the covariate model is misspecified (Moore & van der Laan, 2009).
- The one-step correction makes it asymptotically linear, which is what gives it a standard
  error without bootstrapping and a confidence sequence — and makes it doubly robust.

It gets the most out of continuous covariates, which stratification can only bucket.

**Its limitation is asymptotic.** In simulation its intervals cover at about 93% at 4,000
subscribers against a nominal 95%, reaching nominal by roughly 16,000. sublift warns below
5,000 rather than quietly running narrow. Below a few thousand, use `stratified`.

It also carries inverse-censoring weights, which become unstable if the horizon is long
relative to your follow-up. `sublift.censoring` warns when that happens.

## What about picking whichever gives the tightest interval?

Don't. Running all three and reporting the narrowest is a garden of forking paths, and the
reported interval will no longer have its nominal coverage. Pick the estimator from the design —
sample size, covariate type, whether you will monitor — before you look at the result.

Running all three and finding they *disagree substantially* is different, and informative: it
usually means a covariate model is fitting badly, or a stratum is thin. `result.notes` will
often say which.
