# Changelog

All notable changes to sublift are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Before 1.0, minor versions may change estimator behaviour where the change makes an estimate
more correct. Such changes are always listed under **Changed** with the reasoning, never
slipped into a patch release.

## [Unreleased]

### Added
- **Efficient influence function for `estimator="adjusted"`.** The covariate-adjusted estimator
  is now a one-step (AIPW) estimator rather than plain g-computation, which makes it
  asymptotically linear with a known influence function. Consequences: standard errors without
  200 bootstrap refits, **anytime-valid confidence sequences for the estimator that reduces
  variance the most**, and double robustness — the augmentation keeps it consistent where the
  hazard model is wrong, because assignment is randomized and censoring is known rather than
  modelled. This closes the largest gap listed in the v0.1 roadmap.
- **Exact censoring distribution.** `SubscriberPanel.potential_followup` records how many
  periods each subscriber *could* have been observed for. Under administrative censoring that
  is known at baseline for everybody, including subscribers who churned long before the data
  cut, so `Gbar(s)` is computed exactly instead of estimated by reverse Kaplan–Meier.
  `from_spans` populates it automatically; `from_periods` and `from_subjects` accept it.
- `sublift.censoring` module, with a warning when the horizon is long enough relative to
  follow-up that inverse-censoring weights become unreliable.
- `inference="bootstrap"` on `estimator="adjusted"`, as an escape hatch from the asymptotics.
- Small-sample warning on `estimator="adjusted"`: its influence function is first-order, and
  intervals cover at about 93% at 4,000 subscribers against a nominal 95%, reaching nominal by
  roughly 16,000.

### Changed
- `estimator="adjusted"` now reports `inference="influence"` by default and its point estimate
  carries the one-step correction, so it differs slightly from v0.1. The new estimate is the
  more robust one.

## [0.1.0]

### Added
- Censored incremental LTV over an explicit horizon of billing periods, as a weighted
  restricted mean `sum_t w_t S(t-1)`. Arm-specific price schedules, so a discount-funded save
  offer costs what it actually costs.
- Three estimators over one estimand: `unadjusted`, `stratified`, `adjusted`.
- Hand-derived influence functions, and anytime-valid confidence sequences built on them.
- `churn_decomposition`: voluntary versus involuntary churn, split through the exact identity
  `RMST(H) = H - sum_j L_j`, so causes sum to the headline effect with no residual.
- `check_randomization`: sample ratio mismatch at `p < 0.001` plus baseline covariate balance,
  run automatically on every estimate.
- `SubscriberPanel.from_spans`: subscription dates in, billing periods out, with anniversaries
  clamped to the end of the month the way payment processors bill.
- `LiftResult.relative_ci`: delta-method interval for the percentage lift.
- `duration_to_detect`: enrollment duration needed, under both fixed-sample and always-valid
  analysis, by simulating the real enrollment schedule.
- `qini` / `uplift_scores`: cross-fitted targeting with a pooled, shrunk-interaction learner
  that beats a T-learner whether or not effect modification is real.
- `simulate_experiment`: ground-truth simulator, shipped as public API because the validation
  suite is built on it and you should be able to run it against your own numbers.

[Unreleased]: https://github.com/devjoinedthechat/sublift/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/devjoinedthechat/sublift/releases/tag/v0.1.0
