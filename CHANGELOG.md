# Changelog

All notable changes to sublift are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Before 1.0, minor versions may change estimator behaviour where the change makes an estimate
more correct. Such changes are always listed under **Changed** with the reasoning, never
slipped into a patch release.

## [Unreleased]

### Performance
- **Runs on a real subscriber base.** A million subscribers over twelve billing periods: every
  estimator under two seconds, and `review` — the checks plus both metrics plus the
  competing-risks split — in 1.9s and 160 MB.

  Three changes got it there. The influence function now forms **no subscriber-by-period matrix
  at all**: both of its terms collapse to lookups into length-`horizon` arrays, since a
  subscriber contributes their event term in one period and their at-risk term in a prefix of
  periods. 440 MB to 33 MB at a million subscribers, six times faster, results identical to
  1e-13. The segment odds-ratio fit is **aggregated** — its design is entirely categorical, so
  however many million person-periods a segment holds there are only `2 × horizon` distinct rows
  (`segment_scan`: 1.78s/392 MB to 0.55s/119 MB). And the covariate-adjusted estimator and the
  competing-risks decomposition now work in **blocks**, so peak memory is flat in the size of the
  base; verified identical to twelve decimals at chunk sizes 7, 100,000 and unbounded.

- `fit_logistic` accepts `sample_weight`, which is what makes aggregating a categorical design
  possible. Exact against the expanded fit to 4.6e-15.

- `tests/test_scale.py` asserts the memory budgets, because reintroducing an `(n, horizon)`
  temporary in a hot path is easy and no correctness test would catch it.

### Added
- **`review`: one entry point that runs the checks in the order that matters.** The library had
  grown to a dozen functions where knowing which to call when was most of the skill, and that
  ordering lived only in the documentation — while the failure mode it guards against is
  precisely calling `incremental_ltv` first, getting a tight interval, and shipping a decision on
  an experiment whose randomization was broken.

  `review` runs the diagnostics before the estimate, grades what it finds, and leads with the
  grade. A **blocker** means the number is not measuring what it says, and the verdict reads *do
  not act on this*; the estimate is still computed, because hiding it only invites someone to
  compute it a worse way. A **warning** is a caveat. A **note** is a finding in its own right —
  most usefully, retention up and lifetime value down.

  It also picks the right estimand: a panel with spells gets occupancy rather than
  time-to-first-cancellation, and `monitoring=True` leads with the anytime-valid interval instead
  of the p-value. No new statistics; the existing estimators in the order an experienced analyst
  would run them, with the verdict written down instead of assumed.
- **`SubscriberPanel.from_spells` and `occupancy_lift`: subscriptions that come back.** People
  cancel and resubscribe, or pause over the summer. Time-to-first-cancellation records a
  subscriber who paid for periods 1–3 and 6–12 as churning at period 3.

  The direction of that error is the opposite of what it looks like. Ignoring returns
  **overstates** the treatment, because the subscribers written off as lost are
  disproportionately in the control arm. Measured, holding everything else fixed: at a 10%
  win-back hazard the first-spell estimate is 21% high, at 20% it is 44% high — and it reports
  the *same number* at every rate, because it cannot see returns at all.

  `occupancy_lift` targets expected billing periods *paid for* within the horizon, which does
  not care whether they arrived in one run or three, and tracks the truth at every win-back rate.
  On single-spell data it agrees with `retained_periods_lift` within a fraction of a standard
  error; the product-limit is the more efficient of the two there, which is what you trade away.

  A pause is a gap in the grid: retained, earning nothing. That matters because "pause instead
  of cancel" is itself a retention intervention whose mechanism a survival model cannot express.
- **`segment_scan`: effects by segment, without the slice becoming the finding.** "It didn't
  work overall, but it worked great for annual subscribers on iOS" is the most common way a
  retention experiment produces a false result. On a **null** experiment sliced eight ways,
  per-comparison tests report a segment that "differs" 26.4% of the time; `segment_scan` reports
  one 5.2% of the time, against a nominal 5%.

  It asks three questions in order: is there real variation at all (Cochran's Q per dimension,
  Holm-corrected across dimensions), does the effect differ from zero in this segment
  (simultaneous intervals, calibrated against the correlation between overlapping segments), and
  does it differ from the *pooled* effect — the claim a segment story actually makes, which
  carries its own wider uncertainty.

- **A scale-artefact diagnostic**, which came out of building the above. A *uniform* odds ratio
  produces genuinely different numbers of retained periods per segment, because segments
  churning faster have more to save and the map from hazard to retained periods is not linear.
  Cochran's Q detects that variation reliably at large samples — correctly, it is real — and it
  is almost always misread as "the offer works better for these people".

  So heterogeneity is now tested on two scales: retained periods, which is what the business
  banks, and the per-period churn odds ratio, which is what the treatment does. Variation in the
  first with none in the second sets `scan.scale_artefact` and is reported as such. The
  targeting implication survives; the mechanism story does not.
- **`multi_arm_lift`: several treatment arms against one control, with family-wise error
  control.** Testing three save offers against a holdout and reporting whichever looked best is
  one experiment with three chances to be wrong: with four arms and no real effect, declaring at
  least one a winner happens 13.0% of the time against a nominal 5%. With the correction, 4.3%.

  The default correction is single-step **max-t**, calibrated against the correlation between
  contrasts — they share a control arm, so they are correlated at exactly 0.5 with equal arm
  sizes, and Bonferroni pays for independence the family does not have. The measured saving is
  honest but modest: 1.1% on the critical value at two arms rising to 3.1% at eight, so roughly
  2–6% fewer subscribers for the same power. `"bonferroni"`, `"holm"` and `"none"` are also
  available.

  `best()` returns `None` when nothing survives the correction, because in a null experiment
  some arm always has the largest point estimate.
- `SubscriberPanel` now holds any number of arms, with `n_arms`, `treatment_labels` and
  `contrast("<arm>")` to pull out one comparison. The two-arm estimators refuse to run on a
  multi-arm panel and point at `multi_arm_lift`, so looping over arms is hard to do by accident.
- `check_randomization` generalizes to K arms, with `expected_shares=` for a designed imbalance.
- `simulate_multi_arm` for generating multi-arm experiments with per-arm known truth.
- **`check_censoring`.** The assumption that subscribers are censored because you cut the data,
  rather than because of anything they did, was previously untestable and documented as
  unchecked. It is now checked: if the panel records potential follow-up the question is settled
  by construction, and otherwise sublift fits the censoring hazard with and without your
  covariates and compares the fits. Reports which covariates drive censoring and what to do.
- **`censoring_covariates=`** on `estimator="adjusted"`, reweighting the augmentation term per
  subscriber instead of using one marginal censoring curve for everybody.
- `dropout_hazard` and `dropout_depends_on_engagement` in `simulate_experiment`, so informative
  censoring can be generated and any analysis checked for robustness to it.

  On informative censoring, the measured finding is documented rather than oversold: it is
  **detectable and partly correctable, not solvable**. Covariate adjustment more than halves the
  bias when the offending covariate is in the model (+0.0179 → +0.0077 on a true effect of
  +0.2587); inverse-probability-of-censoring weighting is a partial hedge for an incomplete
  outcome model and buys nothing when the outcome model is already right. A contrast is also far
  more forgiving than a level — on the same simulation the control arm's level is off by −0.15
  periods while the contrast is off by +0.02.
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
