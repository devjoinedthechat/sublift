# Before and after the experiment

## How long until this can answer the question

```python
print(sl.duration_to_detect(
    arrivals_per_period=8_000, horizon=12,
    baseline_hazard=0.06, treatment_odds_ratio=0.90,
))
```

Returns the enrollment duration needed under both fixed-sample and always-valid analysis.
This is the question asked *before* a test, and answering it is what stops the
underpowered 30-day-proxy habit: most retention experiments are called on a proxy because
nobody worked out that the real metric needed nine months.

It is computed by **simulating your actual enrollment schedule** rather than by a closed
form, because the variance of a censored survival contrast depends on the enrollment
pattern in a way no formula captures. Subscribers who joined last month contribute one
period of follow-up each to a twelve-period estimand, and their contribution to the
variance is not proportional to their headcount.

Pick `treatment_odds_ratio` as the smallest effect that would change a decision, not the
effect you hope for.

## Who to target

```python
curve = sl.qini(panel, horizon=12, covariates=["engagement", "plan", "tenure_bucket"])
curve.best_fraction()
```

Per-subscriber effects on the same restricted-mean scale as the headline number,
**cross-fitted** so nobody is ranked by a model that saw them. Each point of the curve is
a real censoring-aware estimate re-run inside the targeted subset — not a sum of
predicted scores. A Qini curve built from a model's own predictions measures the model's
confidence; this one measures the effect.

### Why the default learner is not a T-learner

Fitting a model per arm and differencing them makes predicted heterogeneity depend on
`(γ̂₁ − γ̂₀)′x`, the gap between two independently estimated coefficient vectors. When the
treatment does not genuinely modify a covariate's effect, that gap is **pure noise**, and
it does not average out — it fans the scores around the truth.

Measured against known individual effects:

| | T-learner | pooled + shrunk |
|---|---|---|
| constant odds ratio | 0.26 | **0.62** |
| strong effect modification | 0.94 | **0.97** |

The default fits one pooled model with ridge-penalised treatment×covariate interactions,
with the penalty chosen by held-out likelihood. Real effect modification survives it;
noise does not. It wins in both regimes, which is the argument for it being the default
rather than a bet on heterogeneity being small.

Pass `learner="t"` to see the naive version on your own data.

## Targeting is a hypothesis, not a result

A targeting fraction found after the fact is worth testing in the next experiment. It is
not a result to ship a price change on, and [segment_scan](segments.md) exists because
the same caution applies to any slice discovered after seeing the data.
