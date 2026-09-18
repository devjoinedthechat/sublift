# Slicing the base

## The problem

The offer didn't move the headline number. So someone slices, and finds it worked beautifully
for annual subscribers on iOS in their second year.

Slice a null experiment eight ways and something always looks significant. That is arithmetic.
Measured on sublift's simulator with **no effect at all anywhere**, 8 segments, 250 replications:

| how the analysis is done | reported a segment that "differs" |
|---|---|
| per-comparison tests, no gate — i.e. what people do | **26.4%** |
| heterogeneity gate alone | 6.0% |
| `segment_scan` (gate + simultaneous intervals) | **5.2%** |

Nominal rate: 5%.

## Using it

```python
scan = sl.segment_scan(
    panel, by=["plan", "tenure_bucket", "engagement_bucket"], horizon=12,
)
print(scan)
print(scan.credible_segments())
```

By default each column is scanned separately — the effect for every plan *and* every tenure
bucket — because that is how the question gets asked. All of them count as one multiplicity
family. `cross=True` slices the cross-product instead.

## The three questions, in order

**1. Is there any real variation?** Cochran's Q per dimension, Holm-corrected across dimensions
because asking five dimensions is five chances to find something. If nothing rejects, the
segments are one effect seen through noise and no individual result should be believed however
large it looks. This comes first in the output because it comes first in the reasoning.

**2. Does it work here?** Per-segment effects with intervals simultaneous across every segment
examined. Segments from different columns overlap — a subscriber is both `plan=monthly` and
`tenure=new` — so the contrasts are correlated and the correction is calibrated against that
correlation rather than assuming independence.

**3. Does it work *differently* here?** Usually the claim actually being made. "It works better
for annual" is a statement about the gap between the segment effect and the pooled effect, and
that gap has its own, wider uncertainty because both terms are estimated. `interaction` and
`interaction_ci` carry it. Reporting the segment effect and letting a reader eyeball it against
the headline is how a segment entirely consistent with the average becomes a discovery.

`credible_segments()` returns segments that pass **all three**: real heterogeneity on that
dimension, and a difference from the average that survives the correction.

## The subtle trap: a uniform treatment is not a uniform effect

Heterogeneity in *retained periods* is expected even when the treatment does exactly the same
thing to everybody.

A constant odds ratio on churn buys **more absolute periods in a segment that was churning
faster to begin with**, because the map from hazard to retained periods is not linear. On
sublift's simulator, a perfectly uniform odds ratio of 0.80 produces absolute heterogeneity that
Cochran's Q detects reliably at large samples — and it is real, not a bug.

Real, and almost always misread. "The offer works better for disengaged subscribers" is a claim
about mechanism; what the data shows is that disengaged subscribers had more room to improve.

So heterogeneity is tested on **two scales**:

| scale | what it is | what it means |
|---|---|---|
| retained periods | what the business banks | varies whenever baselines vary |
| churn odds ratio | what the treatment does | varies only if the treatment behaves differently |

```
  Is there real variation between segments?
    dimension                 periods   odds ratio
    plan                           no           no
    tenure_bucket                  no           no
    engagement_bucket             YES           no
```

That pattern — variation in periods, none in the odds ratio — sets `scan.scale_artefact` and the
summary says so in words:

> Retained periods vary between segments; the churn odds ratio does not. The treatment is doing
> the same thing to everyone, and the segments differ because they were churning at different
> rates to begin with.

**The targeting implication still holds.** You really do save more periods in those segments, and
targeting them is a defensible decision. What does not hold is the explanation.

When both scales light up, the treatment genuinely acts differently and the segment story is
real:

```
    engagement_bucket             YES          YES
```

## Limitations

- Two arms. For several arms *and* several segments, the families compound in a way sublift does
  not currently track — run the segment scan within one comparison at a time and treat the
  result as exploratory.
- Nonparametric within each segment. No covariate adjustment inside a slice.
- This controls multiplicity across **segments**. It does not control it across metrics or
  horizons; fix those in advance.
- A scan is still a scan. Finding a credible segment after the fact is a hypothesis worth
  testing in the next experiment, not a result to ship a price change on.
