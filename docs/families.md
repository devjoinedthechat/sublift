# Families the library cannot see

`multi_arm_lift` corrects across arms. `segment_scan` corrects across segments. Neither
knows what *else* you looked at, and the family is whatever you actually examined — not
whatever one function happened to compute.

Three arms scanned across four segments is **twelve** comparisons, not three plus four.
LTV, retained periods and involuntary churn off one panel is three. A horizon you chose
after seeing four of them is four.

```python
sl.correct_family({
    "retained periods": sl.retained_periods_lift(panel, horizon=12, estimator="unadjusted"),
    "LTV":              sl.incremental_ltv(panel, horizon=12, price=12.0, estimator="unadjusted"),
    "involuntary":      ...,
})
```

It takes anything with `.estimate` and `.influence` — every estimator here except
`adjusted` with `inference="bootstrap"`.

## Why max-t matters most here

Every comparison in a family is computed from the same subscribers, so they are
correlated, and for metrics they are *strongly* correlated. With a flat price, LTV is
retained periods multiplied by a constant — correlation **1.00**, literally the same
statistic — and Bonferroni charges for two independent looks at one comparison.

max-t reads the correlation off the influence functions and calibrates against it, giving
intervals about **16% narrower** in that case. For arms sharing a control the correlation
is 0.5 and the saving is 1–3%. For genuinely independent comparisons it converges to
Bonferroni, which is correct.

`effective_multiplicity` exposes the underlying quantity: how many independent
comparisons the family behaves like. Four comparisons at correlation 0 gives 3.95; at
correlation 1 it gives 1.01, because looking twice at the same number is one look.

## Arms crossed with segments

```python
family = {}
for arm in panel.treatment_labels:
    pair = panel.contrast(arm)
    for plan in ("monthly", "annual"):
        mask = (pair.covariates["plan"] == plan).to_numpy()
        family[f"{arm} / {plan}"] = sl.retained_periods_lift(
            pair.subset(mask), horizon=12, estimator="unadjusted"
        )
print(sl.correct_family(family))
```

Every member must cover the same subscribers, because a family correction assumes one
experiment. Comparisons on disjoint or differently-filtered panels are not jointly
calibrated, and `correct_family` refuses them rather than pretending.

## What it cannot do

Know what you looked at and did not report. If you ran the scan, glanced at a segment and
then quietly dropped it, the family is larger than anything in this repository can see.
That part is yours.
