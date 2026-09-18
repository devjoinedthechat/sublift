# Testing several arms at once

## The problem

You test three save offers against one holdout, analyse each as its own experiment at 95%, and
report whichever looked best.

That is not three experiments. It is one experiment with three chances to be wrong. Measured on
sublift's simulator with **four arms, none of which works**, 250 replications:

| correction | declared at least one winner |
|---|---|
| none | **13.0%** |
| max-t | 4.3% |
| Holm | 3.7% |
| Bonferroni | 3.7% |

Nominal rate: 5%. This is the same failure as [peeking](monitoring.md), from a different
direction — there the extra chances come from looking repeatedly, here from looking widely. A
confidence sequence does not help with it, and a multiple-comparison correction does not help
with peeking. If you are doing both, you need both.

## Using it

```python
import sublift as sl

panel = sl.SubscriberPanel.from_spans(..., arm="variant", control="holdout")
print(sl.multi_arm_lift(panel, horizon=12, estimator="stratified", strata=["plan"]))
```

```
3 arms vs holdout, over 8 billing periods
=========================================
  holdout (n=9,993): 5.6794 periods

 *offer_a  n=  9,987  +0.2349  [+0.1388, +0.3311]   p=0.0000
 *offer_b  n=  9,994  +0.1401  [+0.0436, +0.2365]   p=0.0020
  offer_c  n=  9,992  +0.0180  [-0.0791, +0.1151]   p=0.9486

  95% simultaneous intervals, max-t correction (critical value 2.351 vs Bonferroni 2.394)
  Best arm that survives the correction: offer_a
```

The two-arm estimators refuse to run on a multi-arm panel and say why, so this is hard to do by
accident. To analyse one comparison deliberately:

```python
pair = panel.contrast("offer_a")          # control vs one arm, two-arm panel
sl.incremental_ltv(pair, horizon=12)
```

## Why max-t rather than Bonferroni

Every contrast shares the same control arm, so the contrasts are **positively correlated** — if
the control happens to look bad, every treatment arm looks good together. With equal arm sizes
that correlation is exactly 0.5.

Bonferroni assumes the worst about dependence and pays for independence the family does not
have. You can see this in the table above: uncorrected lands at 13%, not the 18.5% that four
independent tests would give, precisely because the tests are not independent.

max-t estimates the correlation from the influence functions and calibrates the critical value
against it. Measured saving over Bonferroni:

| treatment arms | correlation | max-t | Bonferroni | narrower by |
|---|---|---|---|---|
| 2 | 0.50 | 2.216 | 2.241 | 1.1% |
| 3 | 0.50 | 2.352 | 2.394 | 1.7% |
| 5 | 0.50 | 2.514 | 2.576 | 2.4% |
| 8 | 0.50 | 2.649 | 2.734 | 3.1% |

Required sample size scales with the square of the critical value, so that is roughly 2–6% fewer
subscribers for the same power. **Honest framing: worth having and free, but not the difference
between a conclusive test and an inconclusive one.** The reason to prefer max-t is that it is
the correct calibration, not that it rescues an underpowered experiment.

`correction="bonferroni"` and `"holm"` are there for when a reviewer wants the familiar thing.
Holm is more powerful than Bonferroni for *testing* but does not produce simultaneous intervals,
so its intervals fall back to Bonferroni; its adjusted p-values are the Holm ones.

## `best()` returns `None` on purpose

```python
winner = result.best()
if winner is None:
    print("nothing beat the holdout")
```

In a null experiment some arm always has the largest point estimate. Returning it would be the
exact mistake this module exists to prevent, so `best()` returns the best arm that *survives the
correction*, or nothing.

## Monitoring a multi-arm test

```python
result.confidence_sequences(n_target=80_000)
```

Each arm gets `alpha / k`, so the guarantee holds simultaneously over arms **and** over every
interim look. The max-t calibration does not carry over to confidence sequences, so this is
Bonferroni — conservative but correct, rather than tighter and wrong.

## Limitations

- `estimator="unadjusted"` and `"stratified"` only. The adjusted estimator's influence function
  is computed per comparison; use `panel.contrast()` with it and correct the p-values yourself.
- Every contrast is against the control. All-pairs comparisons (Tukey-style) are not supported.
- This controls the family-wise error rate across **arms**. It does not control it across
  metrics, segments, or horizons — see the end of [monitoring](monitoring.md).
