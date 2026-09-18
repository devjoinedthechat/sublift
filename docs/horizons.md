# How the effect develops

Every readout here is at one stated horizon, which is the right discipline — but it
answers *how much* and never *when*. For retention interventions the second question often
decides whether to ship.

A save offer that buys three months and then fades has the same twelve-period number as
one that buys a little every month forever, and they are not the same product.

```python
print(sl.lift_by_horizon(panel, horizons=[3, 6, 9, 12]))
```

```
   horizon      effect         simultaneous band   per period
 *       3     +0.0636  [   +0.0527,    +0.0745]      +0.0212
 *       6     +0.1792  [   +0.1484,    +0.2099]      +0.0385
 *       9     +0.2844  [   +0.2320,    +0.3368]      +0.0351
 *      12     +0.3683  [   +0.2936,    +0.4431]      +0.0280

  The effect accumulated fastest up to horizon 6, and the final stretch added 73% of
  that rate. Most of what this intervention buys, it buys early.
```

## Reading it

The **per period** column is the one that carries the shape. The cumulative effect almost
always rises — more horizon, more accumulated difference — so a rising headline says
nothing. What matters is whether each successive stretch of horizon is still adding at
the rate the earlier ones did.

The shape is read **against the peak, not the start**. Survival differences take time to
open up, so the earliest stretch is nearly always the smallest even for an effect that is
about to fade. Comparing the tail to the first stretch would call almost everything
"growing".

Two readings the summary states explicitly:

- **The last stretch was the most productive.** The data ran out before the effect did.
  The headline is a lower bound, and longer follow-up would report more.
- **The rate peaked earlier.** Most of what the intervention buys, it buys early;
  extending the horizon adds less than the headline implies.

A third shape is worth knowing even though it needs no special handling: an effect that
*declines* in absolute terms, which happens when a treatment pulls churn forward rather
than preventing it. It looks like a win at six periods and a wash at twelve.

## The bands are simultaneous

Reading several horizons and reporting the best one is a multiple-comparisons problem.
The horizons are also nested — the twelve-period effect contains the six-period one — so
they correlate at around 0.85, which is exactly where treating them as independent tests
is most wasteful. The bands here are calibrated on the correlation the influence
functions actually have, roughly 10% tighter than Bonferroni.

## This is not licence to choose the horizon afterwards

Fix the horizon your decision needs beforehand. Read this to understand the shape around
it, and to notice when the data ran out before the effect did. Picking the horizon that
looks best is not something any correction repairs, because the correction is over the
horizons you *report*, not the ones you considered.


## Plotting the curves

```python
frame = sl.survival_curves(panel, horizon=12)
```

One row per arm and period, plus rows for the **difference** between arms — usually the curve
worth plotting, since its distance from zero is the finding. Columns: `survival`, `se`,
`ci_low`/`ci_high` (pointwise) and `band_low`/`band_high` (simultaneous across every period
shown).

Use the band. A pointwise interval is right for one period chosen in advance, and nobody looks at
a plotted curve that way — they look for where the lines separate, which is a search over every
period. Both are computed together because the only honest way to show the pointwise interval is
next to the one that says what looking costs.

Both are verified: over 250 replications the pointwise interval covers each period about 95% of
the time, and the simultaneous band contains the **whole** curve about 95% of the time.

It costs almost nothing. The influence function of `S(t)` depends on a subscriber only through
their last observed period and whether they churned — at most `2H` distinct values however many
million subscribers there are — so the entire covariance across periods is a sum over those
groups. Two million subscribers takes five seconds and 42 MB.
