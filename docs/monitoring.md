# Monitoring a running test

## The problem

A 95% confidence interval is a promise about a procedure: *if you compute this interval once,
it contains the truth 95% of the time*. The "once" is load-bearing.

Growth teams do not look once. They look every morning. And a running estimate is a random
walk, so given enough looks it will eventually wander outside any fixed boundary, whether or not
there is a real effect.

Measured on sublift's own simulator, with a true effect of **exactly zero**, 16 interim looks,
250 replications:

| | called a winner at some point |
|---|---|
| fixed-sample 95% interval, checked at every look | **26.4%** |
| anytime-valid 95% confidence sequence | **0.8%** |

Nominal rate: 5%. Sixteen looks turn a 5% test into a 26% one. Every "we called it early and it
didn't replicate" retrospective is this effect, and no amount of care about the *last* analysis
fixes it, because the damage is done by the looks in between.

## The fix

```python
result = sl.incremental_ltv(panel, horizon=12, strata=["plan"])
print(result.confidence_sequence(n_target=50_000))
```

```
anytime-valid 95% CS at n=40,000: [-13.8343, -10.5347] (excludes 0); 1.55x the fixed-sample width
```

A confidence sequence is valid *simultaneously at every sample size*. Across unlimited looks,
the probability that it ever excludes the truth is at most α. You can stop whenever you like,
including because of what you just saw — which is the thing a fixed-sample interval forbids and
everyone does anyway.

sublift uses the asymptotic confidence sequence of Waudby-Smith, Arbour, Sinha, Kennedy and
Ramdas, applied to the estimator's influence function.

## What it costs

About 1.5–1.7× the fixed-sample width at the tuning point. That is the honest price of being
allowed to look, and it is much cheaper than the alternative, which is a 26% false-positive
rate you were not accounting for.

## `n_target`

The sample size the boundary is tuned to be narrowest at. Normally the size you expect the
experiment to reach.

**Set it before you start and leave it alone.** The guarantee holds at every `n` regardless of
tuning — tuning only decides *where* the sequence is tightest. But re-tuning after seeing the
data is exactly the optional stopping the sequence exists to protect against, and it quietly
gives back the guarantee you paid for.

## Which estimators support it

`unadjusted`, `stratified` and `adjusted` — all three, since `adjusted` gained an efficient
influence function. `adjusted` with `inference="bootstrap"` does not: the bootstrap gives
standard errors, but a confidence sequence needs an i.i.d. per-subject sequence, which only the
influence function provides.

This is the reason sublift derives influence functions by hand rather than bootstrapping
everything. It is more work and it is the whole point.

## Reporting

If you monitored the test, report the confidence sequence. `result.p_value` is a fixed-sample
p-value and it is only valid if the analysis you are reading is the *only* one you ran.
`result.summary()` prints both and says which is which, so the honest number is the one that is
hard to avoid reading.

## Monitoring a family

Peeking and multiplicity are different problems, and a test doing both needs both. A confidence
sequence makes *when* you stop a free variable; it does nothing about *how many things* you
looked at.

```python
result.confidence_sequences(n_target=80_000)      # multi-arm, split across arms
```

The default splits `alpha` by Bonferroni. That is a deliberate retreat from the fixed-sample
case, where max-t is clearly better. Measured under a global null with four arms, watched at
eight interim looks:

| | ever declared a winner | interval width |
|---|---|---|
| no family correction at all | 2.5% | — |
| Bonferroni | 1.0% | baseline |
| effective-multiplicity ("max-t") | 1.0% | ~1% narrower |

Nominal rate: 5%.

Two things worth reading off that table. The correction buys **almost nothing** — `alpha` enters
the sequence boundary inside a logarithm, so dividing it by 3.5 rather than 4 barely moves the
interval. And all three rates sit well under 5%, because a confidence sequence is already
conservative relative to its nominal level; the boundary is tightest at `n_target` and loose
everywhere else.

`calibration="max-t"` is available and validated. It is not the default because defaulting to an
approximation for a one percent interval is not a trade worth making, and Bonferroni's union
bound is provable.

## What this does not fix

Anytime-valid inference protects you against **looking repeatedly at one metric**. It does not
protect against:

- Looking at twenty metrics and reporting the one that moved. That needs multiple-comparison
  control, which sublift does not yet provide.
- Slicing into segments after the fact and reporting the best one. Same problem.
- Changing the horizon after seeing which horizon looks best. Fix the horizon in advance.

A confidence sequence makes *when* you stop a free variable. It does not make *what you look at*
a free variable.
