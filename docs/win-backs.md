# Subscriptions that come back

## The problem

Consumer subscriptions are not one continuous span. People cancel and resubscribe three months
later, pause over the summer, churn from monthly and return on annual. In a lot of consumer
media a meaningful share of cancellations are followed by a return.

Time-to-first-cancellation has no way to express that. A subscriber who paid for periods 1–3 and
6–12 is recorded as churning at period 3.

## The error runs the wrong way

Ignoring returns does not understate your treatment. It **overstates** it.

The subscribers written off as permanently lost are disproportionately in the *control* arm —
that is what the treatment was reducing. A control subscriber who cancelled in March and
resubscribed in May was never really lost, and counting them as lost inflates the gap.

Measured on sublift's simulator, holding everything else fixed and varying only how often
subscribers return:

| win-back hazard | true effect | time-to-first-cancellation | |
|---|---|---|---|
| 0% | +0.2585 | +0.2609 | 1% high |
| 5% | +0.2352 | +0.2609 | **11% high** |
| 10% | +0.2149 | +0.2609 | **21% high** |
| 20% | +0.1814 | +0.2609 | **44% high** |

Look at the middle column falling and the right column standing still. The first-spell estimate
is not merely biased — it reports the *same number* however many subscribers come back, because
it structurally cannot see them.

## The estimand that survives it

```
A(H) = sum_{t=1..H}  E[ w_t · 1{paying in period t} ]
```

Expected billing periods **paid for** within the horizon, or expected revenue when `w_t` is a
price. This is the quantity the finance team already uses, and it does not care whether the
periods came in one run or three.

For a subscription that never returns it is the same quantity restricted mean survival time
estimates, and on single-spell data the two land within a fraction of a standard error of each
other. The product-limit is somewhat more efficient there, because it lets a subscriber censored
in period nine inform the hazard in period three. That efficiency is what you trade away.

## Using it

```python
panel = sl.SubscriberPanel.from_spells(
    df,                              # one row per subscriber-SPELL
    subject="user_id",
    arm="variant",
    assigned_at="experiment_entered_at",
    spell_start="subscription_started_at",
    spell_end="subscription_ended_at",   # date of the last payment; null = still open
    observed_through="2026-09-01",
    billing_interval="month",
    price="mrr",
    covariates=["plan", "tenure_bucket"],
)

sl.occupancy_lift(panel, horizon=12, strata=["plan"])
```

Spells may be listed in any order and may overlap — a plan change often opens a new row before
the old one closes — and overlaps collapse to "paying" rather than being counted twice.

`n_periods` and `event` are still derived from the first spell, so the survival estimators keep
working on the same panel and you can compare the two framings directly. That comparison is
worth running: the gap between them is your win-back exposure.

## Pauses

A pause is a period where the subscriber is retained but pays nothing. `from_spells` represents
it as a gap: they are not active, so the period earns no revenue, and they resume afterwards.

This matters because "pause instead of cancel" **is a retention intervention**, and one whose
whole mechanism is invisible to a survival model. On the occupancy estimand it prices itself
correctly: the retained periods after the pause count, the paused periods do not, and whether it
was worth it falls out of the arithmetic rather than needing an argument.

## What it needs

`occupancy_lift` requires `potential_followup` — how many periods each subscriber *could* have
been observed for. Without it, "not paying in period nine" and "nobody has observed period nine"
are the same row, and the estimator would read every censored subscriber as lapsed.
`from_spells` records it from the assignment date and the data cut, so this is automatic; the
estimator refuses rather than guessing if it is missing.

## Why they lapsed, over a horizon where they come back

```python
panel = sl.SubscriberPanel.from_spells(..., spell_cause="churn_reason")
print(sl.occupancy_decomposition(panel, horizon=12))
```

`churn_decomposition` attributes a subscriber to one cause, because it only ever sees them end
once. Over a horizon in which people leave and return, one subscriber can lapse for different
reasons at different times — cancelled in the spring, card failed in the autumn — and the
periods belong to whichever ending they were living under at the time.

The split is exact for the same reason as the survival one: in every period a subscriber is
either paying or not, and if not, exactly one ending is the most recent. So
`horizon = A + Σ_j L_j` and the contrast decomposes with no residual.

## Covariate adjustment

`occupancy_lift(..., covariates=[...])` fits an outcome model per period and augments it with the
observed residuals. Under randomization the augmentation's expectation is the estimation error
of the model, so a poor model widens the interval and does not move the estimate.

Unlike the covariate-adjusted *survival* estimator, this one carries no large-sample caveat:
each period is an ordinary mean rather than a product-limit, so the influence function is exact
in finite samples.

## Limitations

- Two arms.
- No separate model for *why* a subscriber returned; a win-back is just a resumed spell.
- A lapse is attributed to the most recent ending, which is the right answer when endings are
  distinct events and a simplification when a subscriber's state is genuinely ambiguous.
