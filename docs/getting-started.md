# Getting started

## The question sublift answers

> We ran a retention intervention against a holdout. How much lifetime value did it create,
> per subscriber, over the next *N* billing periods?

Not "did retention go up" — that is a different and easier question, and it is the one that
lets a 50%-off save offer look like a win while it destroys margin.

## 1. Get your data into a panel

Real subscription data arrives as spans: when someone entered the experiment, when their
subscription ended, and when you pulled the extract.

```python
import sublift as sl

panel = sl.SubscriberPanel.from_spans(
    df,
    subject="user_id",
    arm="variant",
    assigned_at="experiment_entered_at",
    ended_at="subscription_ended_at",     # null / NaT = still active
    observed_through="2026-09-01",        # the data cut
    billing_interval="month",
    price="mrr",
    cause="churn_reason",                 # optional but recommended
    covariates=["plan", "tenure_bucket", "engagement_pre"],
)
print(panel)
print(panel.describe())
```

Three things worth getting right:

**`assigned_at` is when they entered the experiment**, not when they signed up. A subscriber
randomized three years into their tenure is in period 1 on that day. Put their prior tenure in
`covariates` if you want to adjust for it.

**`observed_through` is the data cut**, and everyone still active is censored there. This is
what makes the censoring *administrative*, which is the assumption the whole library rests on.

**Covariates must be measured before assignment.** Anything measured afterwards can be a
consequence of the treatment, and conditioning on it reintroduces exactly the bias
randomization removed. The constructors reject covariates that vary within a subscriber.

Already have period counts rather than dates? Use `from_periods` (one row per subscriber-period)
or `from_subjects` (one row per subscriber). Pass `potential_followup=` if you have it — see
[Method](method.md#censoring) for why it is worth carrying.

## 2. Check the experiment before you read it

```python
print(sl.check_randomization(panel, expected_ratio=0.5))
```

This also runs automatically on every estimate and warns. If it reports a sample ratio
mismatch, **stop**. Something upstream is filtering subscribers differently by arm, and no
statistical adjustment rescues that — the estimate is measuring your pipeline.

## 3. Ask both questions

```python
retention = sl.retained_periods_lift(panel, horizon=12, strata=["plan", "tenure_bucket"])
money     = sl.incremental_ltv(panel, horizon=12, strata=["plan", "tenure_bucket"])
```

Report both. When they disagree in sign, the intervention bought retention with margin, and
**that is the finding**.

### About the horizon

`horizon=12` means twelve billing periods from assignment. It is mandatory, because a
"lifetime" value with no horizon is an extrapolation wearing a measurement's clothes. sublift
refuses to estimate past the follow-up both arms actually have unless you pass
`allow_extrapolation=True`, and then you should label the number a projection.

Pick the horizon from the decision, not from the data: if you are deciding whether to fund the
offer for a year, twelve periods is the horizon, and if you do not have twelve periods of
follow-up yet, the honest answer is that the experiment cannot answer that question yet.

## 4. Price the intervention honestly

If the treatment arm got a discount, say so:

```python
schedule = {
    "control":   [12.0] * 12,
    "treatment": [6.0] * 3 + [12.0] * 9,   # 50% off for three periods
}
sl.incremental_ltv(panel, horizon=12, price=schedule, strata=["plan"])
```

Prefer an explicit schedule over letting sublift infer weights from observed revenue. The
observed mean conditions on *being at risk*, and once the treatment has changed who is still
subscribed, the two arms' at-risk populations are no longer comparable.

## 5. If you have been watching the dashboard

```python
print(result.confidence_sequence(n_target=50_000))
```

Read that instead of the p-value. See [Monitoring](monitoring.md).

## 6. Before you run the next one

```python
print(sl.duration_to_detect(arrivals_per_period=8_000, horizon=12,
                            baseline_hazard=0.06, treatment_odds_ratio=0.90))
```

This is the question that stops the underpowered-30-day-proxy habit.
