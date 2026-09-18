# Voluntary vs involuntary churn

## Why it matters

A large share of subscription churn — routinely 20–40% in consumer media — is **involuntary**.
A card expires, a payment fails, dunning runs out of retries, the subscription lapses. The
subscriber never decided anything.

Voluntary churn is somebody clicking cancel.

Treating them as one event corrupts a retention programme in both directions:

- A **save offer acts on voluntary churn only**. Measured against all-cause churn, its effect is
  diluted by an involuntary baseline it cannot move. Real wins look marginal.
- A **card updater or a smarter dunning schedule** shows up as "retention improved", and the
  retention team takes credit for a payments fix. The wrong team gets funded.

## Using it

Give the panel a cause column; sublift needs at least two distinct causes among churned
subscribers.

```python
panel = sl.SubscriberPanel.from_spans(..., cause="churn_reason")
print(sl.churn_decomposition(panel, horizon=12))
```

```
Retention effect by cause of churn, over 12 billing periods
===========================================================
  total  +0.3216 periods per subscriber [+0.2297, +0.4134]

  involuntary  -0.0422  [-0.0908, +0.0063]     -13% of the effect
  voluntary    +0.3638  [+0.2720, +0.4556]     113% of the effect

  Causes sum to the total exactly; the split is an identity, not an attribution.
```

## Reading it

The **voluntary** term is the intervention doing its job.

The **involuntary** term is negative, and that is not an error. Keeping subscribers subscribed
for longer gives their card more billing cycles in which to fail. Some of the retention the
offer buys is handed straight back to your payment processor.

That is a genuine competing-risks trade-off. A single all-cause hazard cannot express it — the
two effects net out into one number and the mechanism disappears. It is also actionable in a way
the total is not: it says that pairing this offer with a card-updater would compound.

## Why the split is exact

Writing `L_j` for billing periods lost to cause `j` within the horizon:

```
RMST(H) = H - sum_j L_j       L_j = sum_{s<H} (H-s) · S(s-1) · h_j(s)
```

where `h_j(s)` is the cause-specific hazard. So the incremental retained periods decompose
additively, with **no residual**:

```
Delta_total = sum_j Delta_j       Delta_j = -(L_j^treatment - L_j^control)
```

This is an algebraic identity, not an attribution heuristic. `churn_decomposition` reports a
total identical to `retained_periods_lift` at the same horizon — the decomposition does not
change the number being decomposed, and the test suite asserts it to ten decimal places.

## What sublift deliberately does not report

A **cause-specific survival curve with the other cause censored out** — "what would retention be
if nobody ever had a failed payment?"

That quantity is not identified from this data without assuming the two causes are independent,
and for subscriptions they plainly are not: the subscriber halfway out the door is also the one
who does not bother updating their card. Software that reports it anyway is answering a question
the data cannot answer.

The cumulative-incidence decomposition above needs no independence assumption, which is why it
is the one sublift computes.

## Limitations

- Nonparametric only in v0.1 — no stratified or covariate-adjusted version yet.
- Two or more causes are supported, but the natural split is two. Splitting into many rare
  causes will give you wide intervals on each.
