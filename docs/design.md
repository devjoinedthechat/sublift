# When the design is not what the analysis assumes

Two ways an experiment's design departs from "each subscriber was independently
randomised and then treated". Both are common, both are silent, and both change the
answer rather than merely blurring it.

## The randomised unit is coarser than the analysed one

A household shares a card and a decision. An account carries several subscriptions. A
business plan has seats that move together. And the randomisation is usually done at the
level that *is* independent — the account — while the analysis is per subscription,
because that is what churns.

```python
panel = sl.SubscriberPanel.from_spans(..., cluster="account_id")
```

Nothing in the data announces this. The estimate stays correct; only the interval is
wrong, and wrong in the direction that matters, because the effective sample size is the
number of accounts rather than the number of rows. On a simulated base of
three-subscription households the reported interval comes out **10% too narrow** with no
warning.

Passing `cluster=` turns the correction on everywhere at once: contrasts, stratification,
competing risks, occupancy, segments, arms, the percentage-lift interval, and the
confidence sequences, which then count clusters rather than subscriptions. The correction
is the standard one — sum influence values within a cluster and treat those sums as the
independent units — and it recovers the truth: **0.90× of the real sampling spread before,
1.01× after**, measured against resampling whole households.

Clustering on a unique id reproduces the independent case exactly, so there is no harm in
passing it when you are not sure.

## Assignment is not exposure

Retention interventions are usually *triggered*. A save offer fires when someone opens the
cancel flow; a win-back email goes only to subscribers who have already lapsed. Everyone
is randomised, but most never meet the thing being tested, and the ones who do are not a
random subset — they are the subscribers who were leaving.

```python
itt = sl.retained_periods_lift(panel, horizon=12, strata=["plan"])
print(sl.complier_effect(itt, panel, exposed="saw_offer"))
```

```
  intention to treat  +0.3155 periods [+0.2760, +0.3549]
  exposed             24.7% of the treatment arm (7,383 subscribers)

  complier effect     +1.2752 periods [+1.1175, +1.4330]
```

**Intention to treat** is what every estimator here reports by default: the effect of
being *assigned*, averaged over everybody including the subscribers the offer never
reached. It is unbiased by construction, and it is the number a launch decision wants,
because shipping the programme means shipping the trigger too.

**The complier effect** is the effect among subscribers who actually saw the offer. It is
what a design decision wants — does this offer work? — and quoting it as though it were
the first overstates the programme by exactly the reciprocal of the exposure rate. Here
that is a factor of four.

The exposure column has to be measured *after* assignment, which is the one place this
library relaxes its own rule about post-assignment columns. That is why it is named in
the call rather than inferred from the panel.

### What it rests on

An **exclusion restriction**: assignment changes nothing for a subscriber who never
triggers. Plausible for a triggered offer; implausible whenever the holdout is visible to
anyone who acts on it — a CRM flag that agents can see, a support queue that routes
differently. If assignment itself changes behaviour, the ratio is not identified and only
the intention-to-treat number survives.

`complier_effect` refuses when assignment moves fewer than 2% of subscribers into
exposure, because dividing by a number that small turns a modest interval into a
meaningless one.
