# Method

The estimand, the estimators, and the influence functions behind every interval sublift prints.

## The estimand

Subscriptions do not decay continuously; they end at renewal. So the natural object is the
**discrete hazard**

```
h(t) = P(T = t | T >= t)
```

read as: given the subscriber paid for period `t`, the probability that `t` is the last period
they pay for. Survival is the product-limit `S(t) = prod_{s<=t} (1 - h(s))`, with `S(0) = 1`.

The quantity anyone actually wants is a **weighted restricted mean**:

```
V(H) = sum_{t=1}^{H} w_t · S(t-1)
```

because "still paying in period `t`" is exactly the event that earns `w_t`. With `w_t = 1` this
is restricted mean survival time — expected billing periods retained. With `w_t` a price it is
lifetime value to horizon `H`. One primitive, both readouts.

Everything sublift reports is a contrast of this functional between arms:

```
Delta = V_treatment(H) - V_control(H)
```

## Why influence functions

An estimator's influence function is the per-subject contribution to its sampling error:

```
theta_hat - theta  ~=  (1/n) sum_i IF_i        Var(theta_hat) = (1/n^2) sum_i IF_i^2
```

Deriving them by hand rather than bootstrapping buys three things:

1. Standard errors that do not depend on the Greenwood-plus-delta-method chain of
   approximations, and do not cost 200 model refits.
2. A contrast variance that is just a sum across independent arms.
3. **An i.i.d. per-subject sequence**, which is exactly what an anytime-valid confidence
   sequence needs. The bootstrap gives you the first two and not the third.

Point 3 is the reason this library is shaped the way it is.

## The product-limit influence function

For `S(t)`, standard:

```
IF_i(S(t)) = -S(t) · sum_{s<=t} [dN_i(s) - Y_i(s) h(s)] / [pi(s) (1 - h(s))]
```

with `Y_i(s) = 1{subscriber i paid for period s}`, `dN_i(s) = 1{s was their last paid period}`,
and `pi(s)` the at-risk share. Summing against the weights and exchanging the order of summation
collapses the double sum to a single pass:

```
IF_i(V) = -sum_s [dN_i(s) - Y_i(s) h(s)] / [pi(s)(1-h(s))] · G(s)
G(s)    = sum_{t>s}^{H} w_t S(t-1)
```

When the revenue weights are estimated from the data rather than supplied as a known schedule,
they carry their own sampling error and a second term appears. Ignoring it understates the
variance of any LTV readout where the treatment moves realized revenue — discounts, downgrades,
win-back pricing — which is most of the interesting ones.

## Stratification

Writing `Delta = sum_k pi_k Delta_k`, the influence function picks up two terms:

```
IF_i = (+/-) iota_i / p_{a|k_i}  +  (Delta_{k_i} - Delta)
```

the within-stratum estimation error rescaled by the stratum's arm shares, and the error in the
stratum shares themselves. Dropping the second term is a common and quiet mistake; it
understates the variance whenever the effect genuinely differs across strata, which is exactly
when someone reaches for stratification.

## Covariate adjustment: the one-step estimator

Plain g-computation — fit `h_a(t|X)`, standardize over the covariate distribution — is
consistent under randomization with arm-specific models and a saturated time baseline
(Moore & van der Laan, 2009), but has no tractable influence function.

Adding the augmentation term of the efficient influence function gives the **one-step (AIPW)**
estimator:

```
V_a = (1/n) sum_i S_a(t|X_i) · (1 - Q_i(t)),   summed against the weights
```

where `Q_i` accumulates inverse-censoring-weighted residuals from subscriber `i`'s own observed
renewal decisions:

```
Q_i(t) = sum_{s<=t}  1{A_i=a} [dN_i(s) - Y_i(s) h_a(s|X_i)] / [pi_a · Gbar(s-1) · S_a(s|X_i)]
```

Dividing by the subscriber's own survival makes the running sum telescope into the ratio
`S_a(t|X)/S_a(s|X)`, which is always ≤ 1 — that is what keeps the augmentation bounded.

This estimator is asymptotically linear with influence function
`D_i = sum_t w_t [S_a(t-1|X_i)(1-Q_i(t-1)) - V_a(t-1)]`, mean zero by construction. It is also
doubly robust: the augmentation keeps it consistent where the hazard model is wrong, because
assignment is randomized and censoring is known rather than modelled.

Its variance is **first-order**, so its coverage is a large-sample property — about 93% at 4,000
subscribers, nominal by 16,000.

## Censoring

The augmentation needs `Gbar(s) = P(C >= s)`, the chance a subscriber is still *observable* in
period `s`, as distinct from still subscribed.

For a retention experiment this is usually not something to estimate. Censoring is
administrative: potential follow-up is fixed the day a subscriber enters, by the distance from
their assignment date to the data cut. That is known for **everyone**, including subscribers who
churned long before the cut. `from_spans` records it and `Gbar` is computed exactly.

When it is absent, sublift falls back on reverse Kaplan–Meier, treating censoring as the event.
One subtlety there: a panel records `n_periods = s, event = True` when a subscriber both churned
at `s` and would have run out of follow-up at `s` — churn wins, because it is what was observed.
Those subscribers are therefore not available to be *seen* censored at `s`, and the risk set for
the censoring event must exclude them. Dividing by the full risk set understates the censoring
hazard and biases `Gbar` upward by about three points by period eight on sublift's own
simulator, which is more than enough to matter inside an inverse weight.

## Competing risks

With cause-specific hazards `h_j(s)`, periods lost to cause `j` within the horizon are

```
L_j = sum_{s<H} (H-s) · S(s-1) · h_j(s)      and      RMST(H) = H - sum_j L_j
```

so the retention effect decomposes additively with no residual. `L_j` perturbs through both the
survival curve and the cause-specific hazard, giving a two-term influence function:

```
IF(L_j) = sum_s (H-s) [ h_j(s) IF(S(s-1)) + S(s-1) IF(h_j(s)) ]
IF(h_j(s)) = [dN_j(s) - Y(s) h_j(s)] / pi(s)
```

Validated two ways: it cancels against the all-cause influence function to machine precision,
and over 400 replications the standard errors land within a few percent of the actual sampling
spread.

## Anytime-valid inference

The asymptotic confidence sequence of Waudby-Smith et al., applied to the influence terms:

```
mean(psi) +/- sigma · sqrt( 2(n rho^2 + 1)/(n^2 rho^2) · log( sqrt(n rho^2 + 1)/alpha ) )
```

with `rho` tuned to a target sample size via `rho^2 = (-W_{-1}(-alpha^2) - 1)/n_target`, using
the lower branch of the Lambert W function. Valid at every `n` simultaneously.

## Validation

Every claim above is asserted by simulation against closed-form truth in
`tests/test_validation.py`, and printed by `examples/validation_report.py`. The simulator is
public API precisely because a reader should be able to check these numbers rather than take
them.

## References

- Waudby-Smith, Arbour, Sinha, Kennedy & Ramdas — *Time-uniform central limit theory and
  asymptotic confidence sequences*.
- Moore & van der Laan (2009) — *Covariate adjustment in randomized trials with binary
  outcomes*, and *Increasing power in randomized trials with right censored outcomes through
  covariate adjustment*.
- Andersen, Borgan, Gill & Keiding — *Statistical Models Based on Counting Processes*.
- Bang & Robins (2005) — *Doubly robust estimation in missing data and causal inference models*.
- Fine & Gray (1999) — *A proportional hazards model for the subdistribution of a competing
  risk*.
