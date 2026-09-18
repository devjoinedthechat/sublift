"""A retention experiment, end to end.

The scenario: subscribers who hit the cancel flow are randomized between the
existing flow and a save offer -- 50% off for three billing periods. The offer
works: people stay longer. The question is whether it was worth the margin.

The whole analysis is one call to `sublift.review`, which is how you would
actually use this. The rest of the script is that same analysis taken apart, to
show what the checks are and why they run in that order.

Run it:

    python examples/save_offer.py
"""

import numpy as np

import sublift as sl

PRICE = 12.0
DISCOUNT_PERIODS = 3
HORIZON = 12

# --- a simulated experiment standing in for your warehouse extract ----------
sim = sl.simulate_experiment(
    n=40_000,
    horizon=HORIZON,
    observation_window=18,
    baseline_hazard=0.065,
    renewal_spike={12: 1.8},  # the annual renewal cliff
    treatment_odds_ratio=0.86,  # the offer genuinely reduces churn
    effect_decay=0.10,  # and the effect fades once the discount ends
    price=PRICE,
    treatment_discount=0.50,
    discount_periods=DISCOUNT_PERIODS,
    involuntary_hazard=0.015,  # some subscribers leave because their card failed
    seed=2024,
)
panel = sim.panel
print(panel)
print(panel.describe().to_string(index=False))
print()

# --- the whole thing, in one call -------------------------------------------
schedule_for_review = {
    "control": np.full(HORIZON, PRICE),
    "treatment": np.concatenate(
        [np.full(DISCOUNT_PERIODS, PRICE * 0.5), np.full(HORIZON - DISCOUNT_PERIODS, PRICE)]
    ),
}
print(
    sl.review(
        panel,
        horizon=HORIZON,
        price=schedule_for_review,
        strata=["plan", "tenure_bucket"],
        monitoring=True,
    )
    .summary()
    .split("Incremental retained")[0]
)
print("=" * 78)
print("Everything below is that same review, taken apart.\n")

# --- 0. is the experiment even readable? -------------------------------------
# Runs automatically on every estimate too, but look at it first: if assignment
# is broken, nothing below is worth reading and no adjustment will rescue it.
print(sl.check_randomization(panel, expected_ratio=0.5))
print("\n" + "=" * 78 + "\n")

# --- 1. did it retain anybody? ----------------------------------------------
retention = sl.retained_periods_lift(
    panel, horizon=HORIZON, estimator="stratified", strata=["plan", "tenure_bucket"]
)
print(retention)
print("\n" + "=" * 78 + "\n")

# --- 2. was it worth it? ----------------------------------------------------
# The treatment arm's price schedule is not the control arm's. Stating it
# explicitly is better than letting the library infer it from observed revenue,
# because once the offer has changed who is still subscribed, the two arms'
# at-risk populations are no longer comparable.
schedule = {
    "control": np.full(HORIZON, PRICE),
    "treatment": np.concatenate(
        [np.full(DISCOUNT_PERIODS, PRICE * 0.5), np.full(HORIZON - DISCOUNT_PERIODS, PRICE)]
    ),
}
ltv = sl.incremental_ltv(
    panel,
    horizon=HORIZON,
    estimator="stratified",
    strata=["plan", "tenure_bucket"],
    price=schedule,
)
print(ltv)
print("\n" + "=" * 78 + "\n")

verdict = "worth the margin" if ltv.estimate > 0 else "NOT worth the margin"
print(f"Verdict: the offer bought {retention.estimate:+.3f} billing periods per subscriber,")
print(f"         and {ltv.estimate:+.2f} in lifetime value. It was {verdict}.")
print(f"         (true effect in this simulation: {sim.true_ltv_lift:+.2f})")
print()

# --- 3. which churn actually moved? -----------------------------------------
# The offer is designed to stop cancellations. It cannot fix a declined card --
# and by keeping subscribers alive longer it gives their card more chances to
# fail, which shows up as a negative involuntary term.
print(sl.churn_decomposition(panel, horizon=HORIZON))
print("\n" + "=" * 78 + "\n")

# --- 4. is it worth it for *everyone*? --------------------------------------
curve = sl.qini(
    panel,
    horizon=HORIZON,
    covariates=["engagement", "plan", "tenure_bucket"],
    metric="ltv",
    price=schedule,
    fractions=np.arange(0.2, 1.01, 0.2),
)
print(curve)
