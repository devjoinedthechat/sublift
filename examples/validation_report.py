"""Measured evidence for the claims in the README.

Everything here is a simulation against a known true effect, so every number is
checkable rather than asserted. This is the same machinery as
``tests/test_validation.py``, printed as a report instead of as assertions.

    python examples/validation_report.py        # ~2 minutes
"""

import warnings

import numpy as np

import sublift as sl

REPS = 400
N = 4000
HORIZON = 8
ADJ_COVS = ["engagement", "plan", "tenure_bucket"]


def rule(title):
    print(f"\n{title}\n{'-' * len(title)}")


# ---------------------------------------------------------------- calibration
rule("1. Are the intervals real? (coverage of a nominal 95% interval)")

configs = [
    ("unadjusted", N, {"estimator": "unadjusted"}),
    ("stratified", N, {"estimator": "stratified", "strata": ["plan", "tenure_bucket"]}),
    # The one-step estimator's influence function is first-order, so its coverage is a
    # large-sample promise. Shown at both sizes, because the gap between them is the point.
    ("adjusted", N, {"estimator": "adjusted", "covariates": ADJ_COVS}),
    ("adjusted", 16_000, {"estimator": "adjusted", "covariates": ADJ_COVS}),
]
for name, n, kw in configs:
    reps = REPS if n <= N else 150
    covered, ests, ses = 0, [], []
    truth = None
    for r in range(reps):
        sim = sl.simulate_experiment(n=n, horizon=HORIZON, observation_window=12, seed=7000 + r)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sl.retained_periods_lift(sim.panel, horizon=HORIZON, **kw)
        covered += res.ci[0] <= sim.true_rmst_lift <= res.ci[1]
        ests.append(res.estimate)
        ses.append(res.se)
        truth = sim.true_rmst_lift
    ests = np.array(ests)
    print(
        f"  {name:<11s} n={n:<6,d} coverage {covered / reps:6.1%}   "
        f"bias {ests.mean() - truth:+.5f}   "
        f"se/sd {np.mean(ses) / ests.std(ddof=1):.3f}"
    )
print("  (`adjusted` is asymptotic: it warns below 5,000 rather than quietly running narrow)")

# ------------------------------------------------------------------ censoring
rule("2. What censoring does to the obvious alternative")
print("   'Naive' = difference in mean observed tenure, capped at the horizon.")
print(f"   {'follow-up':>10s} {'censored':>10s} {'sublift bias':>14s} {'naive bias':>12s}")
for window in (9, 12, 18):
    sub, naive, truth, cens = [], [], None, []
    for r in range(80):
        sim = sl.simulate_experiment(n=N, horizon=HORIZON, observation_window=window, seed=71_000 + r)
        p = sim.panel
        sub.append(sl.retained_periods_lift(p, horizon=HORIZON, estimator="unadjusted").estimate)
        capped = np.minimum(p.n_periods, HORIZON)
        naive.append(capped[p.arm == 1].mean() - capped[p.arm == 0].mean())
        cens.append((~p.event).mean())
        truth = sim.true_rmst_lift
    print(
        f"   {window:>10d} {np.mean(cens):>9.0%} {np.mean(sub) - truth:>+14.4f} "
        f"{np.mean(naive) - truth:>+12.4f}"
    )
print(f"   (true effect {truth:+.4f} periods)")

# -------------------------------------------------------------------- peeking
rule("3. What peeking does, and what the confidence sequence does about it")

reps, n_max = 250, 8000
looks = np.arange(500, n_max + 1, 500)
fixed_alarms = cs_alarms = 0
for r in range(reps):
    sim = sl.simulate_experiment(
        n=n_max,
        horizon=6,
        observation_window=10,
        treatment_odds_ratio=1.0,
        with_covariates=False,
        seed=91_000 + r,
    )
    panel = sim.panel
    fixed_hit = cs_hit = False
    for k in looks:
        mask = np.zeros(panel.n_subjects, dtype=bool)
        mask[:k] = True
        res = sl.retained_periods_lift(
            panel.subset(mask), horizon=6, estimator="unadjusted", allow_extrapolation=True
        )
        fixed_hit |= res.ci[0] > 0 or res.ci[1] < 0
        cs_hit |= res.confidence_sequence(n_target=n_max).excludes_zero
    fixed_alarms += fixed_hit
    cs_alarms += cs_hit

print(f"   True effect: exactly zero. {len(looks)} interim looks, {reps} replications.")
print(f"   fixed-sample 95% interval excluded 0 at some point:  {fixed_alarms / reps:6.1%}")
print(f"   anytime-valid 95% sequence excluded 0 at some point: {cs_alarms / reps:6.1%}")
print("   Nominal false-positive rate: 5%.")

# ----------------------------------------------------------- competing risks
rule("4. Competing risks: are the cause-specific intervals calibrated?")

reps_cr = 300
ests: dict[str, list] = {}
ses: dict[str, list] = {}
truth_cr = None
for r in range(reps_cr):
    sim = sl.simulate_experiment(
        n=8000,
        horizon=HORIZON,
        observation_window=13,
        seed=200_000 + r,
        involuntary_hazard=0.018,
        treatment_odds_ratio=0.80,
    )
    for c in sl.churn_decomposition(sim.panel, horizon=HORIZON).causes:
        ests.setdefault(c.label, []).append(c.estimate)
        ses.setdefault(c.label, []).append(c.se)
    truth_cr = sim.true_periods_saved

print(f"   {'cause':<14s} {'true':>9s} {'mean est':>9s} {'bias':>9s} {'mean se':>9s} {'actual sd':>10s}")
for label, values in ests.items():
    e = np.array(values)
    print(
        f"   {label:<14s} {truth_cr[label]:>+9.4f} {e.mean():>+9.4f} "
        f"{e.mean() - truth_cr[label]:>+9.4f} {np.mean(ses[label]):>9.5f} {e.std(ddof=1):>10.5f}"
    )
print("   (a standard error claims to be the spread of the estimates; these are both)")

# -------------------------------------------------------------------- uplift
rule("5. Uplift: why the default learner is not a T-learner")
print(f"   {'regime':<32s} {'T-learner':>10s} {'pooled+shrunk':>14s}")
covs = ["engagement", "plan", "tenure_bucket"]
for em, label in ((0.0, "constant odds ratio"), (1.5, "strong effect modification")):
    sim = sl.simulate_experiment(
        n=10_000, horizon=HORIZON, observation_window=12, effect_modification=em, seed=77
    )
    truth = sim.true_individual_rmst_lift
    rs = []
    for learner in ("t", "interaction"):
        s = sl.uplift_scores(sim.panel, horizon=HORIZON, covariates=covs, learner=learner)
        rs.append(float(np.corrcoef(s, truth)[0, 1]))
    print(f"   {label:<32s} {rs[0]:>10.2f} {rs[1]:>14.2f}")
print("   (correlation between predicted and true individual effect)")
print()
