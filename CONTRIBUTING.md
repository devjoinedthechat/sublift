# Contributing

```bash
git clone https://github.com/devjoinedthechat/sublift && cd sublift
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install          # optional, runs ruff on commit

pytest                      # fast suite, ~10s
pytest -m slow              # statistical validation, ~2min
ruff check src tests examples
ruff format src tests examples
```

The [docs](docs/) explain the method; [docs/method.md](docs/method.md) has the estimand and
every influence function, which is the fastest way in if you are here to change the statistics.

## The one rule

**A new estimator ships with a coverage test.** `tests/test_validation.py` simulates
experiments whose true effect is known in closed form and asserts that the estimator is
unbiased and that its 95% intervals cover at 95%. Anything that reports an interval has to
earn it there. An estimator that is fast, elegant and miscalibrated is worse than no
estimator, because someone will ship a decision on it.

If you add a generating regime the simulator can't express — competing risks, informative
censoring, a different price schedule — extend `sublift.datasets` first, then test against it.

## Things that would help

- An efficient influence function for the `adjusted` estimator, which would give it a
  confidence sequence. This is the largest known gap; see the roadmap in the README.
- Informative censoring via inverse-probability-of-censoring weights.
- More than two arms, with multiple-comparison control across segments.
- A stratified or covariate-adjusted version of `churn_decomposition`; today it is
  nonparametric only.
- More billing realities in `sublift.datasets`: pauses, plan switches, proration, win-backs.

## Style

Docstrings say *why*, not *what*. The code is short enough to read; what it can't tell you is
which assumption a line is protecting. If you find yourself writing "sets the hazard", write
what breaks when it isn't set that way instead.

Error messages are part of the interface. Say what went wrong, why it matters, and what to do
instead — most people meet a library through its errors before they meet its docs.

## A note on disagreement

This project will have real methodological disagreements, and that is healthy. Two things make
them productive here:

1. **Simulate it.** `sublift.datasets` generates experiments with closed-form truth. Most
   arguments about whether an estimator is biased can be settled in twenty lines rather than in
   a thread.
2. **Nobody has all of statistics.** If you spot something wrong, you have done the project a
   favour regardless of how it is phrased; if you do not follow something, asking is normal and
   welcome. See the [Code of Conduct](CODE_OF_CONDUCT.md).
