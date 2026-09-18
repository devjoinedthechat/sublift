# Contributing

```bash
git clone https://github.com/devjoinedthechat/sublift && cd sublift
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest              # fast suite
pytest -m slow      # statistical validation, ~90s
ruff check src tests
```

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
- Competing risks: voluntary and involuntary churn as separate causes. A failed card and a
  cancellation are different decisions and a save offer acts on only one of them.
- Informative censoring via inverse-probability-of-censoring weights.
- More than two arms.

## Style

Docstrings say *why*, not *what*. The code is short enough to read; what it can't tell you is
which assumption a line is protecting. If you find yourself writing "sets the hazard", write
what breaks when it isn't set that way instead.
