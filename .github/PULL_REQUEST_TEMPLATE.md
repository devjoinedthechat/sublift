## What this changes

<!-- One or two sentences. -->

## Why

<!-- What was wrong, or what could not be measured before. -->

## Checklist

- [ ] `pytest` passes
- [ ] `ruff check src tests examples` passes
- [ ] If this touches an estimator or an interval: `pytest -m slow` passes, and there is a
      coverage test in `tests/test_validation.py` asserting it is unbiased and covers at its
      nominal rate against known truth
- [ ] If this changes an existing estimate: the PR description says so and why
- [ ] Docstrings say *why*, not *what*

<!--
On that third box: it is the one rule this project has. An estimator that is fast, elegant and
miscalibrated is worse than no estimator, because someone will ship a decision on it. If you are
unsure how to write the test, open the PR anyway and say so — that is a normal thing to need help
with, and it is more useful to review together than to guess at alone.
-->
