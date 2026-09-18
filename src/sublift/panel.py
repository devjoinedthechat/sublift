"""The data contract: what a subscription retention experiment looks like to sublift."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["SubscriberPanel"]


class PanelError(ValueError):
    """The input data does not describe a well-formed retention experiment."""


@dataclass(frozen=True)
class SubscriberPanel:
    """One randomized retention experiment, in discrete billing periods.

    The canonical internal form is one record per subject:

    ``n_periods``
        How many billing periods the subject has been observed to pay for,
        counting the period of assignment as period 1. Always ``>= 1``.
    ``event``
        ``True`` if ``n_periods`` is the subject's *last* paid period (they
        churned), ``False`` if they were still active when the data was cut
        (administrative censoring).

    So a subject with ``n_periods=4, event=True`` paid for four periods and
    then cancelled; ``n_periods=4, event=False`` paid for four periods and is
    still subscribed. The distinction is the whole reason this library exists.

    ``cause``
        Optional. Why the subscription ended -- typically voluntary cancellation
        versus involuntary churn from a failed payment. These are different
        events with different remedies, and a save offer acts on one of them, so
        collapsing them understates the intervention and lets a dunning
        improvement masquerade as a retention win. See
        :func:`sublift.churn_decomposition`.

    Build one with :meth:`from_spans` (dates, the usual warehouse shape),
    :meth:`from_periods` (one row per subject-period) or :meth:`from_subjects`
    (one row per subject) rather than calling the constructor directly.
    """

    subject: np.ndarray
    arm: np.ndarray  # 0 = control, 1 = treatment
    n_periods: np.ndarray
    event: np.ndarray
    arm_labels: tuple[str, str] = ("control", "treatment")
    covariates: pd.DataFrame | None = None
    revenue: np.ndarray | None = field(default=None, repr=False)
    cause: np.ndarray | None = field(default=None, repr=False)
    cause_labels: tuple[str, ...] = ()

    # ------------------------------------------------------------------ build

    @classmethod
    def from_subjects(
        cls,
        df: pd.DataFrame,
        *,
        subject: str,
        arm: str,
        periods: str,
        event: str,
        control: object | None = None,
        covariates: Sequence[str] | None = None,
        revenue: str | None = None,
        cause: str | None = None,
    ) -> SubscriberPanel:
        """Build from one row per subject.

        Parameters
        ----------
        periods
            Column of billing periods paid for so far (``>= 1``).
        event
            Column that is truthy when the subject churned, falsy when they are
            still active (censored).
        control
            Which value of ``arm`` is the holdout. Defaults to the
            lexicographically smaller of the two.
        revenue
            Optional column of per-period revenue, assumed constant across the
            subject's periods. For a revenue schedule that varies by period
            (intro pricing, annual step-ups) use :meth:`from_periods`.
        """
        cols = [subject, arm, periods, event] + ([revenue] if revenue else [])
        cols += ([cause] if cause else []) + list(covariates or [])
        _require_columns(df, cols)

        if df[subject].duplicated().any():
            dupes = df[subject][df[subject].duplicated()].unique()[:5]
            raise PanelError(
                f"{subject!r} must be unique in from_subjects(); saw repeats such as "
                f"{list(dupes)}. If you have one row per subject-period, use from_periods()."
            )

        n_periods = _as_periods(df[periods], periods)
        ev = _as_event(df[event], event)
        arm_idx, labels = _as_arm(df[arm], arm, control)

        rev = None
        if revenue is not None:
            per_subject = pd.to_numeric(df[revenue], errors="coerce").to_numpy(dtype=float)
            if np.isnan(per_subject).any():
                raise PanelError(f"{revenue!r} contains non-numeric or missing values.")
            rev = _broadcast_revenue(per_subject, n_periods)

        codes, cause_labels = _as_cause(df[cause] if cause else None, cause, ev)
        return cls(
            subject=df[subject].to_numpy(),
            arm=arm_idx,
            n_periods=n_periods,
            event=ev,
            arm_labels=labels,
            covariates=df[list(covariates)].reset_index(drop=True) if covariates else None,
            revenue=rev,
            cause=codes,
            cause_labels=cause_labels,
        )

    @classmethod
    def from_periods(
        cls,
        df: pd.DataFrame,
        *,
        subject: str,
        period: str,
        churned: str,
        arm: str,
        control: object | None = None,
        covariates: Sequence[str] | None = None,
        revenue: str | None = None,
        cause: str | None = None,
    ) -> SubscriberPanel:
        """Build from one row per subject-period (the long / panel form).

        ``period`` counts from 1 and must be contiguous per subject. ``churned``
        may only be truthy on a subject's final row. Covariates are baseline:
        they are read from each subject's first period and must not vary after
        it, because a covariate measured post-assignment can be affected by the
        treatment and adjusting for it reintroduces exactly the bias the
        randomization removed.
        """
        cols = [subject, period, churned, arm] + ([revenue] if revenue else [])
        cols += ([cause] if cause else []) + list(covariates or [])
        _require_columns(df, cols)

        work = df.copy()
        work["_period"] = _as_periods(work[period], period)
        work["_churned"] = _as_event(work[churned], churned)

        if work.duplicated(subset=[subject, "_period"]).any():
            raise PanelError(f"Duplicate ({subject}, {period}) rows; each subject-period must appear once.")

        work = work.sort_values([subject, "_period"], kind="stable")
        grp = work.groupby(subject, sort=True)

        first = grp.head(1).reset_index(drop=True)
        last = grp.tail(1).reset_index(drop=True)
        n_periods = last["_period"].to_numpy(dtype=np.int64)

        counts = grp.size().to_numpy()
        if not np.array_equal(counts, n_periods):
            bad = first[subject].to_numpy()[counts != n_periods][:5]
            raise PanelError(
                f"{period!r} must run 1..n contiguously for every subject; gaps or a "
                f"non-1 start for subjects such as {list(bad)}. A paused subscription is "
                "still an observed period -- carry it with zero revenue rather than dropping the row."
            )

        churn_counts = grp["_churned"].sum().to_numpy()
        if (churn_counts > 1).any():
            raise PanelError(f"{churned!r} is truthy in more than one period for some subjects.")
        last_period = work[subject].map(dict(zip(first[subject], n_periods, strict=True)))
        early = work[work["_churned"] & (work["_period"] < last_period)]
        if len(early):
            raise PanelError(
                f"{churned!r} is truthy before a subject's final period. Churn ends the "
                "subscription, so it can only occur on the last observed row."
            )

        arm_idx, labels = _as_arm(first[arm], arm, control)

        cov = None
        if covariates:
            varying = [c for c in covariates if grp[c].nunique(dropna=False).gt(1).any()]
            if varying:
                raise PanelError(
                    f"Covariates {varying} vary within a subject. sublift adjusts for *baseline* "
                    "covariates only; a post-assignment covariate can be a consequence of the "
                    "treatment, and conditioning on it biases the effect."
                )
            cov = first[list(covariates)].reset_index(drop=True)

        rev = None
        if revenue is not None:
            wide = work.pivot(index=subject, columns="_period", values=revenue)
            wide = wide.reindex(columns=range(1, int(n_periods.max()) + 1))
            rev = wide.to_numpy(dtype=float)

        event = last["_churned"].to_numpy(dtype=bool)
        codes, cause_labels = _as_cause(last[cause] if cause else None, cause, event)

        return cls(
            subject=first[subject].to_numpy(),
            arm=arm_idx,
            n_periods=n_periods,
            event=event,
            arm_labels=labels,
            covariates=cov,
            revenue=rev,
            cause=codes,
            cause_labels=cause_labels,
        )

    @classmethod
    def from_spans(
        cls,
        df: pd.DataFrame,
        *,
        subject: str,
        arm: str,
        assigned_at: str,
        ended_at: str,
        observed_through: str | object,
        billing_interval: str | int = "month",
        control: object | None = None,
        covariates: Sequence[str] | None = None,
        price: str | float | None = None,
        cause: str | None = None,
    ) -> SubscriberPanel:
        """Build from subscription dates -- the shape warehouse data actually arrives in.

        This is the recommended constructor. Converting dates into billing
        periods by hand is where analyses quietly go wrong: off-by-one on the
        assignment period, month arithmetic that drifts on the 31st, and
        subscribers censored at the wrong date because the extract ran before
        their renewal. Doing it here means the mistakes are made once, in code
        with tests on it.

        Parameters
        ----------
        assigned_at
            When the subscriber entered the experiment. This is the start of
            period 1, not their original signup date -- a subscriber randomized
            three years into their tenure is in period 1 on that day. Put their
            prior tenure in ``covariates`` if you want to adjust for it.
        ended_at
            When the subscription ended. Null/NaT means still active.
        observed_through
            The data cut: a column, or one timestamp for the whole extract.
            Subscribers still active at this date are censored here, which is
            what makes the censoring administrative and the estimates valid.
        billing_interval
            ``"month"`` (default), ``"year"``, ``"week"``, ``"day"``, or an
            integer number of days. Month and year arithmetic uses calendar
            anniversaries, so a subscriber billed on the 31st is not credited
            with an extra period in February.
        price
            A column of per-period revenue, or one number for everybody.
        """
        cols = [subject, arm, assigned_at, ended_at] + list(covariates or [])
        cols += [observed_through] if isinstance(observed_through, str) and observed_through in df else []
        cols += [cause] if cause else []
        cols += [price] if isinstance(price, str) else []
        _require_columns(df, cols)

        start = _as_datetime(df[assigned_at], assigned_at)
        end = _as_datetime(df[ended_at], ended_at, allow_missing=True)
        if isinstance(observed_through, str) and observed_through in df:
            cut = _as_datetime(df[observed_through], observed_through)
        else:
            cut = pd.Series(pd.to_datetime(observed_through), index=df.index).repeat(1)
            cut = pd.Series([pd.to_datetime(observed_through)] * len(df), index=df.index)

        if (cut < start).any():
            raise PanelError(
                f"Some subscribers were assigned after {observed_through!r}. An experiment cannot "
                "be observed before it starts; check the data cut."
            )

        churned = end.notna() & (end <= cut)
        late = end.notna() & (end > cut)
        if late.any():
            warnings.warn(
                f"{int(late.sum())} subscribers have an end date after the data cut. They are "
                "censored at the cut, since anything past it is information the experiment did "
                "not have.",
                stacklevel=3,
            )
        last_seen = end.where(churned, cut)

        elapsed = _periods_between(start, last_seen, billing_interval)
        n_periods = (elapsed + 1).to_numpy(dtype=np.int64)
        ev = churned.to_numpy(dtype=bool)

        arm_idx, labels = _as_arm(df[arm], arm, control)
        codes, cause_labels = _as_cause(df[cause] if cause else None, cause, ev)

        rev = None
        if price is not None:
            per_subject = (
                pd.to_numeric(df[price], errors="coerce").to_numpy(dtype=float)
                if isinstance(price, str)
                else np.full(len(df), float(price))
            )
            if np.isnan(per_subject).any():
                raise PanelError(f"{price!r} contains non-numeric or missing values.")
            rev = _broadcast_revenue(per_subject, n_periods)

        return cls(
            subject=df[subject].to_numpy(),
            arm=arm_idx,
            n_periods=n_periods,
            event=ev,
            arm_labels=labels,
            covariates=df[list(covariates)].reset_index(drop=True) if covariates else None,
            revenue=rev,
            cause=codes,
            cause_labels=cause_labels,
        )

    # --------------------------------------------------------------- inspect

    def __post_init__(self) -> None:
        n = len(self.subject)
        for name in ("arm", "n_periods", "event"):
            if len(getattr(self, name)) != n:
                raise PanelError(f"{name!r} has length {len(getattr(self, name))}, expected {n}.")
        if n == 0:
            raise PanelError("Panel is empty.")
        if self.revenue is not None and self.revenue.shape[0] != n:
            raise PanelError("revenue matrix must have one row per subject.")
        if self.cause is not None:
            if len(self.cause) != n:
                raise PanelError("cause must have one entry per subject.")
            if (self.event & (self.cause < 0)).any():
                raise PanelError("Every churned subscriber needs a cause; some are unlabelled.")
        for a, label in enumerate(self.arm_labels):
            size = int((self.arm == a).sum())
            if size == 0:
                raise PanelError(f"Arm {label!r} has no subjects.")
            if size < 30:
                warnings.warn(
                    f"Arm {label!r} has only {size} subjects; interval coverage relies on "
                    "asymptotics and will be optimistic.",
                    stacklevel=3,
                )

    @property
    def n_subjects(self) -> int:
        return len(self.subject)

    @property
    def max_period(self) -> int:
        """Longest observed follow-up. Estimating past this needs extrapolation."""
        return int(self.n_periods.max())

    @property
    def followup(self) -> int:
        """Longest horizon every arm still has subjects at risk for.

        The usable horizon: past it, at least one arm's survival curve is being
        carried forward on no data, so a contrast there is not a measurement.
        """
        return int(min(self.n_periods[self.arm == a].max() for a in (0, 1)))

    def subset(self, mask: np.ndarray) -> SubscriberPanel:
        """A new panel over the subjects selected by a boolean mask.

        Strictly boolean. An integer array of positions would be silently
        reinterpreted as truthiness -- every nonzero index becoming ``True`` --
        which selects almost the whole panel and looks entirely plausible
        downstream. Use :meth:`take` for positions.
        """
        mask = np.asarray(mask)
        if mask.dtype != bool:
            raise PanelError(
                f"subset() takes a boolean mask, got dtype {mask.dtype}. If those are positions, "
                "use take() -- an integer array here would be read as truthiness and quietly "
                "select nearly everyone."
            )
        if mask.shape != (self.n_subjects,):
            raise PanelError(
                f"mask has shape {mask.shape}, expected ({self.n_subjects},)."
            )
        return self.take(np.flatnonzero(mask))

    def take(self, indices: np.ndarray) -> SubscriberPanel:
        """A new panel over the subjects at the given positions, repeats allowed.

        This is what bootstrap resampling needs and a boolean mask cannot express.
        """
        idx = np.asarray(indices)
        if idx.dtype == bool:
            raise PanelError("take() expects positions; pass a boolean mask to subset() instead.")
        idx = idx.astype(np.intp, copy=False)
        if idx.size and (idx.min() < 0 or idx.max() >= self.n_subjects):
            raise PanelError(f"indices out of range for a panel of {self.n_subjects} subjects.")
        return self._gather(idx)

    def _gather(self, idx: np.ndarray) -> SubscriberPanel:
        return SubscriberPanel(
            subject=self.subject[idx],
            arm=self.arm[idx],
            n_periods=self.n_periods[idx],
            event=self.event[idx],
            arm_labels=self.arm_labels,
            covariates=(
                self.covariates.iloc[idx].reset_index(drop=True)
                if self.covariates is not None
                else None
            ),
            revenue=self.revenue[idx] if self.revenue is not None else None,
            cause=self.cause[idx] if self.cause is not None else None,
            cause_labels=self.cause_labels,
        )

    def describe(self) -> pd.DataFrame:
        """Per-arm sizes, observed churn and follow-up -- read this before trusting anything else."""
        rows = []
        for a, label in enumerate(self.arm_labels):
            m = self.arm == a
            rows.append(
                {
                    "arm": label,
                    "subjects": int(m.sum()),
                    "churned": int(self.event[m].sum()),
                    "censored": int((~self.event[m]).sum()),
                    "censoring_rate": float((~self.event[m]).mean()),
                    "median_followup": float(np.median(self.n_periods[m])),
                    "max_followup": int(self.n_periods[m].max()),
                }
            )
            if self.cause is not None:
                for k, label in enumerate(self.cause_labels):
                    rows[-1][f"churn_{label}"] = int(((self.cause == k) & m).sum())
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        c, t = self.arm_labels
        return (
            f"SubscriberPanel({self.n_subjects} subjects, "
            f"{int((self.arm == 0).sum())} {c} / {int((self.arm == 1).sum())} {t}, "
            f"{self.event.mean():.0%} churned, follow-up {self.followup} periods)"
        )


# ---------------------------------------------------------------- validators


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise PanelError(f"Missing column(s) {missing}. Available: {list(df.columns)}")


def _as_periods(s: pd.Series, name: str) -> np.ndarray:
    vals = pd.to_numeric(s, errors="coerce")
    if vals.isna().any():
        raise PanelError(f"{name!r} contains non-numeric or missing values.")
    arr = vals.to_numpy()
    if not np.allclose(arr, np.round(arr)):
        raise PanelError(f"{name!r} must be whole billing periods, not fractions.")
    arr = np.round(arr).astype(np.int64)
    if (arr < 1).any():
        raise PanelError(f"{name!r} must be >= 1; period 1 is the period of assignment.")
    return arr


def _as_event(s: pd.Series, name: str) -> np.ndarray:
    if s.isna().any():
        raise PanelError(f"{name!r} contains missing values; churned must be known for every row.")
    if s.dtype == bool:
        return s.to_numpy(dtype=bool)
    vals = pd.to_numeric(s, errors="coerce")
    if vals.isna().any() or not set(np.unique(vals.to_numpy())) <= {0, 1}:
        raise PanelError(f"{name!r} must be boolean or 0/1.")
    return vals.to_numpy().astype(bool)


def _as_arm(s: pd.Series, name: str, control: object | None) -> tuple[np.ndarray, tuple[str, str]]:
    levels = pd.unique(s.dropna())
    if s.isna().any():
        raise PanelError(f"{name!r} contains missing values; every subject must have an assignment.")
    if len(levels) != 2:
        raise PanelError(
            f"{name!r} has {len(levels)} distinct values {list(levels[:5])}; sublift v0.1 "
            "compares exactly two arms. Filter to one pair at a time."
        )
    levels = sorted(levels, key=repr)
    if control is None:
        ctrl = levels[0]
    else:
        if control not in levels:
            raise PanelError(f"control={control!r} is not one of the observed arms {levels}.")
        ctrl = control
    treat = levels[1] if levels[0] == ctrl else levels[0]
    return (s.to_numpy() != ctrl).astype(np.int8), (str(ctrl), str(treat))


def _broadcast_revenue(per_subject: np.ndarray, n_periods: np.ndarray) -> np.ndarray:
    width = int(n_periods.max())
    grid = np.arange(1, width + 1)[None, :]
    out = np.where(grid <= n_periods[:, None], per_subject[:, None], np.nan)
    return out


_INTERVAL_DAYS = {"week": 7, "day": 1}


def _as_datetime(s: pd.Series, name: str, *, allow_missing: bool = False) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    bad = out.isna() & s.notna()
    if bad.any():
        raise PanelError(f"{name!r} has {int(bad.sum())} values that are not parseable dates.")
    if not allow_missing and out.isna().any():
        raise PanelError(f"{name!r} has missing values; it is required for every subscriber.")
    return out


def _periods_between(start: pd.Series, end: pd.Series, interval: str | int) -> pd.Series:
    """Complete billing intervals elapsed between two dates.

    Monthly and annual anniversaries are **clamped to the end of the month**,
    which is what payment processors actually do: a subscription started on
    31 January bills on 28 February, then on 31 March. Naive month arithmetic
    (comparing day-of-month) instead skips February entirely and then credits
    two periods at once in March. That is a silent off-by-one on roughly a tenth
    of any subscriber base -- everyone who signed up on the 29th, 30th or 31st --
    and it lands on the periods where the hazard is steepest.
    """
    if isinstance(interval, str) and interval in ("month", "year"):
        step = 1 if interval == "month" else 12
        naive = (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)
        naive = (naive // step).clip(lower=0)

        target = start.dt.to_period("M") + (naive * step)
        day = np.minimum(start.dt.day.to_numpy(), target.dt.days_in_month.to_numpy())
        anniversary = target.dt.to_timestamp() + pd.to_timedelta(day - 1, unit="D")
        elapsed = naive - (anniversary > end).astype(int)
    else:
        days = _INTERVAL_DAYS.get(interval, interval) if isinstance(interval, str) else interval
        if not isinstance(days, int) or days < 1:
            raise PanelError(
                f"billing_interval must be 'month', 'year', 'week', 'day' or a positive number "
                f"of days, got {interval!r}."
            )
        elapsed = (end - start).dt.days // days
    elapsed = elapsed.clip(lower=0)
    return elapsed.astype("int64")


def _as_cause(s: pd.Series | None, name: str | None, event: np.ndarray):
    """Encode the competing cause of churn; -1 marks subscribers who have not churned."""
    if s is None:
        return None, ()
    values = s.reset_index(drop=True)
    churned = pd.Series(event)
    missing = churned & values.isna()
    if missing.any():
        raise PanelError(
            f"{name!r} is missing for {int(missing.sum())} churned subscribers. Every churn needs "
            "a cause, or the decomposition silently attributes it to nothing."
        )
    labels = tuple(sorted({str(v) for v in values[churned].dropna().unique()}))
    if len(labels) < 2:
        raise PanelError(
            f"{name!r} has only {len(labels)} distinct cause(s) among churned subscribers; "
            "competing risks needs at least two. Drop the argument if churn has one cause."
        )
    lookup = {label: i for i, label in enumerate(labels)}
    codes = np.full(len(values), -1, dtype=np.int64)
    as_str = values.astype(str).to_numpy()
    for i in np.flatnonzero(event):
        codes[i] = lookup[as_str[i]]
    return codes, labels
