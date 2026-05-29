"""Dataset trust scanning: the engine for the fourth application of the harness.

This file contains *no* audit logic. It provides the interchangeable pieces the
controller in ``datasetscanner.py`` drives:

* :class:`Dataset`     - a loaded table (pure-stdlib CSV, or built in memory)
* :class:`CheckResult` - the raw outcome of one rule check
* :class:`Check`       - a named rule the scanner runs and re-tests
* the check builders   - ``values_in_range``, ``timestamps_monotonic``,
  ``no_missing_required``, ``duplicate_rate_below``, ``distribution_stationary``

Each check states a *hard, checkable invariant* over the data (the same shape as
``|f(x)| <= tol``). A failing check is the "unexpected" outcome the auditor then
re-tests by segment analysis to localize and classify the anomaly.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field


@dataclass
class Dataset:
    """A loaded table: column names + rows as ``{column: value}`` dicts."""

    columns: list[str]
    rows: list[dict[str, str]]
    name: str = ""

    @property
    def n(self) -> int:
        return len(self.rows)

    def slice(self, start: int, end: int) -> Dataset:
        return Dataset(self.columns, self.rows[start:end], f"{self.name}[{start}:{end}]")

    def numeric_column(self, field_name: str) -> list[tuple[int, float | None]]:
        """Parse ``field_name`` per row to float; ``None`` for missing/unparseable."""
        out: list[tuple[int, float | None]] = []
        for i, row in enumerate(self.rows):
            raw = (row.get(field_name) or "").strip()
            if raw == "":
                out.append((i, None))
                continue
            try:
                out.append((i, float(raw)))
            except ValueError:
                out.append((i, None))
        return out


def load_csv(path: str, *, name: str = "") -> Dataset:
    """Load a CSV file into a :class:`Dataset` (pure stdlib, header row required)."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        rows = [{k: (v if v is not None else "") for k, v in r.items()} for r in reader]
    return Dataset(columns, rows, name or path)


@dataclass
class CheckResult:
    """Raw outcome of one rule check (before the audit wraps it)."""

    ok: bool
    measured: float  # the measured quantity (a fraction or ratio; 0 == clean for most)
    threshold: float
    detail: str
    bad_rows: list[int] = field(default_factory=list)  # 0-based offending row indices


@dataclass
class Check:
    """A named rule: ``run(dataset) -> CheckResult``."""

    name: str
    run: Callable[[Dataset], CheckResult]


def _fraction(count: int, total: int) -> float:
    return count / total if total else 0.0


def values_in_range(field_name: str, lo: float, hi: float, *, max_fraction: float = 0.0) -> Check:
    """Numeric ``field_name`` must lie in ``[lo, hi]`` (missing values are ignored
    here — that is the missing-values check's job)."""

    def run(ds: Dataset) -> CheckResult:
        bad = [i for i, v in ds.numeric_column(field_name) if v is not None and not lo <= v <= hi]
        frac = _fraction(len(bad), ds.n)
        return CheckResult(
            ok=frac <= max_fraction,
            measured=frac,
            threshold=max_fraction,
            detail=f"{len(bad)}/{ds.n} rows have {field_name} outside [{lo:g}, {hi:g}]",
            bad_rows=bad,
        )

    return Check(f"range[{field_name}]", run)


def timestamps_monotonic(field_name: str, *, max_fraction: float = 0.0) -> Check:
    """Numeric ``field_name`` must be non-decreasing from row to row."""

    def run(ds: Dataset) -> CheckResult:
        values = ds.numeric_column(field_name)
        bad: list[int] = []
        prev: float | None = None
        for i, v in values:
            if v is not None and prev is not None and v < prev:
                bad.append(i)
            if v is not None:
                prev = v
        frac = _fraction(len(bad), ds.n)
        return CheckResult(
            ok=frac <= max_fraction,
            measured=frac,
            threshold=max_fraction,
            detail=f"{len(bad)}/{ds.n} rows where {field_name} decreases",
            bad_rows=bad,
        )

    return Check(f"monotonic[{field_name}]", run)


def no_missing_required(fields: Iterable[str], *, max_fraction: float = 0.01) -> Check:
    """At most ``max_fraction`` of rows may be missing any of the required fields."""
    required = list(fields)

    def run(ds: Dataset) -> CheckResult:
        bad = [
            i
            for i, row in enumerate(ds.rows)
            if any((row.get(f) or "").strip() == "" for f in required)
        ]
        frac = _fraction(len(bad), ds.n)
        return CheckResult(
            ok=frac <= max_fraction,
            measured=frac,
            threshold=max_fraction,
            detail=f"{len(bad)}/{ds.n} rows ({frac:.1%}) missing one of {required}",
            bad_rows=bad,
        )

    return Check("missing_required", run)


def duplicate_rate_below(*, max_fraction: float = 0.0) -> Check:
    """At most ``max_fraction`` of rows may be exact duplicates of an earlier row."""

    def run(ds: Dataset) -> CheckResult:
        seen: set[tuple[tuple[str, str], ...]] = set()
        bad: list[int] = []
        for i, row in enumerate(ds.rows):
            key = tuple(sorted(row.items()))
            if key in seen:
                bad.append(i)
            else:
                seen.add(key)
        frac = _fraction(len(bad), ds.n)
        return CheckResult(
            ok=frac <= max_fraction,
            measured=frac,
            threshold=max_fraction,
            detail=f"{len(bad)}/{ds.n} duplicate rows ({frac:.1%})",
            bad_rows=bad,
        )

    return Check("duplicate_rate", run)


def distribution_stationary(field_name: str, *, max_shift: float = 3.0) -> Check:
    """The mean of ``field_name`` must not shift between the first and second half
    by more than ``max_shift`` pooled standard deviations (a regime-shift guard)."""

    def _stats(xs: list[float]) -> tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / len(xs)
        return m, math.sqrt(var)

    def run(ds: Dataset) -> CheckResult:
        present = [(i, v) for i, v in ds.numeric_column(field_name) if v is not None]
        if len(present) < 4:
            return CheckResult(True, 0.0, max_shift, f"too few {field_name} values to assess shift")
        mid = len(present) // 2
        first = [v for _, v in present[:mid]]
        second = [v for _, v in present[mid:]]
        m1, s1 = _stats(first)
        m2, s2 = _stats(second)
        pooled = math.sqrt((s1 * s1 + s2 * s2) / 2.0)
        shift = abs(m2 - m1) / pooled if pooled > 1e-12 else 0.0
        ok = shift <= max_shift
        bad = [i for i, _ in present[mid:]] if not ok else []
        return CheckResult(
            ok=ok,
            measured=shift,
            threshold=max_shift,
            detail=(
                f"{field_name} mean shifts {shift:.2f}σ between halves (1st={m1:.3g}, 2nd={m2:.3g})"
            ),
            bad_rows=bad,
        )

    return Check(f"stationary[{field_name}]", run)
