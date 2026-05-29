"""The Self-Auditing AI applied to datasets: a trust scanner.

The same loop as the other applications, but the "outcome" is now whether a
dataset satisfies an explicit rule:

* **expected** outcome   -> the check passes. Recorded as accepted, no re-test.
* **unexpected** outcome -> the check fails. Re-test by **segment analysis**:
  re-run the failing check on each row-segment to *localize* the anomaly and
  classify it — a contiguous burst (e.g. a regime shift or a faulty-sensor
  window) versus a systemic, dataset-wide problem.

Unlike the solver/validator, a scan never raises: it always returns a
:class:`ScanReport` whose ``trusted`` flag and :class:`AuditLog` explain exactly
what can and cannot be relied on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .audit import Attempt, AuditLog, ExpectationCheck, ReTest
from .datasets import Check, CheckResult, Dataset, load_csv


@dataclass
class ScanReport:
    status: str  # "trusted" | "review" | "untrusted"
    log: AuditLog
    failed_checks: list[str] = field(default_factory=list)  # fail-severity violations
    warnings: list[str] = field(default_factory=list)  # warn/info violations

    @property
    def trusted(self) -> bool:
        return self.status == "trusted"


def _expectation(check: Check, result: CheckResult) -> ExpectationCheck:
    return ExpectationCheck(
        name=check.name,
        measured=result.measured,
        threshold=result.threshold,
        satisfied=result.ok,
        detail=result.detail,
    )


def _row_span(rows: list[int]) -> str:
    return f"{rows[0]}–{rows[-1]}" if rows else "—"


def _segment_retest(check: Check, ds: Dataset, n_segments: int = 5) -> ReTest:
    """Re-run the failing check on each contiguous row-segment to localize it.

    Where the violations concentrate is the discriminator: one segment ->
    localized burst (regime shift / faulty window); every segment -> systemic;
    none individually -> only emerges at full-dataset scale (a boundary or
    aggregate effect).
    """
    n = ds.n
    seg = max(1, -(-n // n_segments))  # ceil division
    segments: list[tuple[int, int, CheckResult]] = []
    for start in range(0, n, seg):
        end = min(start + seg, n)
        segments.append((start, end, check.run(ds.slice(start, end))))
    failing = [(start, end) for start, end, r in segments if not r.ok]
    total = len(segments)

    if not failing:
        concl = (
            "the violation does not appear in any single segment — it only emerges "
            "at full-dataset scale (a boundary or aggregate effect), not a local burst"
        )
    elif len(failing) == 1:
        start, end = failing[0]
        concl = (
            f"violation reproduces in exactly one segment (rows {start}–{end - 1}): "
            f"a localized burst — likely a regime shift or a faulty-sensor window"
        )
    elif len(failing) == total:
        concl = "violation reproduces in every segment: a systemic, dataset-wide problem"
    else:
        spans = ", ".join(f"{s}–{e - 1}" for s, e in failing)
        concl = f"violation reproduces in {len(failing)}/{total} segments (rows {spans})"

    chk = ExpectationCheck(
        name="segment_failures",
        measured=float(len(failing)),
        threshold=0.0,
        satisfied=not failing,
        detail=f"{len(failing)}/{total} row-segments fail the check",
    )
    return ReTest(
        "segment_analysis",
        "re-run the failing check on each row-segment to localize the anomaly",
        True,  # a hard-rule violation reproduces by definition; this locates it
        [chk],
        concl,
    )


class SelfAuditingDatasetScanner:
    """Runs each rule check, re-tests every failure by segment analysis, audits all."""

    def __init__(self, checks: list[Check]) -> None:
        self.checks = checks

    def scan(self, source: str | Dataset) -> ScanReport:
        ds = source if isinstance(source, Dataset) else load_csv(source)
        log = AuditLog(
            problem=ds.name or "dataset",
            tolerance=0.0,
            description="every rule check must pass for the dataset to be trusted",
        )
        failures: list[str] = []  # severity "fail"
        warnings: list[str] = []  # severity "warn"/"info"
        for idx, check in enumerate(self.checks, start=1):
            result = check.run(ds)
            expectation = _expectation(check, result)

            if result.ok:
                log.attempts.append(
                    Attempt(
                        idx,
                        check.name,
                        {},
                        "clean",
                        None,
                        0,
                        [expectation],
                        "expected",
                        [],
                        "accept",
                        "rule satisfied",
                    )
                )
                continue

            is_failure = check.severity == "fail"
            (failures if is_failure else warnings).append(check.name)
            decision = "flag" if is_failure else "warn"
            retest = _segment_retest(check, ds)
            log.attempts.append(
                Attempt(
                    idx,
                    check.name,
                    {},
                    "violation",
                    None,
                    0,
                    [expectation],
                    "unexpected",
                    [retest],
                    decision,
                    f"[{check.severity.upper()}] rule violated in rows "
                    f"{_row_span(result.bad_rows)}; segment-analysed",
                )
            )

        if failures:
            status = "untrusted"
            conclusion = "DO NOT TRUST as-is — failed checks: " + ", ".join(failures)
            if warnings:
                conclusion += "; warnings: " + ", ".join(warnings)
        elif warnings:
            status = "review"
            conclusion = "NEEDS REVIEW — warnings (no hard failures): " + ", ".join(warnings)
        else:
            status = "trusted"
            conclusion = "all checks passed — dataset satisfies every stated rule"
        log.finalize(status, None, None)
        log.conclusion = conclusion
        return ScanReport(status, log, failures, warnings)
