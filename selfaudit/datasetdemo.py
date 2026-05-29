"""Demo: ``python -m selfaudit.datasetdemo``.

Builds a synthetic sensor table (500 rows) with a *planted* fault — rows 420–479
read a stuck-high temperature, a localized regime shift — plus a sprinkle of
missing values. Scans it against explicit rules, prints the audit report (note
the segment-analysis re-test pinning the anomaly to its rows), and writes
``dataset_audit_log.json``. Deterministic; no dependencies beyond the stdlib.
"""

from __future__ import annotations

import math

from .audit import enable_utf8_output
from .datasets import (
    Dataset,
    distribution_stationary,
    duplicate_rate_below,
    no_missing_required,
    timestamps_monotonic,
    values_in_range,
)
from .datasetscanner import SelfAuditingDatasetScanner

_N = 500
_FAULT_START, _FAULT_END = 420, 480  # rows [420, 480): stuck-high temperature


def _synthetic_sensor_table() -> Dataset:
    rows: list[dict[str, str]] = []
    for i in range(_N):
        temp = 20.0 + 5.0 * math.sin(i * 0.05)  # benign daily-ish wobble
        if _FAULT_START <= i < _FAULT_END:
            temp = 215.0  # faulty sensor: stuck above the 150 limit
        # a few benign missing values, well under the 1% budget
        temp_str = "" if i in (10, 250) else f"{temp:.2f}"
        rows.append({"timestamp": str(i), "sensor_id": "S-1", "temperature": temp_str})
    return Dataset(["timestamp", "sensor_id", "temperature"], rows, "sensor-stream")


def _scanner() -> SelfAuditingDatasetScanner:
    return SelfAuditingDatasetScanner(
        checks=[
            no_missing_required(["timestamp", "sensor_id", "temperature"], max_fraction=0.01),
            values_in_range("temperature", -50.0, 150.0),
            timestamps_monotonic("timestamp"),
            duplicate_rate_below(max_fraction=0.0),
            distribution_stationary("temperature", max_shift=3.0),
        ]
    )


def main() -> None:
    enable_utf8_output()
    report = _scanner().scan(_synthetic_sensor_table())
    print(report.log.render())
    report.log.save("dataset_audit_log.json")
    print("\ndataset_audit_log.json written.")
    verdict = "TRUSTED" if report.trusted else "UNTRUSTED"
    print(f"verdict: {verdict}  (failed checks: {report.failed_checks or 'none'})")


if __name__ == "__main__":  # pragma: no cover
    main()
