"""Command-line dataset trust scanner: ``python -m selfaudit.scan``.

Scan a real CSV against rules given as flags, print the audit report, optionally
write the audit log as JSON, and exit ``0`` when the dataset is trusted or ``1``
when any check fails — so it drops straight into a CI pipeline.

Example::

    python -m selfaudit.scan readings.csv \\
        --range temperature:-50:150 \\
        --monotonic timestamp \\
        --missing temperature,sensor_id:0.01 \\
        --duplicates 0 \\
        --stationary temperature:3 \\
        --json audit.json
"""

from __future__ import annotations

import argparse
import sys

from .datasets import (
    Check,
    Dataset,
    distribution_stationary,
    duplicate_rate_below,
    no_missing_required,
    timestamps_monotonic,
    values_in_range,
)
from .datasetscanner import SelfAuditingDatasetScanner
from .sources import SourceUnavailable, open_meteo, usgs_earthquakes


def _parse_range(spec: str) -> Check:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"--range expects FIELD:LO:HI, got {spec!r}")
    field, lo, hi = parts
    try:
        return values_in_range(field, float(lo), float(hi))
    except ValueError as exc:
        raise ValueError(f"--range LO/HI must be numbers, got {spec!r}") from exc


def _parse_missing(spec: str) -> Check:
    if ":" in spec:
        fields_part, frac_part = spec.rsplit(":", 1)
        try:
            frac = float(frac_part)
        except ValueError as exc:
            raise ValueError(f"--missing MAXFRAC must be a number, got {spec!r}") from exc
    else:
        fields_part, frac = spec, 0.01
    fields = [f for f in fields_part.split(",") if f]
    if not fields:
        raise ValueError(f"--missing expects FIELDS[:MAXFRAC], got {spec!r}")
    return no_missing_required(fields, max_fraction=frac)


def _parse_stationary(spec: str) -> Check:
    if ":" in spec:
        field, shift_part = spec.rsplit(":", 1)
        try:
            shift = float(shift_part)
        except ValueError as exc:
            raise ValueError(f"--stationary MAXSHIFT must be a number, got {spec!r}") from exc
    else:
        field, shift = spec, 3.0
    if not field:
        raise ValueError(f"--stationary expects FIELD[:MAXSHIFT], got {spec!r}")
    return distribution_stationary(field, max_shift=shift)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m selfaudit.scan",
        description="Scan a CSV against explicit rules; re-test and audit every violation.",
    )
    p.add_argument(
        "csv",
        nargs="?",
        help="path to the CSV file to scan (omit when using --source)",
    )
    p.add_argument(
        "--source",
        choices=["open-meteo", "usgs"],
        help="fetch a free, real-time dataset instead of reading a CSV",
    )
    p.add_argument("--lat", type=float, default=52.37, help="latitude for --source open-meteo")
    p.add_argument("--lon", type=float, default=4.90, help="longitude for --source open-meteo")
    p.add_argument(
        "--forecast-days", type=int, default=2, help="forecast days for --source open-meteo"
    )
    p.add_argument(
        "--period",
        default="all_hour",
        help="feed period for --source usgs (e.g. all_hour, all_day, significant_week)",
    )
    p.add_argument(
        "--range",
        action="append",
        default=[],
        metavar="FIELD:LO:HI",
        help="numeric FIELD must lie in [LO, HI] (repeatable)",
    )
    p.add_argument(
        "--monotonic",
        action="append",
        default=[],
        metavar="FIELD",
        help="numeric FIELD must be non-decreasing (repeatable)",
    )
    p.add_argument(
        "--missing",
        action="append",
        default=[],
        metavar="FIELDS[:MAXFRAC]",
        help="comma-separated FIELDS missing in at most MAXFRAC of rows (default 0.01)",
    )
    p.add_argument(
        "--duplicates",
        type=float,
        metavar="MAXFRAC",
        help="flag if the exact-duplicate-row rate exceeds MAXFRAC",
    )
    p.add_argument(
        "--stationary",
        action="append",
        default=[],
        metavar="FIELD[:MAXSHIFT]",
        help="FIELD mean must not shift by more than MAXSHIFT sigma (default 3)",
    )
    p.add_argument("--json", metavar="PATH", help="write the audit log to PATH as JSON")
    p.add_argument("--quiet", action="store_true", help="print only the final verdict")
    return p


def _checks_from_args(args: argparse.Namespace) -> list[Check]:
    checks: list[Check] = []
    checks += [_parse_range(s) for s in args.range]
    checks += [timestamps_monotonic(f) for f in args.monotonic]
    checks += [_parse_missing(s) for s in args.missing]
    if args.duplicates is not None:
        checks.append(duplicate_rate_below(max_fraction=args.duplicates))
    checks += [_parse_stationary(s) for s in args.stationary]
    return checks


def _load_source(args: argparse.Namespace) -> Dataset:
    """Resolve the input dataset from --source, fetching live data as needed."""
    if args.source == "open-meteo":
        return open_meteo(args.lat, args.lon, forecast_days=args.forecast_days)
    return usgs_earthquakes(args.period)


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.source and args.csv:
        print("error: give either a CSV path or --source, not both", file=sys.stderr)
        return 2
    if not args.source and not args.csv:
        print("error: provide a CSV path or --source {open-meteo,usgs}", file=sys.stderr)
        return 2

    try:
        checks = _checks_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not checks:
        print(
            "error: no checks specified — use --range/--monotonic/--missing/"
            "--duplicates/--stationary",
            file=sys.stderr,
        )
        return 2

    try:
        source = _load_source(args) if args.source else args.csv
    except SourceUnavailable as exc:
        print(f"error: live source unavailable — {exc}", file=sys.stderr)
        return 3

    report = SelfAuditingDatasetScanner(checks).scan(source)
    if not args.quiet:
        print(report.log.render())
    if args.json:
        report.log.save(args.json)
    verdict = "TRUSTED" if report.trusted else "UNTRUSTED"
    print(f"verdict: {verdict}  (failed: {report.failed_checks or 'none'})")
    return 0 if report.trusted else 1


def main() -> None:  # pragma: no cover
    raise SystemExit(run())


if __name__ == "__main__":  # pragma: no cover
    main()
