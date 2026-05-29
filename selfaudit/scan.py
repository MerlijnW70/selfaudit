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

from .audit import enable_utf8_output
from .datasets import (
    Check,
    Dataset,
    allowed_values,
    distribution_stationary,
    duplicate_rate_below,
    infer_checks,
    load_dataset,
    no_missing_required,
    svg_chart,
    timestamps_monotonic,
    unique_key,
    values_in_range,
    values_of_type,
)
from .datasetscanner import SelfAuditingDatasetScanner
from .sources import SourceUnavailable, crypto_prices, fetch_csv, open_meteo, usgs_earthquakes


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


_TYPE_NAMES = ("int", "float", "bool", "date")


def _parse_type(spec: str) -> Check:
    if ":" not in spec:
        raise ValueError(f"--type expects FIELD:TYPE, got {spec!r}")
    field, type_name = spec.rsplit(":", 1)
    if not field:
        raise ValueError(f"--type expects FIELD:TYPE, got {spec!r}")
    if type_name not in _TYPE_NAMES:
        raise ValueError(f"--type TYPE must be one of {', '.join(_TYPE_NAMES)}, got {type_name!r}")
    return values_of_type(field, type_name)


def _parse_allowed(spec: str) -> Check:
    if ":" not in spec:
        raise ValueError(f"--allowed expects FIELD:v1,v2,..., got {spec!r}")
    field, values_part = spec.split(":", 1)
    allowed = [v for v in values_part.split(",") if v != ""]
    if not field or not allowed:
        raise ValueError(f"--allowed expects FIELD:v1,v2,..., got {spec!r}")
    return allowed_values(field, allowed)


def _parse_unique(spec: str) -> Check:
    fields = [f for f in spec.split(",") if f]
    if not fields:
        raise ValueError(f"--unique expects FIELD[,FIELD...], got {spec!r}")
    return unique_key(fields)


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
        prog="selfaudit",
        description=(
            "Vet a dataset's trustworthiness. Point it at a CSV (path or URL) or a "
            "live --source; with no rules it infers them from the data. "
            "Exit 0 = trusted/review, 1 = untrusted (or any warning with --strict)."
        ),
    )
    p.add_argument(
        "csv",
        nargs="?",
        metavar="DATASET",
        help="path or URL of the dataset to scan: CSV/TSV, JSON, or .xlsx "
        "(omit when using --source)",
    )
    p.add_argument(
        "--source",
        choices=["open-meteo", "usgs", "crypto"],
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
    p.add_argument("--coin", default="bitcoin", help="coin id for --source crypto (CoinGecko)")
    p.add_argument("--vs-currency", default="usd", help="quote currency for --source crypto")
    p.add_argument("--days", type=int, default=1, help="history window in days for --source crypto")
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
    p.add_argument(
        "--unique",
        action="append",
        default=[],
        metavar="FIELD[,FIELD...]",
        help="the (combined) FIELD(s) must be unique — a primary-key check (repeatable)",
    )
    p.add_argument(
        "--type",
        action="append",
        default=[],
        metavar="FIELD:TYPE",
        help="FIELD values must parse as TYPE (int|float|bool|date) (repeatable)",
    )
    p.add_argument(
        "--allowed",
        action="append",
        default=[],
        metavar="FIELD:v1,v2,...",
        help="FIELD values must be one of the listed values (repeatable)",
    )
    p.add_argument(
        "--infer",
        action="store_true",
        help="auto-propose checks from the data itself (zero-config; ignores rule flags)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings as failures too (exit 1 on REVIEW, not just UNTRUSTED)",
    )
    p.add_argument("--json", metavar="PATH", help="write the audit log to PATH as JSON")
    p.add_argument("--html", metavar="PATH", help="write a shareable HTML trust report to PATH")
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
    checks += [_parse_unique(s) for s in args.unique]
    checks += [_parse_type(s) for s in args.type]
    checks += [_parse_allowed(s) for s in args.allowed]
    return checks


def _load_source(args: argparse.Namespace) -> Dataset:
    """Resolve the input dataset from --source, fetching live data as needed."""
    if args.source == "open-meteo":
        return open_meteo(args.lat, args.lon, forecast_days=args.forecast_days)
    if args.source == "crypto":
        return crypto_prices(args.coin, args.vs_currency, days=args.days)
    return usgs_earthquakes(args.period)


def run(argv: list[str] | None = None) -> int:
    enable_utf8_output()
    args = build_parser().parse_args(argv)

    if args.source and args.csv:
        print("error: give either a CSV path or --source, not both", file=sys.stderr)
        return 2
    if not args.source and not args.csv:
        print("error: provide a CSV path or --source {open-meteo,usgs,crypto}", file=sys.stderr)
        return 2

    # Load the dataset first — inference needs to see the data.
    try:
        if args.source:
            ds = _load_source(args)
        elif args.csv.startswith(("http://", "https://")):
            ds = fetch_csv(args.csv)
        else:
            ds = load_dataset(args.csv)
    except SourceUnavailable as exc:
        print(f"error: source unavailable — {exc}", file=sys.stderr)
        return 3
    except (OSError, ValueError) as exc:
        print(f"error: cannot read dataset — {exc}", file=sys.stderr)
        return 2

    # Build checks: explicit rule flags if given, otherwise infer from the data.
    try:
        explicit = _checks_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.infer or not explicit:
        checks = infer_checks(ds)
        if not args.quiet:
            lead = "inferred" if args.infer else "no rules given — inferred"
            print(f"{lead} {len(checks)} checks: {', '.join(c.name for c in checks)}\n")
    else:
        checks = explicit

    report = SelfAuditingDatasetScanner(checks).scan(ds)
    if not args.quiet:
        print(report.log.render())
    if args.json:
        report.log.save(args.json)
    if args.html:
        report.log.save_html(args.html, chart=svg_chart(ds))
    print(
        f"verdict: {report.status.upper()}  "
        f"(failures: {report.failed_checks or 'none'}; warnings: {report.warnings or 'none'})"
    )
    # Exit 1 on a hard failure; with --strict, warnings (REVIEW) also fail the gate.
    if report.status == "untrusted" or (args.strict and report.warnings):
        return 1
    return 0


def main() -> None:  # pragma: no cover
    raise SystemExit(run())


if __name__ == "__main__":  # pragma: no cover
    main()
