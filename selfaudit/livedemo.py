"""Demo: ``python -m selfaudit.livedemo``.

Fetches free, real-time public datasets and scans them with the Self-Auditing
dataset scanner — no API key required. Writes ``live_audit_log.json``. If the
network is unavailable the demo says so and exits cleanly (it never crashes).

* Open-Meteo  — hourly temperature: range / monotonic time / missing / stationary.
* USGS quakes — recent earthquakes: the newest-first feed is *expected* to fail
  the monotonic-time check, a real and explainable finding.
"""

from __future__ import annotations

from collections.abc import Callable

from .audit import enable_utf8_output
from .datasets import (
    Check,
    Dataset,
    distribution_stationary,
    duplicate_rate_below,
    no_missing_required,
    timestamps_monotonic,
    values_in_range,
)
from .datasetscanner import ScanReport, SelfAuditingDatasetScanner
from .sources import SourceUnavailable, crypto_prices, open_meteo, usgs_earthquakes


def _scan_source(
    title: str, fetch: Callable[[], Dataset], checks: list[Check]
) -> ScanReport | None:
    print(f"\n### live source: {title}")
    try:
        ds = fetch()
    except SourceUnavailable as exc:
        print(f"(source unavailable — {exc})")
        return None
    print(f"fetched {ds.n} rows from {ds.name}")
    report = SelfAuditingDatasetScanner(checks).scan(ds)
    print(report.log.render())
    return report


def main() -> None:
    enable_utf8_output()
    weather = _scan_source(
        "Open-Meteo hourly temperature (Amsterdam)",
        open_meteo,
        [
            values_in_range("temperature", -50.0, 60.0),
            timestamps_monotonic("epoch"),
            no_missing_required(["time", "temperature"], max_fraction=0.01),
        ],
    )
    if weather is not None:
        weather.log.save("live_audit_log.json")
        print("\nlive_audit_log.json written.")

    _scan_source(
        "USGS earthquakes (past hour)",
        lambda: usgs_earthquakes("all_hour"),
        [
            values_in_range("mag", -2.0, 10.0),
            timestamps_monotonic("time"),  # newest-first feed -> expected to flag
            duplicate_rate_below(max_fraction=0.0),
        ],
    )

    _scan_source(
        "CoinGecko Bitcoin price (last 24h)",
        lambda: crypto_prices("bitcoin", "usd", days=1),
        [
            values_in_range("price", 0.0, 10_000_000.0),  # sanity bound
            timestamps_monotonic("time"),  # ascending -> should pass
            distribution_stationary("price", max_shift=3.0),  # volatile -> may flag a regime shift
        ],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
