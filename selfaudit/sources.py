"""Live data sources: fetch a free, real-time public dataset into a :class:`Dataset`.

No API key, no third-party HTTP library — just stdlib ``urllib`` + ``json``. TLS
verification is routed through the OS trust store (reusing the LLM app's
``enable_os_truststore``) so it also works behind a TLS-intercepting proxy. A
network failure surfaces as :class:`SourceUnavailable`, never a crash — the
scanner treats an unreachable source the same way the solver treats a clean
numerical failure.

Sources (both free, no auth, real-time):

* :func:`open_meteo`        — hourly 2 m temperature forecast (Open-Meteo).
* :func:`usgs_earthquakes`  — recent earthquakes (USGS GeoJSON feed).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any

from .datasets import Dataset, parse_text
from .llm import enable_os_truststore


class SourceUnavailable(Exception):
    """A live data source could not be reached or returned unusable data."""


_USER_AGENT = "selfaudit/0.1 (+https://github.com/MerlijnW70/selfaudit)"


def _fetch_json(url: str, *, timeout: float = 20.0) -> dict[str, Any]:
    enable_os_truststore()  # secure proxy fix; a no-op when not needed
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https only
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        raise SourceUnavailable(f"could not fetch {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SourceUnavailable(f"unexpected response shape from {url}")
    return payload


def _iso_to_epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def fetch_csv(url: str, *, timeout: float = 20.0) -> Dataset:
    """Fetch a CSV straight from a URL into a :class:`Dataset` — no download step.

    Same stdlib + OS-trust-store path as the JSON sources; a network failure
    surfaces as :class:`SourceUnavailable`.
    """
    enable_os_truststore()
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - http(s) only
            text = resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        raise SourceUnavailable(f"could not fetch {url}: {exc}") from exc
    return parse_text(text, url)


def open_meteo(
    latitude: float = 52.37,
    longitude: float = 4.90,
    *,
    forecast_days: int = 2,
    timeout: float = 20.0,
) -> Dataset:
    """Hourly 2 m temperature from Open-Meteo as a :class:`Dataset`.

    Columns: ``time`` (ISO), ``epoch`` (numeric, for the monotonic check) and
    ``temperature`` (°C). Defaults to Amsterdam; free and key-less.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        f"&hourly=temperature_2m&forecast_days={forecast_days}"
    )
    data = _fetch_json(url, timeout=timeout)
    hourly = data.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly or "temperature_2m" not in hourly:
        raise SourceUnavailable("Open-Meteo response missing the 'hourly' series")
    times = hourly["time"]
    temps = hourly["temperature_2m"]
    rows: list[dict[str, str]] = []
    for t, temp in zip(times, temps, strict=False):
        rows.append(
            {
                "time": str(t),
                "epoch": f"{_iso_to_epoch(t):.0f}",
                "temperature": "" if temp is None else str(temp),
            }
        )
    return Dataset(["time", "epoch", "temperature"], rows, f"open-meteo@{latitude},{longitude}")


def crypto_prices(
    coin: str = "bitcoin",
    vs_currency: str = "usd",
    *,
    days: int = 1,
    timeout: float = 20.0,
) -> Dataset:
    """Recent price time series for a coin from CoinGecko as a :class:`Dataset`.

    Columns: ``time`` (epoch ms, ascending) and ``price``. Free and key-less.
    Genuinely volatile data — good for exercising the range and regime-shift
    checks on numbers that actually move.
    """
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
        f"?vs_currency={vs_currency}&days={days}"
    )
    data = _fetch_json(url, timeout=timeout)
    prices = data.get("prices")
    if not isinstance(prices, list) or not prices:
        raise SourceUnavailable("CoinGecko response missing the 'prices' series")
    rows: list[dict[str, str]] = []
    for point in prices:
        if not isinstance(point, list) or len(point) < 2:
            continue
        try:
            ts = float(point[0])
        except (TypeError, ValueError):
            continue  # skip malformed timestamps rather than crash
        rows.append({"time": f"{ts:.0f}", "price": str(point[1])})
    if not rows:
        raise SourceUnavailable("CoinGecko 'prices' series had no usable points")
    return Dataset(["time", "price"], rows, f"coingecko:{coin}-{vs_currency}")


def usgs_earthquakes(period: str = "all_hour", *, timeout: float = 20.0) -> Dataset:
    """Recent earthquakes from the USGS GeoJSON feed as a :class:`Dataset`.

    ``period`` is one of the feed names, e.g. ``all_hour``, ``all_day``,
    ``significant_week``. Columns: ``time`` (epoch ms), ``mag``, ``depth_km``,
    ``place``. The feed is ordered newest-first, so a monotonic check on ``time``
    is *expected* to flag it — a real, explainable finding.
    """
    url = f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{period}.geojson"
    data = _fetch_json(url, timeout=timeout)
    features = data.get("features")
    if not isinstance(features, list):
        raise SourceUnavailable("USGS response missing 'features'")
    rows: list[dict[str, str]] = []
    for feat in features:
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [None, None, None])
        depth = coords[2] if len(coords) >= 3 else None
        rows.append(
            {
                "time": "" if props.get("time") is None else str(props["time"]),
                "mag": "" if props.get("mag") is None else str(props["mag"]),
                "depth_km": "" if depth is None else str(depth),
                "place": str(props.get("place") or ""),
            }
        )
    return Dataset(["time", "mag", "depth_km", "place"], rows, f"usgs:{period}")
