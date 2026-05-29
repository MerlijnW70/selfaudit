"""Pytest suite for the live data sources (offline: urlopen is mocked)."""

from __future__ import annotations

import json
import urllib.request

import pytest

from selfaudit import sources
from selfaudit.sources import (
    SourceUnavailable,
    crypto_prices,
    fetch_csv,
    open_meteo,
    usgs_earthquakes,
)


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *exc) -> None:
        return None


def _mock_urlopen(monkeypatch, payload: dict) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=20.0: _FakeResp(payload))


_OPEN_METEO = {
    "hourly": {
        "time": ["2026-05-29T00:00", "2026-05-29T01:00", "2026-05-29T02:00"],
        "temperature_2m": [20.7, None, 20.1],  # one missing value
    }
}

_USGS = {
    "features": [
        {
            "properties": {"time": 1717000000000, "mag": 2.3, "place": "near A"},
            "geometry": {"coordinates": [-120.0, 38.0, 5.2]},
        },
        {
            "properties": {"time": 1716999000000, "mag": None, "place": None},
            "geometry": {"coordinates": [-121.0, 39.0]},  # no depth
        },
    ]
}


def test_open_meteo_parses_into_dataset(monkeypatch) -> None:
    _mock_urlopen(monkeypatch, _OPEN_METEO)
    ds = open_meteo(52.0, 4.0)
    assert ds.columns == ["time", "epoch", "temperature"]
    assert ds.n == 3
    assert ds.rows[0]["temperature"] == "20.7"
    assert ds.rows[1]["temperature"] == ""  # None -> missing
    # epoch is numeric and increasing
    epochs = [float(r["epoch"]) for r in ds.rows]
    assert epochs[0] < epochs[1] < epochs[2]


def test_open_meteo_missing_series_raises(monkeypatch) -> None:
    _mock_urlopen(monkeypatch, {"hourly": {"time": []}})  # no temperature_2m
    with pytest.raises(SourceUnavailable):
        open_meteo()


def test_usgs_parses_and_handles_missing_fields(monkeypatch) -> None:
    _mock_urlopen(monkeypatch, _USGS)
    ds = usgs_earthquakes("all_hour")
    assert ds.columns == ["time", "mag", "depth_km", "place"]
    assert ds.n == 2
    assert ds.rows[0]["depth_km"] == "5.2"
    assert ds.rows[1]["mag"] == ""  # None -> empty
    assert ds.rows[1]["depth_km"] == ""  # missing coordinate -> empty


def test_usgs_missing_features_raises(monkeypatch) -> None:
    _mock_urlopen(monkeypatch, {"not_features": 1})
    with pytest.raises(SourceUnavailable):
        usgs_earthquakes()


_CRYPTO = {
    "prices": [
        [1780000000000, 72720.8],
        [1780003600000, 73100.5],
        ["bad", "row"],  # malformed entry -> skipped, still parses
        [1780007200000, 73452.5],
    ]
}


def test_crypto_prices_parses_and_skips_malformed(monkeypatch) -> None:
    _mock_urlopen(monkeypatch, _CRYPTO)
    ds = crypto_prices("bitcoin", "usd", days=1)
    assert ds.columns == ["time", "price"]
    assert ds.n == 3  # the malformed ["bad","row"] entry is skipped
    assert ds.rows[0]["price"] == "72720.8"
    assert ds.name == "coingecko:bitcoin-usd"


def test_crypto_prices_missing_series_raises(monkeypatch) -> None:
    _mock_urlopen(monkeypatch, {"no_prices": []})
    with pytest.raises(SourceUnavailable):
        crypto_prices()


def test_fetch_csv_parses_a_url(monkeypatch) -> None:
    csv_text = "ts,temperature\n0,20.5\n1,21.0\n"

    class _Resp:
        def read(self) -> bytes:
            return csv_text.encode("utf-8")

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *exc) -> None:
            return None

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=20.0: _Resp())
    ds = fetch_csv("https://example.com/d.csv")
    assert ds.columns == ["ts", "temperature"]
    assert ds.n == 2
    assert ds.rows[0]["temperature"] == "20.5"
    assert ds.name == "https://example.com/d.csv"


def test_fetch_rejects_non_http_schemes() -> None:
    # No local-file reads / SSRF via file://, ftp://, etc.
    for bad in ("file:///etc/passwd", "ftp://host/x", "/local/path.csv"):
        with pytest.raises(SourceUnavailable):
            fetch_csv(bad)


def test_fetch_csv_network_error_is_clean(monkeypatch) -> None:
    def boom(req, timeout=20.0):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(SourceUnavailable):
        fetch_csv("https://example.com/d.csv")


def test_fetch_json_network_error_is_clean(monkeypatch) -> None:
    def boom(url, timeout=20.0):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(SourceUnavailable):
        sources._fetch_json("https://example.invalid/x")


def test_fetch_json_non_dict_payload_raises(monkeypatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=20.0: _FakeResp([1, 2, 3]))  # type: ignore[arg-type]
    with pytest.raises(SourceUnavailable):
        sources._fetch_json("https://example.com/x")


# --------------------------------------------------------------------------- #
# Live demo: scans mocked sources end-to-end (still offline)
# --------------------------------------------------------------------------- #


def test_livedemo_runs_with_mocked_sources(tmp_path, monkeypatch, capsys) -> None:
    from selfaudit import livedemo

    monkeypatch.setattr(livedemo, "open_meteo", _meteo_ds)
    monkeypatch.setattr(livedemo, "usgs_earthquakes", lambda period="all_hour": _usgs_ds())
    monkeypatch.chdir(tmp_path)
    livedemo.main()
    out = capsys.readouterr().out
    assert "Open-Meteo" in out
    assert "USGS" in out
    assert (tmp_path / "live_audit_log.json").exists()


def test_livedemo_degrades_when_offline(monkeypatch, capsys) -> None:
    from selfaudit import livedemo

    def offline(*a, **k):
        raise SourceUnavailable("offline")

    monkeypatch.setattr(livedemo, "open_meteo", offline)
    monkeypatch.setattr(livedemo, "usgs_earthquakes", offline)
    livedemo.main()  # must not raise
    assert "source unavailable" in capsys.readouterr().out


def _meteo_ds():
    from selfaudit.datasets import Dataset

    rows = [
        {"time": f"2026-05-29T0{i}:00", "epoch": str(1000 + i), "temperature": str(20 + i)}
        for i in range(5)
    ]
    return Dataset(["time", "epoch", "temperature"], rows, "open-meteo@test")


def _usgs_ds():
    from selfaudit.datasets import Dataset

    rows = [{"time": str(2000 - i), "mag": "2.0", "depth_km": "5", "place": "x"} for i in range(5)]
    return Dataset(["time", "mag", "depth_km", "place"], rows, "usgs:test")
