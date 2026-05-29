"""Pytest suite for the local web UI (selfaudit.serve)."""

from __future__ import annotations

import json
import threading
import urllib.request

from selfaudit import serve
from selfaudit.sources import SourceUnavailable

_CSV = "ts,temperature\n" + "".join(f"{i},{20 + i % 4}\n" for i in range(40))


# --------------------------------------------------------------------------- #
# scan_payload: the pure request handler
# --------------------------------------------------------------------------- #


def test_scan_payload_csv_returns_report() -> None:
    html = serve.scan_payload({"mode": "csv", "value": _CSV, "name": "t.csv"})
    assert "<!doctype html>" in html
    assert "verdict" in html


def test_scan_payload_unknown_mode_is_friendly_error() -> None:
    html = serve.scan_payload({"mode": "bogus"})
    assert "Could not scan" in html
    assert "unknown request" in html


def test_scan_payload_empty_dataset() -> None:
    html = serve.scan_payload({"mode": "csv", "value": "ts,temperature\n"})
    assert "no rows" in html


def test_scan_payload_url_mode(monkeypatch) -> None:
    from selfaudit.datasets import parse_csv

    monkeypatch.setattr(serve, "fetch_csv", lambda url, **k: parse_csv(_CSV, url))
    html = serve.scan_payload({"mode": "url", "value": "https://example.com/d.csv"})
    assert "<!doctype html>" in html


def test_scan_payload_source_unavailable_is_friendly(monkeypatch) -> None:
    def offline():
        raise SourceUnavailable("offline")

    monkeypatch.setitem(serve._SOURCES, "usgs", offline)
    html = serve.scan_payload({"mode": "source", "value": "usgs"})
    assert "source unavailable" in html


def test_scan_payload_catches_arbitrary_errors() -> None:
    # A non-CSV blob still yields a friendly page, never an exception.
    html = serve.scan_payload({"mode": "csv", "value": "\x00not a csv at all"})
    assert "Could not scan" in html or "no rows" in html


# --------------------------------------------------------------------------- #
# End-to-end over a real (localhost) HTTP server
# --------------------------------------------------------------------------- #


def test_server_serves_index_and_scans() -> None:
    server = serve.build_server(("127.0.0.1", 0))  # ephemeral port
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        index = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "selfaudit" in index
        assert "Drop a CSV" in index

        body = json.dumps({"mode": "csv", "value": _CSV, "name": "t.csv"}).encode()
        req = urllib.request.Request(base + "/scan", body, {"Content-Type": "application/json"})
        report = urllib.request.urlopen(req, timeout=5).read().decode()
        assert "<!doctype html>" in report
        assert "verdict" in report
    finally:
        server.shutdown()
        server.server_close()


def test_server_rejects_bad_json() -> None:
    server = serve.build_server(("127.0.0.1", 0))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/scan", b"not json", {"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected HTTP 400")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            assert "invalid request body" in exc.read().decode()
    finally:
        server.shutdown()
        server.server_close()
