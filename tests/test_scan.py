"""Pytest suite for the command-line dataset scanner (selfaudit.scan)."""

from __future__ import annotations

import json

import pytest

from selfaudit import scan
from selfaudit.datasets import Dataset
from selfaudit.scan import (
    _parse_allowed,
    _parse_missing,
    _parse_range,
    _parse_stationary,
    _parse_type,
    _parse_unique,
    run,
)
from selfaudit.sources import SourceUnavailable


def _write_csv(tmp_path, body: str):
    p = tmp_path / "data.csv"
    p.write_text(body, encoding="utf-8")
    return str(p)


_CLEAN = "timestamp,sensor_id,temperature\n0,S1,20\n1,S1,21\n2,S1,22\n"
_DIRTY = "timestamp,sensor_id,temperature\n0,S1,20\n1,S1,999\n2,S1,21\n"  # 999 out of range


# --------------------------------------------------------------------------- #
# Spec parsers
# --------------------------------------------------------------------------- #


def test_parse_range_ok_and_errors() -> None:
    assert _parse_range("temperature:-50:150").name == "range[temperature]"
    with pytest.raises(ValueError):
        _parse_range("temperature:-50")  # wrong arity
    with pytest.raises(ValueError):
        _parse_range("temperature:lo:hi")  # non-numeric


def test_parse_missing_default_and_explicit_fraction() -> None:
    assert _parse_missing("a,b").name == "missing_required"  # default fraction, two fields
    assert _parse_missing("a:0.05").name == "missing_required"  # explicit fraction
    with pytest.raises(ValueError):
        _parse_missing("a:notnum")
    with pytest.raises(ValueError):
        _parse_missing(":0.1")  # no fields


def test_parse_stationary_default_and_explicit() -> None:
    assert _parse_stationary("temp").name == "stationary[temp]"
    assert _parse_stationary("temp:2.5").name == "stationary[temp]"
    with pytest.raises(ValueError):
        _parse_stationary("temp:notnum")


def test_parse_unique_type_allowed() -> None:
    assert _parse_unique("id").name == "unique[id]"
    assert _parse_unique("a,b").name == "unique[a+b]"
    assert _parse_type("age:int").name == "type[age=int]"
    assert _parse_allowed("status:active,churned").name == "allowed[status]"
    with pytest.raises(ValueError):
        _parse_type("age:integer")  # unknown type
    with pytest.raises(ValueError):
        _parse_type("noColon")
    with pytest.raises(ValueError):
        _parse_allowed("status:")  # no values
    with pytest.raises(ValueError):
        _parse_unique(",")  # no fields


# --------------------------------------------------------------------------- #
# run(): exit codes and behaviour
# --------------------------------------------------------------------------- #


def test_run_trusted_returns_zero(tmp_path, capsys) -> None:
    csv = _write_csv(tmp_path, _CLEAN)
    code = run([csv, "--range", "temperature:-50:150", "--monotonic", "timestamp"])
    assert code == 0
    assert "TRUSTED" in capsys.readouterr().out


def test_run_untrusted_returns_one(tmp_path, capsys) -> None:
    csv = _write_csv(tmp_path, _DIRTY)
    code = run([csv, "--range", "temperature:-50:150"])
    assert code == 1
    out = capsys.readouterr().out
    assert "UNTRUSTED" in out
    assert "range[temperature]" in out


def test_run_quiet_suppresses_report(tmp_path, capsys) -> None:
    csv = _write_csv(tmp_path, _CLEAN)
    run([csv, "--duplicates", "0", "--quiet"])
    out = capsys.readouterr().out
    assert "SELF-AUDIT REPORT" not in out
    assert "verdict:" in out


def test_run_writes_json(tmp_path, capsys) -> None:
    csv = _write_csv(tmp_path, _DIRTY)
    out_json = tmp_path / "audit.json"
    run([csv, "--range", "temperature:-50:150", "--json", str(out_json), "--quiet"])
    assert out_json.exists()
    assert json.loads(out_json.read_text(encoding="utf-8"))["final_status"] == "untrusted"


def test_run_no_rules_auto_infers(tmp_path, capsys) -> None:
    # Zero-config: no rule flags -> infer from the data, don't error.
    csv = _write_csv(tmp_path, _CLEAN)
    code = run([csv])
    assert code == 0  # clean data -> trusted
    assert "no rules given — inferred" in capsys.readouterr().out


def test_run_no_input_is_usage_error(capsys) -> None:
    assert run([]) == 2
    assert "provide a CSV path or --source" in capsys.readouterr().err


def test_run_bad_spec_is_usage_error(tmp_path, capsys) -> None:
    csv = _write_csv(tmp_path, _CLEAN)
    assert run([csv, "--range", "temperature:bad:range"]) == 2
    assert "error:" in capsys.readouterr().err


def test_run_with_open_meteo_source(monkeypatch, capsys) -> None:
    rows = [{"time": f"t{i}", "epoch": str(i), "temperature": str(20 + i)} for i in range(6)]
    ds = Dataset(["time", "epoch", "temperature"], rows, "open-meteo@test")
    monkeypatch.setattr(scan, "open_meteo", lambda lat, lon, forecast_days=2: ds)
    code = run(["--source", "open-meteo", "--range", "temperature:-50:60", "--quiet"])
    assert code == 0
    assert "TRUSTED" in capsys.readouterr().out


def test_run_with_usgs_source_warns_newest_first(monkeypatch, capsys) -> None:
    rows = [{"time": str(100 - i), "mag": "2.0"} for i in range(6)]  # decreasing time
    ds = Dataset(["time", "mag"], rows, "usgs:all_hour")
    monkeypatch.setattr(scan, "usgs_earthquakes", lambda period="all_hour": ds)
    # monotonic is a WARNING (a newest-first feed isn't broken) -> REVIEW, exit 0.
    code = run(["--source", "usgs", "--monotonic", "time", "--quiet"])
    assert code == 0
    assert "REVIEW" in capsys.readouterr().out
    # ...but --strict makes warnings fail the gate.
    monkeypatch.setattr(scan, "usgs_earthquakes", lambda period="all_hour": ds)
    assert run(["--source", "usgs", "--monotonic", "time", "--strict", "--quiet"]) == 1


def test_run_with_crypto_source(monkeypatch, capsys) -> None:
    rows = [{"time": str(1000 + i), "price": str(70000 + i)} for i in range(8)]
    ds = Dataset(["time", "price"], rows, "coingecko:bitcoin-usd")
    monkeypatch.setattr(scan, "crypto_prices", lambda coin, vs, days=1: ds)
    code = run(
        ["--source", "crypto", "--coin", "bitcoin", "--range", "price:0:10000000", "--quiet"]
    )
    assert code == 0
    assert "TRUSTED" in capsys.readouterr().out


def test_run_source_unavailable_returns_three(monkeypatch, capsys) -> None:
    def offline(*a, **k):
        raise SourceUnavailable("offline")

    monkeypatch.setattr(scan, "open_meteo", offline)
    code = run(["--source", "open-meteo", "--range", "temperature:-50:60"])
    assert code == 3
    assert "source unavailable" in capsys.readouterr().err


def test_run_rejects_both_csv_and_source(tmp_path, capsys) -> None:
    csv = _write_csv(tmp_path, _CLEAN)
    assert run([csv, "--source", "usgs", "--monotonic", "time"]) == 2
    assert "not both" in capsys.readouterr().err


def test_run_infer_flags_planted_outlier(tmp_path, capsys) -> None:
    body = "ts,temperature\n" + "".join(f"{i},{20 + i % 5}\n" for i in range(60))
    body = body.replace("30,2", "30,9000", 1)  # plant an outlier at ts=30
    csv = _write_csv(tmp_path, body)
    # An outlier is a warning -> REVIEW (exit 0) by default; --strict fails it.
    code = run([csv, "--infer", "--quiet"])
    assert code == 0
    assert "REVIEW" in capsys.readouterr().out
    assert run([csv, "--infer", "--strict", "--quiet"]) == 1


def test_run_infer_announces_inferred_checks(tmp_path, capsys) -> None:
    csv = _write_csv(tmp_path, "ts,temperature\n0,20\n1,21\n2,22\n")
    run([csv, "--infer"])
    assert "inferred" in capsys.readouterr().out


def test_run_writes_html_report(tmp_path) -> None:
    csv = _write_csv(tmp_path, _DIRTY)
    out_html = tmp_path / "report.html"
    run([csv, "--range", "temperature:-50:150", "--html", str(out_html), "--quiet"])
    assert out_html.exists()
    assert out_html.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_run_bad_csv_path_is_error(capsys) -> None:
    assert run(["/no/such/file-12345.csv", "--infer"]) == 2
    assert "cannot read dataset" in capsys.readouterr().err


def test_run_scans_a_csv_url(monkeypatch, capsys) -> None:
    # `selfaudit https://.../data.csv` fetches and scans directly — no download step.
    rows = [{"ts": str(i), "temperature": str(20 + i % 4)} for i in range(40)]
    ds = Dataset(["ts", "temperature"], rows, "https://example.com/d.csv")
    monkeypatch.setattr(scan, "fetch_csv", lambda url, **k: ds)
    code = run(["https://example.com/d.csv"])  # auto-infer
    assert code == 0
    assert "inferred" in capsys.readouterr().out


def test_run_csv_url_unavailable_returns_three(monkeypatch, capsys) -> None:
    def offline(url, **k):
        raise SourceUnavailable("offline")

    monkeypatch.setattr(scan, "fetch_csv", offline)
    assert run(["https://example.com/d.csv"]) == 3
    assert "source unavailable" in capsys.readouterr().err


def test_run_schema_checks_flag_violations(tmp_path, capsys) -> None:
    body = (
        "id,age,status\n"
        "1,30,active\n"
        "2,xx,churned\n"  # age not int
        "1,40,frozen\n"  # duplicate id + status not allowed
    )
    csv = _write_csv(tmp_path, body)
    code = run(
        [
            csv,
            "--unique",
            "id",
            "--type",
            "age:int",
            "--allowed",
            "status:active,churned",
            "--quiet",
        ]
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "UNTRUSTED" in out
    for name in ("unique[id]", "type[age=int]", "allowed[status]"):
        assert name in out


def test_run_all_check_kinds_together(tmp_path, capsys) -> None:
    csv = _write_csv(tmp_path, _CLEAN)
    code = run(
        [
            csv,
            "--range",
            "temperature:-50:150",
            "--monotonic",
            "timestamp",
            "--missing",
            "temperature,sensor_id:0.01",
            "--duplicates",
            "0",
            "--stationary",
            "temperature:3",
        ]
    )
    assert code == 0
