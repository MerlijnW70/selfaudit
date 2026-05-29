"""Pytest suite for the command-line dataset scanner (selfaudit.scan)."""

from __future__ import annotations

import json

import pytest

from selfaudit.scan import (
    _parse_missing,
    _parse_range,
    _parse_stationary,
    run,
)


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


def test_run_no_checks_is_usage_error(tmp_path, capsys) -> None:
    csv = _write_csv(tmp_path, _CLEAN)
    assert run([csv]) == 2
    assert "no checks specified" in capsys.readouterr().err


def test_run_bad_spec_is_usage_error(tmp_path, capsys) -> None:
    csv = _write_csv(tmp_path, _CLEAN)
    assert run([csv, "--range", "temperature:bad:range"]) == 2
    assert "error:" in capsys.readouterr().err


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
