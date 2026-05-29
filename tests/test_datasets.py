"""Pytest suite for the dataset trust scanner (datasets + datasetscanner)."""

from __future__ import annotations

import json

from selfaudit.datasets import (
    Check,
    Dataset,
    distribution_stationary,
    duplicate_rate_below,
    load_csv,
    no_missing_required,
    timestamps_monotonic,
    values_in_range,
)
from selfaudit.datasetscanner import SelfAuditingDatasetScanner, _segment_retest


def _ds(temps: list[str], **extra: list[str]) -> Dataset:
    cols = ["timestamp", "temperature", *extra]
    rows = []
    for i, t in enumerate(temps):
        row = {"timestamp": str(i), "temperature": t}
        for k, vals in extra.items():
            row[k] = vals[i]
        rows.append(row)
    return Dataset(cols, rows, "test")


# --------------------------------------------------------------------------- #
# Dataset basics
# --------------------------------------------------------------------------- #


def test_numeric_column_handles_missing_and_unparseable() -> None:
    ds = _ds(["1.5", "", "abc", "2.0"])
    col = ds.numeric_column("temperature")
    assert col == [(0, 1.5), (1, None), (2, None), (3, 2.0)]


def test_slice_preserves_columns() -> None:
    ds = _ds(["1", "2", "3", "4"])
    sub = ds.slice(1, 3)
    assert sub.n == 2
    assert sub.columns == ds.columns


def test_load_csv_roundtrip(tmp_path) -> None:
    p = tmp_path / "d.csv"
    p.write_text("timestamp,temperature\n0,20.5\n1,21.0\n", encoding="utf-8")
    ds = load_csv(str(p))
    assert ds.n == 2
    assert ds.rows[0]["temperature"] == "20.5"


# --------------------------------------------------------------------------- #
# Individual checks: pass and fail
# --------------------------------------------------------------------------- #


def test_values_in_range_flags_outliers_and_ignores_missing() -> None:
    ds = _ds(["20", "999", "", "abc", "30"])  # 999 out of range; ""/abc ignored
    res = values_in_range("temperature", -50, 150).run(ds)
    assert not res.ok
    assert res.bad_rows == [1]


def test_values_in_range_passes_when_clean() -> None:
    assert values_in_range("temperature", -50, 150).run(_ds(["20", "30", "40"])).ok


def test_timestamps_monotonic_flags_decreases() -> None:
    ds = Dataset(
        ["timestamp"],
        [{"timestamp": v} for v in ["0", "1", "2", "1", "3"]],
        "ts",
    )
    res = timestamps_monotonic("timestamp").run(ds)
    assert not res.ok
    assert res.bad_rows == [3]


def test_no_missing_required_respects_budget() -> None:
    # 1 of 4 missing = 25% > 1% default -> fails
    ds = _ds(["20", "", "30", "40"])
    assert not no_missing_required(["temperature"]).run(ds).ok
    # generous budget -> passes
    assert no_missing_required(["temperature"], max_fraction=0.5).run(ds).ok


def test_duplicate_rate_detects_repeats() -> None:
    rows = [{"a": "1"}, {"a": "1"}, {"a": "2"}]
    ds = Dataset(["a"], rows, "dup")
    res = duplicate_rate_below().run(ds)
    assert not res.ok
    assert res.bad_rows == [1]


def test_distribution_stationary_flags_shift_and_skips_tiny() -> None:
    flat = [0.0, 1.0, 0.0, 1.0, 100.0, 101.0, 100.0, 101.0]
    ds = _ds([str(v) for v in flat])
    assert not distribution_stationary("temperature", max_shift=3.0).run(ds).ok
    # too few points -> trivially ok
    assert distribution_stationary("temperature").run(_ds(["1", "2"])).ok


# --------------------------------------------------------------------------- #
# Segment-analysis re-test classification
# --------------------------------------------------------------------------- #


def _range_check() -> Check:
    return values_in_range("temperature", -50, 150)


def test_segment_retest_localizes_a_burst() -> None:
    temps = ["20"] * 50
    for i in range(20, 30):
        temps[i] = "999"  # one contiguous segment (rows 20-29, with seg size 10)
    retest = _segment_retest(_range_check(), _ds(temps))
    assert "localized burst" in retest.conclusion
    assert "regime shift" in retest.conclusion


def test_segment_retest_detects_systemic() -> None:
    retest = _segment_retest(_range_check(), _ds(["999"] * 20))
    assert "every segment" in retest.conclusion


def test_segment_retest_boundary_only_effect() -> None:
    # Increases within each 4-row segment but resets at boundaries -> the monotonic
    # violation only shows at full scale, not inside any single segment.
    ts = [str(v) for v in [0, 1, 2, 3] * 5]
    ds = Dataset(["timestamp"], [{"timestamp": v} for v in ts], "saw")
    retest = _segment_retest(timestamps_monotonic("timestamp"), ds)
    assert "full-dataset scale" in retest.conclusion


# --------------------------------------------------------------------------- #
# Scanner end-to-end
# --------------------------------------------------------------------------- #


def test_scan_trusted_when_all_pass() -> None:
    ds = _ds([f"{20 + (i % 3)}" for i in range(20)])
    report = SelfAuditingDatasetScanner([values_in_range("temperature", -50, 150)]).scan(ds)
    assert report.trusted
    assert report.log.final_status == "trusted"
    assert report.failed_checks == []
    assert report.log.attempts[0].decision == "accept"


def test_scan_untrusted_flags_and_localizes() -> None:
    temps = ["20"] * 50
    for i in range(20, 30):
        temps[i] = "999"
    report = SelfAuditingDatasetScanner([values_in_range("temperature", -50, 150)]).scan(_ds(temps))
    assert not report.trusted
    assert report.log.final_status == "untrusted"
    assert "range[temperature]" in report.failed_checks
    flagged = report.log.attempts[0]
    assert flagged.classification == "unexpected"
    assert flagged.decision == "flag"
    seg = next(rt for rt in flagged.retests if rt.name == "segment_analysis")
    assert "localized burst" in seg.conclusion


def test_scan_accepts_a_dataset_or_a_path(tmp_path) -> None:
    p = tmp_path / "d.csv"
    p.write_text("timestamp,temperature\n0,20\n1,21\n", encoding="utf-8")
    report = SelfAuditingDatasetScanner([values_in_range("temperature", -50, 150)]).scan(str(p))
    assert report.trusted


def test_scan_log_is_json_serializable_and_renders() -> None:
    report = SelfAuditingDatasetScanner([values_in_range("temperature", -50, 150)]).scan(
        _ds(["999", "999"])
    )
    parsed = json.loads(report.log.to_json())
    assert parsed["final_status"] == "untrusted"
    assert "UNTRUSTED" in report.log.render()


def test_dataset_demo_runs_and_writes_log(tmp_path, monkeypatch, capsys) -> None:
    from selfaudit.datasetdemo import main

    monkeypatch.chdir(tmp_path)
    main()
    out = capsys.readouterr().out
    assert "SELF-AUDIT REPORT" in out
    assert "faulty-sensor window" in out or "regime shift" in out
    written = tmp_path / "dataset_audit_log.json"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["final_status"] == "untrusted"
