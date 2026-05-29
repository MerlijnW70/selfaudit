"""Pytest suite for the dataset trust scanner (datasets + datasetscanner)."""

from __future__ import annotations

import json

import pytest

from selfaudit.datasets import (
    Check,
    Dataset,
    distribution_stationary,
    duplicate_rate_below,
    infer_checks,
    iqr_outliers,
    load_csv,
    load_dataset,
    no_missing_required,
    parse_csv,
    parse_json,
    parse_text,
    svg_chart,
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
# Real-world input: delimiters, encoding, no header, JSON, Excel
# --------------------------------------------------------------------------- #


def test_parse_csv_semicolon_delimiter() -> None:
    # European-style ;-delimited CSV (the #1 "my file isn't a clean comma CSV").
    ds = parse_csv("naam;leeftijd\nAda;36\nGrace;45\n")
    assert ds.columns == ["naam", "leeftijd"]
    assert ds.rows[0]["leeftijd"] == "36"


def test_parse_csv_tab_delimiter() -> None:
    ds = parse_csv("a\tb\n1\t2\n3\t4\n")
    assert ds.columns == ["a", "b"]
    assert ds.rows[1]["a"] == "3"


def test_parse_csv_headerless_numeric() -> None:
    ds = parse_csv("1,2,3\n4,5,6\n")
    assert ds.columns == ["col1", "col2", "col3"]
    assert ds.n == 2


def test_load_csv_handles_bom_and_latin1(tmp_path) -> None:
    bom = tmp_path / "bom.csv"
    bom.write_bytes("﻿city,temp\nUtrecht,20\n".encode())
    assert load_csv(str(bom)).columns == ["city", "temp"]  # BOM stripped
    lat = tmp_path / "lat.csv"
    lat.write_bytes("naam,plaats\nJosé,München\n".encode("latin-1"))
    assert load_csv(str(lat)).rows[0]["naam"] == "José"  # decoded, not crashed


def test_parse_json_list_of_objects() -> None:
    ds = parse_json('[{"a": 1, "b": 2}, {"a": 3, "b": 4}]')
    assert ds.columns == ["a", "b"]
    assert ds.rows[1]["a"] == "3"


def test_parse_json_column_arrays() -> None:
    ds = parse_json('{"time": [1, 2, 3], "temp": [20, 21, 22]}')
    assert ds.columns == ["time", "temp"]
    assert ds.n == 3
    assert ds.rows[2]["temp"] == "22"


def test_parse_json_bad_shape_raises() -> None:
    import pytest as _pytest

    with _pytest.raises(ValueError):
        parse_json('"just a string"')


def test_parse_text_dispatches_by_content_and_name() -> None:
    assert parse_text('[{"x": 1}]').columns == ["x"]  # content sniff -> JSON
    assert parse_text("x,y\n1,2\n").columns == ["x", "y"]  # -> CSV


def test_load_dataset_reads_json_and_xlsx(tmp_path) -> None:
    import pytest as _pytest

    j = tmp_path / "d.json"
    j.write_text('[{"a": 1, "b": 2}, {"a": 3, "b": 4}]', encoding="utf-8")
    assert load_dataset(str(j)).columns == ["a", "b"]

    openpyxl = _pytest.importorskip("openpyxl")
    x = tmp_path / "d.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["city", "temp"])
    ws.append(["Amsterdam", 20])
    ws.append(["Rotterdam", 21])
    wb.save(str(x))
    ds = load_dataset(str(x))
    assert ds.columns == ["city", "temp"]
    assert ds.n == 2
    assert ds.rows[1]["city"] == "Rotterdam"


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


def test_no_missing_required_reports_per_column_breakdown() -> None:
    # Two columns with different missing rates: only the over-budget ones are named,
    # worst-first, each with its own rate (the per-column breakdown).
    rows = [
        {"a": "1" if i else "", "b": "" if i < 5 else "2"}  # a: 1/10 missing, b: 5/10 missing
        for i in range(10)
    ]
    res = no_missing_required(["a", "b"]).run(Dataset(["a", "b"], rows, "m"))
    assert not res.ok
    assert res.measured == pytest.approx(0.5)  # worst column drives the measured value
    assert res.detail.index("b 50") < res.detail.index("a 10")  # worst-first, per-column rates
    # a clean dataset names the worst column for transparency
    clean = no_missing_required(["a"]).run(_ds(["1", "2", "3"], a=["x", "y", "z"]))
    assert clean.ok
    assert "within budget" in clean.detail


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
# Outlier detection + rule inference
# --------------------------------------------------------------------------- #


def test_iqr_outliers_flags_extreme_value() -> None:
    temps = [str(20 + (i % 3)) for i in range(30)]
    temps[15] = "9999"  # blatant outlier
    res = iqr_outliers("temperature").run(_ds(temps))
    assert not res.ok
    assert 15 in res.bad_rows


def test_iqr_outliers_clean_passes_and_tiny_skipped() -> None:
    assert iqr_outliers("temperature").run(_ds([str(20 + (i % 4)) for i in range(40)])).ok
    assert iqr_outliers("temperature").run(_ds(["1", "2"])).ok  # too few -> trivially ok


def test_iqr_outliers_zero_spread_is_not_flagged() -> None:
    # Zero-inflated column (>75% zeros): Q1=Q3=0 -> IQR 0 -> must NOT flag every
    # non-zero value (the Titanic 'Parch' false-positive).
    vals = ["0"] * 38 + ["5"] * 2
    res = iqr_outliers("temperature").run(_ds(vals))
    assert res.ok
    assert "zero IQR" in res.detail


def test_infer_skips_stationarity_on_an_index_column() -> None:
    # A sequential index (1..N) must get monotonic, NOT stationary (the Titanic
    # 'PassengerId' false-positive: a trending mean on an index is expected).
    rows = [{"id": str(i + 1)} for i in range(40)]
    names = [c.name for c in infer_checks(Dataset(["id"], rows, "idx"))]
    assert "monotonic[id]" in names
    assert "stationary[id]" not in names


def test_infer_checks_proposes_sensible_rules() -> None:
    # A numeric, ascending, high-cardinality column -> outliers + stationary + monotonic.
    rows = [{"ts": str(i), "temp": str(20 + (i % 7)), "tag": "A"} for i in range(40)]
    ds = Dataset(["ts", "temp", "tag"], rows, "infer")
    names = [c.name for c in infer_checks(ds)]
    assert "missing_required" in names
    assert "duplicate_rate" in names
    assert "outliers[temp]" in names
    assert "monotonic[ts]" in names  # ascending sequence detected
    assert "monotonic[temp]" not in names  # wobbly column is not treated as ordered
    # 'tag' is categorical -> no numeric checks proposed for it
    assert not any("tag" in n for n in names)


def test_infer_checks_flags_a_planted_outlier_end_to_end() -> None:
    temps = [str(20 + (i % 5)) for i in range(60)]
    temps[30] = "5000"
    rows = [{"ts": str(i), "temperature": temps[i]} for i in range(60)]
    report = SelfAuditingDatasetScanner(infer_checks(Dataset(["ts", "temperature"], rows))).scan(
        Dataset(["ts", "temperature"], rows)
    )
    # An outlier is a WARNING (often legitimate), so the verdict is REVIEW, not a
    # hard UNTRUSTED — and it shows up under warnings, not failures.
    assert report.status == "review"
    assert "outliers[temperature]" in report.warnings
    assert report.failed_checks == []


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


def test_scan_severity_review_vs_untrusted() -> None:
    # A warn-severity violation alone -> REVIEW (not a hard failure).
    warn_only = SelfAuditingDatasetScanner([iqr_outliers("temperature")]).scan(
        _ds([str(20 + i % 4) for i in range(40)] + ["9999"])
    )
    assert warn_only.status == "review"
    assert warn_only.warnings == ["outliers[temperature]"]
    assert warn_only.failed_checks == []
    assert not warn_only.trusted

    # A fail-severity violation -> UNTRUSTED, regardless of warnings.
    hard = SelfAuditingDatasetScanner(
        [values_in_range("temperature", -50, 150), iqr_outliers("temperature")]
    ).scan(_ds([str(20 + i % 4) for i in range(40)] + ["9999"]))
    assert hard.status == "untrusted"
    assert "range[temperature]" in hard.failed_checks
    assert "outliers[temperature]" in hard.warnings  # the outlier is still recorded, as a warning


def test_custom_severity_override() -> None:
    # Callers can promote a normally-warn check to a hard failure.
    strict_outliers = iqr_outliers("temperature", severity="fail")
    report = SelfAuditingDatasetScanner([strict_outliers]).scan(
        _ds([str(20 + i % 4) for i in range(40)] + ["9999"])
    )
    assert report.status == "untrusted"


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


def test_html_report_renders_verdict_and_checks(tmp_path) -> None:
    report = SelfAuditingDatasetScanner([values_in_range("temperature", -50, 150)]).scan(
        _ds(["20", "999", "21"])
    )
    html = report.log.to_html()
    assert "<!doctype html>" in html
    assert "UNTRUSTED" in html
    assert "range[temperature]" in html
    assert "sortBy" in html  # interactive: sortable findings table
    assert "chip fail" in html  # severity chip rendered for the failure
    out = tmp_path / "report.html"
    report.log.save_html(str(out))
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_svg_chart_plots_value_columns_and_skips_index() -> None:
    # temperature varies (a value series); timestamp is an ascending index (skipped).
    rows = [{"timestamp": str(i), "temperature": str(20 + (i % 7))} for i in range(40)]
    svg = svg_chart(Dataset(["timestamp", "temperature"], rows, "w"))
    assert "<svg" in svg
    assert "<polyline" in svg
    assert "temperature" in svg  # legend names the plotted series
    assert "timestamp" not in svg  # the index column is not plotted


def test_svg_chart_empty_when_nothing_to_plot() -> None:
    # all-constant / low-cardinality numeric -> no meaningful series.
    rows = [{"x": "5"} for _ in range(20)]
    assert svg_chart(Dataset(["x"], rows, "flat")) == ""


def test_to_html_embeds_chart_when_provided() -> None:
    report = SelfAuditingDatasetScanner([values_in_range("temperature", -50, 150)]).scan(
        _ds(["20", "21", "22"])
    )
    html = report.log.to_html(chart="<svg id='demo'></svg>")
    assert "<svg id='demo'></svg>" in html
    assert "class='chart'" in html


def test_html_report_escapes_content() -> None:
    # A column name with HTML metacharacters must be escaped, not injected.
    rows = [{"<x>": "1"}, {"<x>": "1"}]
    report = SelfAuditingDatasetScanner([duplicate_rate_below()]).scan(
        Dataset(["<x>"], rows, "<b>evil</b>")
    )
    html = report.log.to_html()
    assert "<b>evil</b>" not in html
    assert "&lt;b&gt;evil&lt;/b&gt;" in html


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
