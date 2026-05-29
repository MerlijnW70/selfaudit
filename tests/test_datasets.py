"""Pytest suite for the dataset trust scanner (datasets + datasetscanner)."""

from __future__ import annotations

import json

import pytest

from selfaudit.datasets import (
    Check,
    Dataset,
    allowed_values,
    check_from_spec,
    checks_from_specs,
    distribution_stationary,
    dump_rules,
    duplicate_rate_below,
    humanize_check,
    infer_checks,
    infer_specs,
    iqr_outliers,
    load_csv,
    load_dataset,
    load_rules,
    no_missing_required,
    parse_csv,
    parse_json,
    parse_text,
    sample_csv_file,
    sample_dataset,
    svg_chart,
    timestamps_monotonic,
    unique_key,
    values_in_range,
    values_of_type,
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


def test_parse_csv_dedupes_duplicate_and_blank_headers() -> None:
    # Duplicate/blank headers must not collapse (data loss) — dogfood: brain_networks.
    ds = parse_csv("a,a,,b\n1,2,3,4\n")
    assert ds.columns == ["a", "a.2", "col3", "b"]
    assert ds.rows[0] == {"a": "1", "a.2": "2", "col3": "3", "b": "4"}  # all 4 values kept


def test_parse_csv_drops_all_blank_filler_rows() -> None:
    # Exported CSVs (dogfood: financial_anomaly_data.csv) carry all-comma filler
    # rows; csv.reader reports ['',''] as truthy, so they used to survive as ghost
    # rows of empty cells ("first row is empty"). They must be dropped — while a
    # row with even one real value is kept (partial data the scanner should flag).
    ds = parse_csv("ts,amount\n,\n   ,  \n2023,100\n2024,\n")
    assert ds.n == 2  # the two all-blank filler rows are gone
    assert ds.rows[0] == {"ts": "2023", "amount": "100"}
    assert ds.rows[1] == {"ts": "2024", "amount": ""}  # partial row kept (missing amount)


def test_report_bounds_wide_offending_preview() -> None:
    # A many-column offending-rows preview must scroll inside its cell, not push the
    # whole report off-screen: fixed table layout + colgroup bound the Detail column
    # and the preview keeps its own horizontal scroller.
    cols = ["id", "amount", "region", "category", "channel", "note"]
    rows = [{c: f"{c}_{i}" for c in cols} for i in range(12)]
    for r in rows:
        r["amount"] = "50"
    rows[5]["amount"] = "99999"  # an outlier so a preview is rendered
    ds = Dataset(cols, rows, "wide")
    html = SelfAuditingDatasetScanner(infer_checks(ds)).scan(ds).log.to_html()
    assert "<colgroup>" in html  # column widths are pinned
    assert "table-layout:fixed" in html  # Detail column can't expand to content width
    assert "overflow-x:auto" in html  # the wide preview scrolls within the cell


def test_humanize_check_gives_plain_titles() -> None:
    assert humanize_check("missing_required") == "Missing values"
    assert humanize_check("duplicate_rate") == "Duplicate rows"
    assert humanize_check("outliers[order_amount]") == "Unusual order_amount values"
    assert humanize_check("stationary[price]") == "price drifts over time"
    assert humanize_check("type[age=int]") == "age has the wrong type"  # field stripped of =type
    assert humanize_check("something_unmapped") == "something_unmapped"  # falls back to code name


def test_report_reads_in_plain_english_no_jargon() -> None:
    # A user-facing report must not leak engineer/statistician notation.
    ds = parse_csv(
        "amount\n" + "".join(f"{v}\n" for v in [40, 45, 50, 48, 52, 47, 99999, 41, 49, 46])
    )
    report = SelfAuditingDatasetScanner(infer_checks(ds)).scan(ds)
    html = report.log.to_html()
    assert "class='plain'" in html  # the plain one-line summary is present
    assert "Unusual amount values" in html  # humanized check title
    assert "IQR fences" not in html  # de-jargoned
    assert "σ" not in html  # de-jargoned
    import re

    assert not re.search(r"\de[+-]\d", html)  # no scientific notation like 1.67e+04


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


def test_sample_csv_file_streams_a_subset(tmp_path) -> None:
    p = tmp_path / "big.csv"
    p.write_text("id,val\n" + "".join(f"{i},{i * 2}\n" for i in range(1000)), encoding="utf-8")
    ds = sample_csv_file(str(p), 50, seed=1)
    assert ds.columns == ["id", "val"]
    assert ds.n == 50  # bounded to the sample size, not 1000
    assert "sample 50 of 1000" in ds.name
    # deterministic for a fixed seed
    assert [r["id"] for r in ds.rows] == [r["id"] for r in sample_csv_file(str(p), 50, seed=1).rows]


def test_sample_dataset_in_memory() -> None:
    big = Dataset(["x"], [{"x": str(i)} for i in range(500)], "big")
    assert sample_dataset(big, 20).n == 20
    small = Dataset(["x"], [{"x": "1"}], "small")
    assert sample_dataset(small, 20) is small  # already small -> unchanged


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
# Schema-level checks: unique key, type, allowed values
# --------------------------------------------------------------------------- #


def test_unique_key_flags_duplicate_keys() -> None:
    rows = [{"id": "A"}, {"id": "B"}, {"id": "A"}, {"id": "C"}]
    res = unique_key(["id"]).run(Dataset(["id"], rows, "k"))
    assert not res.ok
    assert res.bad_rows == [2]  # the repeat of A


def test_unique_key_composite() -> None:
    rows = [{"a": "1", "b": "x"}, {"a": "1", "b": "y"}, {"a": "1", "b": "x"}]
    res = unique_key(["a", "b"]).run(Dataset(["a", "b"], rows, "k"))
    assert res.bad_rows == [2]  # (1,x) repeats; (1,y) is distinct


def test_values_of_type_int_and_date() -> None:
    ints = Dataset(["age"], [{"age": "30"}, {"age": "3.5"}, {"age": "x"}, {"age": ""}], "t")
    res = values_of_type("age", "int").run(ints)
    assert not res.ok
    assert res.bad_rows == [1, 2]  # 3.5 and x are not int; "" ignored
    dates = Dataset(["d"], [{"d": "2023-01-01"}, {"d": "01-01-2023"}, {"d": "nope"}], "t")
    assert values_of_type("d", "date").run(dates).bad_rows == [2]


def test_values_of_type_bool_and_clean() -> None:
    bools = Dataset(["b"], [{"b": "true"}, {"b": "NO"}, {"b": "1"}, {"b": "maybe"}], "t")
    assert values_of_type("b", "bool").run(bools).bad_rows == [3]
    assert values_of_type("temperature", "float").run(_ds(["1.5", "2", "3.0"])).ok


def test_allowed_values_whitelist() -> None:
    rows = [{"s": "active"}, {"s": "churned"}, {"s": "frozen"}, {"s": ""}]
    res = allowed_values("s", ["active", "churned"]).run(Dataset(["s"], rows, "t"))
    assert not res.ok
    assert res.bad_rows == [2]  # 'frozen' not allowed; '' ignored
    assert (
        allowed_values("s", ["active", "churned"])
        .run(Dataset(["s"], [{"s": "active"}, {"s": "churned"}], "t"))
        .ok
    )


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


def test_infer_skips_monotonic_on_a_sorted_value_column() -> None:
    # A pre-sorted value column (ascending, many duplicates) — like 'price' in a
    # price-sorted export, or a day-of-month that resets. Neither monotonic nor
    # stationarity should fire (both falsely trip on merely-sorted data); only the
    # outlier check. (Dogfood finding: diamonds 'price', airquality 'Day'.)
    vals = sorted([10, 20, 30, 40, 50, 60] * 8)  # 48 rows, ascending, 6 distinct
    rows = [{"price": str(v)} for v in vals]
    kinds = {
        s["check"] for s in infer_specs(Dataset(["price"], rows, "d")) if s.get("field") == "price"
    }
    assert kinds == {"outliers"}  # no monotonic, no stationary on a sorted value column


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
# Rules file: specs <-> checks round-trip
# --------------------------------------------------------------------------- #


def test_infer_specs_are_serializable_dicts() -> None:
    rows = [{"ts": str(i), "temp": str(20 + i % 7), "tag": "A"} for i in range(40)]
    specs = infer_specs(Dataset(["ts", "temp", "tag"], rows, "s"))
    kinds = [s["check"] for s in specs]
    assert "missing_required" in kinds and "outliers" in kinds and "monotonic" in kinds
    # round-trips through JSON unchanged
    assert load_rules(dump_rules(specs)) == specs


def test_check_from_spec_builds_each_kind() -> None:
    specs: list[dict] = [
        {"check": "range", "field": "temperature", "lo": -50, "hi": 150},
        {"check": "unique", "fields": ["timestamp"]},
        {"check": "type", "field": "temperature", "type": "float"},
        {"check": "allowed", "field": "temperature", "values": ["20", "21"]},
        {"check": "missing_required", "fields": ["temperature"]},
        {"check": "duplicate_rate"},
        {"check": "outliers", "field": "temperature"},
        {"check": "monotonic", "field": "timestamp"},
        {"check": "stationary", "field": "temperature"},
    ]
    checks = checks_from_specs(specs)
    assert [c.name for c in checks][:3] == [
        "range[temperature]",
        "unique[timestamp]",
        "type[temperature=float]",
    ]
    # severity defaults applied from the spec kind
    assert checks_from_specs([{"check": "range", "field": "x", "lo": 0, "hi": 1}])[0].severity == (
        "fail"
    )
    assert checks_from_specs([{"check": "outliers", "field": "x"}])[0].severity == "warn"
    # explicit severity override survives the round-trip
    assert check_from_spec({"check": "outliers", "field": "x", "severity": "fail"}).severity == (
        "fail"
    )


def test_check_from_spec_errors() -> None:
    with pytest.raises(ValueError):
        check_from_spec({"field": "x"})  # no 'check' key
    with pytest.raises(ValueError):
        check_from_spec({"check": "nonsense"})  # unknown kind
    with pytest.raises(ValueError):
        check_from_spec({"check": "range", "field": "x"})  # missing lo/hi


def test_load_rules_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        load_rules('{"no_checks_key": 1}')
    with pytest.raises(ValueError):
        load_rules("[1, 2, 3]")  # not an object


def test_emit_then_scan_round_trip() -> None:
    # The inferred rules, saved and reloaded, scan a clean dataset to TRUSTED.
    ds = Dataset(
        ["ts", "temp"],
        [{"ts": str(i), "temp": str(20 + i % 5)} for i in range(40)],
        "rt",
    )
    reloaded = checks_from_specs(load_rules(dump_rules(infer_specs(ds))))
    report = SelfAuditingDatasetScanner(reloaded).scan(ds)
    assert report.status in ("trusted", "review")  # no hard failures on clean data


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


def test_scan_records_offending_rows() -> None:
    temps = ["20"] * 50
    for i in (20, 21, 22):
        temps[i] = "999"
    report = SelfAuditingDatasetScanner([values_in_range("temperature", -50, 150)]).scan(_ds(temps))
    # the offending row indices are on the attempt (so they reach JSON) ...
    flagged = report.log.attempts[0]
    assert flagged.rows == [20, 21, 22]
    # ... and the full set is exposed per check for export
    assert report.bad_rows["range[temperature]"] == [20, 21, 22]
    # ... and they render in the text report
    assert "offending rows: 20, 21, 22" in report.log.render()
    # ... and a sample of the actual offending rows' data is embedded for the report
    assert flagged.row_preview[0]["temperature"] == "999"
    assert "999" in report.log.to_html()  # the offending value shows in the HTML preview
    assert "sample offending rows" in report.log.to_html()


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


def test_svg_chart_marks_offending_rows_per_column() -> None:
    # The chart must agree with the verdict: a column's bad rows are marked red on
    # that column's panel only; a clean column's panel has no markers.
    rows = [{"amount": str(40 + (i * 7) % 80), "score": str(50 + i % 9)} for i in range(30)]
    rows[7]["amount"] = "99999"  # a planted outlier in amount, row 7
    ds = Dataset(["amount", "score"], rows, "t")
    svg = svg_chart(ds, bad_rows={"outliers[amount]": [7]})
    assert svg.count("<rect") == 2  # one panel per value column (small multiples)
    assert svg.count("fill='#cf222e'") == 1  # exactly the one offending point, marked
    assert "row 7:" in svg  # marker carries a hover tooltip with the row + value
    assert "offending rows" in svg  # legend key for the red markers
    # real y-axis labels (not 0-1 normalized) are present
    assert "text-anchor='end'" in svg


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
