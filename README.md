# selfaudit

[![CI](https://github.com/MerlijnW70/selfaudit/actions/workflows/ci.yml/badge.svg)](https://github.com/MerlijnW70/selfaudit/actions/workflows/ci.yml)

**Know whether you can trust a dataset — in one command.**

Point `selfaudit` at a file (CSV, JSON, or Excel), a URL, or a live feed. It
figures out sensible checks for you, re-tests every anomaly to tell real problems
from noise, and gives a clear verdict — **TRUSTED**, **NEEDS REVIEW**, or
**UNTRUSTED** — with the exact rows and a plain-English reason.

**What keeps people using it:** the row-level *“where exactly is it wrong,”* it
runs **fully local / private**, and it’s **zero-config — useful on the first run,
with no setup.**

> Beta (v0.0.1) — APIs may still change.

---

## Install

```bash
git clone https://github.com/MerlijnW70/selfaudit && cd selfaudit
pip install -e .            # add ".[excel]" for .xlsx support
```

(`pip install selfaudit` from PyPI is coming; for now install from the clone.)

## 60-second start

```bash
selfaudit yourdata.csv          # scan a file — rules inferred automatically
selfaudit-serve                 # …or open a web UI: drag a CSV, see the report live
```

That's it. No rules to write, no config. You get a verdict and an explanation:

```console
$ selfaudit https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
no rules given — inferred 13 checks: missing_required, duplicate_rate, outliers[Age], ...

[missing_required]  ✗  columns over the 1.0% budget: Cabin 77.1%, Age 19.9%
[outliers[Fare]]    ⚠  53/891 Fare values beyond 3·IQR fences [-61.4, 100.3]
...
verdict: UNTRUSTED  (failures: ['missing_required']; warnings: ['outliers[Fare]'])
```

→ see a finished report: [`examples/sample-audit.html`](examples/sample-audit.html)
(download & open — verdict banner, data chart, severity chips, sortable table).

## Why not just a validation script?

| A normal script | selfaudit |
| --- | --- |
| “183 bad rows” | which rows, which column, and *what kind* of problem |
| every outlier is an error | outliers are **warnings**, not hard failures (no alert fatigue) |
| you hand-write every rule | sensible rules are **inferred from your data** |
| pass / fail | **TRUSTED / NEEDS REVIEW / UNTRUSTED**, with an audit trail |

## What it checks

**Inferred automatically** (zero config) — the statistical health of the data:

- missing-value budget (per column), duplicate rows
- outliers (robust IQR fences), sudden regime shifts
- monotonic order for index/timestamp columns

**Your rules** (add flags for a real contract) — the schema:

| Flag | Checks |
| --- | --- |
| `--range FIELD:LO:HI` | numeric value stays in `[LO, HI]` |
| `--unique FIELD[,FIELD]` | key is unique (no duplicate `customer_id`) |
| `--type FIELD:int\|float\|bool\|date` | every value parses as that type |
| `--allowed FIELD:v1,v2` | value is one of a whitelist (categorical) |
| `--missing F1,F2:0.01` | at most 1% missing in those columns |
| `--monotonic FIELD` · `--stationary FIELD` | order / no regime shift |

Each finding has a **severity**: hard problems (`--range`, `--unique`, `--type`,
missing) **fail** → UNTRUSTED; “probably fine” findings (outliers, shifts) are
**warnings** → REVIEW. Exit codes: `0` trusted/review, `1` untrusted. Use
`--strict` to fail on warnings too.

### A stable rules file (for CI)

Inference is great for a first look, but for a gate you want rules that don’t
change under you. Save them once, edit, and commit:

```bash
selfaudit data.csv --emit-rules selfaudit.json   # writes the inferred rules
# …edit selfaudit.json (tweak thresholds, add --unique/--type/--allowed rules)…
selfaudit data.csv --rules selfaudit.json         # scan against the fixed contract
```

The file is plain JSON (a `"checks"` list), version-controllable, and exactly
what you want a CI gate pinned to.

## Input formats

CSV/TSV (delimiter, encoding & header **auto-detected** — incl. `;`-separated
European files, BOM, latin-1), JSON (list of objects or column arrays), and Excel
`.xlsx` (`pip install ".[excel]"`). From a local path, a URL, or a free,
key-less live feed: `--source open-meteo | usgs | crypto`.

## Use it in CI (GitHub Action)

Gate a dataset on every pull request and get the verdict posted as a comment:

```yaml
- uses: MerlijnW70/selfaudit@v0.0.1
  with:
    dataset: data/customers.csv      # or a URL, or use: source: crypto
    args: --unique id --type age:int --strict
```

The step fails the build on `UNTRUSTED`, exposes a `verdict` output, and can
upload an HTML report. See `.github/workflows/data-trust.yml` for a full example.

## Command cheat-sheet

```bash
selfaudit data.csv                                   # zero-config scan
selfaudit data.csv --html report.html                # + shareable HTML report
selfaudit data.json                                  # JSON (and .xlsx) work too
selfaudit https://host/data.csv                       # scan by URL
selfaudit --source crypto                            # scan a live feed
selfaudit data.csv --unique id --type signup:date \
                   --allowed status:active,churned    # schema rules
selfaudit data.csv --json out.json                   # machine-readable audit log
selfaudit-serve                                      # local web UI (data stays local)
```

## How it works

`selfaudit` is built on one idea — **self-auditing**: hold an explicit
expectation, and when reality deviates, *re-test before believing it*. Each rule
is a checkable invariant; a violation triggers **segment analysis** that localizes
the anomaly and decides whether it’s a real, coherent problem or just noise. That
re-test is what keeps false positives down, and every decision lands in an audit
trail (text, JSON, or HTML).

The same engine also powers three other demos in this repo — a numerical root
finder, sensor-data anomaly detection, and an LLM-output validator — all sharing
the expectation → re-test → escalate → audit-trail loop (`python -m selfaudit.<demo>`).

## Development

```bash
pip install -e ".[dev]"
pytest -q                  # fast tests (also the anvil gate)
pytest --cov=selfaudit     # coverage, 95% floor
ruff check . && ruff format --check . && mypy .
```

CI runs all of the above on every push and pull request.

## License

MIT — see [LICENSE](LICENSE).
