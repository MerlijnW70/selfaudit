# selfaudit

[![CI](https://github.com/MerlijnW70/selfaudit/actions/workflows/ci.yml/badge.svg)](https://github.com/MerlijnW70/selfaudit/actions/workflows/ci.yml)

**Know whether you can trust a dataset — in one command.**

Point `selfaudit` at any CSV (a file, a URL, or a live feed). It figures out the
checks for you, re-tests every anomaly to tell real problems from noise, and
gives a clear verdict — **TRUSTED**, **NEEDS REVIEW**, or **UNTRUSTED** — with the
exact rows and a plain-English reason. No rules to write, no config.

```bash
git clone https://github.com/MerlijnW70/selfaudit && cd selfaudit
pip install -e .

selfaudit yourdata.csv
```

> Beta (v0.0.1) — APIs may still change.

## Why

A normal validation script tells you *“183 bad rows.”* That’s not actionable.
`selfaudit` tells you **which** rows, **what kind** of problem, and **whether it’s
a real issue or just noise** — then writes it to a shareable audit trail.

| A normal script | selfaudit |
| --- | --- |
| “183 bad rows” | “`Cabin` 77% missing; `Fare` has 53 extreme values in rows 28–857” |
| every outlier is an error | outliers are **warnings**, not hard failures (no alert fatigue) |
| pass / fail | **TRUSTED / NEEDS REVIEW / UNTRUSTED**, with evidence |
| you write the rules | rules are **inferred from your data** |

## See it on real data

```console
$ selfaudit https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
no rules given — inferred 13 checks: missing_required, duplicate_rate, outliers[Age], ...

[missing_required]  ✗  columns over the 1.0% budget: Cabin 77.1%, Age 19.9%
[outliers[Fare]]    ⚠  53/891 Fare values beyond 3·IQR fences [-61.4, 100.3]
...
verdict: UNTRUSTED  (failures: ['missing_required']; warnings: ['outliers[SibSp]', 'outliers[Fare]'])
```

Zero config, real public data — it surfaces the genuine issues (missing
`Cabin`/`Age`, the famous extreme fares) and *doesn’t* cry wolf on the index
column or zero-inflated counts.

**Want the visual?** [`examples/sample-audit.html`](examples/sample-audit.html)
is a finished report — download it and open in a browser (verdict banner,
severity chips, a sortable findings table, collapsible re-test details).
Regenerate it any time with `selfaudit examples/sample.csv --html examples/sample-audit.html`.

## What you get

- **Zero-config** — `selfaudit data.csv` infers checks (missing-value budgets,
  duplicates, outliers, regime shifts, monotonic ids/timestamps).
- **Explainable** — exact rows, per-column breakdowns, and a re-test that says
  whether an anomaly is a localized burst or systemic.
- **Low-noise** — severity levels: hard problems fail, “probably fine” findings
  are warnings. The verdict means *needs review*, not *broken*.
- **Anywhere your data is** — a local file, a URL, or a free live feed
  (`--source open-meteo|usgs|crypto`), no API key.
- **Shareable** — `--html report.html` writes a colour-coded report.
- **CI-ready** — exits `0` (trusted/review) / `1` (untrusted); `--strict` fails
  on warnings too.

## Use it in your pipeline (GitHub Action)

`selfaudit` is also a GitHub Action — gate a dataset and get the verdict posted
on the pull request:

```yaml
- uses: MerlijnW70/selfaudit@v0.0.1
  with:
    dataset: data/customers.csv   # or a URL, or use: source: crypto
    args: --strict                # optional
```

It fails the check on `UNTRUSTED`, exposes a `verdict` output, and can upload an
HTML report. See `.github/workflows/data-trust.yml` for a PR-comment example.

## Common commands

```bash
selfaudit data.csv                                  # zero-config scan
selfaudit data.csv --html report.html               # + shareable HTML report
selfaudit https://host/data.csv                      # scan a CSV by URL
selfaudit --source crypto                            # scan a free live feed
selfaudit data.csv --range temperature:-50:150 \
                   --missing id,email:0.01           # explicit rules
selfaudit data.csv --strict                          # warnings fail too
```

## How it works

`selfaudit` is built on one idea — **self-auditing**: hold an explicit
expectation, and when reality deviates, *re-test before believing it*. For a
dataset, each rule is a checkable invariant; a violation triggers **segment
analysis** that localizes the anomaly and decides whether it’s a real, coherent
problem or just noise. That re-test step is what keeps false positives down — and
every decision lands in an audit trail (text, JSON, or HTML).

The same engine powers three other demos in this repo — a numerical root finder,
sensor-data anomaly detection, and an LLM-output validator — all sharing the
expectation → re-test → escalate → audit-trail loop. They’re runnable via
`python -m selfaudit.<demo>` (see `selfaudit/`), but the dataset scanner is the
tool most people want.

## Requirements & development

- Python ≥ 3.10. Runtime deps: `scipy` (only the root-finder demo uses it).
- Dev: `pip install -e ".[dev]"`, then `pytest` (fast) or `pytest --cov=selfaudit`
  (the 95% coverage gate). Lint/type: `ruff check . && ruff format --check . && mypy .`.
- CI runs all of the above on every push and PR.

## License

MIT — see [LICENSE](LICENSE).
