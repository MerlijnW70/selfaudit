# Self-Auditing AI — a self-correcting algorithm

An experiment in *self-correcting algorithms*: a system that holds an
**expectation** about its own outcome and, the moment the actual outcome
deviates from it, **automatically starts a re-test + validation cycle** — all
fully documented in an audit trail.

The "engine" is a numerical root finder, because there the expectation is a
hard, objectively testable invariant: `|f(x)| <= tol`.

## Core idea

| Concept | Concrete in this project |
| --- | --- |
| Expectation | invariant `|f(x)| <= tol` |
| Unexpected outcome | invariant violated **or** method diverges/cycles |
| Re-test | rerun from a perturbed start → does the anomaly reproduce? |
| Corroboration | on success: independent re-evaluation + sign-change check |
| Self-correction | escalate `newton → secant → brentq` (fast/fragile → slow/guaranteed) |
| Proof | `audit_log.json` + a readable report |

## Architecture

Shared core:
- `selfaudit/audit.py` — data structures of the audit trail (JSON + report).

Application 1 — root finder:
- `selfaudit/solver.py` — numerical core (real `scipy.optimize` methods) behind the
  `Method` interface; the audit/escalation harness is engine-agnostic.
- `selfaudit/auditor.py` — the Self-Auditing controller (`SelfAuditingSolver`).
- `selfaudit/__main__.py` — demo runner that produces `audit_log.json`.
- `tests/test_solver.py` — pytest suite that challenges the solver.

Application 2 — anomaly detection in sensor data (physics/engineering):
- `selfaudit/signals.py` — synthetic sensor time series (`TimeSeries`).
- `selfaudit/fitting.py` — physical models + least-squares fit (`Model`).
- `selfaudit/diagnostician.py` — the Self-Auditing fitter (`SelfAuditingFitter`).
- `selfaudit/sensordemo.py` — demo runner that produces `sensor_audit_log.json`.
- `tests/test_fitting.py` — pytest suite for the anomaly detection.

Application 3 — LLM output validation:
- `selfaudit/llm.py` — pluggable model tiers (`AnthropicCaller` / `ScriptedCaller`
  behind the `ModelCaller` interface) + a pure-stdlib JSON-schema validator.
- `selfaudit/llmauditor.py` — the Self-Auditing validator (`SelfAuditingValidator`).
- `selfaudit/llmdemo.py` — demo runner that produces `llm_audit_log.json`.
- `tests/test_llm.py` — pytest suite (deterministic; no API key needed).

Application 4 — dataset trust scanning:
- `selfaudit/datasets.py` — pure-stdlib CSV `Dataset` + pluggable rule checks
  (range, monotonic timestamps, missing-values budget, duplicate rate, regime shift).
- `selfaudit/datasetscanner.py` — the Self-Auditing scanner (`SelfAuditingDatasetScanner`).
- `selfaudit/datasetdemo.py` — demo runner that produces `dataset_audit_log.json`.
- `tests/test_datasets.py` — pytest suite.

## Requirements

- Python ≥ 3.10
- [`scipy`](https://scipy.org/) ≥ 1.10 — the root-finder engine (App 1). App 2 is
  pure standard library.
- Dev/quality gates: `pytest`, `pytest-cov`, `ruff`, `mypy` (run via the ANVIL gate,
  see below). Install with `pip install -e ".[dev]"`.

## Usage

```bash
pip install -e ".[dev]"          # installs scipy + dev gate tools (pytest, cov, ruff, mypy)
python -m selfaudit              # root finder: 6 scenarios, writes audit_log.json
python -m selfaudit.sensordemo   # sensor anomaly: 5 scenarios, writes sensor_audit_log.json
python -m selfaudit.noisedemo    # stochastic noise: Monte-Carlo over the re-test
python -m selfaudit.llmdemo      # LLM validation: 4 scripted scenarios (+ live run if a key is set)
python -m selfaudit.datasetdemo  # dataset scan: planted faulty-sensor window, writes dataset_audit_log.json
python -m selfaudit.scan FILE.csv --range temperature:-50:150 --monotonic timestamp  # scan a real CSV
python -m selfaudit.livedemo     # fetch & scan FREE live data (Open-Meteo + USGS); no API key
pytest -q                        # test suite
pytest --cov=selfaudit -q        # coverage (gate floor: 95%)
ruff check . && ruff format --check . && mypy .   # anvil gates
```

## Scenarios in the demo

1. **Smooth** — `x² − 2 = 0`: Newton finds √2 right away, corroboration succeeds.
2. **Self-correction** — `x³ − 2x + 2 = 0` from `x₀ = 0`: Newton *cycles* (0→1→0→…),
   is classified as unexpected, re-tested, and the system escalates to a method
   that does find the real root (≈ −1.769).
3. **Caveat** — `x² = 0`: the residual is satisfied, but the re-test notes that
   there is no sign change (tangent point / double root) and accepts with a caveat.

## Application 2: from computation error to discovery

The same mechanism, now on sensor time series. The invariant is the fit quality
(`rel-residual ≤ tol`); escalation runs `harmonic → damped → two-resonance`.

| Mapping | root finder | sensor anomaly |
| --- | --- | --- |
| invariant | `\|f(x)\| ≤ tol` | `rel-residual ≤ tol` |
| unexpected | invariant violated | model does not explain the data |
| re-test | perturbed initial guess | **refit per time segment** + bootstrap resampling |
| terminal failure | `unsolved` | **ANOMALY / DISCOVERY** + diagnosis |

The point: if no model fits, that is not a computation error but physics outside
your model. The diagnostician distinguishes two causes via the segment re-test:

- **System change (non-stationary)** — each time segment fits on its own, the
  whole does not (the parameters drift over time). Demo: `regime_shift_signal`.
- **Unmodeled phenomenon** — every segment fits equally poorly: there is coherent
  structure the model family cannot capture (e.g. an unexpected extra resonance).
  Demo: `three_resonance_signal`.

The well-behaved signals are recognised cleanly: `harmonic_signal` (directly),
`damped_signal` (escalates to the damped model), `beat_signal` (escalates to two
resonances).

### Stochastic noise: the re-test as self-correction

With noise close to the fit threshold (`noisy_harmonic_signal`, `sigma=0.105`)
the re-test becomes *consequential* — it changes the decision, and sometimes
reproduces and sometimes does not, depending on the noise realisation:

- **direct-accept** — the fit lands just within tolerance; no re-test needed.
- **accept-after-re-test** — the fit lands just outside tolerance, but the
  residual is white and the **bootstrap re-test** cannot reproduce the deviation
  → attributed to noise → the model is accepted after all (the "self-correcting" step).
- **noise verdict** — the deviation reproduces robustly but the residual stays
  white: `NO DISCOVERY (noise)` — no false-positive physics.

The whiteness test (structured vs. white residual) is the main discriminator;
the bootstrap only runs when the residual is white-ish (possibly noise). Pure
noise (`pure_noise_signal`) therefore always yields the noise verdict, never a
"discovery".

## Application 3: the same loop, applied to LLM output

The third application proves the harness is engine-agnostic: the numerical core
is swapped for **language-model calls**, and `audit.py` + the controller pattern
are reused unchanged. The invariant becomes a checkable validator on the output
(here: "parses as a JSON object with the required keys and types"), and the
escalation ladder becomes a sequence of model tiers.

| Mapping | root finder | LLM validation |
| --- | --- | --- |
| invariant | `\|f(x)\| ≤ tol` | `validator(output)` passes (0 violations) |
| escalation ladder | `newton → secant → brentq` | `haiku → sonnet → opus` |
| unexpected | invariant violated | output fails validation |
| re-test (reproduce) | perturbed start | **retry the same tier** — flaky or deterministic? |
| corroborate success | sign-change check | **re-call the tier** — reproducible or flaky? |
| terminal failure | `unsolved` | `unvalidated` (all tiers exhausted) |

The controller distinguishes a *flaky* failure (the retry validates → a sampling
fluke, accepted after re-test, no escalation) from a *structural* one (the retry
fails again → the tier is too weak → escalate to a stronger model). Model tiers
are pluggable: `AnthropicCaller` hits the real API (lazy SDK import; a missing
key or transport error degrades cleanly to an escalation, never a crash), while
`ScriptedCaller` makes the whole audit/escalation logic deterministically
testable offline. The live run in `llmdemo.py` activates only when
`ANTHROPIC_API_KEY` is set.

Behind a **TLS-intercepting proxy** (corporate networks that re-sign HTTPS),
Python may reject the proxy's CA with `CERTIFICATE_VERIFY_FAILED`. Point
verification at your corporate root CA via the `SSL_CERT_FILE` env var (or
`AnthropicCaller(..., ca_bundle=...)`). TLS verification is never disabled — only
the trust anchor changes.

## Application 4: the same loop, applied to dataset trust

The fourth application turns the harness into a **dataset trust scanner**: it
checks a table (e.g. a CSV of sensor readings) against explicit rules, re-tests
every violation, and writes an audit trail explaining what can and cannot be
trusted. Each rule is a checkable invariant; a violation is the unexpected
outcome; the re-test is **segment analysis** — re-running the failing check on
each row-segment to localize and classify the anomaly.

| Mapping | root finder | dataset scan |
| --- | --- | --- |
| invariant | `\|f(x)\| ≤ tol` | each rule check passes (range, monotonic, missing < 1%, …) |
| unexpected | invariant violated | a check fails |
| re-test | perturbed start | **segment analysis** — which rows reproduce the violation? |
| classification | — | localized burst (regime shift / faulty window) vs systemic vs boundary |
| verdict | `solved` / `unsolved` | `trusted` / `untrusted` |

Where a normal script reports *"found 183 bad rows"*, the scanner reports *which*
rows, *whether* the anomaly is localized or systemic, and *what kind* of problem
it is — then writes it all to an audit log. The demo plants a stuck-high sensor
fault in rows 420–479 and the segment-analysis re-test pins it back to exactly
that window. Checks are pluggable (the `Check` interface), so business or
scientific rules drop straight in.

A command-line front end scans a real CSV and exits `0` (trusted) or `1`
(untrusted), so it drops straight into a CI pipeline:

```bash
python -m selfaudit.scan readings.csv \
    --range temperature:-50:150 \
    --monotonic timestamp \
    --missing temperature,sensor_id:0.01 \
    --duplicates 0 \
    --stationary temperature:3 \
    --json audit.json
```

## Live data sources (free, no API key)

The scanner can pull **real-time public data** instead of a local file — pure
stdlib `urllib`, TLS routed through the OS trust store so it works behind a
corporate proxy, and a clean `SourceUnavailable` (never a crash) when offline:

- `open_meteo(lat, lon)` — hourly 2 m temperature forecast (Open-Meteo).
- `usgs_earthquakes(period)` — recent earthquakes (USGS GeoJSON feed).

`python -m selfaudit.livedemo` fetches both and scans them live. A representative
run: the **Open-Meteo** temperatures pass every rule (`TRUSTED`), while the
**USGS** feed is flagged `UNTRUSTED` because its timestamps decrease — segment
analysis classifies it as "systemic, dataset-wide". That is not a data error: the
USGS feed is documented as newest-first, so the scanner surfaced and explained a
real ordering property automatically.

The CLI scans a live source directly — pass `--source` instead of a CSV path:

```bash
python -m selfaudit.scan --source open-meteo --lat 52.37 --lon 4.90 \
    --range temperature:-50:60 --monotonic epoch        # -> TRUSTED (exit 0)
python -m selfaudit.scan --source usgs --monotonic time --range mag:-2:10
                                                        # -> UNTRUSTED (exit 1)
```

## License

MIT — see [LICENSE](LICENSE).
