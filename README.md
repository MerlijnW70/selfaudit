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

## Requirements

- Python ≥ 3.10
- [`scipy`](https://scipy.org/) ≥ 1.10 — the root-finder engine (App 1). App 2 is
  pure standard library.
- Dev/quality gates: `pytest`, `ruff`, `mypy` (run via the ANVIL gate, see below).

## Usage

```bash
pip install -e .                 # installs scipy (the root-finder engine)
python -m selfaudit              # root finder: 6 scenarios, writes audit_log.json
python -m selfaudit.sensordemo   # sensor anomaly: 5 scenarios, writes sensor_audit_log.json
python -m selfaudit.noisedemo    # stochastic noise: Monte-Carlo over the re-test
pytest -q                        # test suite
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

## License

MIT — see [LICENSE](LICENSE).
