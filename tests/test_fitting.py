"""Pytest suite for the sensor anomaly detection (fitting + diagnostician)."""

from __future__ import annotations

import json

import pytest

from selfaudit.diagnostician import AnomalyDetected, SelfAuditingFitter
from selfaudit.fitting import (
    FitResult,
    _lag1_autocorr,
    _rel_residual,
    _solve,
    default_models,
    fit_harmonic,
)
from selfaudit.signals import (
    beat_signal,
    damped_signal,
    harmonic_signal,
    noisy_harmonic_signal,
    pure_noise_signal,
    regime_shift_signal,
    three_resonance_signal,
)


@pytest.fixture
def fitter() -> SelfAuditingFitter:
    return SelfAuditingFitter()


# --------------------------------------------------------------------------- #
# "Well-behaved" signals: the right model is found, with escalation where needed
# --------------------------------------------------------------------------- #


def test_clean_harmonic_fits_immediately(fitter: SelfAuditingFitter) -> None:
    diag = fitter.diagnose(harmonic_signal(omega=3.0))
    assert diag.model == "harmonic"
    assert diag.log.final_status == "fitted"
    assert len(diag.log.attempts) == 1
    assert diag.log.attempts[0].decision == "accept"
    # frequency recovered correctly.
    assert diag.params["omega"] == pytest.approx(3.0, abs=0.05)


def test_damped_signal_escalates_to_damped_model(fitter: SelfAuditingFitter) -> None:
    diag = fitter.diagnose(damped_signal(omega=3.0, gamma=0.25))
    assert diag.model == "damped"
    assert diag.log.final_status == "fitted"
    # the first attempt (harmonic) is logged as unexpected before the successful escalation.
    first = diag.log.attempts[0]
    assert first.strategy == "harmonic"
    assert first.classification == "unexpected"
    assert first.decision == "escalate"
    assert diag.params["gamma"] > 0.0


def test_two_resonances_escalate_to_two_harmonic(fitter: SelfAuditingFitter) -> None:
    diag = fitter.diagnose(beat_signal(omega1=3.0, omega2=4.3))
    assert diag.model == "two_harmonic"
    assert diag.log.final_status == "fitted"
    assert len(diag.log.attempts) == 3  # harmonic, damped, two_harmonic


# --------------------------------------------------------------------------- #
# Discoveries: no model fits -> a physical anomaly, with a diagnosis
# --------------------------------------------------------------------------- #


def test_regime_shift_is_flagged_as_system_change(fitter: SelfAuditingFitter) -> None:
    with pytest.raises(AnomalyDetected) as excinfo:
        fitter.diagnose(regime_shift_signal(omega_before=3.0, omega_after=4.6))
    exc = excinfo.value
    assert "system change" in exc.reason
    assert exc.log.final_status == "anomaly"
    assert "DISCOVERY" in exc.log.conclusion
    # the stationarity re-test must show that each half fits separately.
    station = next(
        rt for a in exc.log.attempts for rt in a.retests if rt.name == "segment_stationarity"
    )
    assert station.checks[0].satisfied  # worst half <= tol


def test_three_resonances_flagged_as_unmodeled_phenomenon(fitter: SelfAuditingFitter) -> None:
    with pytest.raises(AnomalyDetected) as excinfo:
        fitter.diagnose(three_resonance_signal())
    exc = excinfo.value
    assert "unmodeled phenomenon" in exc.reason
    assert exc.log.final_status == "anomaly"
    # the stationarity re-test shows that each half also fits poorly on its own.
    station = next(
        rt for a in exc.log.attempts for rt in a.retests if rt.name == "segment_stationarity"
    )
    assert station.checks[0].satisfied is False  # both halves fit equally poorly


# --------------------------------------------------------------------------- #
# Re-test branches & audit trail
# --------------------------------------------------------------------------- #


def test_whiteness_caveat_on_imperfect_residual(fitter: SelfAuditingFitter) -> None:
    """The beat fit passes in rms but leaves a small residual structure -> caveat."""
    diag = fitter.diagnose(beat_signal())
    accepted = diag.log.attempts[-1]
    white = next(rt for rt in accepted.retests if rt.name == "residual_whiteness")
    assert white.checks[0].satisfied is False
    assert "residual structure" in accepted.notes


def test_bootstrap_retest_clean_signal_does_not_reproduce(fitter: SelfAuditingFitter) -> None:
    """On a clean signal the (non-existent) deviation does not reproduce under resampling."""
    model = default_models()[0]  # harmonic
    retest, reproduced = fitter._bootstrap_retest(model, harmonic_signal())
    assert reproduced is False
    assert "does NOT reproduce" in retest.conclusion


def test_fit_diagnosis_log_is_json_serializable(fitter: SelfAuditingFitter) -> None:
    diag = fitter.diagnose(harmonic_signal())
    parsed = json.loads(diag.log.to_json())
    assert parsed["final_status"] == "fitted"
    assert parsed["attempts"][0]["params"]["omega"] > 0


def test_anomaly_log_renders_with_conclusion(fitter: SelfAuditingFitter) -> None:
    with pytest.raises(AnomalyDetected) as excinfo:
        fitter.diagnose(regime_shift_signal())
    text = excinfo.value.log.render()
    assert "ANOMALY / DISCOVERY" in text
    assert "parameters:" in text  # fit parameters are shown


# --------------------------------------------------------------------------- #
# Stochastic noise: the re-test sometimes reproduces and sometimes does not
# (deterministic per seed; seeds calibrated at sigma=0.11)
# --------------------------------------------------------------------------- #


def test_noise_within_tolerance_is_directly_accepted(fitter: SelfAuditingFitter) -> None:
    """Noise realisation just within tolerance: direct accept, no re-test needed."""
    diag = fitter.diagnose(noisy_harmonic_signal(seed=1))
    assert diag.model == "harmonic"
    assert diag.log.attempts[-1].outcome == "fitted"


def test_noise_above_tolerance_is_dismissed_after_retest(fitter: SelfAuditingFitter) -> None:
    """Noise realisation just outside tolerance: the bootstrap re-test unmasks the
    deviation as noise -> the model is accepted after all (self-correction)."""
    diag = fitter.diagnose(noisy_harmonic_signal(seed=2))
    assert diag.model == "harmonic"
    accepted = diag.log.attempts[-1]
    assert accepted.outcome == "fitted-after-retest"
    assert accepted.classification == "unexpected"  # unexpected at first...
    assert accepted.decision == "accept"  # ...but accepted after the re-test
    repro = next(rt for rt in accepted.retests if rt.name == "reproduce_under_resampling")
    assert repro.reproduced_anomaly is False
    assert "noise" in diag.log.conclusion


def test_high_noise_is_called_noise_not_discovery(fitter: SelfAuditingFitter) -> None:
    """High noise that robustly fails: no false-positive discovery but the 'noise' verdict."""
    with pytest.raises(AnomalyDetected) as excinfo:
        fitter.diagnose(noisy_harmonic_signal(seed=5))
    assert excinfo.value.log.final_status == "noise"
    assert "measurement noise" in excinfo.value.reason


def test_pure_noise_is_not_a_discovery(fitter: SelfAuditingFitter) -> None:
    with pytest.raises(AnomalyDetected) as excinfo:
        fitter.diagnose(pure_noise_signal())
    assert excinfo.value.log.final_status == "noise"


def test_retest_verdict_flips_across_realisations(fitter: SelfAuditingFitter) -> None:
    """Core of the stochastic behaviour: over different noise realisations the
    re-test gives different outcomes (sometimes reproducible, sometimes not)."""
    outcomes = set()
    for seed in (1, 2, 5):
        try:
            diag = fitter.diagnose(noisy_harmonic_signal(seed=seed))
            outcomes.add(diag.log.attempts[-1].outcome)
        except AnomalyDetected as exc:
            outcomes.add(exc.log.final_status)
    # all three branches occur: direct accept, accept-after-re-test, and noise verdict.
    assert outcomes == {"fitted", "fitted-after-retest", "noise"}


def test_bootstrap_preserves_length() -> None:
    s = harmonic_signal(n=20)
    assert s.bootstrap(seed=7).n == 20


def test_noise_demo_runs(capsys) -> None:
    from selfaudit.noisedemo import main

    main(seeds=5)
    out = capsys.readouterr().out
    assert "noise realisations" in out
    assert "Pure noise" in out


# --------------------------------------------------------------------------- #
# Numerical building blocks (defensive coverage)
# --------------------------------------------------------------------------- #


def test_solve_handles_singular_matrix() -> None:
    """A singular system gives no crash but zero coefficients."""
    x = _solve([[0.0, 0.0], [0.0, 0.0]], [1.0, 2.0])
    assert x == [0.0, 0.0]


def test_rel_residual_on_flat_signal() -> None:
    """Flat signal (variance ~0): the rel-residual falls back to the absolute rms."""
    y = [2.0, 2.0, 2.0, 2.0]
    resid, rel = _rel_residual(y, [2.0, 2.0, 2.0, 2.0])
    assert rel == 0.0
    assert resid == [0.0, 0.0, 0.0, 0.0]


def test_lag1_autocorr_edge_cases() -> None:
    assert _lag1_autocorr([5.0]) == 0.0  # n < 2
    assert _lag1_autocorr([3.0, 3.0, 3.0]) == 0.0  # zero variance


def test_fit_harmonic_returns_fitresult() -> None:
    fr = fit_harmonic(harmonic_signal(omega=2.5))
    assert isinstance(fr, FitResult)
    assert fr.params["omega"] == pytest.approx(2.5, abs=0.05)
    assert 0.0 <= fr.rel_residual < 0.1


def test_timeseries_splitting() -> None:
    s = harmonic_signal(n=10)
    assert s.first_half().n == 5
    assert s.second_half().n == 5
    assert s.n == 10
