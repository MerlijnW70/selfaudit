"""The Self-Auditing AI applied to sensor data: anomaly detection via model fit.

The same loop as the root finder, but the "outcome" is now how well a physical
model explains the time series:

* **expected** outcome -> ``rel-residual <= tol``: the model fits. Corroborate
  with a whiteness test (is the residual structureless?) and accept.
* **unexpected** outcome -> the model does not fit. Re-test (does the deficit
  reproduce on an independent sub-measurement? does each time segment fit on its
  own?) and escalate to a richer model.

If the whole ladder is exhausted, this is not a computation error but a
**discovery**. The diagnosis distinguishes two physical causes:

* **system change** — each time segment fits on its own, the whole does not (the
  parameters change over time: non-stationary).
* **unmodeled phenomenon** — every segment fits equally poorly: there is
  coherent structure in the data the model family cannot capture (e.g. an
  unexpected extra resonance).
"""

from __future__ import annotations

from dataclasses import dataclass

from .audit import Attempt, AuditLog, ExpectationCheck, ReTest
from .fitting import FitResult, Model, default_models
from .signals import TimeSeries


@dataclass
class FitDiagnosis:
    model: str
    params: dict[str, float]
    log: AuditLog


class AnomalyDetected(Exception):
    """No model fits: a physical anomaly/discovery. The log is attached."""

    def __init__(self, label: str, reason: str, log: AuditLog) -> None:
        super().__init__(f"'{label}': no fitting model — {reason}")
        self.reason = reason
        self.log = log


class SelfAuditingFitter:
    """Drives the model ladder, audits every fit, and diagnoses anomalies."""

    def __init__(
        self,
        models: list[Model] | None = None,
        tol_rel: float = 0.15,
        white_tol: float = 0.5,
        n_bootstrap: int = 5,
        boot_seed: int = 20250528,
    ) -> None:
        self.models = models if models is not None else default_models()
        self.tol_rel = tol_rel
        self.white_tol = white_tol
        self.n_bootstrap = n_bootstrap
        self.boot_seed = boot_seed

    def _expect(self, fit: FitResult) -> ExpectationCheck:
        return ExpectationCheck(
            name="rel_residual",
            measured=fit.rel_residual,
            threshold=self.tol_rel,
            satisfied=fit.rel_residual <= self.tol_rel,
            detail=f"rms residual = {fit.rel_residual:.3f} of the signal energy",
        )

    def _whiteness(self, fit: FitResult) -> ReTest:
        mag = abs(fit.lag1_autocorr)
        chk = ExpectationCheck(
            name="lag1_autocorr",
            measured=mag,
            threshold=self.white_tol,
            satisfied=mag <= self.white_tol,
            detail=f"lag-1 autocorrelation of the residual = {fit.lag1_autocorr:.3f}",
        )
        concl = (
            "residual looks like white noise: the model explains the structure fully"
            if chk.satisfied
            else "fit passes, but the residual retains structure: small unmodeled component"
        )
        return ReTest(
            "residual_whiteness",
            "test whether the residual is structureless (noise)",
            None,
            [chk],
            concl,
        )

    def _stationarity(
        self, model: Model, series: TimeSeries, fit: FitResult
    ) -> tuple[ReTest, bool]:
        r1 = model.fit(series.first_half()).rel_residual
        r2 = model.fit(series.second_half()).rel_residual
        seg = max(r1, r2)
        fits = seg <= self.tol_rel
        chk = ExpectationCheck(
            name="worst_half_rel",
            measured=seg,
            threshold=self.tol_rel,
            satisfied=fits,
            detail=f"worst half rel-residual = {seg:.3f} (1st={r1:.3f}, 2nd={r2:.3f})",
        )
        if fits:
            concl = (
                "each time segment fits on its OWN but the whole does not -> "
                "NON-STATIONARY: the system parameters change between the halves"
            )
        else:
            concl = (
                "both halves fit equally (poorly) -> stationary behaviour this model cannot capture"
            )
        return ReTest(
            "segment_stationarity",
            "fit each time segment separately and compare the fit quality",
            None,
            [chk],
            concl,
        ), fits

    def _bootstrap_retest(self, model: Model, series: TimeSeries) -> tuple[ReTest, bool]:
        """Stochastic re-test on a white-ish residual: does the rms exceedance
        reproduce under bootstrap resampling, or does it vanish?

        Reproduces robustly -> the measurement noise structurally exceeds the
        tolerance. Does not reproduce -> incidental noise/outlier fluctuation.
        """
        rels = [
            model.fit(series.bootstrap(self.boot_seed + b)).rel_residual
            for b in range(self.n_bootstrap)
        ]
        fails = sum(1 for r in rels if r > self.tol_rel)
        reproduced = fails > self.n_bootstrap // 2
        median = sorted(rels)[len(rels) // 2]
        chk = ExpectationCheck(
            name="bootstrap_fail_fraction",
            measured=fails / self.n_bootstrap,
            threshold=0.5,
            satisfied=not reproduced,
            detail=f"{fails}/{self.n_bootstrap} resamples fail (median rel = {median:.3f})",
        )
        concl = (
            "deviation reproduces robustly under resampling -> measurement noise too high"
            if reproduced
            else "deviation does NOT reproduce under resampling -> incidental noise; accept"
        )
        return ReTest(
            "reproduce_under_resampling",
            "refit on bootstrap resamples; does the deviation reproduce?",
            reproduced,
            [chk],
            concl,
        ), reproduced

    def diagnose(self, series: TimeSeries) -> FitDiagnosis:
        log = AuditLog(
            problem=series.label,
            tolerance=self.tol_rel,
            description=f"fit a physical model; expect rel-residual <= {self.tol_rel:g}",
        )
        best: tuple[float, bool, str] | None = None  # (rel, segments_fit, model)
        for idx, model in enumerate(self.models, start=1):
            fit = model.fit(series)
            rms = self._expect(fit)
            white = self._whiteness(fit)

            if rms.satisfied:
                note = "model explains the data"
                if not white.checks[0].satisfied:
                    note += "; small residual structure flagged"
                log.attempts.append(
                    Attempt(
                        idx,
                        model.name,
                        {},
                        "fitted",
                        None,
                        0,
                        [rms],
                        "expected",
                        [white],
                        "accept",
                        note,
                        params=fit.params,
                    )
                )
                log.finalize("fitted", None, model.name)
                return FitDiagnosis(model.name, fit.params, log)

            # Unexpected. Is the residual coherent (structural model deficit) or white (noise)?
            if white.checks[0].satisfied:
                # White residual: a richer model would only overfit the noise. Do NOT escalate.
                repro, reproduced = self._bootstrap_retest(model, series)
                if not reproduced:
                    log.attempts.append(
                        Attempt(
                            idx,
                            model.name,
                            {},
                            "fitted-after-retest",
                            None,
                            0,
                            [rms],
                            "unexpected",
                            [white, repro],
                            "accept",
                            "rms above tol, but the residual is noise and does not reproduce "
                            "-> accepted after re-test",
                            params=fit.params,
                        )
                    )
                    log.finalize("fitted", None, model.name)
                    log.conclusion = (
                        "accepted after re-test: the deviation was non-reproducible noise"
                    )
                    return FitDiagnosis(model.name, fit.params, log)
                reason = (
                    "measurement noise too high: the residual lacks coherent "
                    "structure, no physical phenomenon"
                )
                log.attempts.append(
                    Attempt(
                        idx,
                        model.name,
                        {},
                        "too-noisy",
                        None,
                        0,
                        [rms],
                        "unexpected",
                        [white, repro],
                        "reject-noise",
                        "structureless noise that robustly exceeds the tolerance",
                        params=fit.params,
                    )
                )
                log.finalize("noise", None, model.name)
                log.conclusion = "NO DISCOVERY — " + reason
                raise AnomalyDetected(series.label, reason, log)

            # Coherent residual: real model deficit -> escalate to a richer model.
            station, segments_fit = self._stationarity(model, series, fit)
            log.attempts.append(
                Attempt(
                    idx,
                    model.name,
                    {},
                    "insufficient",
                    None,
                    0,
                    [rms],
                    "unexpected",
                    [white, station],
                    "escalate",
                    "coherent residual: model does not explain the data; escalated",
                    params=fit.params,
                )
            )
            if best is None or fit.rel_residual < best[0]:
                best = (fit.rel_residual, segments_fit, model.name)

        assert best is not None  # only coherent (real) deviations reach here
        _, segments_fit, best_model = best
        if segments_fit:
            reason = "system change (non-stationary): each time segment fits, the whole does not"
        else:
            reason = (
                "unmodeled phenomenon (e.g. an unexpected extra resonance): "
                "the residual stays structured in every segment"
            )
        log.finalize("anomaly", None, best_model)
        log.conclusion = "DISCOVERY — " + reason
        raise AnomalyDetected(series.label, reason, log)
