"""Demo: ``python -m selfaudit.sensordemo``.

Runs the Self-Auditing AI over five sensor time series (three "well-behaved",
two with a hidden physical phenomenon), prints the audit reports, and writes the
report of the system change to ``sensor_audit_log.json``.
"""

from __future__ import annotations

from .audit import enable_utf8_output
from .diagnostician import AnomalyDetected, SelfAuditingFitter
from .signals import (
    TimeSeries,
    beat_signal,
    damped_signal,
    harmonic_signal,
    regime_shift_signal,
    three_resonance_signal,
)


def scenarios() -> list[TimeSeries]:
    return [
        harmonic_signal(),
        damped_signal(),
        beat_signal(),
        regime_shift_signal(),
        three_resonance_signal(),
    ]


def main() -> None:
    enable_utf8_output()
    fitter = SelfAuditingFitter()
    saved = False
    for series in scenarios():
        try:
            log = fitter.diagnose(series).log
        except AnomalyDetected as exc:
            log = exc.log
        print(log.render())
        print()
        if not saved and series.label == "regime-shift":
            log.save("sensor_audit_log.json")
            print("sensor_audit_log.json written (scenario: system change).")
            saved = True


if __name__ == "__main__":  # pragma: no cover
    main()
