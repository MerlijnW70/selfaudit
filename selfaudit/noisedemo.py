"""Demo: ``python -m selfaudit.noisedemo``.

A Monte-Carlo trial showing that the re-test is *stochastic*: over many noise
realisations of the same signal, the outcome sometimes lands within tolerance
(direct accept), sometimes just outside, after which the bootstrap re-test
unmasks the deviation as noise (accept after re-test). Pure noise yields the
'no discovery' verdict — no false-positive physics.
"""

from __future__ import annotations

from collections import Counter

from .diagnostician import AnomalyDetected, SelfAuditingFitter
from .signals import noisy_harmonic_signal, pure_noise_signal


def _classify(fitter: SelfAuditingFitter, seed: int) -> str:
    try:
        diag = fitter.diagnose(noisy_harmonic_signal(seed=seed))
    except AnomalyDetected as exc:
        return f"{exc.log.final_status}-verdict"
    out = diag.log.attempts[-1].outcome
    return "direct-accept" if out == "fitted" else "accept-after-re-test (noise)"


def main(seeds: int = 12) -> None:
    fitter = SelfAuditingFitter()
    tally: Counter[str] = Counter()
    for seed in range(1, seeds + 1):
        tally[_classify(fitter, seed)] += 1

    print(f"Re-test over {seeds} noise realisations (noisy-harmonic, sigma=0.105):")
    for outcome, count in tally.most_common():
        print(f"  {outcome:28} {count:3d}x")
    print()
    print("The re-test thus sometimes reproduces and sometimes does not — exactly the point.")
    print()

    try:
        fitter.diagnose(pure_noise_signal())
    except AnomalyDetected as exc:
        print(f"Pure noise -> {exc.log.final_status}: {exc.reason}")


if __name__ == "__main__":  # pragma: no cover
    main()
