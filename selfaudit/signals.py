"""Synthetic sensor time series for the anomaly-detection demo.

Each generator yields a :class:`TimeSeries`. The "well-behaved" signals fit the
model family (harmonic / damped / two resonances); the two "discovery" signals
do not:

* ``regime_shift_signal`` — the frequency jumps halfway through (system change).
* ``three_resonance_signal`` — three resonances, more than the model can handle
  (an unexpected extra resonance remains in the residual).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class TimeSeries:
    """A sampled time series ``y(t)``."""

    t: list[float]
    y: list[float]
    label: str = ""

    @property
    def n(self) -> int:
        return len(self.t)

    def first_half(self) -> TimeSeries:
        m = self.n // 2
        return TimeSeries(self.t[:m], self.y[:m], f"{self.label}[1st half]")

    def second_half(self) -> TimeSeries:
        m = self.n // 2
        return TimeSeries(self.t[m:], self.y[m:], f"{self.label}[2nd half]")

    def bootstrap(self, seed: int) -> TimeSeries:
        """Draw (with replacement) a resample of the same length — for the
        stochastic reproducibility re-test. The timestamps stay real and the
        temporal ordering is preserved (sorted)."""
        rng = random.Random(seed)
        idx = sorted(rng.randrange(self.n) for _ in range(self.n))
        return TimeSeries(
            [self.t[i] for i in idx], [self.y[i] for i in idx], f"{self.label}[bootstrap]"
        )


def _grid(n: int, dt: float) -> list[float]:
    return [i * dt for i in range(n)]


def _noise(n: int, sigma: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0, sigma) for _ in range(n)]


def harmonic_signal(
    n: int = 300,
    dt: float = 0.06,
    omega: float = 3.0,
    amp: float = 1.0,
    phase: float = 0.4,
    offset: float = 0.2,
    noise: float = 0.02,
    seed: int = 1,
) -> TimeSeries:
    """Pure harmonic oscillator + light noise."""
    t = _grid(n, dt)
    e = _noise(n, noise, seed)
    y = [amp * math.sin(omega * ti + phase) + offset + e[i] for i, ti in enumerate(t)]
    return TimeSeries(t, y, "harmonic")


def damped_signal(
    n: int = 300,
    dt: float = 0.06,
    omega: float = 3.0,
    gamma: float = 0.25,
    amp: float = 1.0,
    noise: float = 0.02,
    seed: int = 2,
) -> TimeSeries:
    """Damped oscillator: amplitude decays exponentially."""
    t = _grid(n, dt)
    e = _noise(n, noise, seed)
    y = [amp * math.exp(-gamma * ti) * math.sin(omega * ti) + e[i] for i, ti in enumerate(t)]
    return TimeSeries(t, y, "damped")


def beat_signal(
    n: int = 300,
    dt: float = 0.06,
    omega1: float = 3.0,
    omega2: float = 4.3,
    amp1: float = 1.0,
    amp2: float = 0.8,
    noise: float = 0.02,
    seed: int = 3,
) -> TimeSeries:
    """Two resonances at once (beating) — requires the two-harmonic model."""
    t = _grid(n, dt)
    e = _noise(n, noise, seed)
    y = [
        amp1 * math.sin(omega1 * ti) + amp2 * math.sin(omega2 * ti) + e[i] for i, ti in enumerate(t)
    ]
    return TimeSeries(t, y, "two-resonances")


def regime_shift_signal(
    n: int = 300,
    dt: float = 0.06,
    omega_before: float = 3.0,
    omega_after: float = 4.6,
    amp: float = 1.0,
    noise: float = 0.02,
    seed: int = 4,
) -> TimeSeries:
    """DISCOVERY-1: the frequency jumps halfway through (system change)."""
    t = _grid(n, dt)
    e = _noise(n, noise, seed)
    m = n // 2
    y = []
    for i, ti in enumerate(t):
        w = omega_before if i < m else omega_after
        y.append(amp * math.sin(w * ti) + e[i])
    return TimeSeries(t, y, "regime-shift")


def noisy_harmonic_signal(
    n: int = 300,
    dt: float = 0.06,
    omega: float = 3.0,
    amp: float = 1.0,
    sigma: float = 0.105,
    seed: int = 1,
) -> TimeSeries:
    """Harmonic oscillator with noise close to the fit threshold.

    Depending on the noise realisation (``seed``) the fit lands just below or
    just above the tolerance — the ideal stochastic probe for the re-test.
    """
    t = _grid(n, dt)
    e = _noise(n, sigma, seed)
    y = [amp * math.sin(omega * ti) + e[i] for i, ti in enumerate(t)]
    return TimeSeries(t, y, "noisy-harmonic")


def pure_noise_signal(
    n: int = 300, dt: float = 0.06, sigma: float = 1.0, seed: int = 9
) -> TimeSeries:
    """Noise only, no signal: no model fits, but it is not a discovery."""
    t = _grid(n, dt)
    return TimeSeries(t, _noise(n, sigma, seed), "noise")


def three_resonance_signal(
    n: int = 300,
    dt: float = 0.06,
    omegas: tuple[float, float, float] = (3.0, 4.3, 5.9),
    noise: float = 0.02,
    seed: int = 5,
) -> TimeSeries:
    """DISCOVERY-2: three resonances — more than the model family can capture."""
    t = _grid(n, dt)
    e = _noise(n, noise, seed)
    y = [sum(math.sin(w * ti) for w in omegas) + e[i] for i, ti in enumerate(t)]
    return TimeSeries(t, y, "three-resonances")
