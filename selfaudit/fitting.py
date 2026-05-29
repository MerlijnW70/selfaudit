"""Physical model fitting for sensor time series.

The nonlinear parameters (frequency ``omega``, damping factor ``gamma``) are
found via a grid search; for each candidate the *linear* amplitudes are solved
with least squares (normal equations). Pure stdlib.

A :class:`Model` provides ``fit(series) -> FitResult``; the diagnostician in
``diagnostician.py`` escalates over a ladder of models.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from .signals import TimeSeries

_OMEGA_STEPS = 200
_OMEGA_REFINE = 30
_GAMMA_STEPS = 10
_GAMMA_MAX = 0.8


@dataclass
class FitResult:
    """Outcome of one model fit on one time series."""

    model: str
    params: dict[str, float]
    rel_residual: float  # rms(residual) / rms(y - mean): 0 = perfect, 1 = worthless
    lag1_autocorr: float  # structure in the residual: ~0 = noise, ~1 = missed signal
    residuals: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Linear algebra (small systems)
# --------------------------------------------------------------------------- #


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve ``A x = b`` via Gauss-Jordan with partial pivoting (small k)."""
    k = len(rhs)
    a = [row[:] for row in matrix]
    x = rhs[:]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-15:
            continue
        a[col], a[pivot] = a[pivot], a[col]
        x[col], x[pivot] = x[pivot], x[col]
        diag = a[col][col]
        for r in range(k):
            if r == col:
                continue
            factor = a[r][col] / diag
            for c in range(col, k):
                a[r][c] -= factor * a[col][c]
            x[r] -= factor * x[col]
    return [x[i] / a[i][i] if abs(a[i][i]) > 1e-15 else 0.0 for i in range(k)]


def _lstsq(columns: list[list[float]], y: list[float], ridge: float = 1e-9) -> list[float]:
    """Least-squares coefficients via the normal equations (+ small ridge)."""
    k = len(columns)
    n = len(y)
    gram = [
        [sum(columns[i][p] * columns[j][p] for p in range(n)) for j in range(k)] for i in range(k)
    ]
    for i in range(k):
        gram[i][i] += ridge
    grad = [sum(columns[i][p] * y[p] for p in range(n)) for i in range(k)]
    return _solve(gram, grad)


def _evaluate(columns: list[list[float]], coeffs: list[float]) -> list[float]:
    n = len(columns[0])
    return [sum(coeffs[j] * columns[j][i] for j in range(len(columns))) for i in range(n)]


def _rel_residual(y: list[float], yhat: list[float]) -> tuple[list[float], float]:
    n = len(y)
    resid = [y[i] - yhat[i] for i in range(n)]
    rms_r = math.sqrt(sum(r * r for r in resid) / n)
    mean = sum(y) / n
    rms_y = math.sqrt(sum((v - mean) ** 2 for v in y) / n)
    rel = rms_r / rms_y if rms_y > 1e-12 else rms_r
    return resid, rel


def _lag1_autocorr(resid: list[float]) -> float:
    n = len(resid)
    if n < 2:
        return 0.0
    mean = sum(resid) / n
    num = sum((resid[i] - mean) * (resid[i - 1] - mean) for i in range(1, n))
    den = sum((r - mean) ** 2 for r in resid)
    return num / den if den > 1e-15 else 0.0


# --------------------------------------------------------------------------- #
# Frequency search
# --------------------------------------------------------------------------- #


def _omega_grid(t: list[float], count: int) -> list[float]:
    n = len(t)
    span = t[-1] - t[0]
    dt = span / (n - 1)
    w_min = 2.0 * math.pi / span  # one full cycle over the record
    w_max = 0.95 * math.pi / dt  # just below Nyquist
    return [w_min + (w_max - w_min) * k / (count - 1) for k in range(count)]


def _best_sinusoid(t: list[float], y: list[float], grid: list[float]) -> tuple[float, float]:
    """Find the frequency with the smallest residual (sin/cos/offset basis)."""
    best_omega = grid[0]
    best_rel = float("inf")
    for omega in grid:
        cols = [
            [math.sin(omega * ti) for ti in t],
            [math.cos(omega * ti) for ti in t],
            [1.0] * len(t),
        ]
        coeffs = _lstsq(cols, y)
        _, rel = _rel_residual(y, _evaluate(cols, coeffs))
        if rel < best_rel:
            best_rel, best_omega = rel, omega
    return best_omega, best_rel


def _scan_frequency(t: list[float], y: list[float]) -> float:
    """Coarse grid + local refinement around the best frequency."""
    grid = _omega_grid(t, _OMEGA_STEPS)
    omega, _ = _best_sinusoid(t, y, grid)
    step = grid[1] - grid[0]
    fine = [
        max(1e-6, omega - step + 2.0 * step * k / (_OMEGA_REFINE - 1)) for k in range(_OMEGA_REFINE)
    ]
    omega, _ = _best_sinusoid(t, y, fine)
    return omega


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


@dataclass
class Model:
    name: str
    complexity: int
    fit: Callable[[TimeSeries], FitResult]


def fit_harmonic(series: TimeSeries) -> FitResult:
    """y = A·sin(ωt) + B·cos(ωt) + C."""
    t, y = series.t, series.y
    omega = _scan_frequency(t, y)
    cols = [[math.sin(omega * ti) for ti in t], [math.cos(omega * ti) for ti in t], [1.0] * len(t)]
    coeffs = _lstsq(cols, y)
    resid, rel = _rel_residual(y, _evaluate(cols, coeffs))
    params = {
        "omega": omega,
        "amp": math.hypot(coeffs[0], coeffs[1]),
        "offset": coeffs[2],
    }
    return FitResult("harmonic", params, rel, _lag1_autocorr(resid), resid)


def _best_damped(
    t: list[float], y: list[float], gammas: list[float], omegas: list[float]
) -> tuple[float, float, float]:
    best = (gammas[0], omegas[0], float("inf"))
    for gamma in gammas:
        env = [math.exp(-gamma * ti) for ti in t]
        for omega in omegas:
            cols = [
                [env[i] * math.sin(omega * ti) for i, ti in enumerate(t)],
                [env[i] * math.cos(omega * ti) for i, ti in enumerate(t)],
                [1.0] * len(t),
            ]
            _, rel = _rel_residual(y, _evaluate(cols, _lstsq(cols, y)))
            if rel < best[2]:
                best = (gamma, omega, rel)
    return best


def fit_damped(series: TimeSeries) -> FitResult:
    """y = e^(-γt)·(A·sin(ωt) + B·cos(ωt)) + C (coarse grid + local refinement)."""
    t, y = series.t, series.y
    coarse_omega = _omega_grid(t, _OMEGA_STEPS // 2)
    coarse_gamma = [_GAMMA_MAX * g / (_GAMMA_STEPS - 1) for g in range(_GAMMA_STEPS)]
    gamma, omega, _ = _best_damped(t, y, coarse_gamma, coarse_omega)

    d_omega = coarse_omega[1] - coarse_omega[0]
    d_gamma = coarse_gamma[1] - coarse_gamma[0]
    fine_omega = [max(1e-6, omega - d_omega + 2.0 * d_omega * k / 14) for k in range(15)]
    fine_gamma = [max(0.0, gamma - d_gamma + 2.0 * d_gamma * k / 4) for k in range(5)]
    gamma, omega, _ = _best_damped(t, y, fine_gamma, fine_omega)
    env = [math.exp(-gamma * ti) for ti in t]
    cols = [
        [env[i] * math.sin(omega * ti) for i, ti in enumerate(t)],
        [env[i] * math.cos(omega * ti) for i, ti in enumerate(t)],
        [1.0] * len(t),
    ]
    coeffs = _lstsq(cols, y)
    resid, rel = _rel_residual(y, _evaluate(cols, coeffs))
    params = {
        "omega": omega,
        "gamma": gamma,
        "amp": math.hypot(coeffs[0], coeffs[1]),
    }
    return FitResult("damped", params, rel, _lag1_autocorr(resid), resid)


def fit_two_harmonic(series: TimeSeries) -> FitResult:
    """y = Σ_{k=1,2} A_k·sin(ω_k t) + B_k·cos(ω_k t) + C (matching pursuit)."""
    t, y = series.t, series.y
    omega1 = _scan_frequency(t, y)
    cols1 = [
        [math.sin(omega1 * ti) for ti in t],
        [math.cos(omega1 * ti) for ti in t],
        [1.0] * len(t),
    ]
    resid1, _ = _rel_residual(y, _evaluate(cols1, _lstsq(cols1, y)))
    omega2 = _scan_frequency(t, resid1)
    cols = [
        [math.sin(omega1 * ti) for ti in t],
        [math.cos(omega1 * ti) for ti in t],
        [math.sin(omega2 * ti) for ti in t],
        [math.cos(omega2 * ti) for ti in t],
        [1.0] * len(t),
    ]
    coeffs = _lstsq(cols, y)
    resid, rel = _rel_residual(y, _evaluate(cols, coeffs))
    params = {
        "omega": omega1,
        "omega2": omega2,
        "amp1": math.hypot(coeffs[0], coeffs[1]),
        "amp2": math.hypot(coeffs[2], coeffs[3]),
    }
    return FitResult("two_harmonic", params, rel, _lag1_autocorr(resid), resid)


def default_models() -> list[Model]:
    """The escalation ladder: single -> damped -> two resonances."""
    return [
        Model("harmonic", 1, fit_harmonic),
        Model("damped", 2, fit_damped),
        Model("two_harmonic", 3, fit_two_harmonic),
    ]
