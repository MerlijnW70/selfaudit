"""Numerical core: the problem and the interchangeable solver methods.

This file contains *no* audit logic. It only provides:

* :class:`Problem`        - the equation to solve + its constraints
* :class:`StrategyOutcome`- the raw result of one method
* :class:`Method`         - the interface the auditor uses to escalate
* the methods themselves  - ``newton`` -> ``secant`` -> ``brentq``

The methods are thin adapters over ``scipy.optimize`` (real, battle-tested
numerics). A method signals a *well-defined* failure (divergence, non-
convergence, no sign change) by raising :class:`NumericalFailure`; the auditor
in ``auditor.py`` decides what happens with it. The audit/escalation harness is
engine-agnostic — only this file knows it is scipy underneath.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

from scipy import optimize

Func = Callable[[float], float]


class NumericalFailure(Exception):
    """A method cannot continue, in a clean and explainable way."""


@dataclass
class Problem:
    """Find ``x`` such that ``|f(x)| <= tol``."""

    name: str
    f: Func
    guess: float | None = None
    bracket: tuple[float, float] | None = None
    tol: float = 1e-9
    max_iter: int = 80


@dataclass
class StrategyOutcome:
    """Raw outcome of one method (before any expectation check)."""

    candidate: float
    iterations: int
    status: str  # "converged" | "exhausted"


@dataclass
class Method:
    """Adapter that lets the auditor call a method uniformly.

    ``requires`` determines applicability: "guess" (needs an initial guess) or
    "bracket-or-guess" (needs a bracket, or can search for one from a guess).
    """

    name: str
    requires: str
    solve: Callable[[Problem], StrategyOutcome]

    def is_applicable(self, prob: Problem) -> tuple[bool, str]:
        if self.requires == "guess":
            if prob.guess is None:
                return False, "no initial guess provided"
            return True, ""
        if self.requires == "bracket-or-guess":
            if prob.bracket is None and prob.guess is None:
                return False, "no bracket and no guess to search one from"
            return True, ""
        return False, f"unknown requirement: {self.requires!r}"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _central_diff(f: Func, x: float, h: float = 1e-7) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)


def _guard(f: Func) -> Func:
    """Wrap an objective function so that an arithmetic/domain error becomes a
    clean :class:`NumericalFailure` instead of crashing the auditor.

    The methods evaluate ``f`` at far-away or degenerate points (Newton jumps,
    an expanding bracket search). An ``OverflowError`` (e.g. ``e^x`` for large
    ``x``), ``ZeroDivisionError`` or a domain ``ValueError`` (e.g. ``sqrt`` of a
    negative number) is a *well-defined* failure there — not an unexpected crash.
    """

    def safe(x: float) -> float:
        try:
            return float(f(x))
        except (ArithmeticError, ValueError) as exc:
            raise NumericalFailure(f"f not evaluable at x={x:.6g}: {exc}") from exc

    return safe


def auto_bracket(f: Func, x0: float, max_expand: int = 80) -> tuple[float, float]:
    """Search outward for an interval [a, b] containing a sign change.

    Scans *independently* to both sides of ``x0``. A symmetric interval would
    never reveal a sign change for even functions (such as ``x^2 - 2``) because
    ``f(x0-d) == f(x0+d)``; this asymmetric scan handles that.
    """
    f = _guard(f)
    f0 = f(x0)
    if not math.isfinite(f0):
        raise NumericalFailure(f"f not finite at start point x0={x0:.6g}")
    if f0 == 0.0:
        return (x0, x0)
    step = 1e-2
    left = right = x0
    for _ in range(max_expand):
        left -= step
        right += step
        fl, fr = f(left), f(right)
        if math.isfinite(fr) and f0 * fr < 0:
            return (x0, right)
        if math.isfinite(fl) and f0 * fl < 0:
            return (left, x0)
        step *= 1.6
    raise NumericalFailure(f"no sign change found around x0={x0:.6g}")


# --------------------------------------------------------------------------- #
# Methods (thin adapters over scipy.optimize)
# --------------------------------------------------------------------------- #


def _run_open(prob: Problem, *, use_derivative: bool) -> StrategyOutcome:
    """Open method via ``scipy.optimize.newton`` — Newton-Raphson (with a
    numerical derivative) when ``use_derivative`` else the secant method."""
    label = "newton" if use_derivative else "secant"
    if prob.guess is None:
        raise NumericalFailure(f"{label} requires an initial guess")
    if not math.isfinite(prob.guess):
        raise NumericalFailure(f"non-finite initial guess: {prob.guess!r}")
    f = _guard(prob.f)
    fprime = (lambda x: _central_diff(f, x)) if use_derivative else None
    try:
        with warnings.catch_warnings():
            # We inspect res.converged ourselves; scipy's non-convergence warning
            # ("Tolerance reached") is just noise here.
            warnings.simplefilter("ignore", RuntimeWarning)
            root, res = optimize.newton(
                f,
                prob.guess,
                fprime=fprime,
                tol=prob.tol,
                maxiter=prob.max_iter,
                full_output=True,
                disp=False,
            )
    except RuntimeError as exc:
        # e.g. "Derivative was zero" — a clean, explainable failure of this method.
        raise NumericalFailure(f"{label} failed: {exc}") from exc
    if not math.isfinite(root):
        raise NumericalFailure(f"{label} diverged to {root!r}")
    status = "converged" if res.converged else "exhausted"
    return StrategyOutcome(float(root), int(res.iterations), status)


def newton(prob: Problem) -> StrategyOutcome:
    """Newton-Raphson (scipy) with a numerical derivative. Fast, but fragile."""
    return _run_open(prob, use_derivative=True)


def secant(prob: Problem) -> StrategyOutcome:
    """Secant method (scipy). A bit more robust than Newton, still fragile."""
    return _run_open(prob, use_derivative=False)


def brentq(prob: Problem) -> StrategyOutcome:
    """Brent's method (scipy): robust, guaranteed convergence on a valid bracket.

    Needs an interval enclosing a sign change; if only a guess is given, an
    interval is searched for via :func:`auto_bracket`. Replaces the bisection
    fallback of the original toy engine with a faster, equally guaranteed method.
    """
    f = _guard(prob.f)
    bracket = prob.bracket
    if bracket is None:
        bracket = auto_bracket(f, prob.guess if prob.guess is not None else 0.0)
    a, b = bracket
    if a == b:
        # auto_bracket landed exactly on a root.
        return StrategyOutcome(float(a), 0, "converged")
    if f(a) * f(b) > 0:
        raise NumericalFailure(f"bracket [{a:.6g},{b:.6g}] encloses no sign change")
    try:
        root, res = optimize.brentq(
            f,
            a,
            b,
            xtol=prob.tol,
            maxiter=prob.max_iter,
            full_output=True,
            disp=False,
        )
    except (RuntimeError, ValueError) as exc:
        raise NumericalFailure(f"brentq failed: {exc}") from exc
    return StrategyOutcome(
        float(root), int(res.iterations), "converged" if res.converged else "exhausted"
    )


def default_methods() -> list[Method]:
    """The default escalation ladder: fast/fragile -> slow/guaranteed."""
    return [
        Method("newton", "guess", newton),
        Method("secant", "guess", secant),
        Method("brentq", "bracket-or-guess", brentq),
    ]
