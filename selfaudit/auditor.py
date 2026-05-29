"""The Self-Auditing AI: the controller around the numerical methods.

The auditor runs a method, checks the outcome against the *expectation* (the
invariant ``|f(x)| <= tol``), and acts on what it sees:

* **expected** outcome   -> corroborate with an independent re-test, accept.
* **unexpected** outcome -> re-test to reproduce the anomaly, then force the
  next (more robust) method.

Everything is recorded in an :class:`AuditLog`, the proof that the audit took
place.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .audit import Attempt, AuditLog, ExpectationCheck, ReTest
from .solver import (
    Method,
    NumericalFailure,
    Problem,
    StrategyOutcome,
    default_methods,
)


@dataclass
class Solution:
    root: float
    strategy: str
    log: AuditLog


class SolveFailed(Exception):
    """No strategy satisfied the expectation. The log is attached."""

    def __init__(self, problem: str, log: AuditLog) -> None:
        super().__init__(f"'{problem}' not solved — all strategies exhausted")
        self.log = log


def _invariant(prob: Problem, x: float) -> ExpectationCheck:
    """The expectation: the residual ``|f(x)|`` stays below the tolerance."""
    residual = abs(prob.f(x))
    return ExpectationCheck(
        name="residual_below_tol",
        measured=residual,
        threshold=prob.tol,
        satisfied=residual <= prob.tol,
        detail=f"|f({x:.10g})| = {residual:.3e}",
    )


def _inputs(method: Method, prob: Problem) -> dict[str, object]:
    if method.requires == "guess":
        return {"guess": prob.guess}
    return {"bracket": prob.bracket, "guess": prob.guess}


def _perturb(prob: Problem) -> Problem:
    """Perturb the starting condition for the reproducibility re-test."""
    new_guess = prob.guess
    if prob.guess is not None:
        new_guess = prob.guess + (abs(prob.guess) * 0.05 + 0.1)
    return replace(prob, guess=new_guess)


def _retest_reproduce(method: Method, prob: Problem) -> ReTest:
    """Re-test: does the anomaly reproduce from a perturbed start?

    This distinguishes a *real* shortcoming of the method (deterministic,
    start-insensitive) from an *incidental* failure (only at this start).
    """
    pprob = _perturb(prob)
    name = "reproduce_under_perturbation"
    desc = f"rerun '{method.name}' from perturbed start guess={pprob.guess!r}"
    try:
        out = method.solve(pprob)
    except NumericalFailure as exc:
        return ReTest(name, desc, True, [], f"anomaly reproduced: {exc}")
    check = _invariant(pprob, out.candidate)
    reproduced = (not check.satisfied) or out.status != "converged"
    if reproduced:
        concl = "anomaly reproduced: deterministic shortcoming of this method"
    else:
        concl = (
            "perturbed start did succeed: the failure was start-sensitive, not structural "
            "— escalate to a method that does not depend on the initial guess"
        )
    return ReTest(name, desc, reproduced, [check], concl)


def _retest_corroborate(prob: Problem, x: float) -> ReTest:
    """Re-test: confirm a successful outcome independently.

    1. Re-evaluate the invariant (independent measurement of the residual).
    2. Check for a *transversal* sign change around ``x``. No sign change ->
       likely a tangent point / double root: valid, but flagged as a caveat.
    """
    delta = max(abs(x) * 1e-6, 1e-6)
    left = prob.f(x - delta)
    right = prob.f(x + delta)
    recheck = _invariant(prob, x)
    transversal = left * right < 0
    if transversal:
        concl = "independent re-evaluation confirms the root; sign change verified"
    else:
        concl = (
            f"residual confirmed, but f does not change sign around x "
            f"(left={left:.2e}, right={right:.2e}): likely a tangent point / "
            f"multiple root — accepted with caveat"
        )
    return ReTest(
        "corroborate_root",
        "re-evaluate invariant and check transversal sign change",
        None,
        [recheck],
        concl,
    )


class SelfAuditingSolver:
    """Drives the methods, audits every outcome, corrects itself."""

    def __init__(self, methods: list[Method] | None = None) -> None:
        self.methods = methods if methods is not None else default_methods()

    def solve(self, prob: Problem) -> Solution:
        log = AuditLog(
            problem=prob.name,
            tolerance=prob.tol,
            description=f"find x with |f(x)| <= {prob.tol:g}",
        )
        for idx, method in enumerate(self.methods, start=1):
            applicable, why = method.is_applicable(prob)
            if not applicable:
                log.attempts.append(
                    Attempt(
                        idx,
                        method.name,
                        {},
                        "not_applicable",
                        None,
                        0,
                        [],
                        "skipped",
                        [],
                        "skip",
                        why,
                    )
                )
                continue

            inputs = _inputs(method, prob)

            try:
                out: StrategyOutcome = method.solve(prob)
            except NumericalFailure as exc:
                retest = _retest_reproduce(method, prob)
                log.attempts.append(
                    Attempt(
                        idx,
                        method.name,
                        inputs,
                        "diverged",
                        None,
                        0,
                        [],
                        "unexpected",
                        [retest],
                        "escalate",
                        f"method failed: {exc}",
                    )
                )
                continue

            check = _invariant(prob, out.candidate)

            if check.satisfied and out.status == "converged":
                retest = _retest_corroborate(prob, out.candidate)
                log.attempts.append(
                    Attempt(
                        idx,
                        method.name,
                        inputs,
                        "converged",
                        out.candidate,
                        out.iterations,
                        [check],
                        "expected",
                        [retest],
                        "accept",
                        "invariant satisfied; outcome corroborated",
                    )
                )
                log.finalize("solved", out.candidate, method.name)
                return Solution(out.candidate, method.name, log)

            # A candidate, but the expectation is violated.
            retest = _retest_reproduce(method, prob)
            log.attempts.append(
                Attempt(
                    idx,
                    method.name,
                    inputs,
                    out.status,
                    out.candidate,
                    out.iterations,
                    [check],
                    "unexpected",
                    [retest],
                    "escalate",
                    "invariant violated; re-tested and escalated",
                )
            )

        log.finalize("unsolved", None, None)
        raise SolveFailed(prob.name, log)
