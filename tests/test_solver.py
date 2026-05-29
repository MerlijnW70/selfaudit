"""Pytest suite that challenges both the solver and the Self-Auditing controller."""

from __future__ import annotations

import json

import pytest

from selfaudit.auditor import SelfAuditingSolver, SolveFailed
from selfaudit.solver import (
    Method,
    NumericalFailure,
    Problem,
    auto_bracket,
    brentq,
    newton,
    secant,
)


@pytest.fixture
def solver() -> SelfAuditingSolver:
    return SelfAuditingSolver()


def test_happy_path_newton(solver: SelfAuditingSolver) -> None:
    """Smooth case: Newton solves x^2 - 2 = 0, no escalation needed."""
    prob = Problem("sqrt2", lambda x: x * x - 2.0, guess=1.0)
    sol = solver.solve(prob)
    assert sol.strategy == "newton"
    assert sol.root == pytest.approx(2.0**0.5, abs=1e-8)
    assert sol.log.final_status == "solved"
    # One attempt, accepted, with a corroboration re-test.
    assert len(sol.log.attempts) == 1
    accepted = sol.log.attempts[0]
    assert accepted.decision == "accept"
    assert accepted.classification == "expected"
    assert any(rt.name == "corroborate_root" for rt in accepted.retests)


def test_self_correction_on_newton_cycling(solver: SelfAuditingSolver) -> None:
    """x^3 - 2x + 2 = 0 makes Newton cycle from 0; the system must correct itself."""
    f = lambda x: x**3 - 2.0 * x + 2.0  # noqa: E731 - compact test function
    prob = Problem("cycling", f, guess=0.0)
    sol = solver.solve(prob)

    assert sol.log.final_status == "solved"
    assert abs(f(sol.root)) <= prob.tol
    assert -2.0 < sol.root < -1.0  # the only real root

    # Newton (attempt 1) must be classified as UNEXPECTED and escalated...
    newton_attempt = next(a for a in sol.log.attempts if a.strategy == "newton")
    assert newton_attempt.classification == "unexpected"
    assert newton_attempt.decision == "escalate"
    # ...with a documented reproducibility re-test.
    assert any(rt.name == "reproduce_under_perturbation" for rt in newton_attempt.retests)
    # And a later strategy must have accepted the solution.
    assert sol.strategy != "newton"
    assert any(a.decision == "accept" for a in sol.log.attempts)


def test_unsolvable_raises_with_log(solver: SelfAuditingSolver) -> None:
    """x^2 + 1 = 0 has no real root: all strategies fail, the log is preserved."""
    prob = Problem("no-real-root", lambda x: x * x + 1.0, guess=1.0)
    with pytest.raises(SolveFailed) as excinfo:
        solver.solve(prob)
    log = excinfo.value.log
    assert log.final_status == "unsolved"
    assert log.final_root is None
    assert len(log.attempts) >= 1
    # No attempt may be accepted.
    assert all(a.decision != "accept" for a in log.attempts)


def test_tangency_is_accepted_with_caveat(solver: SelfAuditingSolver) -> None:
    """x^2 = 0: the residual is satisfied, but the re-test flags the missing sign change."""
    prob = Problem("tangency", lambda x: x * x, guess=1.0)
    sol = solver.solve(prob)
    assert sol.log.final_status == "solved"
    assert sol.root == pytest.approx(0.0, abs=1e-4)
    corroboration = next(
        rt for a in sol.log.attempts for rt in a.retests if rt.name == "corroborate_root"
    )
    assert "tangent point" in corroboration.conclusion or "sign change" in corroboration.conclusion


def test_flaky_failure_is_start_sensitive(solver: SelfAuditingSolver) -> None:
    """x^2 - 2 = 0 from 0: Newton fails (f'(0)=0), but the re-test from a perturbed
    start succeeds -> the anomaly is NOT reproducible (start-sensitive).
    This covers the 'reproduced_anomaly is False' branch of the auditor.
    """
    prob = Problem("start-sensitive", lambda x: x * x - 2.0, guess=0.0)
    sol = solver.solve(prob)

    assert sol.log.final_status == "solved"
    assert abs(sol.root**2 - 2.0) <= prob.tol

    newton_attempt = next(a for a in sol.log.attempts if a.strategy == "newton")
    assert newton_attempt.classification == "unexpected"
    assert newton_attempt.decision == "escalate"
    retest = next(rt for rt in newton_attempt.retests if rt.name == "reproduce_under_perturbation")
    assert retest.reproduced_anomaly is False
    assert "start-sensitive" in retest.conclusion


def test_methods_skipped_without_guess(solver: SelfAuditingSolver) -> None:
    """Without a guess but with a bracket: newton & secant are skipped, bisection
    solves it. Covers the 'not_applicable'/'skip' branch.
    """
    prob = Problem("bracket-only", lambda x: x * x - 2.0, bracket=(0.0, 2.0))
    sol = solver.solve(prob)

    assert sol.strategy == "brentq"
    assert sol.log.final_status == "solved"
    skipped = [a for a in sol.log.attempts if a.decision == "skip"]
    assert {a.strategy for a in skipped} == {"newton", "secant"}
    assert all(a.outcome == "not_applicable" for a in skipped)


def test_all_methods_skipped_when_no_inputs(solver: SelfAuditingSolver) -> None:
    """Without a guess and without a bracket no method is applicable -> unsolved."""
    prob = Problem("no-inputs", lambda x: x * x - 2.0)
    with pytest.raises(SolveFailed) as excinfo:
        solver.solve(prob)
    log = excinfo.value.log
    assert log.final_status == "unsolved"
    assert all(a.decision == "skip" for a in log.attempts)


def test_audit_log_is_json_serializable(solver: SelfAuditingSolver) -> None:
    """The audit trail must be writable as valid JSON (persistent proof)."""
    prob = Problem("cycling", lambda x: x**3 - 2.0 * x + 2.0, guess=0.0)
    sol = solver.solve(prob)
    blob = sol.log.to_json()
    parsed = json.loads(blob)
    assert parsed["final_status"] == "solved"
    assert isinstance(parsed["attempts"], list)
    assert parsed["attempts"][0]["strategy"] == "newton"


def test_method_reports_clean_failure_not_crash() -> None:
    """A Newton that cannot make progress (f'(0)=0 for x^2+1, no real root) must
    not crash: it reports a clean non-converged 'exhausted' outcome — or, equally
    cleanly, a NumericalFailure. Either way the auditor can escalate."""
    prob = Problem("deriv-zero", lambda x: x * x + 1.0, guess=0.0)
    try:
        out = newton(prob)
    except NumericalFailure:
        return  # a clean, explainable failure is also acceptable
    assert out.status == "exhausted"


def test_perturb_handles_missing_guess(solver: SelfAuditingSolver) -> None:
    """Bracket without sign change and without a guess: bisection fails -> the
    reproducibility re-test runs _perturb on a guess-less problem.
    Covers the 'guess is None' branch of _perturb.
    """
    prob = Problem("bad-bracket", lambda x: x * x - 2.0, bracket=(3.0, 4.0))
    with pytest.raises(SolveFailed) as excinfo:
        solver.solve(prob)
    log = excinfo.value.log
    brentq_attempt = next(a for a in log.attempts if a.strategy == "brentq")
    assert brentq_attempt.outcome == "diverged"
    assert any(rt.name == "reproduce_under_perturbation" for rt in brentq_attempt.retests)


# --------------------------------------------------------------------------- #
# Direct unit tests on the methods (cover the defensive guards & auto_bracket)
# --------------------------------------------------------------------------- #


def test_secant_on_constant_does_not_falsely_converge() -> None:
    """A constant function has no root; secant must not claim convergence."""
    out = secant(Problem("const", lambda x: 5.0, guess=1.0))
    assert out.status == "exhausted"


def test_auto_bracket_success_and_failure() -> None:
    # Even function from 0: the sign change is to the right.
    f = lambda x: x * x - 2.0  # noqa: E731
    a, b = auto_bracket(f, 0.0)
    assert f(a) * f(b) < 0
    # Root to the left of x0: covers the left-hand return branch.
    g = lambda x: x + 5.0  # noqa: E731
    la, lb = auto_bracket(g, 0.0)
    assert g(la) * g(lb) < 0
    assert la < lb <= 0.0
    # Start point is already exactly a root.
    assert auto_bracket(lambda x: x, 0.0) == (0.0, 0.0)
    # No sign change to be found -> clean failure.
    with pytest.raises(NumericalFailure):
        auto_bracket(lambda x: x * x + 1.0, 0.0)


def test_brentq_auto_brackets_from_guess() -> None:
    out = brentq(Problem("x", lambda x: x * x - 2.0, guess=0.0))
    assert abs(out.candidate**2 - 2.0) <= 1e-9


def test_brentq_rejects_bracket_without_sign_change() -> None:
    with pytest.raises(NumericalFailure):
        brentq(Problem("x", lambda x: x * x - 2.0, bracket=(3.0, 4.0)))


def test_brentq_exhausts_when_iterations_too_few() -> None:
    out = brentq(Problem("x", lambda x: x * x - 2.0, bracket=(0.0, 2.0), max_iter=2))
    assert out.status == "exhausted"


# --------------------------------------------------------------------------- #
# Defensive input: invalid/degenerate input must never crash, always a clean
# NumericalFailure or a documented 'unsolved'.
# --------------------------------------------------------------------------- #

NAN = float("nan")
INF = float("inf")


def test_none_guess_is_rejected_by_methods() -> None:
    """guess=None -> newton & secant refuse cleanly (no TypeError/crash)."""
    f = lambda x: x - 1.0  # noqa: E731
    with pytest.raises(NumericalFailure):
        newton(Problem("none", f, guess=None))
    with pytest.raises(NumericalFailure):
        secant(Problem("none", f, guess=None))


def test_nan_guess_newton_raises_not_crashes() -> None:
    """NaN start -> non-finite iteration -> clean NumericalFailure (covers the isfinite guard)."""
    with pytest.raises(NumericalFailure):
        newton(Problem("nan", lambda x: x * x - 2.0, guess=NAN))


def test_nan_guess_secant_raises_not_crashes() -> None:
    with pytest.raises(NumericalFailure):
        secant(Problem("nan", lambda x: x * x - 2.0, guess=NAN))


def test_inf_guess_newton_raises_not_crashes() -> None:
    with pytest.raises(NumericalFailure):
        newton(Problem("inf", lambda x: x * x - 2.0, guess=INF))


def test_nan_startpoint_auto_bracket_raises() -> None:
    """auto_bracket on a non-finite start point -> clean failure, no infinite loop."""
    with pytest.raises(NumericalFailure):
        auto_bracket(lambda x: x * x - 2.0, NAN)


def test_overflowing_objective_becomes_numerical_failure() -> None:
    """An objective that overflows itself (e^x for large x during the bracket
    search) becomes a clean NumericalFailure instead of an OverflowError crash
    (covers _guard)."""
    import math

    with pytest.raises(NumericalFailure):
        auto_bracket(lambda x: math.exp(x) + 1.0, 0.0)


def test_overflowing_objective_through_auditor_is_unsolved(solver: SelfAuditingSolver) -> None:
    """End-to-end: e^x + 1 = 0 has no real root and makes the objective overflow.
    The system reports a clean 'unsolved' with an audit trail — it does not crash."""
    import math

    with pytest.raises(SolveFailed) as excinfo:
        solver.solve(Problem("overflow: e^x + 1 = 0", lambda x: math.exp(x) + 1.0, guess=0.0))
    log = excinfo.value.log
    assert log.final_status == "unsolved"
    assert all(a.decision != "accept" for a in log.attempts)
    assert any(a.classification == "unexpected" for a in log.attempts)


def test_unknown_requirement_is_not_applicable() -> None:
    """A Method with an unknown 'requires' is cleanly not applicable (covers the fallback)."""
    method = Method("bogus", "something-unknown", newton)
    applicable, reason = method.is_applicable(Problem("x", lambda x: x, guess=1.0))
    assert applicable is False
    assert "unknown requirement" in reason


def test_nan_guess_through_auditor_is_documented(solver: SelfAuditingSolver) -> None:
    """End-to-end: a NaN guess makes all methods fail cleanly. No crash —
    the system reports 'unsolved' with a full audit trail of every deviation.
    """
    prob = Problem("nan-guess", lambda x: x * x - 2.0, guess=NAN)
    with pytest.raises(SolveFailed) as excinfo:
        solver.solve(prob)
    log = excinfo.value.log
    assert log.final_status == "unsolved"
    # Every attempt is classified as unexpected/diverged, nothing accepted.
    assert all(a.decision != "accept" for a in log.attempts)
    assert any(a.classification == "unexpected" for a in log.attempts)


# --------------------------------------------------------------------------- #
# Audit-trail rendering & persistence
# --------------------------------------------------------------------------- #


def test_render_and_save_cover_all_log_shapes(solver: SelfAuditingSolver, tmp_path) -> None:
    """Render all log shapes (skip, diverged, accept, solved & unsolved) and write
    one out as JSON. Covers render()/save() including all re-test tags.
    """
    from selfaudit.__main__ import scenarios

    logs = []
    for prob in scenarios():
        try:
            logs.append(solver.solve(prob).log)
        except SolveFailed as exc:
            logs.append(exc.log)
    # Add a guaranteed 'unsolved' case (all methods skipped).
    try:
        solver.solve(Problem("no-inputs", lambda x: x * x - 2.0))
    except SolveFailed as exc:
        logs.append(exc.log)

    for log in logs:
        text = log.render()
        assert "SELF-AUDIT REPORT" in text
        assert "RESULT" in text

    target = tmp_path / "audit_log.json"
    logs[0].save(str(target))
    parsed = json.loads(target.read_text(encoding="utf-8"))
    assert "attempts" in parsed


def test_demo_main_runs_and_writes_log(tmp_path, monkeypatch, capsys) -> None:
    """The demo runner runs all scenarios and writes audit_log.json (in tmp cwd)."""
    from selfaudit.__main__ import main

    monkeypatch.chdir(tmp_path)
    main()
    out = capsys.readouterr().out
    assert "SELF-AUDIT REPORT" in out
    written = tmp_path / "audit_log.json"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["final_status"] == "solved"
