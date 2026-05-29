"""Demonstration runner: ``python -m selfaudit``.

Runs three scenarios that each show a different behaviour of the Self-Auditing
AI, prints the audit reports, and writes the report of the self-correction
scenario to ``audit_log.json`` as persistent proof.
"""

from __future__ import annotations

from .audit import enable_utf8_output
from .auditor import SelfAuditingSolver, SolveFailed
from .solver import Problem


def scenarios() -> list[Problem]:
    return [
        # 1) Smooth: Newton finds sqrt(2) immediately.
        Problem("sqrt2: x^2 - 2 = 0", lambda x: x * x - 2.0, guess=1.0),
        # 2) Self-correction: Newton cycles (0 -> 1 -> 0 -> ...), escalates to a fallback.
        Problem("cycling: x^3 - 2x + 2 = 0", lambda x: x**3 - 2.0 * x + 2.0, guess=0.0),
        # 3) Caveat: double root at 0 — residual OK, but no sign change.
        Problem("tangency: x^2 = 0", lambda x: x * x, guess=1.0),
        # 4) Flaky/start-sensitive: Newton fails on f'(0)=0, but the re-test from a
        #    perturbed start succeeds -> the failure was not structural.
        Problem("start-sensitive: x^2 - 2 = 0 from 0", lambda x: x * x - 2.0, guess=0.0),
        # 5) No guess, but a bracket: newton & secant are skipped, bisection solves it.
        Problem("bracket-only: x^2 - 2 = 0", lambda x: x * x - 2.0, bracket=(0.0, 2.0)),
        # 6) No real root: all strategies fail -> honestly reported as 'unsolved'.
        Problem("no-real-root: x^2 + 1 = 0", lambda x: x * x + 1.0, guess=1.0),
    ]


def main() -> None:
    enable_utf8_output()
    solver = SelfAuditingSolver()
    saved = False
    for prob in scenarios():
        try:
            log = solver.solve(prob).log
        except SolveFailed as exc:
            log = exc.log
        print(log.render())
        print()
        if not saved and prob.name.startswith("cycling"):
            log.save("audit_log.json")
            print("audit_log.json written (scenario: self-correction).")
            saved = True


if __name__ == "__main__":  # pragma: no cover
    main()
