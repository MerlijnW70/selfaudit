"""Audit-trail data structures for the Self-Auditing AI.

Every run of the solver produces an :class:`AuditLog`: a complete, serializable
account of *what* the system did, *which expectation* it held, *where* reality
deviated from it, and *how* it re-tested and corrected itself. The log is
deliberately both human-readable and machine-readable (JSON).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExpectationCheck:
    """A single check of an expectation against a measured value."""

    name: str
    measured: float
    threshold: float
    satisfied: bool
    detail: str = ""


@dataclass
class ReTest:
    """A re-test: a repeated/independent measurement after an outcome.

    ``reproduced_anomaly`` is ``True``/``False`` for re-tests that try to
    reproduce an *anomaly*, and ``None`` for re-tests that *corroborate* a
    successful outcome.
    """

    name: str
    description: str
    reproduced_anomaly: bool | None
    checks: list[ExpectationCheck]
    conclusion: str


@dataclass
class Attempt:
    """One attempt by one strategy, including its re-tests and the decision taken."""

    index: int
    strategy: str
    inputs: dict[str, Any]
    outcome: str  # "converged" | "exhausted" | "diverged" | "not_applicable"
    candidate: float | None
    iterations: int
    checks: list[ExpectationCheck]
    classification: str  # "expected" | "unexpected" | "skipped"
    retests: list[ReTest]
    decision: str  # "accept" | "escalate" | "skip"
    notes: str
    params: dict[str, Any] | None = None  # fitted model parameters (for model fitting)


@dataclass
class AuditLog:
    """Complete account of a solve run."""

    problem: str
    tolerance: float
    description: str
    attempts: list[Attempt] = field(default_factory=list)
    final_status: str = "pending"  # "solved" | "unsolved" | "fitted" | "anomaly"
    final_root: float | None = None
    winning_strategy: str | None = None
    conclusion: str = ""  # final verdict / discovery (free text)

    def finalize(self, status: str, root: float | None, strategy: str | None) -> None:
        self.final_status = status
        self.final_root = root
        self.winning_strategy = strategy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    def render(self) -> str:
        """A human-readable audit report."""
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append(f"SELF-AUDIT REPORT  ·  problem: {self.problem}")
        lines.append(f"expectation (invariant): {self.description}")
        lines.append("=" * 72)
        for a in self.attempts:
            lines.append("")
            lines.append(f"[attempt {a.index}] strategy = {a.strategy}  ->  {a.outcome.upper()}")
            if a.inputs:
                inputs = ", ".join(f"{k}={v}" for k, v in a.inputs.items())
                lines.append(f"    inputs    : {inputs}")
            if a.candidate is not None:
                lines.append(f"    candidate : x = {a.candidate:.12g}  ({a.iterations} iterations)")
            if a.params:
                params = ", ".join(f"{k}={v:.4g}" for k, v in a.params.items())
                lines.append(f"    parameters: {params}")
            for c in a.checks:
                mark = "OK " if c.satisfied else "!! "
                lines.append(
                    f"    check {mark}: {c.name}  measured={c.measured:.3e} "
                    f"threshold={c.threshold:.1e}  -> {c.detail}"
                )
            lines.append(f"    classification: {a.classification.upper()}")
            for r in a.retests:
                tag = {True: "reproduced", False: "not-reproduced", None: "corroboration"}[
                    r.reproduced_anomaly
                ]
                lines.append(f"    re-test [{tag}] {r.name}: {r.description}")
                for c in r.checks:
                    mark = "OK " if c.satisfied else "!! "
                    lines.append(
                        f"        check {mark}: measured={c.measured:.3e} "
                        f"threshold={c.threshold:.1e}  {c.detail}"
                    )
                lines.append(f"        => {r.conclusion}")
            lines.append(f"    DECISION: {a.decision.upper()}  ({a.notes})")
        labels = {
            "solved": "solved",
            "unsolved": "UNSOLVED — all strategies exhausted",
            "fitted": "model found",
            "anomaly": "ANOMALY / DISCOVERY",
            "noise": "NO DISCOVERY (noise)",
            "validated": "output validated",
            "unvalidated": "NOT validated — all model tiers exhausted",
            "pending": "pending",
        }
        lines.append("")
        lines.append("-" * 72)
        lines.append(f"RESULT: {labels.get(self.final_status, self.final_status)}")
        if self.final_root is not None:
            lines.append(f"  root x = {self.final_root:.12g}")
        if self.winning_strategy is not None:
            lines.append(f"  strategy/model: {self.winning_strategy}")
        if self.conclusion:
            lines.append(f"  {self.conclusion}")
        lines.append("-" * 72)
        return "\n".join(lines)
