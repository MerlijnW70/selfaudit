"""Audit-trail data structures for the Self-Auditing AI.

Every run of the solver produces an :class:`AuditLog`: a complete, serializable
account of *what* the system did, *which expectation* it held, *where* reality
deviated from it, and *how* it re-tested and corrected itself. The log is
deliberately both human-readable and machine-readable (JSON).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any


def enable_utf8_output() -> None:
    """Best-effort: make stdout/stderr emit UTF-8 so audit reports — which contain
    characters like ``σ``, en-dashes and ``…`` — print on any console, including a
    Windows cp1252 terminal. A no-op where the stream cannot be reconfigured."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):  # pragma: no cover - platform/stream dependent
                pass


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

    def _verdict(self) -> tuple[str, bool]:
        """A human label for the final status and whether it is a 'good' outcome."""
        good = self.final_status in ("solved", "fitted", "validated", "trusted")
        labels = {
            "solved": "SOLVED",
            "unsolved": "UNSOLVED",
            "fitted": "MODEL FOUND",
            "anomaly": "ANOMALY / DISCOVERY",
            "noise": "NO DISCOVERY (noise)",
            "validated": "VALIDATED",
            "unvalidated": "NOT VALIDATED",
            "trusted": "TRUSTED",
            "untrusted": "UNTRUSTED",
            "pending": "PENDING",
        }
        return labels.get(self.final_status, self.final_status.upper()), good

    def to_html(self) -> str:
        """A self-contained, shareable HTML report (inline CSS, no dependencies)."""
        from html import escape

        verdict, good = self._verdict()
        accent = "#1a7f37" if good else "#cf222e"
        parts: list[str] = []
        parts.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
        parts.append(f"<title>Self-Audit · {escape(self.problem)}</title>")
        parts.append(
            "<style>"
            "body{font:14px/1.5 system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f6f8fa;"
            "color:#1f2328}.wrap{max-width:920px;margin:0 auto;padding:24px}"
            "h1{font-size:18px;margin:0 0 4px}.sub{color:#656d76;margin:0 0 16px}"
            f".verdict{{display:inline-block;padding:6px 14px;border-radius:6px;color:#fff;"
            f"font-weight:700;background:{accent};margin-bottom:20px}}"
            ".chk{background:#fff;border:1px solid #d0d7de;border-radius:8px;margin:10px 0;"
            "padding:12px 16px}.chk h2{font-size:15px;margin:0 0 6px}"
            ".ok{color:#1a7f37}.bad{color:#cf222e}.pill{font-size:12px;font-weight:700;"
            "padding:2px 8px;border-radius:10px;margin-left:8px}"
            ".pill.ok{background:#dafbe1}.pill.bad{background:#ffebe9}"
            ".meta{color:#656d76;font-size:13px}.rt{margin:8px 0 0 0;padding:8px 12px;"
            "background:#f6f8fa;border-left:3px solid #d0d7de;border-radius:4px}"
            "code{background:#eff1f3;padding:1px 5px;border-radius:4px}"
            ".concl{margin-top:8px}</style></head><body><div class='wrap'>"
        )
        parts.append(f"<h1>Self-Audit · {escape(self.problem)}</h1>")
        parts.append(f"<p class='sub'>{escape(self.description)}</p>")
        parts.append(f"<div class='verdict'>{escape(verdict)}</div>")
        if self.conclusion:
            parts.append(f"<p class='meta'>{escape(self.conclusion)}</p>")
        for a in self.attempts:
            passed = a.classification == "expected" or a.decision == "accept"
            cls = "ok" if passed else "bad"
            pill = "PASS" if passed else a.decision.upper()
            parts.append("<div class='chk'>")
            parts.append(
                f"<h2>{escape(a.strategy)} <span class='pill {cls}'>{escape(pill)}</span></h2>"
            )
            for c in a.checks:
                mark = "ok" if c.satisfied else "bad"
                parts.append(
                    f"<div class='meta'><span class='{mark}'>"
                    f"{'✓' if c.satisfied else '✗'}</span> "
                    f"<code>{escape(c.name)}</code> — {escape(c.detail)}</div>"
                )
            for r in a.retests:
                parts.append(f"<div class='rt'><b>re-test:</b> {escape(r.name)}")
                parts.append(f"<div class='concl'>{escape(r.conclusion)}</div></div>")
            parts.append(
                f"<div class='meta'>decision: <b>{escape(a.decision)}</b> — {escape(a.notes)}</div>"
            )
            parts.append("</div>")
        parts.append("</div></body></html>")
        return "".join(parts)

    def save_html(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_html())

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
            "trusted": "dataset TRUSTED — all checks passed",
            "untrusted": "dataset UNTRUSTED — one or more checks failed",
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
