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


def _plural(n: int, noun: str) -> str:
    """``1 problem`` / ``2 problems`` — count with a correctly pluralized noun."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


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
    rows: list[int] = field(default_factory=list)  # offending row indices (dataset checks)
    row_preview: list[dict[str, Any]] = field(default_factory=list)  # sample of offending rows
    title: str = ""  # optional plain-English heading for reports (falls back to strategy)


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

    def _verdict(self) -> tuple[str, str]:
        """A human label for the final status and a tone (good/warn/bad) for styling."""
        labels = {
            "solved": "SOLVED",
            "unsolved": "UNSOLVED",
            "fitted": "MODEL FOUND",
            "anomaly": "ANOMALY / DISCOVERY",
            "noise": "NO DISCOVERY (noise)",
            "validated": "VALIDATED",
            "unvalidated": "NOT VALIDATED",
            "trusted": "TRUSTED",
            "review": "NEEDS REVIEW",
            "untrusted": "UNTRUSTED",
            "pending": "PENDING",
        }
        tone = {
            "solved": "good",
            "fitted": "good",
            "validated": "good",
            "trusted": "good",
            "review": "warn",
            "noise": "warn",
        }.get(self.final_status, "bad")
        return labels.get(self.final_status, self.final_status.upper()), tone

    @staticmethod
    def _attempt_status(attempt: Attempt) -> tuple[str, str]:
        """Map an attempt's decision to a (label, css-class) status chip."""
        return {
            "accept": ("PASS", "pass"),
            "warn": ("WARN", "warn"),
            "flag": ("FAIL", "fail"),
            "escalate": ("ESCALATED", "warn"),
            "skip": ("SKIPPED", "muted"),
        }.get(attempt.decision, (attempt.decision.upper(), "muted"))

    def to_html(self, *, chart: str = "") -> str:
        """A self-contained, shareable HTML report: summary banner, severity chips,
        an optional data chart, a sortable findings table, and collapsible re-test
        details. One portable file — inline CSS, a touch of vanilla JS, no build."""
        from html import escape

        verdict, tone = self._verdict()
        accent = {"good": "#1a7f37", "warn": "#9a6700", "bad": "#cf222e"}[tone]

        counts: dict[str, int] = {}
        for a in self.attempts:
            label = self._attempt_status(a)[0]
            counts[label] = counts.get(label, 0) + 1
        chip_class = {"PASS": "pass", "WARN": "warn", "FAIL": "fail"}
        summary = "".join(
            f"<span class='chip {chip_class.get(lbl, 'muted')}'>{counts[lbl]} {escape(lbl.lower())}"
            f"</span>"
            for lbl in ("FAIL", "WARN", "PASS", "ESCALATED", "SKIPPED")
            if counts.get(lbl)
        )

        css = (
            "body{font:14px/1.6 system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f6f8fa;"
            "color:#1f2328}.wrap{max-width:960px;margin:0 auto;padding:28px 20px}"
            "h1{font-size:20px;margin:0 0 2px;overflow-wrap:anywhere}"
            ".sub{color:#656d76;margin:0 0 18px;overflow-wrap:anywhere}"
            f".verdict{{display:inline-block;padding:8px 18px;border-radius:8px;color:#fff;"
            f"font-weight:800;font-size:16px;letter-spacing:.3px;background:{accent}}}"
            ".counts{margin:14px 0 6px}.concl{color:#424a53;margin:6px 0 18px}"
            ".plain{font-size:15px;font-weight:600;color:#1f2328;margin:12px 0 2px}"
            ".chip{display:inline-block;font-size:12px;font-weight:700;padding:2px 9px;"
            "border-radius:999px;margin-right:6px}"
            ".chip.pass{background:#dafbe1;color:#1a7f37}.chip.warn{background:#fff1cc;color:#9a6700}"
            ".chip.fail{background:#ffebe9;color:#cf222e}.chip.muted{background:#eaeef2;color:#656d76}"
            "table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #d0d7de;"
            "border-radius:10px;overflow:hidden}thead th{text-align:left;background:#f0f3f6;"
            "padding:10px 12px;font-size:12px;text-transform:uppercase;letter-spacing:.4px;"
            "color:#57606a;cursor:pointer;user-select:none}tbody td{padding:11px 12px;"
            "border-top:1px solid #eaeef2;vertical-align:top;overflow-wrap:anywhere}"
            "tbody tr:hover{background:#fafbfc}"
            # fixed layout + a colgroup bound the Detail column, so a wide offending-rows
            # preview scrolls inside its cell instead of pushing the whole report off-screen.
            "#findings{table-layout:fixed}td.idx{color:#8c959f}td.name{font-weight:600}"
            "details{margin-top:6px}summary{cursor:pointer;color:#57606a;font-size:13px}"
            ".notes{color:#656d76;font-size:13px;margin-top:4px}"
            "code{background:#eff1f3;padding:1px 5px;border-radius:4px}"
            ".chart{background:#fff;border:1px solid #d0d7de;border-radius:10px;"
            "padding:14px;margin:0 0 16px}.legend{font-size:12px;color:#424a53;margin-top:8px}"
            ".muted{color:#8c959f}footer{color:#8c959f;font-size:12px;margin-top:18px}"
            ".preview{font-size:12px;border-collapse:collapse;margin-top:6px;display:block;"
            "overflow-x:auto;max-width:100%}.preview th,.preview td{border:1px solid #e1e6ea;"
            "padding:3px 7px;white-space:nowrap;text-align:left}.preview th{background:#f6f8fa}"
        )

        parts: list[str] = []
        parts.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
        parts.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
        parts.append(f"<title>selfaudit · {escape(self.problem)}</title>")
        parts.append(f"<style>{css}</style></head><body><div class='wrap'>")
        parts.append(f"<h1>selfaudit · {escape(self.problem)}</h1>")
        parts.append(f"<p class='sub'>{escape(self.description)}</p>")
        parts.append(f"<div class='verdict'>{escape(verdict)}</div>")
        parts.append(f"<div class='counts'>{summary}</div>")
        n_fail, n_warn = counts.get("FAIL", 0), counts.get("WARN", 0)
        if n_fail:
            plain = (
                f"We found {_plural(n_fail, 'problem')}"
                + (f" and {_plural(n_warn, 'warning')}" if n_warn else "")
                + " — don't trust this data until they're resolved."
            )
        elif n_warn:
            plain = f"{_plural(n_warn, 'warning')} to review — no outright failures."
        else:
            plain = f"No problems found — all {_plural(len(self.attempts), 'check')} passed."
        parts.append(f"<p class='plain'>{escape(plain)}</p>")
        if self.conclusion:
            parts.append(f"<p class='concl'>{escape(self.conclusion)}</p>")
        if chart:
            parts.append(f"<div class='chart'>{chart}</div>")

        parts.append("<table id='findings'>")
        parts.append(
            "<colgroup><col style='width:38px'><col style='width:26%'>"
            "<col style='width:96px'><col></colgroup>"
        )
        parts.append(
            "<thead><tr><th onclick='sortBy(0)'>#</th><th onclick='sortBy(1)'>Check</th>"
            "<th onclick='sortBy(2)'>Status</th><th>Detail</th></tr></thead><tbody>"
        )
        for a in self.attempts:
            label, klass = self._attempt_status(a)
            detail_bits = [f"<code>{escape(c.name)}</code> — {escape(c.detail)}" for c in a.checks]
            cell = "<br>".join(detail_bits) if detail_bits else ""
            for r in a.retests:
                cell += (
                    f"<details><summary>re-test: {escape(r.name)}</summary>"
                    f"<div class='notes'>{escape(r.conclusion)}</div></details>"
                )
            if a.rows:
                shown = ", ".join(str(r) for r in a.rows[:15])
                more = " …" if len(a.rows) > 15 else ""
                cell += f"<div class='notes'>offending rows: {escape(shown)}{more}</div>"
            if a.row_preview:
                cols = list(a.row_preview[0].keys())
                head = "".join(f"<th>{escape(str(c))}</th>" for c in cols)
                body = ""
                for prow in a.row_preview:
                    cells = "".join(f"<td>{escape(str(prow.get(c, '')))}</td>" for c in cols)
                    body += f"<tr>{cells}</tr>"
                cell += (
                    f"<details><summary>show {len(a.row_preview)} sample offending rows</summary>"
                    f"<table class='preview'><thead><tr>{head}</tr></thead>"
                    f"<tbody>{body}</tbody></table></details>"
                )
            if a.notes:
                cell += f"<div class='notes'>{escape(a.notes)}</div>"
            parts.append(
                f"<tr><td class='idx'>{a.index}</td>"
                f"<td class='name'>{escape(a.title or a.strategy)}</td>"
                f"<td><span class='chip {klass}'>{escape(label)}</span></td>"
                f"<td>{cell}</td></tr>"
            )
        parts.append("</tbody></table>")
        parts.append("<footer>Generated by selfaudit · click a column header to sort.</footer>")
        parts.append(
            "<script>function sortBy(n){var t=document.getElementById('findings'),"
            "b=t.tBodies[0],rows=Array.prototype.slice.call(b.rows),"
            "asc=t.getAttribute('data-s')!==n+'a';"
            "rows.sort(function(p,q){var x=p.cells[n].innerText.trim(),"
            "y=q.cells[n].innerText.trim(),nx=parseFloat(x),ny=parseFloat(y);"
            "if(!isNaN(nx)&&!isNaN(ny))return asc?nx-ny:ny-nx;"
            "return asc?x.localeCompare(y):y.localeCompare(x);});"
            "rows.forEach(function(r){b.appendChild(r);});"
            "t.setAttribute('data-s',asc?n+'a':n+'d');}</script>"
        )
        parts.append("</div></body></html>")
        return "".join(parts)

    def save_html(self, path: str, *, chart: str = "") -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_html(chart=chart))

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
            if a.rows:
                shown = ", ".join(str(r) for r in a.rows[:15])
                more = " …" if len(a.rows) > 15 else ""
                lines.append(f"    offending rows: {shown}{more}")
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
            "review": "dataset NEEDS REVIEW — warnings only (no hard failures)",
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
