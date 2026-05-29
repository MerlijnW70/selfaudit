"""Demo: ``python -m selfaudit.llmdemo``.

Shows the Self-Auditing AI validating LLM output. The scripted scenarios are
deterministic (no API key needed) and exercise every branch of the controller:
direct accept, accept-after-retry, escalate-then-validate, and all-tiers-
exhausted. They print the audit reports and write ``llm_audit_log.json``.

If the Anthropic SDK and an API key are available, a final live run sends a real
JSON task through haiku -> sonnet -> opus and prints whatever happens.
"""

from __future__ import annotations

import os

from .audit import AuditLog
from .llm import (
    AnthropicCaller,
    ScriptedCaller,
    Task,
    default_tiers,
    json_schema_validator,
)
from .llmauditor import SelfAuditingValidator, ValidationFailed

# The task: a JSON object with these keys and types. Same validator everywhere.
_REQUIRED = {"name": str, "age": int, "email": str}
_PROMPT = (
    "Return ONLY a JSON object with keys 'name' (string), 'age' (integer) and "
    "'email' (string). No prose, no markdown fences."
)
_GOOD = '{"name": "Ada", "age": 36, "email": "ada@example.com"}'
_BAD_PROSE = "Sure! Here is the person: Ada, age 36, ada@example.com."
_BAD_JSON = '{"name": "Ada", "age": "thirty-six"}'  # wrong type + missing key


def _task() -> Task:
    return Task("person-json", _PROMPT, json_schema_validator(_REQUIRED))


def scripted_scenarios() -> list[tuple[str, SelfAuditingValidator]]:
    """Each scenario pins the tiers' replies to drive one controller branch."""
    return [
        (
            "direct-accept (haiku nails it)",
            SelfAuditingValidator([ScriptedCaller("haiku", [_GOOD])]),
        ),
        (
            "accept-after-retry (haiku flaky, retry validates)",
            SelfAuditingValidator([ScriptedCaller("haiku", [_BAD_PROSE, _GOOD])]),
        ),
        (
            "escalate (haiku always fails, sonnet validates)",
            SelfAuditingValidator(
                [
                    ScriptedCaller("haiku", [_BAD_JSON]),
                    ScriptedCaller("sonnet", [_GOOD]),
                ]
            ),
        ),
        (
            "unvalidated (every tier fails)",
            SelfAuditingValidator(
                [
                    ScriptedCaller("haiku", [_BAD_PROSE]),
                    ScriptedCaller("sonnet", [_BAD_JSON]),
                ]
            ),
        ),
    ]


def _run(validator: SelfAuditingValidator, task: Task) -> AuditLog:
    try:
        return validator.run(task).log
    except ValidationFailed as exc:
        return exc.log


def main() -> None:
    task = _task()
    saved = False
    for title, validator in scripted_scenarios():
        print(f"### scenario: {title}")
        log = _run(validator, task)
        print(log.render())
        print()
        if not saved and title.startswith("escalate"):
            log.save("llm_audit_log.json")
            print("llm_audit_log.json written (scenario: escalate-then-validate).")
            saved = True

    # Optional live run against real models, if the environment allows it.
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\n### live run (real models: haiku -> sonnet -> opus)")
        try:
            import anthropic  # noqa: F401
        except ImportError:
            print("anthropic SDK not installed; skipping live run.")
            return
        log = _run(SelfAuditingValidator(default_tiers()), task)
        print(log.render())
    else:
        print("\n(no ANTHROPIC_API_KEY set — skipping the live run.)")


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["AnthropicCaller", "main", "scripted_scenarios"]
