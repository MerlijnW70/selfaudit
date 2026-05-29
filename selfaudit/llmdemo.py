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
    enable_os_truststore,
    exact_field_validator,
    json_schema_validator,
    load_dotenv,
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


# A deliberately hard live task: multi-digit multiplication with "JSON only, no
# prose" suppresses the model's scratch space. The validator computes the truth
# itself, so it cannot be fooled by a confident-but-wrong answer — in practice
# every tier tends to miscompute, so the ladder escalates fully and honestly
# reports `unvalidated` rather than accepting a plausible wrong number. That is
# the whole point: an objective invariant prevents false positives.
_A, _B = 739_613, 856_447
_HARD_PROMPT = (
    f'Compute {_A} * {_B} exactly. Return ONLY a JSON object {{"product": <integer>}} '
    f"with the exact integer result. No prose, no markdown, no commas in the number."
)


def _hard_task() -> Task:
    return Task("hard-arithmetic", _HARD_PROMPT, exact_field_validator("product", _A * _B))


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
    # Load .env (if present) so the live run can read ANTHROPIC_API_KEY /
    # SSL_CERT_FILE; the file is authoritative over any stale inherited value.
    load_dotenv(override=True)
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
        if enable_os_truststore():
            print("(TLS verification routed through the OS trust store via truststore.)")
        print("\n-- live task A: produce valid person JSON --")
        print(_run(SelfAuditingValidator(default_tiers()), task).render())
        print("\n-- live task B: exact multi-digit arithmetic (stresses weaker tiers) --")
        print(_run(SelfAuditingValidator(default_tiers()), _hard_task()).render())
    else:
        print("\n(no ANTHROPIC_API_KEY set — skipping the live run.)")


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["AnthropicCaller", "main", "scripted_scenarios"]
