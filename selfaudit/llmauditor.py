"""The Self-Auditing AI applied to LLM output: validate-and-escalate controller.

The same loop as the root finder, but the "outcome" is now whether a model's
output satisfies a checkable validator:

* **expected** outcome   -> the output validates. Corroborate by re-calling the
  same tier (is it reproducible, or flaky?), then accept.
* **unexpected** outcome -> the output fails validation. Re-test by retrying the
  same tier once: if the retry passes the failure was a sampling fluke (accept
  after re-test); if it fails again the tier is structurally too weak -> escalate
  to a stronger model.

If the whole ladder is exhausted, the task is reported as ``unvalidated`` with a
full audit trail. Everything is recorded in an :class:`AuditLog` — the same
structure the other two applications use.
"""

from __future__ import annotations

from dataclasses import dataclass

from .audit import Attempt, AuditLog, ExpectationCheck, ReTest
from .llm import LLMUnavailable, ModelCaller, Task, ValidationResult, default_tiers


@dataclass
class Validation:
    output: str
    tier: str
    log: AuditLog


class ValidationFailed(Exception):
    """No model tier produced output that satisfies the validator. Log attached."""

    def __init__(self, task: str, log: AuditLog) -> None:
        super().__init__(f"'{task}' not validated — all model tiers exhausted")
        self.log = log


def _check(result: ValidationResult) -> ExpectationCheck:
    """The expectation: the output has zero validator violations."""
    return ExpectationCheck(
        name="output_valid",
        measured=float(result.violations),
        threshold=0.0,
        satisfied=result.ok,
        detail=result.detail,
    )


def _preview(text: str, limit: int = 80) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


def _retest_corroborate(caller: ModelCaller, task: Task) -> ReTest:
    """Re-call the same tier and re-validate: is the success reproducible?"""
    name = "corroborate_output"
    desc = "re-call the same tier and re-validate the output"
    try:
        again = caller.call(task.prompt)
    except LLMUnavailable as exc:
        return ReTest(name, desc, None, [], f"could not corroborate (re-call failed: {exc})")
    res = task.validator(again)
    check = _check(res)
    if res.ok:
        concl = "re-call produced valid output again: the result is reproducible"
    else:
        concl = (
            "re-call produced INVALID output: this tier is flaky for the task "
            "— accepted with caveat (the first output did validate)"
        )
    return ReTest(name, desc, None, [check], concl)


def _retest_reproduce(caller: ModelCaller, task: Task) -> tuple[ReTest, bool, str | None]:
    """Retry the same tier once: does the validation failure reproduce?

    Returns the re-test record, whether the anomaly reproduced, and the retry's
    output (only meaningful — and valid — when it did *not* reproduce).
    """
    name = "reproduce_under_retry"
    desc = "retry the same tier once and re-validate"
    try:
        retry = caller.call(task.prompt)
    except LLMUnavailable as exc:
        return ReTest(name, desc, True, [], f"retry also failed: {exc}"), True, None
    res = task.validator(retry)
    check = _check(res)
    reproduced = not res.ok
    if reproduced:
        concl = "retry also failed validation: deterministic deficiency of this tier — escalate"
    else:
        concl = (
            "retry validated: the failure was a sampling fluke, not structural "
            "— accepted after re-test (no escalation needed)"
        )
    return ReTest(name, desc, reproduced, [check], concl), reproduced, retry


class SelfAuditingValidator:
    """Drives the model tiers, audits every output, escalates on real failure."""

    def __init__(self, tiers: list[ModelCaller] | None = None) -> None:
        self.tiers = tiers if tiers is not None else default_tiers()

    def run(self, task: Task) -> Validation:
        log = AuditLog(
            problem=task.name,
            tolerance=0.0,
            description="produce output with 0 validator violations",
        )
        for idx, caller in enumerate(self.tiers, start=1):
            # 1) Call the tier. A clean unavailability is an unexpected outcome.
            try:
                output = caller.call(task.prompt)
            except LLMUnavailable as exc:
                retest, _, _ = _retest_reproduce(caller, task)
                log.attempts.append(
                    Attempt(
                        idx,
                        caller.name,
                        {"prompt": _preview(task.prompt)},
                        "unavailable",
                        None,
                        0,
                        [],
                        "unexpected",
                        [retest],
                        "escalate",
                        f"tier unavailable: {exc}",
                    )
                )
                continue

            result = task.validator(output)
            check = _check(result)

            # 2) Expected: the output validates -> corroborate and accept.
            if result.ok:
                retest = _retest_corroborate(caller, task)
                log.attempts.append(
                    Attempt(
                        idx,
                        caller.name,
                        {"prompt": _preview(task.prompt)},
                        "validated",
                        None,
                        0,
                        [check],
                        "expected",
                        [retest],
                        "accept",
                        f"output validated; '{_preview(output)}'",
                    )
                )
                log.finalize("validated", None, caller.name)
                return Validation(output, caller.name, log)

            # 3) Unexpected: failed validation -> retry the same tier once.
            retest, reproduced, retry_out = _retest_reproduce(caller, task)
            if not reproduced and retry_out is not None:
                # Flaky failure: the retry validated -> accept after re-test.
                log.attempts.append(
                    Attempt(
                        idx,
                        caller.name,
                        {"prompt": _preview(task.prompt)},
                        "validated-after-retry",
                        None,
                        0,
                        [check],
                        "unexpected",
                        [retest],
                        "accept",
                        f"failed then validated on retry; '{_preview(retry_out)}'",
                    )
                )
                log.finalize("validated", None, caller.name)
                log.conclusion = "accepted after re-test: the first failure was a sampling fluke"
                return Validation(retry_out, caller.name, log)

            # Deterministic failure of this tier -> escalate to the next.
            log.attempts.append(
                Attempt(
                    idx,
                    caller.name,
                    {"prompt": _preview(task.prompt)},
                    "invalid",
                    None,
                    0,
                    [check],
                    "unexpected",
                    [retest],
                    "escalate",
                    f"validation failed ({result.detail}); escalated",
                )
            )

        log.finalize("unvalidated", None, None)
        raise ValidationFailed(task.name, log)
