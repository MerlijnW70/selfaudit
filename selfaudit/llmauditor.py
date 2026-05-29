"""The Self-Auditing AI applied to LLM output: validate-and-escalate controller.

The same loop as the root finder, but the "outcome" is now whether a model's
output satisfies a checkable validator:

* **expected** outcome   -> the output validates. Corroborate by re-calling the
  same tier (is it reproducible, or flaky?), then accept.
* **unexpected** outcome -> the output fails validation. Re-test by **self-repair**:
  re-prompt the *same* tier with the exact validation error so it can fix its own
  output. If the repaired output validates, accept it (the tier self-corrected);
  if it still fails, the tier is structurally too weak -> escalate to a stronger
  model.

The self-repair step is the audit loop's re-test turned intelligent: instead of a
blind retry, the model is *told what was wrong* and given one chance to correct —
genuine self-correction, not luck.

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


def _retest_retry(caller: ModelCaller, task: Task) -> ReTest:
    """Plain retry, used when the tier was *unavailable* (no output to repair).

    Even if the retry succeeds we still escalate — a flaky/unreachable tier is an
    infrastructure risk — so this re-test only documents whether it was transient.
    """
    name = "retry_tier"
    desc = "retry the unavailable tier once"
    try:
        caller.call(task.prompt)
    except LLMUnavailable as exc:
        return ReTest(name, desc, True, [], f"retry also failed: {exc}")
    return ReTest(
        name, desc, False, [], "tier responded on retry, but the first call failed — escalate"
    )


def _repair_prompt(task: Task, failure_detail: str) -> str:
    return (
        f"{task.prompt}\n\n"
        f"Your previous response FAILED validation: {failure_detail}. "
        f"Return ONLY the corrected output that fixes this — no prose, no markdown."
    )


def _retest_self_repair(
    caller: ModelCaller, task: Task, failure_detail: str
) -> tuple[ReTest, bool, str | None]:
    """Re-prompt the same tier *with the validation error* so it can self-correct.

    Returns the re-test record, whether the anomaly reproduced (i.e. repair did
    NOT fix it), and the repaired output (only when it *did* validate).
    """
    name = "self_repair"
    desc = "re-prompt the same tier with the validation error and re-validate"
    try:
        repaired = caller.call(_repair_prompt(task, failure_detail))
    except LLMUnavailable as exc:
        return ReTest(name, desc, True, [], f"repair attempt failed: {exc}"), True, None
    res = task.validator(repaired)
    check = _check(res)
    if res.ok:
        concl = "self-repair succeeded: told what was wrong, the tier corrected its output"
        return ReTest(name, desc, False, [check], concl), False, repaired
    concl = "self-repair still failed: deterministic deficiency of this tier — escalate"
    return ReTest(name, desc, True, [check], concl), True, None


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
                retest = _retest_retry(caller, task)
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

            # 3) Unexpected: failed validation -> let the tier self-repair.
            #    Feed the *hint* (never the full detail, which may leak the answer).
            retest, reproduced, repaired = _retest_self_repair(caller, task, result.hint)
            if not reproduced and repaired is not None:
                # The tier fixed its own output once told what was wrong -> accept.
                log.attempts.append(
                    Attempt(
                        idx,
                        caller.name,
                        {"prompt": _preview(task.prompt)},
                        "validated-after-repair",
                        None,
                        0,
                        [check],
                        "unexpected",
                        [retest],
                        "accept",
                        f"failed then self-repaired; '{_preview(repaired)}'",
                    )
                )
                log.finalize("validated", None, caller.name)
                log.conclusion = (
                    "accepted after self-repair: the tier corrected its output once "
                    "told what was wrong"
                )
                return Validation(repaired, caller.name, log)

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
