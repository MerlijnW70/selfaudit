"""LLM output validation: the engine for the third application of the harness.

This file contains *no* audit logic. It provides the interchangeable pieces the
controller in ``llmauditor.py`` drives:

* :class:`Task`        - the prompt to satisfy + a checkable validator
* :class:`ModelCaller` - the interface a model tier implements (``name`` + ``call``)
* :class:`AnthropicCaller` - a real tier backed by the Anthropic API (lazy import)
* :class:`ScriptedCaller`  - a deterministic tier for tests/offline demos
* :func:`json_schema_validator` - a pure-stdlib "output must be JSON like this"

The invariant the auditor checks is binary: ``validator(output)`` either passes
(0 violations) or not. The escalation ladder is the list of model tiers — cheap
and fast first, stronger (and pricier) as a fallback.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# Model IDs of the default escalation ladder (cheap/fast -> strong).
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"
OPUS = "claude-opus-4-8"


@dataclass
class ValidationResult:
    """Outcome of checking one model output against a :class:`Task`'s validator."""

    ok: bool
    violations: int  # 0 == satisfied; the auditor uses this as the measured value
    detail: str


Validator = Callable[[str], ValidationResult]


@dataclass
class Task:
    """Produce text that satisfies ``validator``; escalate model tiers if needed."""

    name: str
    prompt: str
    validator: Validator


@runtime_checkable
class ModelCaller(Protocol):
    """A model tier: it has a ``name`` and turns a prompt into a text reply."""

    name: str

    def call(self, prompt: str) -> str: ...


class LLMUnavailable(Exception):
    """A real model tier cannot run (no SDK / no API key / transport error)."""


def _resolve_ca_bundle(explicit: str | None) -> str | None:
    """Pick the CA bundle to verify TLS against: an explicit path wins, else the
    standard ``SSL_CERT_FILE`` env var, else ``None`` (the SDK's own default).

    This is the secure escape hatch for hosts behind a TLS-intercepting proxy:
    point it at the corporate root CA. Verification is *never* disabled.
    """
    return explicit or os.environ.get("SSL_CERT_FILE") or None


class AnthropicCaller:
    """A real model tier backed by the Anthropic API.

    The ``anthropic`` SDK is imported lazily so the package imports fine without
    it; a missing SDK or key surfaces as :class:`LLMUnavailable`, which the
    controller treats as a clean tier failure (it escalates) rather than a crash.

    ``ca_bundle`` (or the ``SSL_CERT_FILE`` env var) points TLS verification at a
    custom root CA — needed behind corporate proxies that re-sign HTTPS. TLS
    verification stays on; this only changes *which* trust anchor is used.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        max_tokens: int = 1024,
        ca_bundle: str | None = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.ca_bundle = ca_bundle

    def _client_kwargs(self) -> dict[str, Any]:
        """Build kwargs for ``anthropic.Anthropic`` — an explicit verifying HTTP
        client only when a custom CA bundle is configured."""
        bundle = _resolve_ca_bundle(self.ca_bundle)
        if bundle is None:
            return {}
        import httpx

        return {"http_client": httpx.Client(verify=bundle)}

    def call(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the SDK
            raise LLMUnavailable("the 'anthropic' SDK is not installed") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
        try:
            client = anthropic.Anthropic(**self._client_kwargs())
            msg = client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # pragma: no cover - network/transport errors
            raise LLMUnavailable(f"API call failed: {exc}") from exc
        return "".join(
            getattr(block, "text", "")
            for block in msg.content
            if getattr(block, "type", None) == "text"
        )


class ScriptedCaller:
    """A deterministic model tier for tests and offline demos.

    ``replies`` is either a fixed list (consumed one per ``call``, the last reply
    repeating once exhausted) or a callable ``prompt -> reply``. No network, no
    key — so the audit/escalation logic is fully testable.
    """

    def __init__(self, name: str, replies: list[str] | Callable[[str], str]) -> None:
        self.name = name
        self._replies = replies
        self._i = 0

    def call(self, prompt: str) -> str:
        if callable(self._replies):
            return self._replies(prompt)
        if not self._replies:
            raise LLMUnavailable(f"scripted tier {self.name!r} has no replies")
        reply = self._replies[min(self._i, len(self._replies) - 1)]
        self._i += 1
        return reply


def _strip_code_fence(text: str) -> str:
    """Models often wrap JSON in ```json ... ``` fences; peel them off."""
    s = text.strip()
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def json_schema_validator(required: dict[str, type]) -> Validator:
    """Build a validator: the output must parse as a JSON object that contains
    each ``required`` key with a value of the given Python type.

    Counts one violation per problem (unparseable, not an object, missing key,
    wrong type), so ``violations == 0`` exactly when the output is well-formed.
    """

    def validate(text: str) -> ValidationResult:
        cleaned = _strip_code_fence(text)
        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError) as exc:
            return ValidationResult(False, 1, f"not valid JSON: {exc}")
        if not isinstance(data, dict):
            return ValidationResult(False, 1, "JSON is not an object")
        problems: list[str] = []
        for key, typ in required.items():
            if key not in data:
                problems.append(f"missing key '{key}'")
            elif not isinstance(data[key], typ):
                got = type(data[key]).__name__
                problems.append(f"key '{key}' should be {typ.__name__}, got {got}")
        if problems:
            return ValidationResult(False, len(problems), "; ".join(problems))
        keys = ", ".join(required)
        return ValidationResult(True, 0, f"valid JSON object with keys: {keys}")

    return validate


def default_tiers() -> list[ModelCaller]:
    """The default escalation ladder of real model tiers: haiku -> sonnet -> opus."""
    return [
        AnthropicCaller("haiku", HAIKU),
        AnthropicCaller("sonnet", SONNET),
        AnthropicCaller("opus", OPUS),
    ]
