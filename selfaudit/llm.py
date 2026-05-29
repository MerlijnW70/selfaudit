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


def load_dotenv(path: str = ".env", *, override: bool = False) -> dict[str, str]:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into ``os.environ``.

    Blank lines and ``#`` comments are ignored; an optional ``export`` prefix and
    surrounding quotes around the value are stripped. Existing environment
    variables are left untouched unless ``override`` is true. A missing file is a
    no-op. Returns the parsed pairs (also when nothing was applied to the env).

    Pure stdlib — no python-dotenv dependency.
    """
    parsed: dict[str, str] = {}
    if not os.path.exists(path):
        return parsed
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if not key:
                continue
            parsed[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
    return parsed


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


def enable_os_truststore() -> bool:
    """Route Python's TLS verification through the operating system's trust store
    (via the optional ``truststore`` package). Returns ``True`` when enabled.

    This is the secure fix for TLS-intercepting corporate proxies: the OS often
    trusts the proxy's re-signing CA even when Python's bundled OpenSSL trust does
    not. Verification stays **on** — it is delegated to the platform, not
    disabled. A no-op returning ``False`` when ``truststore`` is not installed.
    """
    try:
        import truststore
    except ImportError:
        return False
    truststore.inject_into_ssl()
    return True


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


def _parse_json_object(text: str) -> tuple[dict[str, Any] | None, ValidationResult | None]:
    """Fence-strip and parse ``text`` as a JSON object.

    Returns ``(data, None)`` on success, or ``(None, failure)`` with the
    :class:`ValidationResult` describing why it is not a JSON object. Shared by
    the validators so the unparseable / not-an-object branches live in one place.
    """
    cleaned = _strip_code_fence(text)
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, ValidationResult(False, 1, f"not valid JSON: {exc}")
    if not isinstance(data, dict):
        return None, ValidationResult(False, 1, "JSON is not an object")
    return data, None


def json_schema_validator(required: dict[str, type]) -> Validator:
    """Build a validator: the output must parse as a JSON object that contains
    each ``required`` key with a value of the given Python type.

    Counts one violation per problem (unparseable, not an object, missing key,
    wrong type), so ``violations == 0`` exactly when the output is well-formed.
    """

    def validate(text: str) -> ValidationResult:
        data, failure = _parse_json_object(text)
        if failure is not None:
            return failure
        assert data is not None
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


def exact_field_validator(key: str, expected: object) -> Validator:
    """Build a validator: the output must parse as a JSON object whose ``key``
    equals ``expected``. Useful for tasks with one objectively correct answer
    (e.g. arithmetic) — the validator *computes* the truth, so it cannot drift.
    """

    def validate(text: str) -> ValidationResult:
        data, failure = _parse_json_object(text)
        if failure is not None:
            return failure
        assert data is not None
        if key not in data:
            return ValidationResult(False, 1, f"missing key '{key}'")
        if data[key] != expected:
            return ValidationResult(False, 1, f"key '{key}' = {data[key]!r}, expected {expected!r}")
        return ValidationResult(True, 0, f"key '{key}' equals the expected value")

    return validate


def default_tiers() -> list[ModelCaller]:
    """The default escalation ladder of real model tiers: haiku -> sonnet -> opus."""
    return [
        AnthropicCaller("haiku", HAIKU),
        AnthropicCaller("sonnet", SONNET),
        AnthropicCaller("opus", OPUS),
    ]
