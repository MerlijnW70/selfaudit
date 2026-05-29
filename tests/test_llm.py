"""Pytest suite for the LLM output validator (llm + llmauditor).

Every test is deterministic: model tiers are ScriptedCaller stubs, so no API key
or network is needed. The one real-engine path (AnthropicCaller) is exercised
with a monkeypatched client — still offline.
"""

from __future__ import annotations

import json
import os

import pytest

from selfaudit.llm import (
    AnthropicCaller,
    LLMUnavailable,
    ScriptedCaller,
    Task,
    _resolve_ca_bundle,
    _strip_code_fence,
    enable_os_truststore,
    exact_field_validator,
    json_schema_validator,
    load_dotenv,
)
from selfaudit.llmauditor import SelfAuditingValidator, ValidationFailed

_REQUIRED = {"name": str, "age": int}
_GOOD = '{"name": "Ada", "age": 36}'
_FENCED = "```json\n" + _GOOD + "\n```"
_PROSE = "Sure, here you go: Ada is 36."
_WRONGTYPE = '{"name": "Ada", "age": "old"}'


def _task() -> Task:
    return Task("person", "give me JSON", json_schema_validator(_REQUIRED))


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


def test_validator_accepts_well_formed_json() -> None:
    res = json_schema_validator(_REQUIRED)(_GOOD)
    assert res.ok
    assert res.violations == 0


def test_validator_strips_markdown_fences() -> None:
    res = json_schema_validator(_REQUIRED)(_FENCED)
    assert res.ok


def test_validator_rejects_non_json() -> None:
    res = json_schema_validator(_REQUIRED)(_PROSE)
    assert not res.ok
    assert res.violations == 1
    assert "not valid JSON" in res.detail


def test_validator_rejects_non_object() -> None:
    res = json_schema_validator(_REQUIRED)("[1, 2, 3]")
    assert not res.ok
    assert "not an object" in res.detail


def test_validator_reports_missing_and_wrong_type() -> None:
    res = json_schema_validator(_REQUIRED)(_WRONGTYPE)
    assert not res.ok
    assert res.violations == 1  # age wrong type (name present & correct)
    assert "age" in res.detail


def test_strip_code_fence_plain_passthrough() -> None:
    assert _strip_code_fence("  hello  ") == "hello"
    assert _strip_code_fence("```\nx\n```") == "x"


def test_exact_field_validator() -> None:
    v = exact_field_validator("product", 42)
    assert v('{"product": 42}').ok
    assert v('```json\n{"product": 42}\n```').ok  # fence-stripped
    assert not v("not json").ok
    assert not v("[1,2]").ok  # not an object
    assert not v('{"other": 42}').ok  # missing key
    wrong = v('{"product": 41}')
    assert not wrong.ok
    assert "expected 42" in wrong.detail  # full detail records the truth (audit log)
    # ...but the self-repair hint must NOT leak the expected value:
    assert "42" not in wrong.hint
    assert "incorrect" in wrong.hint


def test_value_validator_hint_does_not_leak_into_repair_prompt() -> None:
    # End-to-end: a tier that keeps answering wrong is never handed the answer,
    # so it cannot "self-repair" by echoing it -> the failure escalates honestly.
    from selfaudit.llm import exact_field_validator
    from selfaudit.llmauditor import _repair_prompt

    res = exact_field_validator("product", 33679062)('{"product": 33682062}')
    prompt = _repair_prompt(Task("calc", "compute it", lambda _t: res), res.hint)
    assert "33679062" not in prompt  # the correct answer never appears in the hint


# --------------------------------------------------------------------------- #
# Controller branches
# --------------------------------------------------------------------------- #


def test_direct_accept_when_first_tier_validates() -> None:
    v = SelfAuditingValidator([ScriptedCaller("haiku", [_GOOD])])
    res = v.run(_task())
    assert res.tier == "haiku"
    assert res.log.final_status == "validated"
    assert len(res.log.attempts) == 1
    accepted = res.log.attempts[0]
    assert accepted.classification == "expected"
    assert accepted.decision == "accept"
    assert accepted.outcome == "validated"
    assert any(rt.name == "corroborate_output" for rt in accepted.retests)


def test_accept_after_self_repair() -> None:
    # First reply fails; re-prompted with the error, the tier returns valid output.
    v = SelfAuditingValidator([ScriptedCaller("haiku", [_PROSE, _GOOD])])
    res = v.run(_task())
    assert res.tier == "haiku"
    assert res.log.final_status == "validated"
    accepted = res.log.attempts[-1]
    assert accepted.outcome == "validated-after-repair"
    assert accepted.classification == "unexpected"
    assert accepted.decision == "accept"
    repair = next(rt for rt in accepted.retests if rt.name == "self_repair")
    assert repair.reproduced_anomaly is False
    assert "self-repair succeeded" in repair.conclusion
    assert "self-repair" in res.log.conclusion


def test_escalation_when_tier_is_deterministically_too_weak() -> None:
    v = SelfAuditingValidator(
        [
            ScriptedCaller("haiku", [_WRONGTYPE]),  # always fails (repeats last reply)
            ScriptedCaller("sonnet", [_GOOD]),
        ]
    )
    res = v.run(_task())
    assert res.tier == "sonnet"
    assert res.log.final_status == "validated"
    assert len(res.log.attempts) == 2
    first = res.log.attempts[0]
    assert first.strategy == "haiku"
    assert first.classification == "unexpected"
    assert first.decision == "escalate"
    repair = next(rt for rt in first.retests if rt.name == "self_repair")
    assert repair.reproduced_anomaly is True


def test_all_tiers_exhausted_raises_with_log() -> None:
    v = SelfAuditingValidator(
        [
            ScriptedCaller("haiku", [_PROSE]),
            ScriptedCaller("sonnet", [_WRONGTYPE]),
        ]
    )
    with pytest.raises(ValidationFailed) as excinfo:
        v.run(_task())
    log = excinfo.value.log
    assert log.final_status == "unvalidated"
    assert len(log.attempts) == 2
    assert all(a.decision != "accept" for a in log.attempts)


def test_flaky_corroboration_is_accepted_with_caveat() -> None:
    # First call validates (accept), but the corroboration re-call fails -> caveat.
    v = SelfAuditingValidator([ScriptedCaller("haiku", [_GOOD, _PROSE])])
    res = v.run(_task())
    assert res.log.final_status == "validated"
    corr = next(rt for rt in res.log.attempts[0].retests if rt.name == "corroborate_output")
    assert "flaky" in corr.conclusion
    assert "caveat" in res.log.attempts[0].retests[0].conclusion


def test_unavailable_tier_escalates() -> None:
    # An empty scripted tier raises LLMUnavailable on call -> escalate to the next.
    v = SelfAuditingValidator(
        [
            ScriptedCaller("haiku", []),
            ScriptedCaller("sonnet", [_GOOD]),
        ]
    )
    res = v.run(_task())
    assert res.tier == "sonnet"
    first = res.log.attempts[0]
    assert first.outcome == "unavailable"
    assert first.decision == "escalate"


def test_scripted_caller_callable_form() -> None:
    caller = ScriptedCaller("echo", lambda prompt: f"got: {prompt}")
    assert caller.call("hi") == "got: hi"


def test_log_is_json_serializable_and_renders() -> None:
    v = SelfAuditingValidator([ScriptedCaller("haiku", [_GOOD])])
    log = v.run(_task()).log
    parsed = json.loads(log.to_json())
    assert parsed["final_status"] == "validated"
    text = log.render()
    assert "SELF-AUDIT REPORT" in text
    assert "output validated" in text


# --------------------------------------------------------------------------- #
# Real engine (AnthropicCaller) — offline, via a monkeypatched client
# --------------------------------------------------------------------------- #


def test_anthropic_caller_without_key_is_unavailable(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable):
        AnthropicCaller("x", "model-id").call("hi")


def test_anthropic_caller_happy_path_with_fake_client(monkeypatch) -> None:
    anthropic = pytest.importorskip("anthropic")

    class _Block:
        def __init__(self, type_: str, text: str) -> None:
            self.type = type_
            self.text = text

    class _Msg:
        content = [_Block("text", "hello "), _Block("image", "ignored"), _Block("text", "world")]

    class _Messages:
        def create(self, **kwargs):
            return _Msg()

    class _FakeClient:
        def __init__(self, *a, **k) -> None:
            self.messages = _Messages()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)  # default client, no custom CA
    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)
    out = AnthropicCaller("x", "model-id").call("hi")
    assert out == "hello world"  # only text blocks, concatenated


def test_load_dotenv_parses_and_respects_override(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "export QUOTED='hello world'",
                'PLAIN="bare"',
                "ANTHROPIC_API_KEY=sk-from-file",
                "noequalsline",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("QUOTED", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-already-set")

    parsed = load_dotenv(str(env))  # override=False
    assert parsed["QUOTED"] == "hello world"  # quotes + export stripped
    assert parsed["PLAIN"] == "bare"
    assert "noequalsline" not in parsed
    assert os.environ["QUOTED"] == "hello world"  # newly set
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-already-set"  # not overridden

    load_dotenv(str(env), override=True)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-from-file"  # now overridden


def test_load_dotenv_missing_file_is_noop() -> None:
    assert load_dotenv("does-not-exist-12345.env") == {}


def test_enable_os_truststore_missing_returns_false(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "truststore":
            raise ImportError("simulated: truststore not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert enable_os_truststore() is False


def test_enable_os_truststore_injects_when_present(monkeypatch) -> None:
    truststore = pytest.importorskip("truststore")
    called: dict[str, bool] = {}
    monkeypatch.setattr(truststore, "inject_into_ssl", lambda: called.setdefault("done", True))
    assert enable_os_truststore() is True
    assert called["done"] is True


def test_resolve_ca_bundle_precedence(monkeypatch) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    assert _resolve_ca_bundle(None) is None
    monkeypatch.setenv("SSL_CERT_FILE", "/env/ca.pem")
    assert _resolve_ca_bundle(None) == "/env/ca.pem"
    assert _resolve_ca_bundle("/explicit/ca.pem") == "/explicit/ca.pem"  # explicit wins


def test_ca_bundle_builds_a_verifying_http_client(monkeypatch) -> None:
    """A configured CA bundle is passed to a verifying httpx client — TLS stays on,
    only the trust anchor changes (the fix for TLS-intercepting proxies)."""
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx")

    recorded: dict[str, object] = {}

    class _FakeHttpx:
        def __init__(self, *a, verify=None, **k) -> None:
            recorded["verify"] = verify

    class _Block:
        type = "text"
        text = "ok"

    class _Msg:
        content = [_Block()]

    class _Messages:
        def create(self, **kwargs):
            return _Msg()

    class _FakeClient:
        def __init__(self, *a, **k) -> None:
            recorded["http_client"] = k.get("http_client")
            self.messages = _Messages()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "Client", _FakeHttpx)
    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)

    out = AnthropicCaller("x", "model-id", ca_bundle="/corp/ca.pem").call("hi")
    assert out == "ok"
    assert recorded["verify"] == "/corp/ca.pem"
    assert isinstance(recorded["http_client"], _FakeHttpx)


# --------------------------------------------------------------------------- #
# Demo runner (scripted scenarios only)
# --------------------------------------------------------------------------- #


def test_llm_demo_runs_and_writes_log(tmp_path, monkeypatch, capsys) -> None:
    from selfaudit.llmdemo import main

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # skip the live run
    monkeypatch.chdir(tmp_path)
    main()
    out = capsys.readouterr().out
    assert "direct-accept" in out
    assert "escalate" in out
    written = tmp_path / "llm_audit_log.json"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["final_status"] == "validated"
