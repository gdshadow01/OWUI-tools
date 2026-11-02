#!/usr/bin/env python3
"""Tests for enforce_effort_policy (managed-alias client-effort strip, plan 261031).

Runs BOTH ways:
  - standalone:  python3 test_enforce_effort_policy.py   (exit 0 = all pass)
  - under pytest: pytest test_enforce_effort_policy.py

No litellm/fastapi install required on the host: both imports are stubbed
before the module import when not already importable (same pattern as
test_strip_thinking.py / test_codex_session_id.py).

Sections:
  T1.1 strip-vs-reject matrix       T1.4 idempotency
  T1.2 non-managed passthrough      T1.5 no effort-level constants (static)
  T1.3 unrelated-field preservation T1.6 config.yml drift/sync (a-e + registration)
  T1.7 real-proxy call-type contract ('acompletion' + enum normalization)
"""
import ast
import asyncio
import copy
import io
import os
import sys
import tokenize
import types

try:
    import litellm  # noqa: F401
except ImportError:
    _cl = types.ModuleType("litellm.integrations.custom_logger")
    _cl.CustomLogger = object
    sys.modules.setdefault("litellm", types.ModuleType("litellm"))
    sys.modules.setdefault("litellm.integrations", types.ModuleType("litellm.integrations"))
    sys.modules.setdefault("litellm.integrations.custom_logger", _cl)

try:
    import fastapi  # noqa: F401
except ImportError:
    _fa = types.ModuleType("fastapi")

    class _HTTPException(Exception):
        def __init__(self, status_code, detail=None):
            super().__init__(f"{status_code}: {detail}")
            self.status_code = status_code
            self.detail = detail

    _fa.HTTPException = _HTTPException
    sys.modules.setdefault("fastapi", _fa)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import enforce_effort_policy as eep  # noqa: E402
from fastapi import HTTPException  # noqa: E402

MANAGED = [
    "hermes-plan",
    "hermes-execute",
    "hermes-tasks",
    "hermes-coding-plan",
    "hermes-coding-execute",
    "hermes-heartbeat-plan",
    "hermes-heartbeat-execute",
    "hermes-heartbeat-tasks",
    "hermes-vision",
    "hermes-fallback",
]

# The only hermes-* aliases allowed OUTSIDE MANAGED_GROUPS (rollback twins,
# client-controlled). Any other hermes-* model_name in config.yml must FAIL
# drift test_hermes_names_classified until consciously classified.
ALLOWED_HERMES_TWINS = {"hermes-plan-zai", "hermes-execute-zai", "hermes-tasks-zai"}

# Frozen effort-encoding map: alias -> (backing prefix, carrier path or None,
# expected level or None). Mismatch = FAIL (drift test e).
EXPECTED_EFFORT = {
    "hermes-plan": ("mistral/", ("extra_body", "reasoning_effort"), "high"),
    "hermes-execute": ("mistral/", ("extra_body", "reasoning_effort"), "high"),
    "hermes-tasks": ("mistral/", ("extra_body", "reasoning_effort"), "high"),
    "hermes-coding-plan": ("zai/", ("extra_body", "reasoning_effort"), "high"),
    "hermes-coding-execute": ("zai/", ("extra_body", "reasoning_effort"), "high"),
    "hermes-heartbeat-plan": ("zai/", ("extra_body", "reasoning_effort"), "high"),
    "hermes-heartbeat-execute": ("zai/", ("extra_body", "reasoning_effort"), "high"),
    "hermes-heartbeat-tasks": ("zai/", ("extra_body", "reasoning_effort"), "high"),
    "hermes-fallback": ("openai/", ("extra_body", "reasoning_effort"), "high"),
    "hermes-vision": ("mistral/", ("extra_body", "reasoning_effort"), "low"),
}

FROZEN_CALLBACK_HEAD = [
    "strip_hermes.strip_hermes_handler",
    "fallback_notify.fallback_notifier",
    "strip_thinking.strip_thinking_handler",
    "codex_session_id.codex_session_id_hook",
]

THINKING = {"type": "enabled", "clear_thinking": False}
MESSAGES = [{"role": "user", "content": "hi"}]


def _base(model, **extra):
    d = {
        "model": model,
        "messages": copy.deepcopy(MESSAGES),
        "max_tokens": 64,
        "metadata": {"existing": "keep-me"},
    }
    d.update(extra)
    return d


def _run(data, call_type="completion"):
    return asyncio.run(
        eep.effort_policy_handler.async_pre_call_hook(None, None, data, call_type)
    )


def _run_expect_reject(data, call_type="completion"):
    try:
        _run(data, call_type)
    except HTTPException as exc:
        assert exc.status_code == 400, f"expected 400, got {exc.status_code}"
        detail = exc.detail
        assert isinstance(detail, dict) and detail.get("error") == "managed-alias-effort-policy", detail
        return detail
    raise AssertionError(f"expected policy rejection, got none for {data!r}")


def _walk(obj, path):
    for key in path:
        obj = obj[key]
    return obj


# ---------------------------------------------------------------- T1.1 matrix


def test_no_carriers_is_noop():
    for m in MANAGED:
        d = _base(m)
        before = copy.deepcopy(d)
        out = _run(d)
        assert out is d, m
        assert out == before, m


def test_reasoning_effort_any_value_popped():
    for value in ["low", "medium", "high", "xhigh", "none", "default", {"effort": "low"}, None, 123, True]:
        for m in MANAGED:
            d = _base(m, reasoning_effort=copy.deepcopy(value))
            out = _run(d)
            assert "reasoning_effort" not in out, (m, value)
            assert out["messages"] == MESSAGES and out["max_tokens"] == 64, (m, value)


def test_thinking_any_form_popped():
    for value in [dict(THINKING), True, False, "auto", None, []]:
        d = _base("hermes-plan", thinking=copy.deepcopy(value))
        out = _run(d)
        assert "thinking" not in out, value


def test_reasoning_effort_only_removed_whole_key():
    d = _base("hermes-execute", reasoning={"effort": "xhigh"})
    out = _run(d)
    assert "reasoning" not in out
    assert out["messages"] == MESSAGES


def test_reasoning_empty_dict_removed():
    d = _base("hermes-tasks", reasoning={})
    out = _run(d)
    assert "reasoning" not in out


def test_reasoning_residual_rejected():
    for reasoning in [{"effort": "low", "summary": "auto"}, {"summary": "auto"}, {"summary": "auto", "other": 1}]:
        d = _base("hermes-plan", reasoning=dict(reasoning))
        detail = _run_expect_reject(d)
        assert detail["carrier"] == "reasoning", detail
        assert detail["reason"] == "non-effort-residual", detail


def test_reasoning_non_dict_rejected():
    for value in ["high", True, False, 7, ["auto"]]:
        d = _base("hermes-plan", reasoning=value)
        detail = _run_expect_reject(d)
        assert detail["carrier"] == "reasoning", detail
        assert detail["reason"] == "unsupported-type", detail


def test_output_config_effort_popped_residual_preserved():
    d = _base("hermes-fallback", output_config={"effort": "high"})
    out = _run(d)
    assert "output_config" not in out

    verifiers = {"max_output_tokens": 4096}
    d = _base("hermes-fallback", output_config={"effort": "high", "verifiers": verifiers})
    out = _run(d)
    assert out["output_config"] == {"verifiers": verifiers}, "residual sub-keys must be preserved"
    assert "effort" not in out["output_config"]


def test_output_config_non_dict_rejected():
    for value in ["high", True, 3, []]:
        d = _base("hermes-vision", output_config=value)
        detail = _run_expect_reject(d)
        assert detail["carrier"] == "output_config", detail
        assert detail["reason"] == "unsupported-type", detail


def test_extra_body_effort_only_forms_removed_entirely():
    forms = [
        {"reasoning_effort": "high"},
        {"reasoning": {"effort": "low"}},
        {"thinking": dict(THINKING)},
        {"output_config": {"effort": "high"}},
        {},
        {
            "reasoning_effort": "high",
            "reasoning": {"effort": "low"},
            "thinking": dict(THINKING),
            "output_config": {"effort": "high"},
        },
    ]
    for eb in forms:
        d = _base("hermes-heartbeat-plan", extra_body=copy.deepcopy(eb))
        out = _run(d)
        assert "extra_body" not in out, eb


def test_extra_body_residual_rejected():
    for eb, carrier in [
        ({"foo": 1}, "extra_body"),
        ({"reasoning_effort": "high", "foo": 1}, "extra_body"),
        ({"reasoning": {"summary": "auto"}}, "extra_body.reasoning"),
        ({"reasoning": {"effort": "high", "summary": "auto"}}, "extra_body.reasoning"),
        ({"output_config": {"effort": "high", "verifiers": {}}}, "extra_body"),
        ({"reasoning_effort": "high", "reasoning": {"summary": "auto"}}, "extra_body.reasoning"),
    ]:
        d = _base("hermes-execute", extra_body=copy.deepcopy(eb))
        detail = _run_expect_reject(d)
        assert detail["carrier"] == carrier, (eb, detail)


def test_extra_body_non_dict_rejected():
    for value in ["high", True, 5, ["reasoning_effort"]]:
        d = _base("hermes-coding-execute", extra_body=value)
        detail = _run_expect_reject(d)
        assert detail["carrier"] == "extra_body", detail
        assert detail["reason"] == "unsupported-type", detail


def test_combined_carriers_all_stripped():
    d = _base(
        "hermes-plan",
        reasoning_effort="medium",
        thinking=dict(THINKING),
        reasoning={"effort": "low"},
        output_config={"effort": "high", "verifiers": {"enabled": True}},
        extra_body={"reasoning_effort": "low", "reasoning": {"effort": "high"}},
        stream=True,
    )
    out = _run(d)
    for key in ("reasoning_effort", "thinking", "reasoning", "extra_body"):
        assert key not in out, key
    assert out["output_config"] == {"verifiers": {"enabled": True}}
    assert out["stream"] is True and out["messages"] == MESSAGES


# ------------------------------------------------------- T1.2 non-managed pass


def test_non_managed_passthrough_byte_identical():
    others = [
        "hermes-plan-zai",
        "hermes-execute-zai",
        "hermes-tasks-zai",
        "zai-glm-5.3",
        "glm-5.3-flash",
        "deepseek-v4-flash(0.14/0.28)",
        "deepseek-v4-pro(0.435/0.87)",
        "openrouter-zai-glm-4.7(0.38/1.98)",
        "mistral-vibe-glm-5.3-flash(0.15/0.5)",
        "chatgpt-gpt-6.1-sol",
        "unknown-model",
        "",
    ]
    client_control = {
        "reasoning_effort": "high",
        "thinking": dict(THINKING),
        "reasoning": {"effort": "low"},
        "extra_body": {"foo": 1},
    }
    for m in others:
        d = _base(m, **copy.deepcopy(client_control))
        before = copy.deepcopy(d)
        out = _run(d)
        assert out is d, m
        assert out == before, m


def test_weird_model_values_passthrough():
    client_control = {"reasoning_effort": "high", "thinking": dict(THINKING)}
    for bad in (None, 123, ["hermes-plan"], {"model": "hermes-plan"}):
        d = _base(bad, **copy.deepcopy(client_control))
        before = copy.deepcopy(d)
        out = _run(d)
        assert out is d and out == before, bad
    d = _base(None, **copy.deepcopy(client_control))
    del d["model"]
    before = copy.deepcopy(d)
    out = _run(d)
    assert out is d and out == before, "missing model must pass through untouched"


def test_other_call_types_untouched():
    for call_type in ("text_completion", "embeddings", "image_generation", "moderation", "audio_transcription"):
        d = _base("hermes-plan", reasoning_effort="high", thinking=dict(THINKING))
        before = copy.deepcopy(d)
        out = _run(d, call_type=call_type)
        assert out is d and out == before, call_type


# ------------------------------------- T1.7 real-proxy call-type contract


def test_real_proxy_call_type_acompletion_engages():
    """Runtime-proven v1.103.2 contract: the REAL /v1/chat/completions route
    presents call_type='acompletion' to litellm_settings callbacks (an
    instrumented probe on the pinned image logged it for every real-path
    request; the original 'completion' gate made the hook inert there).
    The strip/reject matrix must behave identically under that form."""
    d = _base("hermes-plan", reasoning_effort="medium", thinking=dict(THINKING))
    out = _run(d, call_type="acompletion")
    assert "reasoning_effort" not in out and "thinking" not in out

    d = _base("hermes-execute", reasoning={"effort": "low"})
    out = _run(d, call_type="acompletion")
    assert "reasoning" not in out

    d = _base("hermes-tasks", output_config={"effort": "high", "verifiers": {"v": 1}})
    out = _run(d, call_type="acompletion")
    assert out["output_config"] == {"verifiers": {"v": 1}}

    # binding rejection forms under the real call_type
    d = _base("hermes-plan", reasoning={"effort": "low", "summary": "auto"})
    detail = _run_expect_reject(d, call_type="acompletion")
    assert detail["carrier"] == "reasoning" and detail["reason"] == "non-effort-residual"

    d = _base("hermes-execute", extra_body={"foo": 1})
    detail = _run_expect_reject(d, call_type="acompletion")
    assert detail["reason"] == "residual-keys"

    # THE binding acceptance probe of the failed deploy (t_7ebff3fe):
    # managed alias + reasoning:false -> policy rejection, on ALL ten aliases.
    for m in MANAGED:
        d = _base(m, reasoning=False)
        detail = _run_expect_reject(d, call_type="acompletion")
        assert detail["carrier"] == "reasoning", m
        assert detail["reason"] == "unsupported-type", m
        assert detail["model"] == m


def test_enum_call_type_members_normalized():
    """If litellm ever hands the CallTypes enum member instead of its string
    value, .value normalization must keep completion forms working and
    non-completion forms untouched."""

    class _FakeEnumMember:
        def __init__(self, value):
            self.value = value

    d = _base("hermes-plan", reasoning_effort="medium")
    out = _run(d, call_type=_FakeEnumMember("acompletion"))
    assert "reasoning_effort" not in out

    d = _base("hermes-plan", reasoning_effort="medium", thinking=dict(THINKING))
    out = _run(d, call_type=_FakeEnumMember("completion"))
    assert "reasoning_effort" not in out and "thinking" not in out

    d = _base("hermes-plan", reasoning_effort="medium")
    before = copy.deepcopy(d)
    out = _run(d, call_type=_FakeEnumMember("text_completion"))
    assert out is d and out == before


def test_non_dict_data_passthrough():
    for bad in (None, [], "str", 42):
        out = asyncio.run(
            eep.effort_policy_handler.async_pre_call_hook(None, None, bad, "completion")
        )
        assert out is bad


# --------------------------------------------- T1.3 unrelated-field preservation


def test_unrelated_fields_survive_every_strip_case():
    tools = [{"type": "function", "function": {"name": "get_time", "parameters": {}}}]
    fixed = {
        "messages": [{"role": "user", "content": "what time is it?"}],
        "tools": tools,
        "metadata": {"session_id": "abc-123", "trace": 1},
        "stream": True,
        "max_tokens": 128,
        "temperature": 0.2,
    }
    cases = [
        {"reasoning_effort": "medium"},
        {"thinking": dict(THINKING)},
        {"reasoning": {"effort": "low"}},
        {"output_config": {"effort": "high", "verifiers": {"x": 1}}},
        {"extra_body": {"reasoning_effort": "low"}},
        {
            "reasoning_effort": "high",
            "thinking": dict(THINKING),
            "reasoning": {"effort": "low"},
            "output_config": {"effort": "high"},
            "extra_body": {"reasoning": {"effort": "low"}},
        },
    ]
    for extra in cases:
        d = _base("hermes-coding-plan", **copy.deepcopy(fixed), **copy.deepcopy(extra))
        out = _run(d)
        for key, value in fixed.items():
            assert out[key] == value, (extra, key)


# ---------------------------------------------------------- T1.4 idempotency


def test_idempotent_strip_cases():
    cases = [
        {"reasoning_effort": "medium", "thinking": dict(THINKING)},
        {"reasoning": {"effort": "low"}},
        {"output_config": {"effort": "high", "verifiers": {"v": 1}}},
        {"extra_body": {"reasoning": {"effort": "high"}}},
        {
            "reasoning_effort": "none",
            "reasoning": {"effort": "low"},
            "output_config": {"effort": "high"},
            "extra_body": {"reasoning_effort": "low", "thinking": dict(THINKING)},
        },
    ]
    for extra in cases:
        once = _run(_base("hermes-tasks", **copy.deepcopy(extra)))
        twice_target = _base("hermes-tasks", **copy.deepcopy(extra))
        _run(twice_target)
        again = _run(twice_target)
        assert again == once, extra


def test_rejects_are_deterministic_and_leave_no_pass():
    payload = {"summary": "auto"}
    for _ in range(2):
        detail = _run_expect_reject(_base("hermes-plan", reasoning=dict(payload)))
        assert detail["model"] == "hermes-plan"
    for _ in range(2):
        detail = _run_expect_reject(_base("hermes-plan", extra_body={"foo": 1}))
        assert detail["reason"] == "residual-keys"


# ---------------------------------------- T1.5 static: no effort-level constants


def test_no_effort_level_constants():
    LEVELS = {"low", "medium", "high", "xhigh", "max", "none"}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enforce_effort_policy.py")
    with open(path, "rb") as fh:
        source = fh.read()
    offenders = []
    for tok in tokenize.tokenize(io.BytesIO(source).readline):
        if tok.type != tokenize.STRING:
            continue
        try:
            value = ast.literal_eval(tok.string)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, str) and value.strip().casefold() in LEVELS:
            offenders.append((tok.start, value))
    assert not offenders, f"effort level constants found in module: {offenders}"


# --------------------------------------- T1.6 drift / sync against config.yml


def _load_cfg():
    import yaml

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def _entries(cfg):
    entries = {}
    for entry in cfg.get("model_list", []):
        entries[entry["model_name"]] = entry
    return entries


def test_managed_groups_exact_membership():
    assert eep.MANAGED_GROUPS == frozenset(MANAGED), sorted(
        set(eep.MANAGED_GROUPS) ^ set(MANAGED)
    )


def test_managed_aliases_exist_in_config():
    cfg = _load_cfg()
    entries = _entries(cfg)
    missing = sorted(set(MANAGED) - set(entries))
    assert not missing, f"managed aliases missing from config.yml model_list: {missing}"


def test_hermes_names_classified():
    cfg = _load_cfg()
    hermes_names = {n for n in _entries(cfg) if n.startswith("hermes-")}
    allowed = set(MANAGED) | ALLOWED_HERMES_TWINS
    unclassified = sorted(hermes_names - allowed)
    assert not unclassified, (
        f"new hermes-* alias(es) in config.yml not classified in "
        f"enforce_effort_policy.MANAGED_GROUPS or the -zai twin set: {unclassified}"
    )


def test_effort_encoding_freeze():
    cfg = _load_cfg()
    entries = _entries(cfg)
    for alias, (backing, path, level) in EXPECTED_EFFORT.items():
        assert alias in entries, alias
        params = entries[alias]["litellm_params"]
        assert params["model"].startswith(backing), (alias, params["model"], backing)
        extra_body = params.get("extra_body")
        if path is None:
            assert extra_body is None or (
                "reasoning_effort" not in extra_body and "reasoning" not in extra_body
            ), (
                f"{alias}: explicit no-effort policy violated: {extra_body}"
            )
            continue
        assert extra_body is not None, f"{alias}: missing extra_body"
        value = _walk(params, path)
        assert value == level, f"{alias}: {path} = {value!r}, expected {level!r}"


def test_fallback_targets_are_managed():
    cfg = _load_cfg()
    fallbacks = cfg.get("litellm_settings", {}).get("fallbacks", [])
    assert fallbacks, "config.yml fallbacks came back empty — parser broken?"
    for entry in fallbacks:
        assert isinstance(entry, dict) and len(entry) == 1, entry
        for source, targets in entry.items():
            stray = sorted(set(targets) - set(MANAGED))
            assert not stray, f"fallback chain {source} targets non-managed alias(es): {stray}"


def test_registration_line_present_and_last():
    cfg = _load_cfg()
    callbacks = cfg.get("litellm_settings", {}).get("callbacks", [])
    assert callbacks[-1] == "enforce_effort_policy.effort_policy_handler", callbacks
    assert callbacks[: len(FROZEN_CALLBACK_HEAD)] == FROZEN_CALLBACK_HEAD, callbacks
    assert len(callbacks) == len(FROZEN_CALLBACK_HEAD) + 1, callbacks


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()
