#!/usr/bin/env python3
"""Tests for strip_mistral_reasoning.

Runs BOTH ways:
  - standalone:  python3 test_strip_mistral_reasoning.py   (exit 0 = all pass)
  - under pytest: pytest test_strip_mistral_reasoning.py

No litellm install required: the litellm imports are stubbed before the
module import when litellm is not already importable.
"""
import asyncio
import os
import sys
import types

try:
    import litellm  # noqa: F401
except ImportError:
    _cl = types.ModuleType("litellm.integrations.custom_logger")
    _cl.CustomLogger = object
    sys.modules.setdefault("litellm", types.ModuleType("litellm"))
    sys.modules.setdefault("litellm.integrations", types.ModuleType("litellm.integrations"))
    sys.modules.setdefault("litellm.integrations.custom_logger", _cl)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import strip_mistral_reasoning as smr  # noqa: E402

HISTORY = [
    {"role": "user", "content": "first question"},
    {
        "role": "assistant",
        "content": "first answer",
        "reasoning": "long scratchpad from turn one",
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}],
        "reasoning_content": "tool-turn scratchpad",
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "file.txt"},
    {
        "role": "user",
        "content": "follow-up question",
        "reasoning": "user messages must NEVER be touched (nonstandard but harmless)",
    },
]


def _base(model, messages=None):
    return {
        "model": model,
        "messages": [dict(m) for m in (messages if messages is not None else HISTORY)],
        "max_tokens": 512,
    }


def _run(data, call_type="acompletion"):
    hook = smr.strip_mistral_reasoning_handler.async_pre_call_hook
    return asyncio.run(hook(None, None, data, call_type))


def _all_groups():
    return sorted(smr.StripMistralReasoningHandler.MISTRAL_STRICT_GROUPS)


def test_strips_reasoning_from_assistant_history_all_groups():
    for m in _all_groups():
        out = _run(_base(m))
        msgs = out["messages"]
        assert msgs[0] == HISTORY[0], m  # user turn untouched
        assert "reasoning" not in msgs[1], m
        assert msgs[1]["content"] == "first answer", m
        assert "reasoning_content" not in msgs[2], m
        assert msgs[2]["tool_calls"] == HISTORY[2]["tool_calls"], m  # tool calls preserved
        assert msgs[3] == HISTORY[3], m  # tool result untouched
        assert msgs[4] == HISTORY[4], m  # user message reasoning preserved


def test_no_carriers_is_noop():
    for m in _all_groups():
        d = _base(m, messages=[{"role": "user", "content": "hi"}])
        before = dict(d)
        out = _run(d)
        assert out is d and out == before, m


def test_non_member_groups_untouched():
    others = [
        "hermes-fallback",  # openai/deepseek backing
        "hermes-heartbeat-plan",  # zai backing
        "hermes-heartbeat-execute",
        "hermes-plan-zai",  # rollback twin, zai backing
        "zai-glm-5.3-flash",
        "deepseek-v4-flash(0.14/0.28)",
        "openrouter-zai-glm-4.5-air-free",
        "chatgpt-gpt-6.1-sol",
        "totally-unknown-model",
    ]
    for m in others:
        out = _run(_base(m))
        assert out["messages"] == HISTORY, m


def test_body_level_reasoning_popped_any_form():
    for value in [False, True, {"effort": "low"}, "auto", None]:
        for m in ("hermes-execute", "mistral-vibe-glm-5.3(1.4/4.4)"):
            d = _base(m, messages=[{"role": "user", "content": "hi"}])
            d["reasoning"] = value
            out = _run(d)
            assert "reasoning" not in out, (m, value)


def test_extra_body_reasoning_carriers_popped_others_kept():
    d = _base("hermes-plan", messages=[{"role": "user", "content": "hi"}])
    d["extra_body"] = {
        "reasoning": False,
        "reasoning_content": "x",
        "other_param": "keep-me",
    }
    out = _run(d)
    assert out["extra_body"] == {"other_param": "keep-me"}


def test_call_type_gate():
    d = _base("hermes-execute")
    for ct in ("text_completion", "embeddings", "moderation"):
        out = _run(_base("hermes-execute"), ct)
        assert out["messages"] == HISTORY, ct
    # both completion forms engage; enum member normalizes via .value
    for ct in ("completion", "acompletion"):
        out = _run(_base("hermes-execute"), ct)
        assert "reasoning" not in out["messages"][1], ct


def test_non_str_model_untouched():
    for model in (None, 42, {"model": "hermes-execute"}):
        d = _base("hermes-execute")
        d["model"] = model
        assert _run(d)["messages"] == HISTORY
    d = _base("hermes-execute")
    del d["model"]
    assert _run(d)["messages"] == HISTORY


def test_idempotent_double_run():
    d = _base("hermes-execute")
    once = _run(d)
    twice = _run(once)
    assert twice["messages"] == once["messages"]
    assert "reasoning" not in twice["messages"][1]


def _config_pairs():
    """(model_name, model) pairs scraped from config.yml in this dir."""
    pairs = []
    cur = None
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("- model_name:"):
                cur = s.split(":", 1)[1].strip().strip('"').strip("'")
            elif s.startswith("model:") and cur is not None:
                pairs.append((cur, s.split(":", 1)[1].strip().strip('"').strip("'")))
                cur = None
    return pairs


def test_mistral_groups_synced_with_config():
    pairs = _config_pairs()
    assert pairs, "config.yml parse came back empty — scanner broken"
    mistral = {n for n, m in pairs if m.startswith("mistral/")}
    nonmistral = {n for n, m in pairs if not m.startswith("mistral/")}
    groups = set(smr.StripMistralReasoningHandler.MISTRAL_STRICT_GROUPS)
    assert mistral == groups, f"MISTRAL_STRICT_GROUPS out of sync with config.yml: {mistral ^ groups}"
    assert not (nonmistral & groups), f"non-mistral groups wrongly listed: {nonmistral & groups}"


def test_registered_before_effort_policy():
    """The hook must be registered, and before enforce_effort_policy."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yml")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")) as fh:
        lines = [ln.strip() for ln in fh]
    mine = next(
        (i for i, ln in enumerate(lines) if "strip_mistral_reasoning.strip_mistral_reasoning_handler" in ln),
        None,
    )
    assert mine is not None, "strip_mistral_reasoning not registered in config.yml callbacks"
    enforce = next(
        (i for i, ln in enumerate(lines) if "enforce_effort_policy.effort_policy_handler" in ln),
        None,
    )
    assert enforce is not None, "enforce_effort_policy registration not found"
    assert mine < enforce, "strip_mistral_reasoning must run before enforce_effort_policy"
    del path


def test_compose_mount_present():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docker-compose.yml")
    with open(path) as fh:
        compose = fh.read()
    assert "./config/strip_mistral_reasoning.py:/app/strip_mistral_reasoning.py:ro" in compose, (
        "docker-compose.yml missing the strip_mistral_reasoning.py mount"
    )


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(
        (k, v) for k, v in vars().items() if k.startswith("test_") and callable(v)
    ):
        try:
            fn()
            print("PASS", name)
        except AssertionError as err:
            fails += 1
            print("FAIL", name, "::", err)
    print("VERDICT:", "ALL_PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
