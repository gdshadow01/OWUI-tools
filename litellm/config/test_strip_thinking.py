#!/usr/bin/env python3
"""Tests for strip_thinking.

Runs BOTH ways:
  - standalone:  python3 test_strip_thinking.py   (exit 0 = all pass)
  - under pytest: pytest test_strip_thinking.py

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

import strip_thinking  # noqa: E402

THINKING = {"type": "enabled", "clear_thinking": False}


def _base(model, thinking=True):
    d = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
    }
    if thinking:
        d["thinking"] = dict(THINKING)
    return d


def _run(data):
    hook = strip_thinking.strip_thinking_handler.async_pre_call_hook
    return asyncio.run(hook(None, None, data, "completion"))


def test_strips_all_zai_groups():
    for m in sorted(strip_thinking.StripThinkingHandler.ZAI_GROUPS):
        out = _run(_base(m))
        assert "thinking" not in out, m
        assert out["model"] == m
        assert out["messages"] == [{"role": "user", "content": "hi"}]


def test_preserves_non_zai_groups():
    others = [
        "hermes-vision",  # deepseek-backed — card listed it, config.yml wins
        "hermes-fallback",
        "deepseek-v4-flash(0.14/0.28)",
        "deepseek-v4-pro(0.435/0.87)",
        "openrouter-zai-glm-4.7(0.38/1.98)",
        "openrouter-zai-glm-4.5-air-free",
        "openrouter-z-ai/glm-5.1(1.05/3.50)",
        "openrouter-z-ai/glm-4.7-flash(0.06/0.4)",
        "groq-gpt-oss-120b",
        "totally-unknown-model",
    ]
    for m in others:
        out = _run(_base(m))
        assert out.get("thinking") == THINKING, m


def test_no_thinking_is_noop():
    d = _base("zai-glm-5.3", thinking=False)
    assert _run(d) == d


def test_non_str_model_untouched():
    d = {"model": None, "thinking": dict(THINKING)}
    assert _run(d).get("thinking") == THINKING
    d = {"thinking": dict(THINKING)}
    assert _run(d).get("thinking") == THINKING


def test_returns_same_dict_object():
    d = _base("hermes-coding-plan")
    assert _run(d) is d


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


def test_zai_groups_synced_with_config():
    pairs = _config_pairs()
    assert pairs, "config.yml parse came back empty — scanner broken"
    zai = {n for n, m in pairs if m.startswith("zai/")}
    nonzai = {n for n, m in pairs if not m.startswith("zai/")}
    groups = set(strip_thinking.StripThinkingHandler.ZAI_GROUPS)
    assert zai == groups, f"ZAI_GROUPS out of sync with config.yml: {zai ^ groups}"
    assert not (nonzai & groups), f"non-zai groups wrongly listed: {nonzai & groups}"


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
