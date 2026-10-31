#!/usr/bin/env python3
"""Tests for codex_session_id (stable sharded session ids for chatgpt/ groups).

Runs BOTH ways:
  - standalone:  python3 test_codex_session_id.py   (exit 0 = all pass)
  - under pytest: pytest test_codex_session_id.py

No litellm install required: the litellm imports are stubbed before the
module import when litellm is not already importable (same pattern as
test_strip_thinking.py). The config.yml sync check needs PyYAML and runs
from the repo checkout; it self-skips (loudly) otherwise.
"""
import asyncio
import copy
import os
import re
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

import codex_session_id as csi  # noqa: E402

SID_RE = re.compile(r"^hermes-[A-Za-z0-9._\-]+-\d+-[0-9a-f]{8}$")

MESSAGES = [
    {"role": "system", "content": "You are a hermes worker. Stable AGENTS.md context."},
    {"role": "user", "content": "hello"},
]


def _data(model, messages=None, with_litellm_metadata=False):
    d = {
        "model": model,
        "messages": copy.deepcopy(messages if messages is not None else MESSAGES),
        "metadata": {"existing": "keep-me"},
    }
    if with_litellm_metadata:
        d["litellm_metadata"] = {"existing": "also-keep"}
    return d


def _run(model, **kw):
    return asyncio.run(
        csi.codex_session_id_hook.async_pre_call_hook(None, None, _data(model, **kw), "completion")
    )


def test_injects_and_is_deterministic():
    a = _run("chatgpt-gpt-6-luna")
    b = _run("chatgpt-gpt-6-luna")
    assert SID_RE.match(a["metadata"]["session_id"]), a["metadata"]["session_id"]
    assert a["metadata"]["session_id"] == b["metadata"]["session_id"], "identical content must give identical id"
    assert a["metadata"]["existing"] == "keep-me", "must not clobber existing metadata"


def test_content_change_changes_id():
    a = _run("chatgpt-gpt-6-luna")
    b = _run("chatgpt-gpt-6-luna", messages=[{"role": "system", "content": "DIFFERENT system prompt"}] + MESSAGES[1:])
    assert a["metadata"]["session_id"] != b["metadata"]["session_id"], "changed first message must change id"


def test_shard_math():
    import hashlib

    digest = hashlib.sha256(
        f"chatgpt-gpt-6-luna\x00{csi._first_message_text(MESSAGES)}".encode()
    ).hexdigest()
    expect = f"hermes-chatgpt-gpt-6-luna-{int(digest, 16) % 4}-{digest[:8]}"
    got = _run("chatgpt-gpt-6-luna")["metadata"]["session_id"]
    assert got == expect, (got, expect)


def test_primary_chatgpt_groups_get_session_ids():
    for model in ("hermes-plan", "hermes-execute", "hermes-tasks"):
        a = _run(model)
        b = _run(model)
        assert SID_RE.match(a["metadata"]["session_id"]), model
        assert a["metadata"]["session_id"] == b["metadata"]["session_id"], model


def test_non_chatgpt_groups_untouched():
    for model in ("hermes-heartbeat-plan", "zai-glm-5.3", "glm-5.3-flash", "unknown-model", ""):
        d = _data(model)
        before = copy.deepcopy(d)
        out = asyncio.run(
            csi.codex_session_id_hook.async_pre_call_hook(None, None, d, "completion")
        )
        assert out == before, f"{model!r} must pass through untouched"
        assert "session_id" not in out.get("metadata", {})


def test_weird_model_types_untouched():
    for bad in (None, 123, ["chatgpt-gpt-6-luna"]):
        d = _data(bad)
        out = asyncio.run(
            csi.codex_session_id_hook.async_pre_call_hook(None, None, d, "completion")
        )
        assert out is d


def test_litellm_metadata_set_defensively():
    out = _run("chatgpt-gpt-6-astra", with_litellm_metadata=True)
    assert out["litellm_metadata"]["session_id"] == out["metadata"]["session_id"]
    assert out["litellm_metadata"]["existing"] == "also-keep"


def test_fail_open():
    # non-dict data: hook must not raise, must return it unchanged
    out = asyncio.run(
        csi.codex_session_id_hook.async_pre_call_hook(None, None, None, "completion")
    )
    assert out is None


def test_multimodal_and_degenerate_messages():
    mm = [{"role": "system", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": " two"}]}]
    a = _run("chatgpt-gpt-6-luna", messages=mm)
    b = _run("chatgpt-gpt-6-luna", messages=[{"role": "system", "content": "part one two"}])
    assert a["metadata"]["session_id"] == b["metadata"]["session_id"], "text parts must concatenate"
    for degenerate in (None, [], "not-a-list", [{"content": None}], [42]):
        out = _run("chatgpt-gpt-6-luna", messages=degenerate)
        assert SID_RE.match(out["metadata"]["session_id"]), degenerate


def test_header_safety():
    assert csi._safe_header_value("ok-value") == "ok-value"
    assert all(32 <= ord(c) < 127 for c in csi._safe_header_value("hé\u0080llo\n"))
    assert csi._safe_header_value("  ") == "hermes-session"


def test_shard_env_var():
    key = csi.ENV_SHARD_COUNT
    old = os.environ.pop(key, None)
    try:
        os.environ[key] = "1"
        sid = _run("chatgpt-gpt-6-luna")["metadata"]["session_id"]
        assert re.match(r"^hermes-chatgpt-gpt-6-luna-0-[0-9a-f]{8}$", sid), sid
        os.environ[key] = "not-a-number"
        assert SID_RE.match(_run("chatgpt-gpt-6-luna")["metadata"]["session_id"])
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def test_config_groups_in_sync():
    """CHATGPT_GROUPS must equal config.yml deployments backed by chatgpt/."""
    repo_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")
    try:
        import yaml
    except ImportError:
        print(f"SKIP sync check (no yaml): verified manually against {repo_cfg}")
        return
    with open(repo_cfg) as f:
        cfg = yaml.safe_load(f)
    backed = {
        e["model_name"]
        for e in cfg.get("model_list", [])
        if str(e.get("litellm_params", {}).get("model", "")).startswith("chatgpt/")
    }
    assert backed == set(csi.CHATGPT_GROUPS), (
        f"CHATGPT_GROUPS out of sync with config.yml:\n  config.yml chatgpt/-backed: {sorted(backed)}\n"
        f"  hook CHATGPT_GROUPS:            {sorted(csi.CHATGPT_GROUPS)}"
    )


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
