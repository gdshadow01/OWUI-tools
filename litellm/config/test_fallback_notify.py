import asyncio
import logging
import os
import sys
import time
import types

import pytest

if "pytest" not in sys.modules:
    raise RuntimeError("This file is a pytest module; run it via pytest, not directly.")

try:
    import litellm  # noqa: F401
except ImportError:
    # Stub only the CustomLogger base when litellm is not installed
    # (same pattern as test_strip_thinking.py) — the state machine under
    # test does not touch litellm beyond subclassing it.
    _cl = types.ModuleType("litellm.integrations.custom_logger")
    _cl.CustomLogger = object
    sys.modules.setdefault("litellm", types.ModuleType("litellm"))
    sys.modules.setdefault("litellm.integrations", types.ModuleType("litellm.integrations"))
    sys.modules.setdefault("litellm.integrations.custom_logger", _cl)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fallback_notify


class FakeResponse:
    status_code = 200


class FakeAsyncClient:
    """Records every post() call; usable as an async context manager."""

    posts = []
    fail = False  # when True, post() raises instead of recording

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, timeout=None):
        if FakeAsyncClient.fail:
            raise ConnectionError("gotify unreachable")
        FakeAsyncClient.posts.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse()


class FakeSyncClient:
    """Sync twin of FakeAsyncClient for the log_success_event path."""

    posts = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None, timeout=None):
        FakeSyncClient.posts.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse()


def _reset(monkeypatch):
    fallback_notify.LAST_FAILOVER = 0.0
    fallback_notify.LAST_RECOVERY = 0.0
    fallback_notify._warned = False
    fallback_notify._fallback_active.clear()
    FakeAsyncClient.posts = []
    FakeAsyncClient.fail = False
    FakeSyncClient.posts = []
    monkeypatch.setattr(fallback_notify.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(fallback_notify.httpx, "Client", FakeSyncClient)


def _run_fallback_hook(group="m1", served="m2"):
    hook = fallback_notify.fallback_notifier.log_success_fallback_event
    asyncio.run(
        hook(
            original_model_group=group,
            kwargs={"model": served},
            original_exception=ValueError("boom"),
        )
    )


def _success_kwargs(group="m1", fallback_depth=None):
    kwargs = {"litellm_params": {"metadata": {"model_group": group}}}
    if fallback_depth is not None:
        kwargs["fallback_depth"] = fallback_depth
    return kwargs


def _run_success_hook(group="m1", fallback_depth=None):
    hook = fallback_notify.fallback_notifier.async_log_success_event
    asyncio.run(hook(_success_kwargs(group, fallback_depth), None, None, None))


def _run_sync_success_hook(group="m1", fallback_depth=None):
    hook = fallback_notify.fallback_notifier.log_success_event
    hook(_success_kwargs(group, fallback_depth), None, None, None)


def test_env_unset_noop(monkeypatch, caplog):
    monkeypatch.delenv("GOTIFY_URL", raising=False)
    monkeypatch.delenv("GOTIFY_TOKEN", raising=False)
    _reset(monkeypatch)

    with caplog.at_level(logging.WARNING):
        _run_fallback_hook()
        _run_fallback_hook()

    assert FakeAsyncClient.posts == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_throttle_one_post(monkeypatch):
    monkeypatch.setenv("GOTIFY_URL", "http://gotify.example.com")
    monkeypatch.setenv("GOTIFY_TOKEN", "tok123")
    _reset(monkeypatch)

    _run_fallback_hook()
    _run_fallback_hook()

    assert len(FakeAsyncClient.posts) == 1
    post = FakeAsyncClient.posts[0]
    assert "/message?token=tok123" in post["url"]
    assert post["json"]["title"] == "LiteLLM fallback"
    assert post["json"]["priority"] == 8
    assert "m1" in post["json"]["message"]
    assert "ValueError" in post["json"]["message"]
    assert "m2" in post["json"]["message"]


# --- recovery state machine (back on primary) ---------------------------


def test_fallback_event_sets_flag(monkeypatch):
    """(a) every fallback event sets the fallback-active flag."""
    monkeypatch.setenv("GOTIFY_URL", "http://gotify.example.com")
    monkeypatch.setenv("GOTIFY_TOKEN", "tok123")
    _reset(monkeypatch)

    assert "m1" not in fallback_notify._fallback_active
    _run_fallback_hook()
    assert "m1" in fallback_notify._fallback_active
    assert fallback_notify._fallback_active["m1"] > 0


def test_flag_set_even_when_throttled(monkeypatch):
    """Flag tracks serving state: throttled fallback events still set it."""
    monkeypatch.setenv("GOTIFY_URL", "http://gotify.example.com")
    monkeypatch.setenv("GOTIFY_TOKEN", "tok123")
    _reset(monkeypatch)

    fallback_notify.LAST_FAILOVER = time.time()  # failover ping fully throttled
    _run_fallback_hook()

    assert FakeAsyncClient.posts == []
    assert "m1" in fallback_notify._fallback_active


def test_recovery_first_success_one_ping_and_clears_flag(monkeypatch):
    """(b) first direct-primary success: exactly one recovery ping, flag cleared."""
    monkeypatch.setenv("GOTIFY_URL", "http://gotify.example.com")
    monkeypatch.setenv("GOTIFY_TOKEN", "tok123")
    _reset(monkeypatch)

    fallback_notify._fallback_active["m1"] = time.time() - 125  # 2m ago

    _run_success_hook(group="m1")
    _run_success_hook(group="m1")

    assert len(FakeAsyncClient.posts) == 1
    post = FakeAsyncClient.posts[0]
    assert "/message?token=tok123" in post["url"]
    assert post["json"]["title"] == "LiteLLM recovery"
    assert post["json"]["priority"] == 8
    assert "m1" in post["json"]["message"]
    assert "back on primary" in post["json"]["message"]
    assert "(2m on fallback)" in post["json"]["message"]
    assert "m1" not in fallback_notify._fallback_active


def test_fallback_served_success_keeps_flag(monkeypatch):
    """(c) fallback_depth >= 1 is fallback traffic: no ping, flag kept."""
    monkeypatch.setenv("GOTIFY_URL", "http://gotify.example.com")
    monkeypatch.setenv("GOTIFY_TOKEN", "tok123")
    _reset(monkeypatch)

    fallback_notify._fallback_active["m1"] = time.time() - 300

    _run_success_hook(group="m2", fallback_depth=1)  # served BY the fallback
    _run_success_hook(group="m1", fallback_depth=2)

    assert FakeAsyncClient.posts == []
    assert "m1" in fallback_notify._fallback_active


def test_second_recovery_throttled_but_flag_cleared(monkeypatch):
    """(d) second recovery inside 600 s: no ping, flag still cleared."""
    monkeypatch.setenv("GOTIFY_URL", "http://gotify.example.com")
    monkeypatch.setenv("GOTIFY_TOKEN", "tok123")
    _reset(monkeypatch)

    fallback_notify._fallback_active["m1"] = time.time() - 60
    _run_success_hook(group="m1")  # recovery #1 -> ping, window consumed
    assert len(FakeAsyncClient.posts) == 1

    # New outage + recovery inside the throttle window
    fallback_notify._fallback_active["m1"] = time.time() - 30
    _run_success_hook(group="m1")  # recovery #2 -> throttled

    assert len(FakeAsyncClient.posts) == 1
    assert "m1" not in fallback_notify._fallback_active


def test_failed_post_consumes_throttle_window(monkeypatch, caplog):
    """(e) a failed gotify POST still consumes the recovery window."""
    monkeypatch.setenv("GOTIFY_URL", "http://gotify.example.com")
    monkeypatch.setenv("GOTIFY_TOKEN", "tok123")
    _reset(monkeypatch)
    FakeAsyncClient.fail = True

    fallback_notify._fallback_active["m1"] = time.time() - 60
    with caplog.at_level(logging.ERROR):
        _run_success_hook(group="m1")
    assert "m1" not in fallback_notify._fallback_active

    # Window was consumed: a fresh recovery attempt sends nothing
    FakeAsyncClient.fail = False
    fallback_notify._fallback_active["m1"] = time.time() - 60
    _run_success_hook(group="m1")

    assert FakeAsyncClient.posts == []
    assert "m1" not in fallback_notify._fallback_active
    assert any("failed to send" in r.message for r in caplog.records)


def test_recovery_via_sync_hook(monkeypatch):
    """The sync log_success_event path sends the same single ping."""
    monkeypatch.setenv("GOTIFY_URL", "http://gotify.example.com")
    monkeypatch.setenv("GOTIFY_TOKEN", "tok123")
    _reset(monkeypatch)

    fallback_notify._fallback_active["m1"] = time.time() - 125
    _run_sync_success_hook(group="m1")

    assert len(FakeSyncClient.posts) == 1
    post = FakeSyncClient.posts[0]
    assert post["json"]["title"] == "LiteLLM recovery"
    assert "(2m on fallback)" in post["json"]["message"]
    assert FakeAsyncClient.posts == []
    assert "m1" not in fallback_notify._fallback_active


def test_no_model_group_noop(monkeypatch):
    """Successes without a model_group never touch the state machine."""
    monkeypatch.setenv("GOTIFY_URL", "http://gotify.example.com")
    monkeypatch.setenv("GOTIFY_TOKEN", "tok123")
    _reset(monkeypatch)

    hook = fallback_notify.fallback_notifier.async_log_success_event
    asyncio.run(hook({"litellm_params": {"metadata": {}}}, None, None, None))
    asyncio.run(hook({}, None, None, None))

    assert FakeAsyncClient.posts == []
    assert fallback_notify._fallback_active == {}


def test_recovery_independent_from_failover_throttle(monkeypatch):
    """A recent failover ping cannot swallow the recovery ping."""
    monkeypatch.setenv("GOTIFY_URL", "http://gotify.example.com")
    monkeypatch.setenv("GOTIFY_TOKEN", "tok123")
    _reset(monkeypatch)

    fallback_notify.LAST_FAILOVER = time.time()  # failover ping just sent
    fallback_notify._fallback_active["m1"] = time.time() - 60

    _run_success_hook(group="m1")

    assert len(FakeAsyncClient.posts) == 1
    assert FakeAsyncClient.posts[0]["json"]["title"] == "LiteLLM recovery"

# --- quota-"until" cooldown (Phase B, Option Q1) -------------------------


class FakeCooldownCache:
    """Records add_deployment_to_cooldown calls, like CooldownCache."""

    calls = []

    def add_deployment_to_cooldown(
        self, model_id, original_exception, exception_status, cooldown_time
    ):
        FakeCooldownCache.calls.append(
            {
                "model_id": model_id,
                "original_exception": original_exception,
                "exception_status": exception_status,
                "cooldown_time": cooldown_time,
            }
        )


class FakeRouter:
    cooldown_cache = None  # set per-test via _reset_q1()


class FakeRateLimitError(Exception):
    """RateLimitError stand-in: str() carries the quota deadline."""

    def __init__(self, message, status_code=429):
        super().__init__(message)
        self.status_code = status_code


def _reset_q1(monkeypatch):
    FakeCooldownCache.calls = []
    FakeRouter.cooldown_cache = FakeCooldownCache()
    monkeypatch.setattr(fallback_notify, "_get_llm_router", lambda: FakeRouter())


def _failure_kwargs(
    deadline_text,
    deployment_id="dep-primary",
    status_code=429,
    group="q1g",
    provider=None,
    deployment_model=None,
):
    exc = FakeRateLimitError(deadline_text, status_code=status_code)
    model_info = {"id": deployment_id}
    if provider is not None:
        model_info["litellm_provider"] = provider
    metadata = {
        "model_group": group,
        "model_info": model_info,
    }
    if deployment_model is not None:
        metadata["deployment"] = deployment_model
    kwargs = {
        "litellm_params": {
            "metadata": metadata,
        },
        "exception": exc,
    }
    return kwargs, exc


# --- parse table ----------------------------------------------------------


def test_parse_iso_zulu():
    from datetime import datetime, timezone as tz

    ts = fallback_notify.parse_quota_until(
        "You exceeded your current quota. Usage limit until "
        "2027-09-15T23:30:00Z. Try again after that."
    )
    assert ts == datetime(2027, 9, 15, 23, 30, tzinfo=tz.utc).timestamp()


def test_parse_iso_offset():
    from datetime import datetime, timezone as tz

    ts = fallback_notify.parse_quota_until(
        "quota exceeded, reset at 2027-03-01T12:00:00+02:00"
    )
    assert ts == datetime(2027, 3, 1, 10, 0, tzinfo=tz.utc).timestamp()


def test_parse_iso_space_separator():
    from datetime import datetime, timezone as tz

    ts = fallback_notify.parse_quota_until("Limit resets 2027-03-01 12:00:00")
    assert ts == datetime(2027, 3, 1, 12, 0, tzinfo=tz.utc).timestamp()


def test_parse_iso_naive_assumes_utc():
    """Default zone for a tz-naive stamp: UTC — unchanged for unlisted providers.

    Z.AI-backed deployments pass their real offset instead (see the pinned
    incident tests below); this asserts the default stays the historical one.
    """
    from datetime import datetime, timezone as tz

    ts = fallback_notify.parse_quota_until("Limit resets 2027-03-01T12:00:00")
    assert ts == datetime(2027, 3, 1, 12, 0, tzinfo=tz.utc).timestamp()


def test_parse_rfc2822():
    from datetime import datetime, timezone as tz

    ts = fallback_notify.parse_quota_until(
        "Rate limit will reset on Mon, 15 Sep 2027 23:30:00 GMT"
    )
    assert ts == datetime(2027, 9, 15, 23, 30, tzinfo=tz.utc).timestamp()


def test_parse_relative_seconds():
    t0 = time.time()
    ts = fallback_notify.parse_quota_until("Try again in 90 seconds")
    assert ts == pytest.approx(t0 + 90, abs=5)


def test_parse_relative_minutes():
    t0 = time.time()
    ts = fallback_notify.parse_quota_until("retry after 5m")
    assert ts == pytest.approx(t0 + 300, abs=5)


def test_parse_past_deadline_returns_none():
    assert (
        fallback_notify.parse_quota_until("quota until 2020-01-01T00:00:00Z")
        is None
    )


def test_parse_unparseable_returns_none():
    assert fallback_notify.parse_quota_until("Something went wrong") is None
    assert fallback_notify.parse_quota_until("") is None
    assert fallback_notify.parse_quota_until(None) is None


def test_parse_past_deadline_does_not_shadow_relative():
    # A stale ISO timestamp in the text must not swallow the relative hint
    t0 = time.time()
    ts = fallback_notify.parse_quota_until(
        "request at 2020-01-01T00:00:00Z failed; retry in 30 seconds"
    )
    assert ts == pytest.approx(t0 + 30, abs=5)


# --- provider zone for tz-naive quota stamps -----------------------------
#
# 2026-10-12 incident: Z.AI answered HTTP 429 with
#   "ZaiException - Usage limit reached for 5 hour. Your limit will reset at
#    2026-10-12 19:48:48."
# at 08:46:46 UTC. The bare stamp is Beijing local time (UTC+8), but the parser
# stamped it UTC, so the deadline landed 8 h late and hermes-plan was cooled
# down for 39721.74 s (11.03 h) instead of 10921.74 s (3.03 h) — an ~11 h
# lockout while the upstream was demonstrably healthy. Numbers below are the
# measured ones (scratch/tz_math_fix.py), pinned so the regression cannot return.

ZAI_QUOTA_MSG = (
    "litellm.RateLimitError: RateLimitError: ZaiException - Usage limit "
    "reached for 5 hour. Your limit will reset at 2026-10-12 19:48:48. "
    "Received Model Group=hermes-plan"
)
# The exact failure instant; 1791794806.255878 epoch.
ZAI_FAILURE_MONOTONIC_FREE_NOW = 1791794806.255878
ZAI_TRUE_RESET_EPOCH = 1791805728.0       # 2026-10-12 11:48:48 UTC
ZAI_MISREAD_EPOCH = 1791834528.0          # 2026-10-12 19:48:48 UTC (wrong)
ZAI_TRUE_COOLDOWN_S = 10921.74412202835   # reported live value was 39721.74412226677


def test_zai_naive_stamp_uses_beijing_offset(monkeypatch):
    """Z.AI's bare wall clock is UTC+8: 19:48:48 Beijing = 11:48:48 UTC."""
    monkeypatch.setattr(
        fallback_notify.time, "time", lambda: ZAI_FAILURE_MONOTONIC_FREE_NOW
    )
    ts = fallback_notify.parse_quota_until(
        ZAI_QUOTA_MSG, tz_offset_s=fallback_notify._PROVIDER_UTC_OFFSET_S["zai"]
    )
    assert ts == ZAI_TRUE_RESET_EPOCH
    assert ts != ZAI_MISREAD_EPOCH
    cooldown_s = ts - ZAI_FAILURE_MONOTONIC_FREE_NOW
    assert cooldown_s == pytest.approx(ZAI_TRUE_COOLDOWN_S, abs=1)
    assert cooldown_s == pytest.approx(10922, abs=1)
    # The bug: reading the stamp as UTC produced exactly the reported 39722.
    assert cooldown_s != pytest.approx(39722, abs=1)


def test_zai_naive_stamp_without_offset_still_reads_utc(monkeypatch):
    """Guard against a blanket shift: only the per-provider offset moves it."""
    monkeypatch.setattr(
        fallback_notify.time, "time", lambda: ZAI_FAILURE_MONOTONIC_FREE_NOW
    )
    ts = fallback_notify.parse_quota_until(ZAI_QUOTA_MSG)
    assert ts == ZAI_MISREAD_EPOCH


def test_zai_explicit_offset_in_text_wins(monkeypatch):
    """An explicit tz in the text overrides the provider default."""
    from datetime import datetime, timezone as tz

    monkeypatch.setattr(
        fallback_notify.time, "time", lambda: ZAI_FAILURE_MONOTONIC_FREE_NOW
    )
    ts = fallback_notify.parse_quota_until(
        "quota until 2027-03-01T12:00:00Z", tz_offset_s=8 * 3600
    )
    assert ts == datetime(2027, 3, 1, 12, 0, tzinfo=tz.utc).timestamp()


def test_quota_window_phrase_is_not_a_wait(monkeypatch):
    """"limit reached for 5 hour" is the window LENGTH, never an 18000 s wait.

    Once the offset shifts a stale deadline into the past, the relative branch
    is the next candidate in line — it must not read the window phrase.
    """
    # now is AFTER the true reset: nothing valid remains, so -> None
    monkeypatch.setattr(
        fallback_notify.time, "time",
        lambda: ZAI_TRUE_RESET_EPOCH + 60,
    )
    assert (
        fallback_notify.parse_quota_until(
            ZAI_QUOTA_MSG, tz_offset_s=fallback_notify._PROVIDER_UTC_OFFSET_S["zai"]
        )
        is None
    )


def test_real_relative_hint_still_parsed(monkeypatch):
    """The window-phrase guard must not disable genuine relative hints."""
    t0 = time.time()
    ts = fallback_notify.parse_quota_until(
        "Usage limit reached for 5 hour. Please retry in 90 seconds."
    )
    assert ts == pytest.approx(t0 + 90, abs=5)


def test_provider_zone_offset_scoped_per_provider():
    def kwargs_for(
        provider=None, deployment=None, model=None, key=None
    ):
        model_info = {}
        if provider is not None:
            model_info["litellm_provider"] = provider
        if key is not None:
            model_info["key"] = key
        metadata = {"model_info": model_info}
        if deployment is not None:
            metadata["deployment"] = deployment
        litellm_params = {"metadata": metadata}
        if model is not None:
            litellm_params["model"] = model
        return {"litellm_params": litellm_params}

    off = fallback_notify.provider_zone_offset_s
    # live hermes-plan deployment shape (verified via GET /model/info + router)
    assert off(kwargs_for(provider="zai", key="zai/glm-5.3", model="zai/glm-5.3")) == 28800
    assert off(kwargs_for(deployment="zai/glm-5.3")) == 28800
    # unlisted provider -> UTC default, no shift
    assert off(kwargs_for(provider="openai", model="openai/deepseek-v4-flash")) == 0
    assert off(kwargs_for(deployment="deepseek-v4-flash-vision-exp")) == 0
    # malformed / absent metadata must degrade, never raise
    assert off({}) == 0
    assert off({"litellm_params": None}) == 0
    assert off({"litellm_params": {"metadata": {"model_info": "nonsense"}}}) == 0


def test_format_zone_offset():
    assert fallback_notify.format_zone_offset(28800) == "+08:00"
    assert fallback_notify.format_zone_offset(0) == "+00:00"
    assert fallback_notify.format_zone_offset(-19800) == "-05:30"


# --- cooldown step -------------------------------------------------------


def _future_iso(offset_s):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + offset_s))


def test_quota_cooldown_registers_cooldown(monkeypatch):
    _reset_q1(monkeypatch)
    kwargs, exc = _failure_kwargs(
        f"You exceeded quota. Usage limit until {_future_iso(3600)}."
    )
    ok = fallback_notify._quota_cooldown_from_failure(kwargs)
    assert ok is True
    assert len(FakeCooldownCache.calls) == 1
    call = FakeCooldownCache.calls[0]
    assert call["model_id"] == "dep-primary"
    assert call["original_exception"] is exc
    assert call["exception_status"] == 429
    assert call["cooldown_time"] == pytest.approx(3600, abs=5)


def test_quota_cooldown_over_24h_ignored(monkeypatch, caplog):
    _reset_q1(monkeypatch)
    kwargs, _ = _failure_kwargs("usage limit until 2030-01-01T00:00:00Z")
    with caplog.at_level(logging.INFO):
        ok = fallback_notify._quota_cooldown_from_failure(kwargs)
    assert ok is False
    assert FakeCooldownCache.calls == []
    assert any("exceeds" in r.message for r in caplog.records)


def test_quota_cooldown_via_async_hook(monkeypatch):
    _reset_q1(monkeypatch)
    kwargs, _ = _failure_kwargs(f"usage limit until {_future_iso(1200)}")
    asyncio.run(
        fallback_notify.fallback_notifier.async_log_failure_event(
            kwargs, None, None, None
        )
    )
    assert len(FakeCooldownCache.calls) == 1
    assert FakeCooldownCache.calls[0]["model_id"] == "dep-primary"


def test_quota_cooldown_via_sync_hook(monkeypatch):
    _reset_q1(monkeypatch)
    kwargs, _ = _failure_kwargs(f"usage limit until {_future_iso(1200)}")
    fallback_notify.fallback_notifier.log_failure_event(kwargs, None, None, None)
    assert len(FakeCooldownCache.calls) == 1
    assert FakeCooldownCache.calls[0]["model_id"] == "dep-primary"


def test_quota_cooldown_missing_router_no_crash(monkeypatch):
    FakeCooldownCache.calls = []
    monkeypatch.setattr(fallback_notify, "_get_llm_router", lambda: None)
    kwargs, _ = _failure_kwargs(f"usage limit until {_future_iso(600)}")
    ok = fallback_notify._quota_cooldown_from_failure(kwargs)
    assert ok is False
    assert FakeCooldownCache.calls == []


def test_quota_cooldown_missing_cache_no_crash(monkeypatch):
    FakeCooldownCache.calls = []
    FakeRouter.cooldown_cache = None
    monkeypatch.setattr(fallback_notify, "_get_llm_router", lambda: FakeRouter())
    kwargs, _ = _failure_kwargs(f"usage limit until {_future_iso(600)}")
    ok = fallback_notify._quota_cooldown_from_failure(kwargs)
    assert ok is False
    assert FakeCooldownCache.calls == []


def test_quota_cooldown_wrong_status_ignored(monkeypatch):
    _reset_q1(monkeypatch)
    kwargs, _ = _failure_kwargs(
        f"usage limit until {_future_iso(600)}", status_code=500
    )
    ok = fallback_notify._quota_cooldown_from_failure(kwargs)
    assert ok is False
    assert FakeCooldownCache.calls == []


def test_quota_cooldown_no_exception_noop(monkeypatch):
    _reset_q1(monkeypatch)
    ok = fallback_notify._quota_cooldown_from_failure({"litellm_params": {}})
    assert ok is False
    assert FakeCooldownCache.calls == []


def test_quota_cooldown_unparseable_body_noop(monkeypatch):
    _reset_q1(monkeypatch)
    kwargs, _ = _failure_kwargs("This model is overloaded")
    ok = fallback_notify._quota_cooldown_from_failure(kwargs)
    assert ok is False
    assert FakeCooldownCache.calls == []


def test_quota_cooldown_str_status(monkeypatch):
    """litellm exceptions can carry str status codes — int() must coerce."""
    _reset_q1(monkeypatch)
    kwargs, _ = _failure_kwargs(
        f"usage limit until {_future_iso(600)}", status_code="429"
    )
    ok = fallback_notify._quota_cooldown_from_failure(kwargs)
    assert ok is True
    assert FakeCooldownCache.calls[0]["exception_status"] == 429


def test_quota_cooldown_end_to_end_zai_uses_beijing_duration(monkeypatch):
    """The incident path, end to end: a zai-stamped deployment must register the
    UTC+8-correct cooldown, not the 8 h-overcooldown."""
    _reset_q1(monkeypatch)
    monkeypatch.setattr(
        fallback_notify.time, "time", lambda: ZAI_FAILURE_MONOTONIC_FREE_NOW
    )
    kwargs, _ = _failure_kwargs(
        ZAI_QUOTA_MSG, provider="zai", deployment_model="zai/glm-5.3"
    )
    ok = fallback_notify._quota_cooldown_from_failure(kwargs)
    assert ok is True
    assert len(FakeCooldownCache.calls) == 1
    cooldown_s = FakeCooldownCache.calls[0]["cooldown_time"]
    assert cooldown_s == pytest.approx(ZAI_TRUE_COOLDOWN_S, abs=1)
    assert cooldown_s < 4 * 3600  # hours, not the 11.03 h lockout


def test_quota_cooldown_end_to_end_unknown_provider_keeps_utc(monkeypatch):
    """Same text from an unlisted provider: the historical UTC read is retained
    (no blanket shift)."""
    _reset_q1(monkeypatch)
    monkeypatch.setattr(
        fallback_notify.time, "time", lambda: ZAI_FAILURE_MONOTONIC_FREE_NOW
    )
    kwargs, _ = _failure_kwargs(ZAI_QUOTA_MSG, provider="openai")
    ok = fallback_notify._quota_cooldown_from_failure(kwargs)
    assert ok is True
    assert FakeCooldownCache.calls[0]["cooldown_time"] == pytest.approx(39722, abs=1)


def test_quota_cooldown_zai_message_logs_zone(monkeypatch, caplog):
    """The zone must be visible in the log line that a future incident reads."""
    _reset_q1(monkeypatch)
    monkeypatch.setattr(
        fallback_notify.time, "time", lambda: ZAI_FAILURE_MONOTONIC_FREE_NOW
    )
    kwargs, _ = _failure_kwargs(ZAI_QUOTA_MSG, provider="zai")
    with caplog.at_level(logging.INFO):
        assert fallback_notify._quota_cooldown_from_failure(kwargs) is True
    assert any("zone +08:00" in r.getMessage() for r in caplog.records)


def test_quota_cooldown_over_24h_guard_survives_offset(monkeypatch, caplog):
    """The 24 h garbage guard must still fire for a zai stamp that is genuinely
    too far out — the offset must not be able to smuggle a cooldown past it."""
    _reset_q1(monkeypatch)
    kwargs, _ = _failure_kwargs(
        "quota exceeded until 2030-01-01T00:00:00", provider="zai"
    )
    with caplog.at_level(logging.INFO):
        ok = fallback_notify._quota_cooldown_from_failure(kwargs)
    assert ok is False
    assert FakeCooldownCache.calls == []
    assert any("exceeds" in r.getMessage() for r in caplog.records)
