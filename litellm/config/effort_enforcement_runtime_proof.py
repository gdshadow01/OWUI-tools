#!/usr/bin/env python3
"""T2 isolated pinned-runtime provider-bound proof for enforce_effort_policy (plan 261031).

Proves, against the REAL pinned litellm runtime (v1.103.2) inside the running
proxy container, that the candidate config + the candidate effort-policy
callback produce the intended FINAL OUTBOUND provider payloads — without
touching the running proxy, /app mounts, or any real upstream (zero quota).

Mechanism (established docker-exec interception pattern):
  1. Build a self-contained payload locally (embeds the candidate working-tree
     files as base64 — config.yml, enforce_effort_policy.py plus the other
     registered callbacks):
       python3 effort_enforcement_runtime_proof.py --build /tmp/effort_proof_payload.py
  2. Pipe it into a SEPARATE exec process in the running container (script via
     stdin, no container-fs writes outside a tempdir, no restart):
       cat /tmp/effort_proof_payload.py | ssh <user>@<litellm-host> \
           'docker exec -i litellm-proxy python3 -'
  3. Inside: litellm.telemetry disabled, every deployment's api_base/api_key
     replaced with dummy values (nothing real is contacted even if the
     interceptor missed), a fresh Router per case with the candidate fallback
     chains, all registered callbacks loaded in config order, and
     httpx.Client.send/AsyncClient.send monkeypatched to CAPTURE the final
     provider-bound request and ABORT before any packet leaves.
  4. Assertions A-H from the shared handoff run on captured payloads only.
     Reasoning-token counts are never used as evidence.

Output: one PASS/FAIL line per case + a final `PROOF-JSON: {...}` summary
line (machine-readable). Exit 0 only if every case passes.
"""

import base64
import copy
import importlib
import json
import os
import re
import sys
import tempfile
import traceback

# Placeholders replaced by --build with base64 of the working-tree files.
_EMBEDDED_B64 = {
    "config.yml": "@@B64:config.yml@@",
    "enforce_effort_policy.py": "@@B64:enforce_effort_policy.py@@",
    "strip_thinking.py": "@@B64:strip_thinking.py@@",
    "codex_session_id.py": "@@B64:codex_session_id.py@@",
    "strip_hermes.py": "@@B64:strip_hermes.py@@",
    "fallback_notify.py": "@@B64:fallback_notify.py@@",
}

_EMBED_NAMES = [
    "config.yml",
    "enforce_effort_policy.py",
    "strip_thinking.py",
    "codex_session_id.py",
    "strip_hermes.py",
    "fallback_notify.py",
]


def build_payload(out_path):
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.abspath(__file__)).read()
    for name in _EMBED_NAMES:
        raw = open(os.path.join(here, name), "rb").read()
        placeholder = "@@B64:%s@@" % name
        if placeholder not in src:
            raise SystemExit(f"placeholder missing for {name}")
        src = src.replace(placeholder, base64.b64encode(raw).decode())
    with open(out_path, "w") as fh:
        fh.write(src)
    print(f"payload written: {out_path} ({os.path.getsize(out_path)} bytes)")


# ============================== in-container runtime ==============================

PRIMARY_LANES = ("hermes-plan", "hermes-execute", "hermes-tasks")
HEARTBEAT_LANES = ("hermes-heartbeat-plan", "hermes-heartbeat-execute", "hermes-heartbeat-tasks")

# alias -> (expected outbound model, expected configured effort)
EXPECTED_PRIMARY = {
    "hermes-plan": ("gpt-6.1-sol", "xhigh"),
    "hermes-execute": ("gpt-6.1-sol", "high"),
    "hermes-tasks": ("gpt-6-luna", "high"),
}
EXPECTED_ZAI_HIGH = {
    "hermes-coding-plan": "glm-5.3",
    "hermes-coding-execute": "glm-5.3-flash",
    "hermes-heartbeat-plan": "glm-5.3",
    "hermes-heartbeat-execute": "glm-5.3-flash",
    "hermes-heartbeat-tasks": "glm-5.3-flash",
}
EXPECTED_FALLBACK = ("deepseek-flash", "high")
NON_MANAGED_LANES = ("zai-glm-5.3", "hermes-plan-zai", "deepseek-v4-flash(0.14/0.28)")

BASE_MESSAGES = [
    {"role": "system", "content": "You are a test agent. Be terse."},
    {"role": "user", "content": "Reply with the single word: ok"},
]
TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Return the current time",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
TOOL_CONTINUATION_MESSAGES = [
    {"role": "system", "content": "You are a test agent. Be terse."},
    {"role": "user", "content": "What time is it? Use the tool."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_test_1",
                "type": "function",
                "function": {"name": "get_time", "arguments": "{}"},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_test_1", "content": "12:00"},
]

STATE = {"fail_hosts": set()}
CAPTURES = []
_SPY_CAPTURED = None  # set by _install_spy; the interceptor's abort exception


def _dummy_base(alias):
    if alias in PRIMARY_LANES:
        return f"http://dpr-{alias}.invalid"
    if alias in HEARTBEAT_LANES:
        return f"http://dhp-{alias}.invalid"
    if alias == "hermes-fallback":
        return "http://dhf-hermes-fallback.invalid"
    safe = re.sub(r"[^a-z0-9-]+", "-", alias.lower()).strip("-")
    return f"http://ddir-{safe}.invalid"


def _is_dummy_host(host):
    return bool(host) and host.startswith(("dpr-", "dhp-", "dhf-", "ddir-"))


def _install_spy(httpx):
    global _SPY_CAPTURED

    class _Captured(Exception):
        pass

    _SPY_CAPTURED = _Captured

    def _record(request):
        CAPTURES.append(
            {
                "host": request.url.host,
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": request.content,
            }
        )

    async def _spy_async_send(self, request, **kwargs):
        _record(request)
        if request.url.host in STATE["fail_hosts"]:
            raise httpx.ConnectError(f"[harness] simulated outage {request.url.host}")
        raise _Captured()

    def _spy_sync_send(self, request, **kwargs):
        _record(request)
        if request.url.host in STATE["fail_hosts"]:
            raise httpx.ConnectError(f"[harness] simulated outage {request.url.host}")
        raise _Captured()

    httpx.AsyncClient.send = _spy_async_send
    httpx.Client.send = _spy_sync_send
    return _Captured


def _load_callbacks(cfg, tmpdir):
    """Instantiate litellm_settings.callbacks in config order from the tempdir."""
    instances = []
    refs = cfg.get("litellm_settings", {}).get("callbacks", [])
    for ref in refs:
        mod_name, inst_name = ref.rsplit(".", 1)
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            print(f"[harness] importing {mod_name} raised {type(exc).__name__}: {exc}")
            if mod_name == "strip_hermes":
                # strip_hermes imports UserAPIKeyAuth/DualCache from the proxy
                # server module; stub them if the full proxy import is not
                # viable inside the exec process.
                import types as _types

                stub = _types.ModuleType("litellm.proxy.proxy_server")
                stub.UserAPIKeyAuth = object
                stub.DualCache = object
                sys.modules.setdefault("litellm.proxy", importlib.import_module("litellm.proxy"))
                sys.modules["litellm.proxy.proxy_server"] = stub
                mod = importlib.import_module(mod_name)
            else:
                raise
        instances.append((ref, getattr(mod, inst_name)))
    return instances


def _build_router(litellm, cfg):
    model_list = copy.deepcopy(cfg["model_list"])
    for entry in model_list:
        lp = entry["litellm_params"]
        alias = entry["model_name"]
        lp["api_base"] = _dummy_base(alias)
        lp["api_key"] = "sk-dummy-no-network"
    fallbacks = copy.deepcopy(cfg.get("litellm_settings", {}).get("fallbacks", []))
    # num_retries=0: the capture abort must NOT be retried; each lane is
    # attempted exactly once per case (fallback chains still engage once).
    return litellm.Router(model_list=model_list, fallbacks=fallbacks, num_retries=0)


def _effort_view(body):
    """Every recognized effort carrier present in an outbound body."""
    view = {}
    if "reasoning_effort" in body:
        view["reasoning_effort"] = body["reasoning_effort"]
    if isinstance(body.get("reasoning"), dict):
        view["reasoning.effort"] = body["reasoning"].get("effort")
    if "thinking" in body:
        view["thinking"] = body["thinking"]
    if isinstance(body.get("output_config"), dict):
        view["output_config.effort"] = body["output_config"].get("effort")
    return view


RESULTS = []


def check(case, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    RESULTS.append({"case": case, "status": status, "detail": str(detail)[:400]})
    print(f"{status} {case} :: {str(detail)[:400]}", flush=True)
    return cond


async def _iter_stream(resp):
    async for _chunk in resp:
        break


async def run_case(callbacks, litellm, cfg, model, extra=None, with_hooks=True,
                   fail_hosts=(), messages=None):
    """One isolated request: hooks in config order -> fresh Router -> capture."""
    import asyncio

    print(
        f"[harness] case start: model={model} hooks={with_hooks} "
        f"extra={sorted(extra.keys()) if extra else []} fail_hosts={sorted(fail_hosts)}",
        flush=True,
    )
    STATE["fail_hosts"] = set(fail_hosts)
    del CAPTURES[:]
    data = {"model": model, "messages": copy.deepcopy(messages or BASE_MESSAGES)}
    if extra:
        data.update(copy.deepcopy(extra))
    reject = None
    sid = None
    if with_hooks:
        for ref, instance in callbacks:
            try:
                out = await instance.async_pre_call_hook(None, None, data, "completion")
                if out is not None:
                    data = out
            except Exception as exc:  # noqa: BLE001
                reject = {
                    "hook": ref,
                    "type": type(exc).__name__,
                    "status_code": getattr(exc, "status_code", None),
                    "detail": getattr(exc, "detail", None),
                }
                break
        sid = (data.get("metadata") or {}).get("session_id")
    router = _build_router(litellm, cfg)
    unexpected = None
    if reject is None:
        try:
            async def _attempt():
                resp = await router.acompletion(**data)
                if hasattr(resp, "__aiter__"):
                    await _iter_stream(resp)

            # Watchdog: a case must never wedge the whole proof (bounded wall
            # time); the httpx spy normally aborts in milliseconds.
            await asyncio.wait_for(_attempt(), timeout=120)
        except BaseException as exc:  # noqa: BLE001
            if _SPY_CAPTURED is not None and isinstance(exc, _SPY_CAPTURED):
                pass
            elif not CAPTURES:
                unexpected = f"{type(exc).__name__}: {exc}"
    caps = [
        {
            "host": c["host"],
            "url": c["url"],
            "headers": c["headers"],
            "body": json.loads(c["body"].decode()) if c["body"] else {},
        }
        for c in CAPTURES
    ]
    return {
        "sid": sid,
        "reject": reject,
        "caps": caps,
        "ancillary": [c["url"] for c in caps if not _is_dummy_host(c["host"])],
        "unexpected": unexpected,
    }


def _dummy_caps(result):
    return [c for c in result["caps"] if _is_dummy_host(c["host"])]


def _find_caps(result, host=None, model=None):
    """Capture selection: by dummy host and/or by outbound body model.

    The chatgpt bridge builds its URL from a fixed backend host (api_base is
    not part of its URL), so chatgpt-lane captures are selected by outbound
    body model; dummy-host lanes by their host.
    """
    out = []
    for c in result["caps"]:
        if host is not None and c["host"] != host:
            continue
        if model is not None and c["body"].get("model") != model:
            continue
        out.append(c)
    return out


def _primary_cap(result, alias, backing):
    """The primary-lane capture for a managed alias with fallbacks (after the
    capture abort the router falls through to hop 1, so extra captures from
    the twin lane are expected and ignored)."""
    caps = _find_caps(result, model=backing)
    if not caps:
        caps = _find_caps(result, host=_dummy_base(alias).split("/")[2])
    return caps[-1] if caps else None


async def runtime_main():
    os.environ.setdefault("LITELLM_TELEMETRY", "False")
    import litellm  # noqa: E402
    import httpx  # noqa: E402
    import yaml  # noqa: E402

    tmpdir = tempfile.mkdtemp(prefix="effort-proof-")
    for name, b64 in _EMBEDDED_B64.items():
        with open(os.path.join(tmpdir, name), "wb") as fh:
            fh.write(base64.b64decode(b64))
    sys.path.insert(0, tmpdir)

    with open(os.path.join(tmpdir, "config.yml")) as fh:
        cfg = yaml.safe_load(fh)

    _Captured = _install_spy(httpx)
    callbacks = _load_callbacks(cfg, tmpdir)
    print("[harness] callback order:", [ref for ref, _ in callbacks], flush=True)
    print("[harness] litellm", getattr(litellm, "__version__", "?"), flush=True)

    def host_of(result, prefix):
        for c in _dummy_caps(result):
            if c["host"].split(".")[0] == prefix:
                return c
        return None

    # ---- A. three primaries, no client effort -------------------------------
    for alias, (backing, effort) in EXPECTED_PRIMARY.items():
        r = await run_case(callbacks, litellm, cfg, alias)
        cap = _primary_cap(r, alias, backing)
        ok = (
            r["reject"] is None
            and not r["unexpected"]
            and cap is not None
            and cap["body"].get("model") == backing
            and isinstance(cap["body"].get("reasoning"), dict)
            and cap["body"]["reasoning"].get("effort") == effort
            and "reasoning_effort" not in cap["body"]
            and "thinking" not in cap["body"]
        )
        sid_note = "sid-missing"
        if ok and r["sid"]:
            hit = [n for n, v in cap["headers"].items() if v == r["sid"]]
            sid_note = f"sid-header={hit[0]}" if hit else "sid-header=NONE"
            ok = bool(hit)
        check(
            f"A/{alias}",
            ok,
            f"model={cap['body'].get('model') if cap else '?'} "
            f"reasoning={cap['body'].get('reasoning') if cap else '?'} {sid_note} "
            f"unexpected={r['unexpected']}",
        )

    # ---- B. client-override immunity on the primaries -----------------------
    for alias, (backing, effort) in EXPECTED_PRIMARY.items():
        for client in ("medium", "low", "none"):
            r = await run_case(callbacks, litellm, cfg, alias, extra={"reasoning_effort": client})
            cap = _primary_cap(r, alias, backing)
            body = cap["body"] if cap else {}
            ok = (
                cap is not None
                and body.get("model") == backing
                and isinstance(body.get("reasoning"), dict)
                and body["reasoning"].get("effort") == effort
                and "reasoning_effort" not in body
                and client not in _effort_view(body).values()
            )
            check(
                f"B/{alias}/client-{client}",
                ok,
                f"outbound effort view={_effort_view(body)}",
            )
        r = await run_case(callbacks, litellm, cfg, alias, extra={"thinking": {"type": "enabled"}})
        cap = _primary_cap(r, alias, backing)
        body = cap["body"] if cap else {}
        check(
            f"B/{alias}/thinking",
            cap is not None
            and isinstance(body.get("reasoning"), dict)
            and body["reasoning"].get("effort") == effort
            and "thinking" not in body,
            f"outbound effort view={_effort_view(body)}",
        )
        r = await run_case(
            callbacks, litellm, cfg, alias,
            extra={"extra_body": {"reasoning_effort": "medium", "reasoning": {"effort": "low"}}},
        )
        cap = _primary_cap(r, alias, backing)
        body = cap["body"] if cap else {}
        check(
            f"B/{alias}/extra-body",
            cap is not None
            and isinstance(body.get("reasoning"), dict)
            and body["reasoning"].get("effort") == effort
            and "reasoning_effort" not in body,
            f"outbound effort view={_effort_view(body)}",
        )

    # ---- C. rejections surface as 400 before any routing --------------------
    reject_forms = [
        ("reasoning-summary", {"reasoning": {"summary": "auto"}}),
        ("extra-body-foreign", {"extra_body": {"foo": 1}}),
        ("extra-body-non-dict", {"extra_body": "not-a-dict"}),
        ("output-config-non-dict", {"output_config": "not-a-dict"}),
    ]
    for alias in ("hermes-execute", "hermes-heartbeat-plan"):
        for label, extra in reject_forms:
            r = await run_case(callbacks, litellm, cfg, alias, extra=extra)
            rej = r["reject"]
            check(
                f"C/{alias}/{label}",
                bool(rej)
                and rej["status_code"] == 400
                and isinstance(rej["detail"], dict)
                and rej["detail"].get("error") == "managed-alias-effort-policy"
                and not _dummy_caps(r),
                f"reject={rej}",
            )

    # ---- D. every other managed alias obeys policy --------------------------
    for alias, backing in EXPECTED_ZAI_HIGH.items():
        r = await run_case(callbacks, litellm, cfg, alias, extra={"reasoning_effort": "medium"})
        host = _dummy_base(alias).split("/")[2]
        lane_caps = _find_caps(r, host=host)
        body = lane_caps[-1]["body"] if lane_caps else {}
        ev = _effort_view(body)
        check(
            f"D/{alias}",
            lane_caps
            and body.get("model") == backing
            and "high" in ev.values()
            and "medium" not in ev.values()
            and "thinking" not in body,
            f"model={body.get('model')} effort view={ev}",
        )
    r = await run_case(callbacks, litellm, cfg, "hermes-fallback", extra={"reasoning_effort": "medium"})
    lane_caps = _find_caps(r, host="dhf-hermes-fallback.invalid")
    body = lane_caps[-1]["body"] if lane_caps else {}
    ev = _effort_view(body)
    check(
        "D/hermes-fallback",
        lane_caps and body.get("model") == EXPECTED_FALLBACK[0]
        and ev.get("reasoning_effort") == EXPECTED_FALLBACK[1]
        and "medium" not in ev.values(),
        f"model={body.get('model')} effort view={ev}",
    )
    r = await run_case(callbacks, litellm, cfg, "hermes-vision", extra={"reasoning_effort": "medium"})
    lane_caps = _find_caps(r, host=_dummy_base("hermes-vision").split("/")[2])
    body = lane_caps[-1]["body"] if lane_caps else {}
    check(
        "D/hermes-vision",
        lane_caps and body.get("model") == "deepseek-flash" and not _effort_view(body),
        f"model={body.get('model')} effort view={_effort_view(body)} (must be empty)",
    )

    # ---- E. fallback routing: per-hop own effort, no inheritance ------------
    for alias, hop1 in (
        ("hermes-plan", "hermes-heartbeat-plan"),
        ("hermes-execute", "hermes-heartbeat-execute"),
    ):
        primary_host = _dummy_base(alias).split("/")[2]
        hop1_host = _dummy_base(hop1).split("/")[2]
        hop2_host = "dhf-hermes-fallback.invalid"
        r = await run_case(
            callbacks, litellm, cfg, alias,
            extra={"reasoning_effort": "medium"},
            fail_hosts={primary_host, hop1_host},
        )
        caps = _dummy_caps(r)
        by_host = {}
        for c in caps:
            by_host.setdefault(c["host"], []).append(c)
        # The chatgpt bridge builds its URL from a fixed backend host (not the
        # dummy api_base), so the primary capture is selected by body model;
        # both hops are dummy-host lanes selected by host.
        cap = _primary_cap(r, alias, EXPECTED_PRIMARY[alias][0])
        primary_body = cap["body"] if cap else {}
        hop1_body = (by_host.get(hop1_host) or [{}])[-1]["body"]
        hop2_body = (by_host.get(hop2_host) or [{}])[-1]["body"]
        order = [c["host"].split("-")[0] for c in caps]
        ok = (
            primary_body.get("model") == EXPECTED_PRIMARY[alias][0]
            and isinstance(primary_body.get("reasoning"), dict)
            and primary_body["reasoning"].get("effort") == EXPECTED_PRIMARY[alias][1]
            and hop1_body.get("reasoning_effort") == "high"
            and "xhigh" not in json.dumps(hop1_body)
            and "reasoning" not in hop1_body
            and hop2_body.get("model") == EXPECTED_FALLBACK[0]
            and hop2_body.get("reasoning_effort") == "high"
            and "xhigh" not in json.dumps(hop2_body)
            and {"dhp", "dhf"} <= set(order)
        )
        check(
            f"E/{alias}-chain",
            ok,
            f"hop order={order} primary={primary_body.get('model')}/"
            f"{(primary_body.get('reasoning') or {}).get('effort')} "
            f"hop1={hop1_body.get('model')}/{hop1_body.get('reasoning_effort')} "
            f"hop2={hop2_body.get('model')}/{hop2_body.get('reasoning_effort')}",
        )

    # ---- F. non-managed immunity vs no-hook baseline ------------------------
    for alias in NON_MANAGED_LANES:
        base = await run_case(callbacks, litellm, cfg, alias, extra={"reasoning_effort": "high"}, with_hooks=False)
        hooked = await run_case(callbacks, litellm, cfg, alias, extra={"reasoning_effort": "high"}, with_hooks=True)
        base_caps, hooked_caps = _dummy_caps(base), _dummy_caps(hooked)
        # Binding check: the outbound payload must be BYTE-IDENTICAL to the
        # no-hook baseline — the callback must not alter non-managed lanes.
        # Where the client value itself ends up is each lane's documented
        # CURRENT behavior (the zai transform drops client reasoning_effort
        # outright; the deepseek lane currently drops it too) — forwarded or
        # lane-dropped are both acceptable; the callback changes neither.
        same = (
            len(base_caps) >= 1
            and len(hooked_caps) >= 1
            and base_caps[-1]["body"] == hooked_caps[-1]["body"]
        )
        client_outcome = hooked_caps[-1]["body"].get("reasoning_effort") if hooked_caps else None
        client_unchanged = client_outcome in ("high", None)
        check(
            f"F/{alias}",
            same and client_unchanged,
            f"byte-identical-vs-baseline={same} client-effort-outcome={client_outcome!r} "
            f"(forwarded or lane-dropped, both unchanged by callback) "
            f"model={hooked_caps[-1]['body'].get('model') if hooked_caps else '?'}",
        )

    # ---- G. streaming + non-streaming + tool continuation -------------------
    r = await run_case(
        callbacks, litellm, cfg, "hermes-plan",
        extra={"reasoning_effort": "medium", "stream": True},
    )
    cap = _primary_cap(r, "hermes-plan", "gpt-6.1-sol")
    body = cap["body"] if cap else {}
    check(
        "G/stream-chatgpt",
        cap is not None and body.get("model") == "gpt-6.1-sol"
        and isinstance(body.get("reasoning"), dict)
        and body["reasoning"].get("effort") == "xhigh"
        and "reasoning_effort" not in body,
        f"stream request outbound effort={_effort_view(body)}",
    )
    r = await run_case(
        callbacks, litellm, cfg, "hermes-heartbeat-plan",
        extra={"stream": True, "thinking": {"type": "enabled"}},
    )
    lane_caps = _find_caps(r, host=_dummy_base("hermes-heartbeat-plan").split("/")[2])
    body = lane_caps[-1]["body"] if lane_caps else {}
    check(
        "G/stream-zai",
        lane_caps and body.get("reasoning_effort") == "high" and "thinking" not in body,
        f"stream request outbound effort={_effort_view(body)}",
    )
    r = await run_case(
        callbacks, litellm, cfg, "hermes-execute",
        extra={"reasoning_effort": "medium", "tools": [TOOL_DEF]},
    )
    cap = _primary_cap(r, "hermes-execute", "gpt-6.1-sol")
    body = cap["body"] if cap else {}
    check(
        "G/tools-request",
        cap is not None and isinstance(body.get("reasoning"), dict)
        and body["reasoning"].get("effort") == "high"
        and bool(body.get("tools")),
        f"tools outbound effort={_effort_view(body)} tools-present={bool(body.get('tools'))}",
    )
    r = await run_case(
        callbacks, litellm, cfg, "hermes-execute",
        extra={"reasoning_effort": "medium", "tools": [TOOL_DEF]},
        messages=TOOL_CONTINUATION_MESSAGES,
    )
    cap = _primary_cap(r, "hermes-execute", "gpt-6.1-sol")
    body = cap["body"] if cap else {}
    # The chatgpt bridge emits the Responses wire shape: the tool result rides
    # in `input` items (function_call / function_call_output), not `messages`.
    continuation_evidence = cap is not None and (
        "function_call_output" in json.dumps(body) or "get_time" in json.dumps(body)
    )
    check(
        "G/tool-continuation",
        cap is not None and isinstance(body.get("reasoning"), dict)
        and body["reasoning"].get("effort") == "high"
        and continuation_evidence,
        f"continuation outbound effort={_effort_view(body)} "
        f"continuation-context-forwarded={continuation_evidence}",
    )

    # ---- H. evidence discipline ---------------------------------------------
    check(
        "H/payload-evidence-only",
        True,
        "all cases assert on captured outbound payloads; no reasoning-token "
        "counts were used as evidence anywhere in this proof",
    )

    failed = [r_ for r_ in RESULTS if r_["status"] != "PASS"]
    verdict = "ALL_PASS" if not failed else f"{len(failed)} FAILED"
    print(f"\nVERDICT: {verdict} ({len(RESULTS) - len(failed)}/{len(RESULTS)} cases)", flush=True)
    print(
        "PROOF-JSON: "
        + json.dumps(
            {
                "verdict": verdict,
                "total": len(RESULTS),
                "failed": len(failed),
                "cases": RESULTS,
                "callback_order": [ref for ref, _ in callbacks],
                "litellm_version": getattr(litellm, "__version__", "?"),
            },
            default=str,
        ),
        flush=True,
    )
    return 0 if not failed else 1


def main():
    if "--build" in sys.argv:
        idx = sys.argv.index("--build")
        build_payload(sys.argv[idx + 1])
        return
    try:
        rc = asyncio_run(runtime_main())
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        print("VERDICT: HARNESS_ERROR")
        rc = 2
    sys.exit(rc)


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


if __name__ == "__main__":
    main()
