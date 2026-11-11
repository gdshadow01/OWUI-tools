#!/usr/bin/env python3
"""T3 REAL-proxy-path HTTP proof for enforce_effort_policy (plan 261031 re-patch).

The harness gap that let the original defect ship: T1/T2 drove the callback
in-process with call_type='completion', while the REAL litellm v1.103.2 proxy
route presents call_type='acompletion' — the original gate silently
early-returned there (deploy t_7ebff3fe: HTTP 200 where policy required 400,
zero hook log lines). This harness drives the REAL HTTP boundary:

  1. Builds a scratch config from the working-tree config.yml with chatgpt
     backings text-swapped to zai/glm-5.3 and mode responses->chat (dummy
     upstreams; prevents the chatgpt OAuth device-code startup hang). The
     callbacks section and every policy value are byte-untouched.
  2. Starts a local capture server; the scratch config's upstream api_bases
     are redirected to it. It records every FINAL provider-bound request body
     verbatim (the strip proof) and serves canned completions / SSE so probes
     resolve cleanly. A body-conditioned outage rule forces the fallback
     chain for the per-hop effort proof.
  3. Runs the pinned v1.103.2 image (digest-checked) in an ISOLATED scratch
     container with no published ports and dummy creds, mounting the working-tree
     callback files. Never touches .209 or any running proxy. An internal
     per-run Docker bridge prevents external upstream routing. Capture binds
     only its gateway and requires a run token and owned-container source.
     T3 substitutes Z.ai; T2 actual ChatGPT bridge interception is complementary.
  4. Probes over HTTP with the master key and asserts:
       - BINDING: all ten managed aliases + reasoning:false -> HTTP 400
         managed-alias-effort-policy (the exact probe that failed on deploy);
       - strip proof: client effort carriers absent from captured FINAL
         provider-bound payloads; configured per-deployment effort present;
       - reject matrix over the real path (residual/unsupported forms);
       - streaming + non-streaming + tool continuation;
       - manual aliases and -zai twins pass through (no policy 400, client
         effort intact in the captured payload);
       - fallback hop uses ITS OWN config effort (primary xhigh vs heartbeat
         high captured on the wire, no inheritance);
       - sibling callbacks still active on the real path (codex_session_id);
       - config.yml drift checks (alias classification, registration order).
     Reasoning-token counts are never used as evidence.

Usage: python3 test_enforce_effort_policy_proxy_http.py [--out DIR] [--keep]
Exit 0 only if every case passes. Requires local rootful Linux Docker bridge
access and the already-local pinned image; no image pulls. Evidence is written
under a unique run directory beneath --out. No proxy ports are published
(--proxy-port was removed). --keep retains only this run's labeled container
and network, never the capture listener; remove the reported container ID and
then network ID after inspection. This is not a sandbox against host root or
Docker administrators, who can read tokens and join Docker networks.
"""

import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path
from proxy_http_resources import ScratchResources, sh
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_YML = os.path.join(HERE, "config.yml")
PROXY_BASE = None  # Direct host access to the internal run bridge; no published ports.
IMAGE = (
    "docker.litellm.ai/berriai/litellm:v1.103.2"
    "@sha256:f63fb81b831b170ec16851e23c36ac5bf52ef106b271406429524a2ed730bbfd"
)
IMAGE_DIGEST = "sha256:f63fb81b831b170ec16851e23c36ac5bf52ef106b271406429524a2ed730bbfd"
MASTER_KEY = "sk-t3-scratch"
CALLBACK_FILES = [
    "enforce_effort_policy.py",
    "strip_thinking.py",
    "codex_session_id.py",
    "fallback_notify.py",
    "strip_hermes.py",
]
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
THINKING_OPENCODE = {"type": "enabled", "clear_thinking": False}
MESSAGES = [{"role": "user", "content": "Reply with the single word: ok"}]

LOCAL_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))

RESULTS = []
CAPTURES = []
RULES = []  # outage rules: {"contains": <substring>, "remaining": <int>}
_seq_lock = threading.Lock()


def check(case, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    RESULTS.append({"case": case, "status": status, "detail": str(detail)[:500]})
    print(f"{status} {case} :: {str(detail)[:500]}", flush=True)
    return bool(cond)


# ------------------------------- capture server -------------------------------

CANNED = {
    "id": "chatcmpl-t3",
    "object": "chat.completion",
    "created": 1,
    "model": "captured",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
SSE_CHUNKS = (
    'data: {"id":"chatcmpl-t3","object":"chat.completion.chunk","choices":[{"delta":{"content":"ok"}}]}\n\n'
    "data: [DONE]\n\n"
)


class CaptureHandler(BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):  # silence per-request stderr
        return

    def _reply_json(self, obj, code=200, ctype="application/json"):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        if (self.headers.get("Authorization") != f"Bearer {self.server.token}"
                or self.client_address[0] != self.server.allowed_source):
            self._reply_json({"error": "unauthorized capture request"}, code=403)
            return False
        return True

    def do_GET(self):
        if not self.authorized():
            return
        # /models-style answers for any upstream introspection
        self._reply_json({"object": "list", "data": []})

    def do_POST(self):
        if not self.authorized():
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            body = {"_unparsed": raw.decode("utf-8", "replace")[:2000]}
        with _seq_lock:
            seq = len(CAPTURES)
            CAPTURES.append(
                {"seq": seq, "path": self.path, "ts": time.time(), "body": body}
            )
            fail = False
            for rule in RULES:
                if rule["remaining"] > 0 and rule["contains"] in raw.decode("utf-8", "replace"):
                    rule["remaining"] -= 1
                    fail = True
                    break
        if fail:
            self._reply_json(
                {"error": {"message": "simulated outage (harness rule)", "type": "server_error"}},
                code=500,
            )
            return
        if body.get("stream"):
            payload = SSE_CHUNKS.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self._reply_json(CANNED)


def start_capture_server(bind_host, token):
    server = ThreadingHTTPServer((bind_host, 0), CaptureHandler)
    server.token = token
    server.allowed_source = None  # Fail closed until the owned container is inspected.
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    try:
        thread.start()
    except BaseException:
        server.server_close()
        raise
    return server, thread


def stop_capture_server(server, thread):
    try:
        server.shutdown()
    finally:
        server.server_close()
        thread.join(timeout=5)


# ------------------------------- scratch setup --------------------------------


def build_scratch_config(cap_base):
    with open(CONFIG_YML) as fh:
        text = fh.read()
    # Dummy-upstream transforms ONLY (scratch copy): chatgpt backings would
    # hang startup in the OAuth device-code flow without auth.json.
    text = text.replace("chatgpt/gpt-6.1-sol", "zai/glm-5.3")
    text = text.replace("chatgpt/gpt-6-luna", "zai/glm-5.3")
    text = text.replace("chatgpt/gpt-6-astra", "zai/glm-5.3")
    text = text.replace("mode: responses", "mode: chat")
    # Inline deepseek api_base -> capture server (zai lanes follow the
    # ZAI_API_BASE env var, set on the container below).
    text = text.replace("https://api.deepseek.com/v1", cap_base + "/v1")
    residual = [
        line
        for line in text.splitlines()
        if "chatgpt/" in line.split("#", 1)[0]
    ]
    assert not residual, f"scratch config still references chatgpt backings: {residual[:3]}"
    return text


def wait_health(timeout_s=240):
    deadline = time.time() + timeout_s
    url = f"{PROXY_BASE}/health/liveliness"
    while time.time() < deadline:
        try:
            with LOCAL_HTTP.open(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return False


def http_post(payload, timeout=120):
    req = urllib.request.Request(
        f"{PROXY_BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {MASTER_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with LOCAL_HTTP.open(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def effort_view(body):
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


def captured_since(marker):
    with _seq_lock:
        return [c for c in CAPTURES if c["seq"] >= marker]


# ------------------------------- proof cases ----------------------------------


def run_proofs():
    ok = True

    # S1 — BINDING: every managed alias + reasoning:false -> policy 400.
    # This is the exact acceptance probe that returned 200 on deploy t_7ebff3fe.
    for alias in MANAGED:
        code, text = http_post(
            {"model": alias, "messages": MESSAGES, "reasoning": False, "max_tokens": 8}
        )
        good = code == 400 and "managed-alias-effort-policy" in text and '"unsupported-type"' in text
        ok = check(f"S1.binding-400[{alias}]", good, f"HTTP {code}: {text[:220]}") and ok

    # ---- strip + configured-effort proof on captured FINAL payloads ----
    # expected maps body key -> required value (None = key must be absent).
    # Client-carrier stripping is proven by the VALUE checks: the configured
    # level (from config.yml, never the client-sent one) must be what lands
    # on the wire.
    def strip_case(tag, model, extra, expected):
        marker = len(CAPTURES)
        code, text = http_post(dict({"model": model, "messages": MESSAGES, "max_tokens": 16}, **extra))
        if code != 200:
            return check(tag, False, f"HTTP {code}: {text[:220]}")
        caps = captured_since(marker)
        if not caps:
            return check(tag, False, "no provider-bound payload captured")
        body = caps[-1]["body"]
        good = True
        detail = f"view={effort_view(body)}"
        for key, want in expected.items():
            got = body.get(key)
            if want is None:
                if key in body:
                    good = False
                    detail += f" {key} must be absent, got {body[key]!r}"
            elif got != want:
                good = False
                detail += f" {key}={got!r} != {want!r}"
        return check(tag, good, detail)

    # S2: conflicting client level + opencode thinking -> stripped; execute=high configured.
    ok = strip_case(
        "S2.execute-client-medium+thinking-stripped",
        "hermes-execute",
        {"reasoning_effort": "medium", "thinking": THINKING_OPENCODE},
        {"reasoning": {"effort": "high"}, "reasoning_effort": None, "thinking": None},
    ) and ok

    # S3: absent effort -> configured xhigh (plan) reaches the wire.
    ok = strip_case(
        "S3.plan-absent-effort-configured-xhigh",
        "hermes-plan",
        {},
        {"reasoning": {"effort": "xhigh"}, "reasoning_effort": None, "thinking": None},
    ) and ok

    # S4: conflicting client dict level -> stripped; configured xhigh wins.
    ok = strip_case(
        "S4.plan-client-reasoning-effort-stripped",
        "hermes-plan",
        {"reasoning": {"effort": "low"}},
        {"reasoning": {"effort": "xhigh"}, "reasoning_effort": None},
    ) and ok

    # S5: output_config effort popped, residual sub-keys preserved; tasks=high.
    marker = len(CAPTURES)
    code, text = http_post(
        {
            "model": "hermes-tasks",
            "messages": MESSAGES,
            "max_tokens": 16,
            "output_config": {"effort": "high", "verifiers": {"enabled": True}},
        }
    )
    if code == 200 and captured_since(marker):
        body = captured_since(marker)[-1]["body"]
        ok = check(
            "S5.tasks-output_config-residual-preserved",
            body.get("output_config") == {"verifiers": {"enabled": True}}
            and body.get("reasoning") == {"effort": "high"},
            f"output_config={body.get('output_config')!r} reasoning={body.get('reasoning')!r}",
        ) and ok
    else:
        ok = check("S5.tasks-output_config-residual-preserved", False, f"HTTP {code}: {text[:200]}") and ok

    # S6: zai lane (top-level reasoning_effort config) — client "none" stripped,
    # configured high lands on the wire (client value "none" != "high" proves it).
    ok = strip_case(
        "S6.coding-plan-client-none-stripped",
        "hermes-coding-plan",
        {"reasoning_effort": "none"},
        {"reasoning_effort": "high", "thinking": None},
    ) and ok

    # S7-S12: reject matrix over the REAL path (must 400, never silently pass).
    rejects = [
        ("S7.fallback-residual-reasoning-400", "hermes-fallback", {"reasoning": {"effort": "low", "summary": "auto"}}, "non-effort-residual"),
        ("S8.vision-output_config-type-400", "hermes-vision", {"output_config": "high"}, "unsupported-type"),
        ("S9.plan-extra_body-residual-400", "hermes-plan", {"extra_body": {"foo": 1}}, "residual-keys"),
        ("S10.plan-reasoning-type-400", "hermes-plan", {"reasoning": "high"}, "unsupported-type"),
        ("S11.plan-reasoning-false-400", "hermes-plan", {"reasoning": False}, "unsupported-type"),
        ("S12.execute-extra_body-type-400", "hermes-execute", {"extra_body": ["reasoning_effort"]}, "unsupported-type"),
    ]
    for tag, model, extra, reason in rejects:
        code, text = http_post(dict({"model": model, "messages": MESSAGES, "max_tokens": 8}, **extra))
        good = code == 400 and f'"{reason}"' in text and "managed-alias-effort-policy" in text
        ok = check(tag, good, f"HTTP {code}: {text[:220]}") and ok

    # S13: fully-consumable extra_body form is stripped, NOT rejected.
    ok = strip_case(
        "S13.plan-extra_body-effort-only-stripped",
        "hermes-plan",
        {"extra_body": {"reasoning_effort": "low"}},
        {"reasoning": {"effort": "xhigh"}, "reasoning_effort": None},
    ) and ok

    # S14: manual alias passthrough — the policy hook must NOT touch
    # non-managed aliases. extra_body bypasses litellm's supported-param
    # machinery, so a marker + client effort sent via extra_body must land
    # on the wire VERBATIM for a manual alias (a managed alias would be
    # policy-400'd for the residual extra_body instead — see S9).
    marker = len(CAPTURES)
    code, text = http_post(
        {
            "model": "zai-glm-5.3",
            "messages": MESSAGES,
            "max_tokens": 16,
            "extra_body": {"marker_keep": 1, "reasoning_effort": "low"},
        }
    )
    if code == 200 and captured_since(marker):
        body = captured_since(marker)[-1]["body"]
        ok = check(
            "S14.manual-alias-client-effort-intact",
            body.get("reasoning_effort") == "low" and body.get("marker_keep") == 1,
            f"captured reasoning_effort={body.get('reasoning_effort')!r} marker_keep={body.get('marker_keep')!r}",
        ) and ok
    else:
        ok = check("S14.manual-alias-client-effort-intact", False, f"HTTP {code}: {text[:200]}") and ok

    # S15: remaining manual flash alias also bypasses managed policy.
    marker = len(CAPTURES)
    code, text = http_post(
        {
            "model": "zai-glm-5.3-flash",
            "messages": MESSAGES,
            "max_tokens": 16,
            "extra_body": {"marker_keep": 1, "reasoning_effort": "low"},
        }
    )
    if code == 200 and captured_since(marker):
        body = captured_since(marker)[-1]["body"]
        ok = check(
            "S15.manual-flash-passthrough",
            body.get("reasoning_effort") == "low" and body.get("marker_keep") == 1,
            f"HTTP {code}, captured reasoning_effort={body.get('reasoning_effort')!r}",
        ) and ok
    else:
        ok = check("S15.manual-flash-passthrough", False, f"HTTP {code}: {text[:200]}") and ok

    # S16: streaming — stripped payload, SSE response.
    marker = len(CAPTURES)
    code, text = http_post(
        {
            "model": "hermes-execute",
            "messages": MESSAGES,
            "max_tokens": 16,
            "stream": True,
            "reasoning_effort": "medium",
        }
    )
    stream_ok = code == 200 and "data:" in text and "[DONE]" in text
    if stream_ok and captured_since(marker):
        body = captured_since(marker)[-1]["body"]
        stream_ok = (
            body.get("stream") is True
            and "reasoning_effort" not in body
            and body.get("reasoning") == {"effort": "high"}
        )
    ok = check("S16.streaming-stripped-sse", stream_ok, f"HTTP {code}: {text[:160]}") and ok

    # S17: tool continuation — tools + tool role preserved, client effort gone.
    tool = {
        "type": "function",
        "function": {"name": "get_time", "description": "time", "parameters": {"type": "object", "properties": {}}},
    }
    tool_msgs = [
        {"role": "user", "content": "What time is it? Use the tool."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_t3", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_t3", "content": "12:00"},
    ]
    marker = len(CAPTURES)
    code, text = http_post(
        {
            "model": "hermes-execute",
            "messages": tool_msgs,
            "tools": [tool],
            "max_tokens": 32,
            "reasoning_effort": "medium",
        }
    )
    if code == 200 and captured_since(marker):
        body = captured_since(marker)[-1]["body"]
        ok = check(
            "S17.tool-continuation-preserved",
            len(body.get("tools", [])) == 1
            and any(m.get("role") == "tool" for m in body.get("messages", []))
            and "reasoning_effort" not in body
            and body.get("reasoning") == {"effort": "high"},
            f"tools={len(body.get('tools', []))} msgs={len(body.get('messages', []))} view={effort_view(body)}",
        ) and ok
    else:
        ok = check("S17.tool-continuation-preserved", False, f"HTTP {code}: {text[:200]}") and ok

    # S18: fallback routing — body-conditioned outage: EVERY provider-bound
    # body carrying the primary's configured xhigh fails (that substring only
    # occurs on hermes-plan primary hops; the heartbeat hop carries its own
    # top-level reasoning_effort=high). Whatever the retry count, the chain
    # must eventually hop and the final body must use the HOP'S OWN config
    # effort — never the primary's xhigh.
    marker = len(CAPTURES)
    RULES.append({"contains": '"xhigh"', "remaining": 12})
    try:
        code, text = http_post(
            {"model": "hermes-plan", "messages": MESSAGES, "max_tokens": 16},
            timeout=300,
        )
    finally:
        RULES.clear()
    caps = captured_since(marker)
    xhigh_hops = [c for c in caps if isinstance(c["body"].get("reasoning"), dict) and c["body"]["reasoning"].get("effort") == "xhigh"]
    final = caps[-1]["body"] if caps else {}
    good = (
        code == 200
        and len(xhigh_hops) >= 1
        and final.get("reasoning_effort") == "high"
        and not (isinstance(final.get("reasoning"), dict) and final["reasoning"].get("effort") == "xhigh")
    )
    ok = check(
        "S18.fallback-per-hop-effort",
        good,
        f"HTTP {code}, hops={len(caps)}, xhigh_hops={len(xhigh_hops)}, final_view={effort_view(final)}",
    ) and ok
    return ok and all(r["status"] == "PASS" for r in RESULTS)


def log_evidence_checks(ok, container_id):
    logs = sh(["docker", "logs", container_id], timeout=60)
    combined = logs.stdout + logs.stderr
    ok = check(
        "S19.fix-hook-logs-on-real-path",
        "[enforce_effort_policy]" in combined and ("stripped" in combined or "rejecting" in combined),
        "hook log lines present on the real proxy path",
    ) and ok
    ok = check(
        "S20.sibling-codex-session-active",
        "[codex_session_id] hermes-plan ->" in combined,
        "sibling callback active in the same container/run",
    ) and ok
    return ok


def drift_checks(ok, container_id):
    # Parse YAML in the pinned runtime, not in the host/pytest environment.
    parsed = sh(["docker", "exec", container_id, "python", "/app/proxy_http_resources.py",
                 "/app/config.yml"], timeout=60)
    if parsed.returncode:
        raise RuntimeError(f"config parse failed: {parsed.stderr[:500]}")
    cfg = json.loads(parsed.stdout)
    entries = {e["model_name"]: e for e in cfg.get("model_list", [])}
    missing = sorted(set(MANAGED) - set(entries))
    ok = check("S21.managed-aliases-exist", not missing, f"missing={missing}") and ok
    unclassified = sorted(
        n for n in entries if n.startswith("hermes-") and n not in set(MANAGED)
    )
    ok = check("S22.hermes-names-classified", not unclassified, f"unclassified={unclassified}") and ok
    callbacks = cfg.get("litellm_settings", {}).get("callbacks", [])
    ok = check(
        "S23.registration-present-last",
        callbacks[-1:] == ["enforce_effort_policy.effort_policy_handler"],
        f"callbacks={callbacks}",
    ) and ok
    fallbacks = cfg.get("litellm_settings", {}).get("fallbacks", [])
    stray = []
    for entry in fallbacks:
        for source, targets in entry.items():
            stray += [t for t in targets if t not in MANAGED]
    ok = check("S24.fallback-targets-managed", not stray, f"stray={stray}") and ok
    return ok


# ------------------------------- orchestration --------------------------------


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".", help="parent for unique run evidence directory")
    parser.add_argument("--keep", action="store_true", help="keep this run's owned container/network")
    args = parser.parse_args(argv)
    return args


def write_captures(path):
    with open(path, "w") as fh:
        for cap in CAPTURES:
            fh.write(json.dumps(cap) + "\n")


def main(argv=None):
    global PROXY_BASE
    args = parse_args(argv)
    RESULTS.clear()
    CAPTURES.clear()
    RULES.clear()
    owner = ScratchResources()
    out = Path(args.out).resolve() / owner.name
    out.mkdir(parents=True, exist_ok=False)
    captures_path = out / "t3_captures.jsonl"
    server = thread = None
    exit_code = 2
    print("=== T3 real proxy HTTP proof (SUBSTITUTED Z.ai, not actual ChatGPT bridge) ===")
    print(f"run={owner.run_id} evidence={out}")
    try:
        if sh(["docker", "info"], timeout=60).returncode:
            raise RuntimeError("docker must be usable")
        check("T0.docker-available", True)
        insp = sh(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], timeout=60)
        if not check("T0.pinned-image-digest", insp.returncode == 0 and insp.stdout.strip() == IMAGE_DIGEST,
                     insp.stdout.strip() or insp.stderr.strip()[:200]):
            return 2
        gateway = owner.create_network()
        token = "sk-t3-" + secrets.token_hex(24)
        server, thread = start_capture_server(gateway, token)
        cap_base = f"http://{gateway}:{server.server_port}"
        print(f"capture listener {cap_base} (internal run bridge, token + owned-source required)")
        cfg_path = out / "t3_scratch_config.yaml"
        cfg_path.write_text(build_scratch_config(cap_base))
        for name in CALLBACK_FILES + ["config.yml"]:
            path = Path(HERE) / name
            print(f"  {name} sha256={hashlib.sha256(path.read_bytes()).hexdigest()[:16]}")
        owner.create_container(IMAGE, cfg_path,
                               [Path(HERE) / n for n in CALLBACK_FILES + ["proxy_http_resources.py", "config.yml"]],
                               token, cap_base, MASTER_KEY)
        server.allowed_source = owner.source
        PROXY_BASE = f"http://{owner.source}:4000"
        print(f"scratch proxy {PROXY_BASE} (internal bridge only); container={owner.container_id}")
        if not check("T0.scratch-healthy", wait_health(), "health/liveliness"):
            logs = sh(["docker", "logs", owner.container_id], timeout=60)
            print((logs.stdout + logs.stderr)[-2000:])
        else:
            ok = run_proofs()
            ok = log_evidence_checks(ok, owner.container_id) and ok
            ok = drift_checks(ok, owner.container_id) and ok
            exit_code = 0 if ok else 1
    except Exception as exc:
        check("T0.harness-error", False, str(exc))
    finally:
        # Independent guards: a failed artifact write must not leak resources.
        try:
            write_captures(captures_path)
        except Exception as exc:
            check("T0.evidence-write", False, str(exc))
            exit_code = 2
        try:
            if server is not None:
                stop_capture_server(server, thread)
        except Exception as exc:
            check("T0.capture-cleanup", False, str(exc))
            exit_code = 2
        try:
            if not args.keep:
                owner.cleanup()
            else:
                print(f"KEPT owned resources: {owner.name} {owner.network_name}")
        except Exception as exc:
            check("T0.resource-cleanup", False, str(exc))
            exit_code = 2
        failed = [r for r in RESULTS if r["status"] != "PASS"]
        summary = {
            "proof": "T3 substituted Z.ai real proxy HTTP; NOT actual ChatGPT bridge proof",
            "complement": "T2 actual ChatGPT bridge interception remains complementary",
            "image_digest": IMAGE_DIGEST, "run_id": owner.run_id,
            "container_id": owner.container_id, "network_id": owner.network_id,
            "kept": args.keep, "cases": len(RESULTS), "failed": len(failed),
            "captures_written": str(captures_path), "results": RESULTS,
            "status": "PASS" if exit_code == 0 and not failed else "FAIL",
        }
        print("PROXY-HTTP-PROOF-JSON:", json.dumps(summary))
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return exit_code if exit_code else (1 if failed else 0)


def test_proxy_http_real_boundary(tmp_path):
    """pytest entry point: ignore pytest's argv and isolate output paths."""
    assert main(["--out", str(tmp_path)]) == 0


if __name__ == "__main__":
    sys.exit(main())
