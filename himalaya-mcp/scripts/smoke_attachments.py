#!/usr/bin/env python3
"""Attachment smoke harness for the himalaya MCP server.

Offline (default): expects the smoke stack from smoke.docker-compose.yml on
--url; populates the m2dir fixture store via `docker run ... himalaya` and
asserts the full attachment verdict matrix (A1..A8). --live: skips store
population, runs only stack-independent assertions plus best-effort
listing/export against the newest INBOX message.

Host python3 stdlib only — no third-party imports.
"""

import argparse
import base64
import email
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVICE_ROOT)

PROTOCOL_VERSION = "2025-03-26"
SMOKE_CONFIG = (
    '[accounts.smoke]\ndefault = true\nemail = "smoke@fixture.invalid"\n'
    'downloads-dir = "/tmp/dl"\n\n[accounts.smoke.m2dir]\nroot = "/store"\n')

def die(msg):
    print(f"FATAL: {msg}")
    raise SystemExit(2)

def check(name, ok, detail="", ctx=None):
    """Print PASS/FAIL; on FAIL print offending JSON, then exit 1."""
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        if ctx is not None:
            print("  offending JSON:", json.dumps(ctx, ensure_ascii=False)[:400])
        sys.exit(1)

def skip(name, detail=""):
    print(f"[SKIP] {name}" + (f" — {detail}" if detail else ""))

class McpClient:
    """Minimal MCP Streamable HTTP client (JSON-RPC over POST)."""

    def __init__(self, url, timeout):
        self.url, self.timeout = url, timeout
        self.session, self.next_id = None, 0

    def _post(self, payload):
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream"}
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                ctype = resp.headers.get("Content-Type", "")
                body = resp.read().decode("utf-8", errors="replace")
                resp_headers = dict(resp.headers)
        except urllib.error.HTTPError as e:
            excerpt = e.read().decode(errors="replace")[:400]
            raise RuntimeError(f"HTTP {e.code} from {self.url}: {excerpt}") from e
        if "text/event-stream" in ctype:
            for line in body.splitlines():
                if line.startswith("data:"):
                    return json.loads(line[5:].strip()), resp_headers
            raise RuntimeError(f"no data: line in SSE body: {body[:400]!r}")
        return (None if not body.strip() else json.loads(body)), resp_headers

    def rpc(self, method, params=None, notify=False):
        self.next_id += 1
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notify:
            payload["id"] = self.next_id
        resp, resp_headers = self._post(payload)
        if notify:
            return None
        sid = resp_headers.get("Mcp-Session-Id") or resp_headers.get("mcp-session-id")
        if sid:
            self.session = sid
        if isinstance(resp, dict) and resp.get("error"):
            raise RuntimeError(f"RPC error on {method}: {json.dumps(resp['error'])[:400]}")
        return (resp or {}).get("result", {})

    def initialize(self):
        result = self.rpc("initialize", {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "smoke-attachments", "version": "1.0"}})
        server_info = result.get("serverInfo", {})
        if "himalaya" not in json.dumps(server_info):
            die(f"initialize serverInfo lacks 'himalaya': {json.dumps(server_info)[:400]}")
        self.rpc("notifications/initialized", notify=True)
        return result

    def tools_list(self):
        return self.rpc("tools/list", {}).get("tools", [])

    def call(self, tool, args):
        result = self.rpc("tools/call", {"name": tool, "arguments": args})
        return json.loads(result["content"][0]["text"])

def _load_module(name):
    try:
        return importlib.import_module("tests.fixtures.make_fixture")
    except Exception:
        path = os.path.join(SERVICE_ROOT, "tests", "fixtures", "make_fixture.py")
        if not os.path.isfile(path):
            return None
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

def fixture_raw_and_payloads():
    """(full eml bytes, {filename: bytes}) — via make_fixture, else fixture.eml."""
    mod = _load_module("make_fixture")
    if mod is not None:
        return mod.build_message().as_bytes(), {
            fn: payload for fn, (_, payload) in mod.build_payloads().items()}
    path = os.path.join(SERVICE_ROOT, "tests", "fixtures", "fixture.eml")
    if not os.path.isfile(path):
        die(f"make_fixture not importable and {path} missing — run make_fixture.py first")
    raw = open(path, "rb").read()
    payloads = {}
    for part in email.message_from_bytes(raw).walk():
        if part.get_filename():
            payloads[part.get_filename()] = part.get_payload(decode=True)
    return raw, payloads

def docker_run(store, fixtures, image, *himalaya_args):
    cfg_path = os.path.join(store, ".smoke-config.toml")
    with open(cfg_path, "w") as f:
        f.write(SMOKE_CONFIG)
    cmd = ["docker", "run", "--rm",
           "-v", f"{store}:/store",
           "-v", f"{cfg_path}:/smoke-config.toml",
           "-v", f"{fixtures}:/f",
           "--entrypoint", "himalaya", image,
           "--config", "/smoke-config.toml", *himalaya_args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        die(f"docker himalaya {' '.join(himalaya_args[:3])} failed rc={proc.returncode}: "
            f"{(proc.stderr or proc.stdout)[:400]}")
    return proc.stdout

def setup_store(store, fixtures, image):
    """Populate the m2dir store; return (envelope_id, flags) of the single message."""
    # m2dir store root must carry a `.m2store` marker; `.smoke-config.toml`
    # lives INSIDE the store dir so it rides the same volume mount.
    os.makedirs(store, exist_ok=True)
    # m2dir layout markers (verified against himalaya v2.0.0: 0-byte files):
    # <root>/.m2store, <root>/<folder>/.m2dir, <root>/<folder>/.meta/
    for marker in (os.path.join(store, ".m2store"),
                   os.path.join(store, "Inbox", ".m2dir")):
        if not os.path.exists(marker):
            with open(marker, "w") as f:
                f.write("")
    os.makedirs(os.path.join(store, "Inbox", ".meta"), exist_ok=True)
    # Reset: drop messages saved by previous runs (keep layout markers).
    for base in (os.path.join(store, "Inbox"), os.path.join(store, "Inbox", ".meta")):
        for name in os.listdir(base):
            if not name.startswith("."):
                os.remove(os.path.join(base, name))
    mod = _load_module("make_fixture")
    if mod is not None:
        with open(os.path.join(fixtures, "fixture.eml"), "wb") as f:
            f.write(mod.build_message().as_bytes())
    docker_run(store, fixtures, image,
               "m2dir", "messages", "save", "-m", "Inbox", "--", "/f/fixture.eml")
    out = docker_run(store, fixtures, image,
                     "--json", "envelope", "list", "-m", "Inbox", "-p", "1", "-s", "10")
    envelopes = None
    try:
        envelopes = json.loads(out)
    except json.JSONDecodeError:
        die(f"envelope list returned non-JSON: {out[:400]}")
    if isinstance(envelopes, dict) and "envelopes" in envelopes:
        envelopes = envelopes["envelopes"]  # raw CLI wraps; server unwraps
    if not isinstance(envelopes, list) or len(envelopes) != 1:
        die(f"expected exactly 1 envelope in smoke store, got: {json.dumps(envelopes)[:400]}")
    return str(envelopes[0]["id"]), envelopes[0].get("flags")

def run(args):
    client = McpClient(args.url, args.timeout)
    client.initialize()

    # A1 — tool surface
    names = {t.get("name") for t in client.tools_list()}
    check("A1 tools/list has attachment_list+attachment_export, no attachment_download",
          {"attachment_list", "attachment_export"} <= names and "attachment_download" not in names,
          f"tools={sorted(n for n in names if n)}"[:400])

    health = client.call("health_check", {})

    # A8 — health/config (cap value expectation depends on stack)
    wl = health.get("attachment_whitelist_exts")
    tools_reg = health.get("tools_registered", [])
    cap = health.get("config", {}).get("ATTACHMENT_MAX_BYTES")
    check("A8 health_check ok + tool surface + whitelist",
          health.get("status") == "ok"
          and {"attachment_list", "attachment_export"} <= set(tools_reg)
          and "attachment_download" not in tools_reg
          and isinstance(cap, int)
          and isinstance(wl, list) and wl == sorted(wl)
          and "pdf" in wl and "png" in wl,
          f"cap={cap!r} wl_len={len(wl) if isinstance(wl, list) else '?'}", ctx=health)
    if not args.live:
        check("A8b smoke env ATTACHMENT_MAX_BYTES == 1000", cap == 1000, f"cap={cap!r}", ctx=health)

    mid, flags = None, None
    folder = "Inbox"  # m2dir smoke store folder (case-sensitive)
    _, payloads = fixture_raw_and_payloads()
    entries = []

    if args.live:
        try:
            envs = client.call("envelope_list", {"folder": "INBOX", "page_size": 10})
            mid = str(envs[0]["id"])
        except Exception as e:
            skip("live attachment_list", f"no INBOX message usable: {e}")
    else:
        mid, flags = setup_store(args.store, args.fixtures, args.image)
        check("A2 envelope_list Inbox == 1 message", True, f"id={mid} flags={flags}")

        # A3 — verdict matrix
        listing = client.call("attachment_list", {"id": mid, "folder": folder, "include_inline": False})
        entries = listing.get("attachments", [])
        expected = [("report.pdf", True, None), ("bild.png", True, None),
                    ("invite.ics", False, "calendar"), ("setup.exe", False, "extension"),
                    ("grafik.png", True, None), ("gross.pdf", False, "exceeds cap")]
        ok = len(entries) == 6 and all(
            e.get("filename") == fn and e.get("exportable") is exp
            and (sub is None or sub in (e.get("block_reason") or ""))
            for e, (fn, exp, sub) in zip(entries, expected))
        detail = "; ".join(f"{i + 1}:{e.get('filename')}={e.get('exportable')}"
                           f"({e.get('block_reason')})" for i, e in enumerate(entries))
        check("A3 attachment_list 6-entry verdict matrix", ok, detail[:400], ctx=listing)

        # A4 — happy-path export byte-exactness
        exp = client.call("attachment_export", {"id": mid, "folder": folder, "attachment_id": 1})
        data = base64.b64decode(exp.get("content_b64", ""))
        check("A4 export report.pdf byte-exact",
              exp.get("filename") == "report.pdf" and exp.get("encoding") == "base64"
              and data == payloads["report.pdf"]
              and exp.get("size") == len(payloads["report.pdf"]),
              f"size={exp.get('size')} len={len(payloads['report.pdf'])} "
              f"bytes_match={data == payloads['report.pdf']}", ctx=exp)

        # A5 — export refusals
        for aid, needle in [(3, "calendar"), (4, "extension"), (5, "markup"), (6, "exceeds cap")]:
            r = client.call("attachment_export", {"id": mid, "folder": folder, "attachment_id": aid})
            check(f"A5 export refusal id={aid} ({needle})",
                  "error" in r and needle in r.get("error", "")
                  and r.get("stays_on_server") is True,
                  f"error={r.get('error')!r}", ctx=r)

        # A6 — out-of-range id
        r = client.call("attachment_export", {"id": mid, "folder": folder, "attachment_id": 999})
        check("A6 export id=999 refused with stays_on_server",
              "error" in r and r.get("stays_on_server") is True,
              f"error={r.get('error')!r}", ctx=r)

        # A7 — no \Seen side effect
        envs = client.call("envelope_list", {"folder": "Inbox", "page_size": 10})
        check("A7 flags unchanged after peek-safe ops",
              len(envs) == 1 and envs[0].get("flags") == flags,
              f"before={flags} after={envs[0].get('flags') if envs else '?'}", ctx=envs)

    if args.live and mid is not None:
        try:
            entries = client.call(
                "attachment_list", {"id": mid, "folder": folder, "include_inline": False}).get("attachments", [])
        except Exception as e:
            skip("live attachment_list", str(e)[:200])
        exported = False
        for idx, e in enumerate(entries, start=1):
            if not e.get("exportable"):
                continue
            try:
                r = client.call("attachment_export", {"id": mid, "folder": folder, "attachment_id": idx})
                if "content_b64" in r:
                    base64.b64decode(r["content_b64"])
                    print(f"[PASS] live export {e.get('filename')} "
                          f"({r.get('size')} bytes, base64 decodes)")
                    exported = True
                    break
            except Exception as ex:
                skip(f"live export {e.get('filename')}", str(ex)[:200])
                break
        if not exported:
            skip("live export", "no exportable attachment on newest INBOX message")

    print("ALL MANDATORY ASSERTIONS PASSED")

def main():
    p = argparse.ArgumentParser(description="Attachment smoke harness (stdlib MCP client)")
    p.add_argument("--url", default="http://localhost:9202/mcp")
    p.add_argument("--live", action="store_true", help="run against the real stack")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--image", default="owui-himalaya-smoke:latest")
    p.add_argument("--store", default=os.path.join(SERVICE_ROOT, "tests", "fixtures", "store"))
    p.add_argument("--fixtures", default=os.path.join(SERVICE_ROOT, "tests", "fixtures"))
    p.add_argument("--keep-stack", action="store_true",
                   help="offline: leave the smoke stack running after the run")
    args = p.parse_args()
    if not args.live:
        os.makedirs(args.store, exist_ok=True)
        up_smoke_stack(args.image)
        try:
            run(args)
        finally:
            if not args.keep_stack:
                down_smoke_stack()
    else:
        run(args)


def up_smoke_stack(image: str) -> None:
    """Build + start the offline smoke stack (compose project himalaya-smoke)."""
    compose = ["docker", "compose", "-p", "himalaya-smoke",
               "-f", os.path.join(SERVICE_ROOT, "smoke.docker-compose.yml")]
    for step in (["build"], ["up", "-d"]):
        proc = subprocess.run(compose + step, capture_output=True, text=True)
        if proc.returncode != 0:
            die(f"docker compose {step[0]} failed: {(proc.stderr or proc.stdout)[:400]}")
    # Wait for the MCP endpoint to answer initialize (up to 60 s).
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            McpClient("http://localhost:9202/mcp", 5).initialize()
            return
        except Exception:
            time.sleep(2)
    die("smoke stack did not become ready on :9202 within 60 s")


def down_smoke_stack() -> None:
    subprocess.run(["docker", "compose", "-p", "himalaya-smoke",
                    "-f", os.path.join(SERVICE_ROOT, "smoke.docker-compose.yml"),
                    "down", "-v"], capture_output=True, text=True)

if __name__ == "__main__":
    main()
