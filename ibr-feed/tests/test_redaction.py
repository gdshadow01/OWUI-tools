"""Regression tests for the B1 finding (parent t_e21e9a51): the feed token
leaked into uvicorn's access log because the old redact_paths middleware was
a response-side no-op that never touched what uvicorn logs. These tests
exercise the two layers that matter:

1. the ASGI scope uvicorn's access logger formats its line from, checked
   through uvicorn's own get_path_with_query_string(), and
2. a real uvicorn subprocess with access logging on -- the exact channel
   that exposed the token through `docker logs` on the live deployment.

All tokens are synthetic fixed strings, never production secrets.
"""

import asyncio
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TOKEN = "b" * 48
WRONG = "c" * 48
REDACTED = "/<redacted>/leitsaetze.atom"


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _drive_app(scope):
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    async def call():
        import app

        await app.app(scope, receive, send)

    asyncio.run(call())
    return messages


def test_access_log_path_is_redacted():
    from uvicorn.protocols.utils import get_path_with_query_string

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": f"/{TOKEN}/leitsaetze.atom",
        "raw_path": f"/{TOKEN}/leitsaetze.atom".encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    messages = _drive_app(scope)
    assert messages and messages[0]["type"] == "http.response.start"
    assert scope["path"] == REDACTED
    assert scope["raw_path"] == REDACTED.encode()
    # This is the exact string uvicorn's AccessLogger prints for the request;
    # uvicorn percent-quotes raw_path, so decode before comparing.
    logged = get_path_with_query_string(scope)
    assert TOKEN not in logged
    assert urllib.parse.unquote(logged) == REDACTED


def test_real_uvicorn_access_log_never_contains_token(tmp_path):
    port = _free_port()
    (tmp_path / "leitsaetze.atom").write_bytes(b"<feed>placeholder</feed>")
    env = {
        **os.environ,
        "IBR_DATA_DIR": str(tmp_path),
        "IBR_GITHUB_PAT": "dummy",
        "IBR_FEED_TOKEN": TOKEN,
    }
    service_dir = Path(__file__).resolve().parents[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1",
         "--port", str(port), "--lifespan", "off", "--log-level", "info"],
        cwd=str(service_dir), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            if proc.poll() is not None:
                pytest.fail(f"uvicorn exited early:\n{proc.communicate()[0]}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                if time.monotonic() > deadline:
                    pytest.fail("uvicorn did not become ready in time")
                time.sleep(0.2)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/{TOKEN}/leitsaetze.atom", timeout=5) as response:
            assert response.status == 200
            assert response.headers["content-type"].startswith("application/atom+xml")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/{WRONG}/leitsaetze.atom", timeout=5)
            pytest.fail("wrong token unexpectedly served")
        except urllib.error.HTTPError as error:
            assert error.code == 404
    finally:
        proc.terminate()
        try:
            out = proc.communicate(timeout=15)[0]
        except subprocess.TimeoutExpired:
            proc.kill()
            out = proc.communicate()[0]
    assert TOKEN not in out, f"feed token leaked to uvicorn access log:\n{out}"
    assert WRONG not in out, f"request token leaked to uvicorn access log:\n{out}"
    # uvicorn percent-quotes the path ('<' -> '%3C'), so decode before counting.
    assert urllib.parse.unquote(out).count(REDACTED) >= 2
    assert '" 200 OK' in out
    assert '" 404 Not Found' in out
