"""Opt-in local Docker negatives: T3_DOCKER_TESTS=1 python3 -m unittest ...

Uses only the already-local pinned image, no pulls and no external requests.
"""

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_enforce_effort_policy_proxy_http as harness
from proxy_http_resources import ScratchResources, sh


@unittest.skipUnless(os.environ.get("T3_DOCKER_TESTS") == "1", "opt-in local Docker negatives")
class DockerIsolationTests(unittest.TestCase):
    def test_real_collision_preserves_sentinel(self):
        sentinel = ScratchResources()
        # Create, but do not start: this script owns its immutable ID and label.
        sentinel.attempted.add("container")
        try:
            sentinel.container_id = sentinel.command(
                "container", "create", "--pull", "never", "--network", "none",
                "--name", sentinel.name, "--label", f"owui.t3.run={sentinel.run_id}",
                harness.IMAGE,
            )
            intruder = ScratchResources()
            intruder.name = sentinel.name
            with self.assertRaisesRegex(RuntimeError, "refusing existing"):
                intruder.create_network()
            intruder.cleanup()
            info = sentinel.inspect("container", sentinel.container_id)
            self.assertEqual(info["Id"], sentinel.container_id)
            self.assertTrue(sentinel.owned("container", info))
        finally:
            sentinel.cleanup()

    def test_real_bridge_listener_and_no_egress_route(self):
        owner = ScratchResources()
        server = thread = None
        try:
            gateway = owner.create_network()
            server, thread = harness.start_capture_server(gateway, "sk-test-token")
            self.assertEqual(server.server_address[0], gateway)
            owner.attempted.add("container")
            owner.container_id = owner.command(
                "container", "create", "--pull", "never", "--name", owner.name,
                "--label", f"owui.t3.run={owner.run_id}", "--network", owner.network_id,
                "--entrypoint", "sleep", harness.IMAGE, "300",
            )
            owner.command("container", "start", owner.container_id)
            info = owner.inspect("container", owner.container_id)
            server.allowed_source = info["NetworkSettings"]["Networks"][owner.network_name]["IPAddress"]
            self.assertFalse(info["HostConfig"].get("PortBindings"))
            # Read kernel routes without attempting any Internet connection.
            routes = owner.command("exec", owner.container_id, "cat", "/proc/net/route")
            self.assertFalse(any(line.split()[1] == "00000000" for line in routes.splitlines()[1:]))
            # Execute a file, not inline code, in the owned container.
            with tempfile.TemporaryDirectory() as tmp:
                probe = Path(tmp) / "probe.py"
                probe.write_text(
                    "import urllib.request, urllib.error\n"
                    f"url = 'http://{gateway}:{server.server_port}/v1/chat/completions'\n"
                    "opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))\n"
                    "for token, expected in [('wrong', 403), ('sk-test-token', 200)]:\n"
                    "    req = urllib.request.Request(url, data=b'{}', headers={'Authorization': 'Bearer ' + token})\n"
                    "    try:\n"
                    "        with opener.open(req, timeout=3) as response: code = response.status\n"
                    "    except urllib.error.HTTPError as exc: code = exc.code\n"
                    "    assert code == expected, (code, expected)\n"
                )
                owner.command("cp", str(probe), f"{owner.container_id}:/tmp/t3_probe.py")
                owner.command("exec", owner.container_id, "python", "/tmp/t3_probe.py")
            # Host LAN address is not a capture listener; test localhost only,
            # which must also be closed on the gateway-bound ephemeral port.
            import socket
            with socket.socket() as sock:
                sock.settimeout(1)
                self.assertNotEqual(sock.connect_ex(("127.0.0.1", server.server_port)), 0)
        finally:
            if server:
                harness.stop_capture_server(server, thread)
            owner.cleanup()
        self.assertIsNone(owner.inspect("container", owner.container_id))
        self.assertIsNone(owner.inspect("network", owner.network_id))

    def test_real_cleanup_failure_stages(self):
        for stage in ("config", "start", "health", "proof", "evidence"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as out, contextlib.ExitStack() as stack:
                owner = ScratchResources()
                stack.enter_context(patch.object(harness, "ScratchResources", return_value=owner))
                stack.enter_context(patch.object(harness, "wait_health", return_value=stage != "health"))
                stack.enter_context(patch.object(harness, "run_proofs", return_value=True))
                stack.enter_context(patch.object(harness, "log_evidence_checks", return_value=True))
                stack.enter_context(patch.object(harness, "drift_checks", return_value=True))
                if stage == "config":
                    stack.enter_context(patch.object(harness, "build_scratch_config", side_effect=OSError("injected config failure")))
                if stage == "start":
                    real_command = owner.command
                    def command(*args, **kwargs):
                        if args[:2] == ("container", "start"):
                            raise RuntimeError("injected start failure after successful create")
                        return real_command(*args, **kwargs)
                    stack.enter_context(patch.object(owner, "command", side_effect=command))
                if stage == "proof":
                    stack.enter_context(patch.object(harness, "run_proofs", side_effect=RuntimeError("injected proof failure")))
                if stage == "evidence":
                    stack.enter_context(patch.object(harness, "write_captures", side_effect=OSError("injected disk failure")))
                original_start = harness.start_capture_server
                listeners = []
                def start(*args):
                    result = original_start(*args)
                    listeners.append(result)
                    return result
                stack.enter_context(patch.object(harness, "start_capture_server", side_effect=start))
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertNotEqual(harness.main(["--out", out]), 0)
                    self.assertEqual(len(listeners), 1)
                    server, thread = listeners[0]
                    self.assertEqual(server.fileno(), -1)
                    self.assertFalse(thread.is_alive())
                    if owner.container_id:
                        self.assertIsNone(owner.inspect("container", owner.container_id))
                    self.assertIsNone(owner.inspect("network", owner.network_id))
                    summary = json.loads((Path(out) / owner.name / "summary.json").read_text())
                    self.assertEqual(summary["status"], "FAIL")
                finally:
                    owner.cleanup()


if __name__ == "__main__":
    unittest.main()
