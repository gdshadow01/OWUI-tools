"""Negative T3 ownership/isolation checks; stdlib-only, no external upstreams."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import test_enforce_effort_policy_proxy_http as harness
from proxy_http_resources import OWNER_LABEL, ScratchResources


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.runner = Mock()
        self.owner = ScratchResources(self.runner)

    def info(self, kind, owned=True):
        labels = {OWNER_LABEL: self.owner.run_id if owned else "some-other-run"}
        return {"Id": "a" * 64, **({"Config": {"Labels": labels}} if kind == "container" else {"Labels": labels})}

    def test_unique_names(self):
        other = ScratchResources()
        self.assertNotEqual(self.owner.name, other.name)
        self.assertNotEqual(self.owner.run_id, other.run_id)

    def test_collision_refuses_even_matching_label(self):
        with patch.object(self.owner, "inspect", return_value=self.info("container")):
            with self.assertRaisesRegex(RuntimeError, "refusing existing"):
                self.owner.create_network()
        self.owner.cleanup()
        self.runner.assert_not_called()

    def test_network_collision_does_not_delete(self):
        with patch.object(self.owner, "inspect", side_effect=[None, self.info("network")]):
            with self.assertRaisesRegex(RuntimeError, "refusing existing"):
                self.owner.create_network()
        self.owner.cleanup()
        self.runner.assert_not_called()

    def test_container_collision_after_network_does_not_claim_it(self):
        with patch.object(self.owner, "inspect", return_value=self.info("container")):
            with self.assertRaisesRegex(RuntimeError, "refusing existing"):
                self.owner.create_container("image", "config", [], "token", "base", "master")
        self.assertNotIn("container", self.owner.attempted)
        self.runner.assert_not_called()

    def test_cleanup_unowned_rejected_and_network_still_attempted(self):
        self.owner.attempted = {"container", "network"}
        with patch.object(self.owner, "inspect", side_effect=[self.info("container", False), self.info("network")]), patch.object(self.owner, "command") as cmd:
            with self.assertRaisesRegex(RuntimeError, "unowned container"):
                self.owner.cleanup()
        cmd.assert_called_once_with("network", "rm", "a" * 64, timeout=60)

    def test_partial_create_recovered_by_label_and_removed_by_id(self):
        self.owner.attempted = {"container"}
        with patch.object(self.owner, "inspect", return_value=self.info("container")) as inspect, patch.object(self.owner, "command") as cmd:
            self.owner.cleanup()
        inspect.assert_called_once_with("container", self.owner.name)
        cmd.assert_called_once_with("container", "rm", "-f", "a" * 64, timeout=60)

    def test_failed_container_remove_still_attempts_network(self):
        self.owner.attempted = {"container", "network"}
        with patch.object(self.owner, "inspect", side_effect=[self.info("container"), self.info("network")]), patch.object(self.owner, "command", side_effect=[RuntimeError("remove failed"), ""]) as cmd:
            with self.assertRaisesRegex(RuntimeError, "remove failed"):
                self.owner.cleanup()
        self.assertEqual(cmd.call_count, 2)

    def test_inspect_daemon_error_is_not_absence(self):
        self.runner.return_value = subprocess.CompletedProcess([], 1, "", "permission denied")
        with self.assertRaisesRegex(RuntimeError, "permission denied"):
            self.owner.refuse_collision("container", "name")

    def test_absent_cleanup_is_idempotent(self):
        self.owner.attempted = {"container", "network"}
        self.runner.side_effect = [subprocess.CompletedProcess([], 1, "", "No such container: c"), subprocess.CompletedProcess([], 1, "", f"network {self.owner.network_name} not found")]
        self.owner.cleanup()
        self.assertEqual(self.runner.call_count, 2)

    def test_invalid_cleanup_id_fails_closed(self):
        self.owner.attempted = {"container"}
        info = self.info("container")
        info["Id"] = "some-name"
        with patch.object(self.owner, "inspect", return_value=info), patch.object(self.owner, "command") as cmd:
            with self.assertRaisesRegex(RuntimeError, "invalid container ID"):
                self.owner.cleanup()
        cmd.assert_not_called()


    def test_network_create_timeout_recovers_only_owned_resource(self):
        self.runner.side_effect = [
            subprocess.CompletedProcess([], 1, "", "No such container: c"),
            subprocess.CompletedProcess([], 1, "", f"network {self.owner.network_name} not found"),
            subprocess.TimeoutExpired("docker network create", 120),
            subprocess.CompletedProcess([], 0, json.dumps([self.info("network")]), ""),
            subprocess.CompletedProcess([], 0, "a" * 64, ""),
        ]
        with self.assertRaises(subprocess.TimeoutExpired):
            self.owner.create_network()
        self.owner.cleanup()
        self.assertEqual(self.runner.call_args.args[0], ["docker", "network", "rm", "a" * 64])

    def test_container_create_timeout_recovers_only_owned_resource(self):
        self.owner.network_id = "b" * 64
        self.runner.side_effect = [
            subprocess.CompletedProcess([], 1, "", "No such container: c"),
            subprocess.TimeoutExpired("docker container create", 120),
            subprocess.CompletedProcess([], 0, json.dumps([self.info("container")]), ""),
            subprocess.CompletedProcess([], 0, "a" * 64, ""),
        ]
        with self.assertRaises(subprocess.TimeoutExpired):
            self.owner.create_container("image", Path("/config"), [], "token", "base", "master")
        create_args = self.runner.call_args_list[1].args[0]
        self.assertIn("never", create_args)
        self.assertNotIn("-p", create_args)
        self.owner.cleanup()
        self.assertEqual(self.runner.call_args.args[0], ["docker", "container", "rm", "-f", "a" * 64])


class CaptureTests(unittest.TestCase):
    def setUp(self):
        harness.CAPTURES.clear()
        self.server, self.thread = harness.start_capture_server("127.0.0.1", "test-token")
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"

    def tearDown(self):
        harness.stop_capture_server(self.server, self.thread)
        self.assertFalse(self.thread.is_alive())
        self.assertEqual(self.server.fileno(), -1)

    def request(self, token=None):
        req = harness.urllib.request.Request(self.url, data=b'{"messages":[]}', headers={"Authorization": f"Bearer {token}"})
        try:
            with harness.LOCAL_HTTP.open(req, timeout=3) as resp:
                return resp.status
        except harness.urllib.error.HTTPError as exc:
            return exc.code

    def test_binds_only_requested_interface(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_source_unset_fails_closed(self):
        self.assertEqual(self.request("test-token"), 403)
        self.assertFalse(harness.CAPTURES)

    def test_wrong_or_missing_token_not_captured(self):
        self.server.allowed_source = "127.0.0.1"
        for token in (None, "wrong"):
            self.assertEqual(self.request(token), 403)
        self.assertFalse(harness.CAPTURES)

    def test_wrong_source_not_captured(self):
        self.server.allowed_source = "192.0.2.1"
        self.assertEqual(self.request("test-token"), 403)
        self.assertFalse(harness.CAPTURES)

    def test_authorized_request_captured(self):
        self.server.allowed_source = "127.0.0.1"
        self.assertEqual(self.request("test-token"), 200)
        self.assertEqual(len(harness.CAPTURES), 1)


class OrchestrationTests(unittest.TestCase):
    def invoke(self, failure=None, keep=False):
        with tempfile.TemporaryDirectory() as out, contextlib.ExitStack() as stack:
            owner = Mock(name="owner")
            owner.name = "test-owned-run"
            owner.network_name = "test-owned-network"
            owner.run_id = "test-run"
            owner.container_id = "a" * 64
            owner.network_id = "b" * 64
            owner.source = "127.0.0.1"
            owner.create_network.return_value = "127.0.0.1"
            server, thread = harness.start_capture_server("127.0.0.1", "test-token")
            self.addCleanup(harness.stop_capture_server, server, thread)
            stack.enter_context(patch.object(harness, "ScratchResources", return_value=owner))
            stack.enter_context(patch.object(harness, "start_capture_server", return_value=(server, thread)))
            stop = stack.enter_context(patch.object(harness, "stop_capture_server", wraps=harness.stop_capture_server))
            stack.enter_context(patch.object(harness, "sh", return_value=subprocess.CompletedProcess([], 0, harness.IMAGE_DIGEST, "")))
            for name in ("wait_health", "run_proofs", "log_evidence_checks", "drift_checks"):
                stack.enter_context(patch.object(harness, name, return_value=True))
            if failure == "config":
                stack.enter_context(patch.object(harness, "build_scratch_config", side_effect=OSError("config failed")))
            if failure == "start":
                owner.create_container.side_effect = RuntimeError("start failed")
            if failure == "health":
                stack.enter_context(patch.object(harness, "wait_health", return_value=False))
            if failure == "proof":
                stack.enter_context(patch.object(harness, "run_proofs", side_effect=RuntimeError("proof failed")))
            if failure == "evidence":
                stack.enter_context(patch.object(harness, "write_captures", side_effect=OSError("disk full")))
            if failure == "cleanup":
                owner.cleanup.side_effect = RuntimeError("cleanup failed")
            with contextlib.redirect_stdout(io.StringIO()):
                result = harness.main(["--out", out] + (["--keep"] if keep else []))
            stop.assert_called_once_with(server, thread)
            self.assertEqual(server.fileno(), -1)
            if keep:
                owner.cleanup.assert_not_called()
            else:
                owner.cleanup.assert_called_once()
            summary = json.loads((Path(out) / owner.name / "summary.json").read_text())
            self.assertEqual(summary["status"], "FAIL" if failure else "PASS")
            return result

    def test_failure_cleanup_matrix(self):
        for stage in ("config", "start", "health", "proof", "evidence", "cleanup"):
            with self.subTest(stage=stage):
                self.assertNotEqual(self.invoke(stage), 0)

    def test_success_cleanup(self):
        self.assertEqual(self.invoke(), 0)

    def test_keep_still_closes_capture(self):
        self.assertEqual(self.invoke(keep=True), 0)

    def test_pytest_entrypoint_ignores_ambient_flags(self):
        with patch.object(sys, "argv", ["pytest", "-q", "--maxfail=1"]), patch.object(harness, "main", return_value=0) as main:
            harness.test_proxy_http_real_boundary(Path("/dummy"))
        main.assert_called_once_with(["--out", "/dummy"])

    def test_failed_early_case_is_not_overwritten(self):
        harness.RESULTS.clear()
        with patch.object(harness, "http_post", return_value=(200, "ok")), patch.object(harness, "captured_since", return_value=[{"body": {}}]), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(harness.run_proofs())
        self.assertTrue(any(r["status"] == "FAIL" for r in harness.RESULTS))


if __name__ == "__main__":
    unittest.main()
