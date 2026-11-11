"""Owned, local-only Docker resources for the T3 HTTP proof (stdlib only)."""

import ipaddress
import json
import re
import subprocess
import uuid

OWNER_LABEL = "owui.t3.run"


def sh(cmd, timeout=120):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


class ScratchResources:
    def __init__(self, runner=sh):
        self.runner = runner
        self.run_id = uuid.uuid4().hex
        self.name = f"litellm-t3-{self.run_id}"
        self.network_name = f"{self.name}-net"
        self.container_id = None
        self.network_id = None
        self.attempted = set()
        self.gateway = None
        self.source = None

    def command(self, *args, timeout=120):
        result = self.runner(["docker", *args], timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"docker {' '.join(args[:2])}: {result.stderr.strip()[:500]}")
        return result.stdout.strip()

    def inspect(self, kind, ref):
        result = self.runner(["docker", kind, "inspect", ref], timeout=60)
        if result.returncode:
            # Do not interpret daemon/permission failures as absence.
            if "No such" in result.stderr or f"network {ref} not found" in result.stderr:
                return None
            raise RuntimeError(f"inspect {kind}: {result.stderr.strip()[:500]}")
        return json.loads(result.stdout)[0]

    def refuse_collision(self, kind, name):
        if self.inspect(kind, name) is not None:
            raise RuntimeError(f"refusing existing {kind} name: {name}")

    def create_network(self):
        self.refuse_collision("container", self.name)
        self.refuse_collision("network", self.network_name)
        self.attempted.add("network")
        self.network_id = self.command(
            "network", "create", "--driver", "bridge", "--internal",
            "--label", f"{OWNER_LABEL}={self.run_id}", self.network_name,
        )
        info = self.inspect("network", self.network_id)
        if info is None or not self.owned("network", info) or not info.get("Internal"):
            raise RuntimeError("scratch network ownership/isolation mismatch")
        self.gateway = info["IPAM"]["Config"][0]["Gateway"]
        ipaddress.IPv4Address(self.gateway)
        return self.gateway

    def create_container(self, image, config, callbacks, token, cap_base, master_key):
        self.refuse_collision("container", self.name)
        self.attempted.add("container")
        args = [
            "container", "create", "--pull", "never", "--name", self.name,
            "--label", f"{OWNER_LABEL}={self.run_id}", "--network", self.network_id,
            "-e", f"LITELLM_MASTER_KEY={master_key}",
            "-e", f"ZAI_API_BASE={cap_base}",
            "-e", "LITELLM_LOCAL_MODEL_COST_MAP=True",
            "-e", "LITELLM_TELEMETRY=False",
            "-v", f"{config}:/app/config.yaml:ro",
        ]
        for provider in ("ZAI", "OPENAI", "DEEPSEEK", "OPENROUTER", "GEMINI", "GROQ", "MISTRAL", "MISTRAL_VIBE"):
            args += ["-e", f"{provider}_API_KEY={token}"]
        for path in callbacks:
            args += ["-v", f"{path}:/app/{path.name}:ro"]
        args += [image, "--config", "/app/config.yaml", "--port", "4000"]
        self.container_id = self.command(*args)
        self.command("container", "start", self.container_id)
        info = self.inspect("container", self.container_id)
        if info is None or not self.owned("container", info):
            raise RuntimeError("scratch container ownership mismatch")
        self.source = info["NetworkSettings"]["Networks"][self.network_name]["IPAddress"]
        ipaddress.IPv4Address(self.source)
        if info["HostConfig"].get("PortBindings"):
            raise RuntimeError("scratch proxy must not publish host ports")

    def owned(self, kind, info):
        if not info:
            return False
        labels = info.get("Config", {}).get("Labels", {}) if kind == "container" else info.get("Labels", {})
        return (labels or {}).get(OWNER_LABEL) == self.run_id

    def cleanup(self):
        errors = []
        for kind, ref, name in (
            ("container", self.container_id, self.name),
            ("network", self.network_id, self.network_name),
        ):
            if kind not in self.attempted:
                continue
            try:
                # A create timeout can leave a resource without returning its ID.
                # Recover by name only when the unguessable run label also matches.
                info = self.inspect(kind, ref or name)
                if info is None:
                    continue
                if not self.owned(kind, info):
                    raise RuntimeError(f"refusing cleanup of unowned {kind}: {ref or name}")
                resource_id = info["Id"]
                if not re.fullmatch(r"[0-9a-f]{64}", resource_id):
                    raise RuntimeError(f"invalid {kind} ID")
                args = [kind, "rm"] + (["-f"] if kind == "container" else []) + [resource_id]
                self.command(*args, timeout=60)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    # This entry point runs only inside the pinned image (PyYAML preinstalled).
    import sys
    import yaml

    with open(sys.argv[1]) as config_file:
        print(json.dumps(yaml.safe_load(config_file)))
