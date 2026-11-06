#!/usr/bin/env python3
"""Mistral Vibe quota runner — prints remaining included allowance as JSON.

Reads MISTRAL_VIBE_KEY from env (name only; the value never appears in code
or output). GET-only against the documented billing endpoints:
  GET https://api.mistral.ai/v1/billing/budget        (x-api-key header)
  GET https://api.mistral.ai/v1/billing/subscription  (x-api-key header)

Stdout: ONE JSON object. Stderr: actionable errors. Non-200 or malformed
responses -> non-zero exit, no partial JSON on stdout.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

BUDGET_URL = "https://api.mistral.ai/v1/billing/budget"
SUBSCRIPTION_URL = "https://api.mistral.ai/v1/billing/subscription"
TIMEOUT_S = 15
ENV_VAR = "MISTRAL_VIBE_KEY"


def die(msg: str) -> None:
    print(f"vibe-quota: {msg}", file=sys.stderr)
    sys.exit(1)


def get_json(url: str) -> dict:
    key = os.environ.get(ENV_VAR)
    if not key:
        die(
            f"environment variable {ENV_VAR} is not set — provide the Mistral "
            "API key via --env-file (see ~/.config/vibe-quota/vibe-quota.env)"
        )
    req = urllib.request.Request(url, method="GET")
    req.add_header("x-api-key", key)
    req.add_header("Accept", "application/json")
    status: int = 0
    body: bytes = b""
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            status = resp.status
            body = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 401:
            die(f"GET {url} returned HTTP 401 — invalid {ENV_VAR}")
        if e.code == 403:
            die(
                f"GET {url} returned HTTP 403 — key may have lost the "
                "see_org_inference_usage permission"
            )
        die(f"GET {url} returned HTTP {e.code}")
    except urllib.error.URLError as e:
        die(f"GET {url} failed: {e.reason}")
    except TimeoutError:
        die(f"GET {url} timed out after {TIMEOUT_S}s")
    if status != 200:
        die(f"GET {url} returned HTTP {status}")
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        die(f"GET {url} returned malformed JSON")


def main() -> None:
    budget = get_json(BUDGET_URL)
    subscription = get_json(SUBSCRIPTION_URL)

    vibe = budget.get("vibe_budget")
    if not isinstance(vibe, dict):
        die("budget response has no vibe_budget object")
    for field in ("usage_percentage", "initial_budget", "currency", "reset_at", "payg_enabled"):
        if field not in vibe:
            die(f"vibe_budget is missing field '{field}'")

    usage_pct = vibe["usage_percentage"]
    initial = vibe["initial_budget"]
    try:
        remaining_pct = 100.0 - float(usage_pct)
        remaining_amount = (1.0 - float(usage_pct) / 100.0) * float(initial)
    except (TypeError, ValueError):
        die("vibe_budget usage_percentage/initial_budget are not numeric")

    out = {
        "plan": subscription.get("plan_name"),
        "vibe": {
            "usage_percentage": usage_pct,
            "initial_budget": initial,
            "currency": vibe["currency"],
            "reset_at": vibe["reset_at"],
            "payg_enabled": vibe["payg_enabled"],
            "remaining_pct": round(remaining_pct, 6),
            "remaining_amount": round(remaining_amount, 6),
        },
        "api_budget": budget.get("api_budget"),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
