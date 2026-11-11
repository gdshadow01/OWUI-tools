"""Proxy-side client-effort enforcement for managed hermes-* aliases (plan 261031).

WHY THIS EXISTS
At the pinned litellm v1.103.2 the router deliberately gives CLIENT
`reasoning_effort` precedence over per-deployment configured effort
(Router._deployment_params_with_request_reasoning_override pops the
deployment's effort carriers whenever the request carries a non-None
top-level `reasoning_effort` — the literal string "none" counts as an
override too). Hermes clients send `reasoning_effort: medium` from their
restored profile settings, which silently defeated every configured
effort on the managed aliases. This pre-call hook runs BEFORE routing
(proxy common_request_processing.py invokes async_pre_call_hook before
route_request), so stripping the client carriers here lets each selected
deployment — including every fallback hop — supply effort from its own
config.yml entry. config.yml is the sole source of effort levels; this
module contains NO effort level constants.

CALL-TYPE CONTRACT (runtime-proven at the v1.103.2 pin, 2026-10-31)
The REAL proxy /v1/chat/completions route presents call_type='acompletion'
(plain str) to callbacks registered under litellm_settings.callbacks; the
original `call_type != "completion"` gate therefore silently early-returned
on every real-path request (zero hook log lines while sibling callbacks
without a gate worked — see Kanban t_6ce36901 evidence: instrumented
rc_probe callback logged call_type='acompletion' for every real-path
probe on the pinned image). In-process callers (unit tests, the T2
runtime proof) present 'completion'. Exactly these two completion forms
are accepted — a litellm CallTypes enum member is normalized via .value
if ever handed as the enum — and every other call type (text_completion,
embeddings, moderation, ...) stays byte-untouched.

SCOPE
Exactly the ten managed aliases in MANAGED_GROUPS, matched by exact
membership on `data["model"]` (hooks see the client-facing group name
pre-routing). No substrings, no prefixes — the fleet has lookalike
names. Ordinary/manual aliases stay client-controlled and pass through
byte-identically.

POLICY (strip-vs-reject matrix, plan 261031 D1/D3/D4)
- `reasoning_effort` (any value)  -> popped.
- `thinking` (any form)           -> popped.
- `reasoning` (dict)              -> its "effort" sub-key popped; if any
  other sub-keys remain the request is REJECTED (a residual client
  `reasoning` object per-key-shadows the deployment's configured
  `extra_body.reasoning` on the chatgpt bridge); if nothing remains the
  key is removed. Non-dict values are rejected as unsupported forms.
- `output_config` (dict)          -> its "effort" sub-key popped, all
  other sub-keys preserved (folds as a distinct extra_body key no
  deployment uses); key removed once empty. Non-dict values rejected.
- `extra_body` (literal dict)     -> nested effort carriers popped
  (reasoning_effort, thinking, reasoning.effort, output_config.effort);
  a literal client extra_body WHOLESALE-REPLACES the deployment
  extra_body at the router merge, so ANY residual content is REJECTED
  and an emptied one is removed entirely (even `extra_body: {}` would
  kill the configured effort). Non-dict values rejected.
- everything else                 -> untouched.

REJECTIONS raise fastapi.HTTPException(400): the pre-call loop only
catches SensitiveDataRouteException, everything else propagates and
aborts the request; the surfaced client-visible status is 400. Ambiguous
forms are never silently permitted to bypass the policy.

The hook mutates `data` in place and returns it, is idempotent (hooks
can re-run on pre-call retry paths), never reads config.yml at runtime,
and logs at most the alias and carrier names — never message content.
"""

from typing import Optional

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger

# The two completion call-type forms this hook engages on: the real proxy
# route presents 'acompletion', in-process callers present 'completion'
# (see CALL-TYPE CONTRACT in the module docstring).
_COMPLETION_CALL_TYPES = ("completion", "acompletion")

# Client-facing model group names under centrally enforced effort policy.
# EXACT membership match on data["model"] only. Keep in sync with
# litellm/config/config.yml — test_enforce_effort_policy.py enforces the
# sync in both directions (every member exists; every hermes-* alias in
# config.yml is consciously classified here or as a -zai rollback twin).
MANAGED_GROUPS = frozenset(
    {
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
    }
)

# Carrier key names (NOT effort levels — levels live only in config.yml).
_EFFORT = "effort"
_REASONING_EFFORT = "reasoning_effort"
_THINKING = "thinking"
_REASONING = "reasoning"
_OUTPUT_CONFIG = "output_config"
_EXTRA_BODY = "extra_body"


def _policy_reject(model, carrier, reason):
    """Abort the request with the policy rejection contract (400)."""
    print("[enforce_effort_policy] rejecting", model, carrier, reason)
    raise HTTPException(
        status_code=400,
        detail={
            "error": "managed-alias-effort-policy",
            "model": model,
            "carrier": carrier,
            "reason": reason,
        },
    )


class EffortPolicyHandler(CustomLogger):
    """Strip client effort carriers from managed-alias requests."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type: str,
    ) -> Optional[dict]:
        if not isinstance(data, dict):
            return data
        model = data.get("model")
        # Exact membership only — lookalike names (openrouter-zai-*,
        # -zai twins, bare zai-*/glm-*/deepseek-*) must never match.
        if not isinstance(model, str) or model not in MANAGED_GROUPS:
            return data
        # Real proxy path presents 'acompletion', in-process callers
        # 'completion'; normalize an enum member via .value defensively.
        call_type_name = getattr(call_type, "value", call_type)
        if call_type_name not in _COMPLETION_CALL_TYPES:
            return data

        stripped = []

        # 1. Top-level reasoning_effort — popped regardless of value
        #    (any non-None value overrides deployment config at the
        #    router; even absent, popping is idempotent).
        if _REASONING_EFFORT in data:
            data.pop(_REASONING_EFFORT)
            stripped.append(_REASONING_EFFORT)

        # 2. thinking — popped in every form (dict/bool/other).
        if _THINKING in data:
            data.pop(_THINKING)
            stripped.append(_THINKING)

        # 3. reasoning — dict only. Pop its effort sub-key; a residual
        #    non-effort object would shadow the deployment's configured
        #    extra_body.reasoning per-key on the chatgpt bridge → reject.
        if _REASONING in data:
            reasoning = data[_REASONING]
            if not isinstance(reasoning, dict):
                _policy_reject(model, _REASONING, "unsupported-type")
            reasoning.pop(_EFFORT, None)
            if reasoning:
                _policy_reject(
                    model,
                    _REASONING,
                    "non-effort-residual",
                )
            data.pop(_REASONING)
            stripped.append(_REASONING)

        # 4. output_config — dict only. Pop the effort sub-key, preserve
        #    every other sub-key (folds as a distinct extra_body key no
        #    deployment configures); drop the key once empty.
        if _OUTPUT_CONFIG in data:
            output_config = data[_OUTPUT_CONFIG]
            if not isinstance(output_config, dict):
                _policy_reject(model, _OUTPUT_CONFIG, "unsupported-type")
            output_config.pop(_EFFORT, None)
            if not output_config:
                data.pop(_OUTPUT_CONFIG)
            stripped.append(_OUTPUT_CONFIG)

        # 5. Literal extra_body — wholesale-replaces the deployment
        #    extra_body at the router merge, so it may only contain
        #    effort carriers we can fully remove. Any residual content
        #    → reject; emptied → remove entirely (an empty dict would
        #    still clobber the configured effort).
        if _EXTRA_BODY in data:
            extra_body = data[_EXTRA_BODY]
            if not isinstance(extra_body, dict):
                _policy_reject(model, _EXTRA_BODY, "unsupported-type")
            extra_body.pop(_REASONING_EFFORT, None)
            extra_body.pop(_THINKING, None)
            if _REASONING in extra_body:
                nested_reasoning = extra_body[_REASONING]
                if not isinstance(nested_reasoning, dict):
                    _policy_reject(model, "extra_body.reasoning", "unsupported-type")
                nested_reasoning.pop(_EFFORT, None)
                if nested_reasoning:
                    _policy_reject(model, "extra_body.reasoning", "non-effort-residual")
                extra_body.pop(_REASONING)
            if _OUTPUT_CONFIG in extra_body:
                nested_output_config = extra_body[_OUTPUT_CONFIG]
                if not isinstance(nested_output_config, dict):
                    _policy_reject(model, "extra_body.output_config", "unsupported-type")
                nested_output_config.pop(_EFFORT, None)
                if not nested_output_config:
                    extra_body.pop(_OUTPUT_CONFIG)
            if extra_body:
                _policy_reject(model, _EXTRA_BODY, "residual-keys")
            data.pop(_EXTRA_BODY)
            stripped.append(_EXTRA_BODY)

        if stripped:
            # Alias + carrier names only, never message content.
            print("[enforce_effort_policy] stripped", model, stripped)
        return data  # MUST return data, even if unchanged


# Module-level instance — this is what config.yml references
# (litellm_settings.callbacks: enforce_effort_policy.effort_policy_handler).
effort_policy_handler = EffortPolicyHandler()
