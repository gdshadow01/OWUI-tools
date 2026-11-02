from typing import Optional

from litellm.integrations.custom_logger import CustomLogger

# NOTE: same minimal-import pattern as strip_thinking.py — no proxy_server
# imports, so the module stays unit-testable with a stubbed CustomLogger.

# The two completion call-type forms this hook engages on: the real proxy
# route presents 'acompletion', in-process callers present 'completion'
# (same runtime-proven contract as enforce_effort_policy, v1.103.2).
_COMPLETION_CALL_TYPES = ("completion", "acompletion")

# Message-body keys carrying replayed reasoning. `reasoning` is the Hermes
# client carrier (observed live: Mistral 422 extra_forbidden at
# body.messages.N.assistant.reasoning); `reasoning_content` is the
# deepseek/openrouter-style twin, rejected by the same strict schema.
_MSG_REASONING_KEYS = ("reasoning", "reasoning_content")


class StripMistralReasoningHandler(CustomLogger):
    """Pop replayed reasoning from request bodies for mistral-backed groups.

    WHY THIS EXISTS (2026-11-02): Mistral Vibe (api.mistral.ai) runs a strict
    request schema — any assistant message carrying a `reasoning` key is
    rejected with HTTP 422 `extra_forbidden`, and a body-level `reasoning`
    key (e.g. `false`) is rejected the same way. Hermes clients replay their
    own past reasoning traces in conversation history, so EVERY multi-turn
    request on a mistral-backed lane died at the primary: either Hermes's
    client-side auto-recovery absorbed it (one doomed round trip, manual
    aliases) or — worse — the proxy-side 2-hop fallback chain silently
    absorbed it (managed aliases), serving long conversations from the Z.AI
    heartbeat twin while paging gotify (2026-11-02 incident: hermes-execute /
    hermes-plan BadRequestError fallbacks 11:14–11:55 CET).

    This is the SAME transform Hermes performs in its own client-side
    auto-recovery (strip reasoning, retry) — relocated to the proxy so it
    happens on the first attempt, before routing, for every client. Current-
    turn reasoning is untouched: `reasoning_effort` still flows from each
    deployment's config.yml extra_body; only PAST-turn scratchpads are
    dropped from the replayed history. Z.AI accepts reasoning echo-back and
    keeps receiving it — this hook never touches zai-backed groups.

    Scope: ONLY the groups enumerated in MISTRAL_STRICT_GROUPS (exact
    membership on the client-facing group name, no substrings) — keep in
    sync with config.yml deployments whose litellm_params.model starts with
    `mistral/` (test_strip_mistral_reasoning.py enforces this in both
    directions). Also pops body-level and extra_body-nested reasoning keys
    for the same groups (same strict-schema class; harmless if absent).

    Registered BEFORE enforce_effort_policy so managed mistral groups lose
    the whole `reasoning` carrier before the effort policy evaluates it —
    popping the carrier removes any effort shadowing it could carry, so the
    config-owns-effort invariant is preserved. Idempotent; mutates `data`
    in place; logs alias + counts only, never message content.
    """

    # mistral-backed model groups (client-facing names) — keep in sync with
    # litellm/config/config.yml (`model: mistral/...` deployments).
    MISTRAL_STRICT_GROUPS = frozenset(
        {
            "hermes-plan",
            "hermes-execute",
            "hermes-tasks",
            "hermes-vision",
            "mistral-vibe-glm-5.3-flash(0.15/0.5)",
            "mistral-vibe-deepseek-v4.1-flash(0.3/1.2)",
            "mistral-vibe-glm-5.3(1.4/4.4)",
            "mistral-vibe-mistral-large-4(0.7/2.1)",
            # legacy direct-Mistral twins (same api.mistral.ai strict schema):
            "mistral-mistral-large(0.5/1.5)",
            "mistral-mistral-codestral(0.3/0.9)",
            "mistral-mistral-small(0.1/0.3)",
            "mistral-mistral-medium(0.4/2)",
        }
    )

    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type,
    ) -> Optional[dict]:
        if not isinstance(data, dict):
            return data
        model = data.get("model")
        # Exact membership only — lookalike names (openrouter-*,
        # zai-backed twins, bare zai-glm-*/glm-*) must never match.
        if not isinstance(model, str) or model not in self.MISTRAL_STRICT_GROUPS:
            return data
        call_type_name = getattr(call_type, "value", call_type)
        if call_type_name not in _COMPLETION_CALL_TYPES:
            return data

        dropped = 0

        # 1. Message-level replayed reasoning (assistant messages only —
        #    the observed carrier; user/tool messages are never touched).
        messages = data.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                if message.get("role") != "assistant":
                    continue
                for key in _MSG_REASONING_KEYS:
                    if key in message:
                        message.pop(key, None)
                        dropped += 1

        # 2. Body-level `reasoning` — mistral forbids the key entirely
        #    (probe-verified 422 on `reasoning: false`), any value form.
        if "reasoning" in data:
            data.pop("reasoning", None)
            dropped += 1

        # 3. extra_body-nested reasoning keys — extra_body merges into the
        #    outbound request body, same strict-schema class. Other
        #    extra_body keys are preserved (only the carriers are ours).
        extra_body = data.get("extra_body")
        if isinstance(extra_body, dict):
            for key in _MSG_REASONING_KEYS:
                if key in extra_body:
                    extra_body.pop(key, None)
                    dropped += 1

        if dropped:
            # Alias + count only, never message content.
            print(
                "[strip_mistral_reasoning] dropped",
                dropped,
                "replayed-reasoning carrier(s) for",
                model,
            )
        return data  # MUST return data, even if unchanged


# Module-level instance — this is what config.yml references
# (litellm_settings.callbacks: strip_mistral_reasoning.strip_mistral_reasoning_handler).
strip_mistral_reasoning_handler = StripMistralReasoningHandler()
